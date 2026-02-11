import dataclasses
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Optional, Union

import numpy as np
import jax
import jax.numpy as jnp
from jax import core as jax_core

import sbh
import sbh_gpu


@dataclasses.dataclass(frozen=True)
class SbhGpuTree:
    dim: int
    level: int
    alpha: float
    cpu_tree: object
    gpu_tree: object


@dataclasses.dataclass
class SbhGpuWorker:
    dim: int
    level: int
    alpha: float
    _conn: Connection
    _proc: mp.Process

    def close(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.send(("close",))
        except Exception:
            pass
        try:
            self._conn.close()
        finally:
            self._conn = None
        if self._proc is not None:
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _as_f32(x: np.ndarray, *, copy: bool = False) -> np.ndarray:
    return np.array(x, dtype=np.float32, copy=copy)


def _worker_main(conn: Connection) -> None:
    import numpy as _np
    import sbh as _sbh
    import sbh_gpu as _sbh_gpu

    tree = None
    dim = None
    level = None

    def _get_classes(_dim: int, _level: int):
        if _dim == 2:
            cpu_map = {1: _sbh.Quadtree1, 2: _sbh.Quadtree2, 3: _sbh.Quadtree3}
            gpu_map = {1: _sbh_gpu.QuadtreeGPU1, 2: _sbh_gpu.QuadtreeGPU2, 3: _sbh_gpu.QuadtreeGPU3}
        elif _dim == 3:
            cpu_map = {1: _sbh.Octree1, 2: _sbh.Octree2, 3: _sbh.Octree3}
            gpu_map = {1: _sbh_gpu.OctreeGPU1, 2: _sbh_gpu.OctreeGPU2, 3: _sbh_gpu.OctreeGPU3}
        else:
            raise ValueError(f"Unsupported dim: {_dim}")
        if _level not in cpu_map:
            raise ValueError(f"Unsupported level: {_level}")
        return cpu_map[_level], gpu_map[_level]

    while True:
        msg = conn.recv()
        op = msg[0]
        if op == "close":
            break
        if op == "init":
            _, points, normals, masses, min_corner, max_corner, alpha, level = msg
            dim = points.shape[1]
            level = int(level)
            cpu_cls, gpu_cls = _get_classes(dim, level)
            cpu_tree = cpu_cls(points, normals, masses, min_corner, max_corner, float(alpha))
            tree = gpu_cls(cpu_tree)
            conn.send(("ok", dim, level))
            continue
        if tree is None:
            conn.send(("error", "worker not initialized"))
            continue

        _, eval_op, queries, beta, samples_per_subdomain = msg
        queries = _np.asarray(queries, dtype=_np.float32)
        if eval_op == "brute_force_gravity":
            result, _ = _sbh_gpu.eval_brute_force_gravity(tree, queries)
        elif eval_op == "barnes_hut_gravity":
            result, _ = _sbh_gpu.eval_barnes_hut_gravity(tree, queries, float(beta))
        elif eval_op == "mlpcv_gravity":
            result, _ = _sbh_gpu.multi_level_prefix_control_variate_gravity(
                tree, queries, int(samples_per_subdomain)
            )
        else:
            conn.send(("error", f"Unsupported op: {eval_op}"))
            continue
        conn.send(("ok", _np.asarray(result, dtype=_np.float32)))


def _get_tree_classes(dim: int, level: int):
    if dim == 2:
        cpu_map = {1: sbh.Quadtree1, 2: sbh.Quadtree2, 3: sbh.Quadtree3}
        gpu_map = {1: sbh_gpu.QuadtreeGPU1, 2: sbh_gpu.QuadtreeGPU2, 3: sbh_gpu.QuadtreeGPU3}
    elif dim == 3:
        cpu_map = {1: sbh.Octree1, 2: sbh.Octree2, 3: sbh.Octree3}
        gpu_map = {1: sbh_gpu.OctreeGPU1, 2: sbh_gpu.OctreeGPU2, 3: sbh_gpu.OctreeGPU3}
    else:
        raise ValueError(f"Unsupported dim: {dim}")

    if level not in cpu_map:
        raise ValueError(f"Unsupported level: {level}")

    return cpu_map[level], gpu_map[level]


def build_gpu_tree(
    points: np.ndarray,
    normals: np.ndarray,
    masses: np.ndarray,
    min_corner: Optional[np.ndarray] = None,
    max_corner: Optional[np.ndarray] = None,
    *,
    alpha: float = 1.0,
    level: int = 1,
) -> SbhGpuTree:
    points = _as_f32(points)
    normals = _as_f32(normals)
    masses = _as_f32(masses)

    if points.ndim != 2:
        raise ValueError("points must be (N, dim)")
    dim = points.shape[1]

    if min_corner is None:
        min_corner = points.min(axis=0) - 1e-3
    if max_corner is None:
        max_corner = points.max(axis=0) + 1e-3

    min_corner = _as_f32(min_corner)
    max_corner = _as_f32(max_corner)

    cpu_cls, gpu_cls = _get_tree_classes(dim, level)
    cpu_tree = cpu_cls(points, normals, masses, min_corner, max_corner, float(alpha))
    gpu_tree = gpu_cls(cpu_tree)

    return SbhGpuTree(dim=dim, level=level, alpha=float(alpha), cpu_tree=cpu_tree, gpu_tree=gpu_tree)


def build_gpu_worker(
    points: np.ndarray,
    normals: np.ndarray,
    masses: np.ndarray,
    min_corner: Optional[np.ndarray] = None,
    max_corner: Optional[np.ndarray] = None,
    *,
    alpha: float = 1.0,
    level: int = 1,
) -> SbhGpuWorker:
    # Use a separate process to avoid JAX/CUDA context conflicts.
    points = _as_f32(points, copy=True)
    normals = _as_f32(normals, copy=True)
    masses = _as_f32(masses, copy=True)

    if points.ndim != 2:
        raise ValueError("points must be (N, dim)")
    dim = points.shape[1]

    if min_corner is None:
        min_corner = points.min(axis=0) - 1e-3
    if max_corner is None:
        max_corner = points.max(axis=0) + 1e-3

    min_corner = _as_f32(min_corner, copy=True)
    max_corner = _as_f32(max_corner, copy=True)

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_worker_main, args=(child_conn,), daemon=True)
    proc.start()
    parent_conn.send(("init", points, normals, masses, min_corner, max_corner, float(alpha), int(level)))
    status = parent_conn.recv()
    if status[0] != "ok":
        raise RuntimeError(f"worker init failed: {status}")
    return SbhGpuWorker(dim=dim, level=level, alpha=float(alpha), _conn=parent_conn, _proc=proc)


def _jax_eval(
    tree: Union[SbhGpuTree, SbhGpuWorker],
    queries: jnp.ndarray,
    *,
    op: str,
    beta: Optional[float] = None,
    samples_per_subdomain: Optional[int] = None,
) -> jnp.ndarray:
    if isinstance(queries, jax_core.Tracer):
        raise ValueError("jax_sbh eval_* functions are eager-only (no jit).")

    queries = jnp.asarray(queries, dtype=jnp.float32)
    if queries.ndim != 2 or queries.shape[1] != tree.dim:
        raise ValueError(f"queries must be (M, {tree.dim})")

    q_np = _as_f32(jax.device_get(queries), copy=True)

    if isinstance(tree, SbhGpuWorker):
        tree._conn.send(("eval", op, q_np, beta, samples_per_subdomain))
        status = tree._conn.recv()
        if status[0] != "ok":
            raise RuntimeError(status[1])
        result = status[1]
    else:
        if op == "brute_force_gravity":
            result, _ = sbh_gpu.eval_brute_force_gravity(tree.gpu_tree, q_np)
        elif op == "barnes_hut_gravity":
            result, _ = sbh_gpu.eval_barnes_hut_gravity(tree.gpu_tree, q_np, float(beta))
        elif op == "mlpcv_gravity":
            result, _ = sbh_gpu.multi_level_prefix_control_variate_gravity(
                tree.gpu_tree, q_np, int(samples_per_subdomain)
            )
        else:
            raise ValueError(f"Unsupported op: {op}")

    return jnp.asarray(result, dtype=jnp.float32)


def eval_brute_force_gravity(tree: Union[SbhGpuTree, SbhGpuWorker], queries: jnp.ndarray) -> jnp.ndarray:
    return _jax_eval(tree, queries, op="brute_force_gravity")


def eval_barnes_hut_gravity(
    tree: Union[SbhGpuTree, SbhGpuWorker], queries: jnp.ndarray, *, beta: float = 2.0
) -> jnp.ndarray:
    return _jax_eval(tree, queries, op="barnes_hut_gravity", beta=beta)


def eval_mlpcv_gravity(
    tree: Union[SbhGpuTree, SbhGpuWorker], queries: jnp.ndarray, *, samples_per_subdomain: int = 1
) -> jnp.ndarray:
    return _jax_eval(tree, queries, op="mlpcv_gravity", samples_per_subdomain=samples_per_subdomain)
