import ctypes
import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


_LIB_NAME = "sbh_ffi"
_TARGET_BF = "sbh_bruteforce_gravity"
_TARGET_BH = "sbh_barnes_hut_gravity"
_TARGET_BH_FORCE = "sbh_barnes_hut_force"
_TARGET_BH_3D = "sbh_barnes_hut_gravity_3d"
_TARGET_BH_FORCE_3D = "sbh_barnes_hut_force_3d"
_TARGET_BH_FORCE_SOFT = "sbh_barnes_hut_force_soft"
_TARGET_BH_FORCE_SOFT_3D = "sbh_barnes_hut_force_soft_3d"
_TARGET_SBH = "sbh_stochastic_bh_gravity"
_TARGET_SBH_3D = "sbh_stochastic_bh_gravity_3d"
_TARGET_SBH_FORCE = "sbh_stochastic_bh_force"
_TARGET_SBH_FORCE_3D = "sbh_stochastic_bh_force_3d"
_TARGET_SBH_FORCE_SOFT = "sbh_stochastic_bh_force_soft"
_TARGET_SBH_FORCE_SOFT_3D = "sbh_stochastic_bh_force_soft_3d"
_TARGET_BUILD_OCTREE_L3 = "sbh_build_octree_level3_buffers"
_TARGET_BUILD_OCTREE8 = "sbh_build_octree8_buffers"
_TARGET_BH_FORCE_OCTREE8 = "sbh_barnes_hut_force_octree8"
_TARGET_BH_FORCE_OCTREE8_SOFT = "sbh_barnes_hut_force_octree8_soft"
_TARGET_BH_FORCE_OCTREE8_SOFT_VJP = "sbh_barnes_hut_force_octree8_soft_vjp"
_TARGET_SBH_FORCE_OCTREE8 = "sbh_stochastic_bh_force_octree8"
_TARGET_SBH_FORCE_OCTREE8_SOFT = "sbh_stochastic_bh_force_octree8_soft"

_LEAF3_SIZE: Optional[int] = None
_NODE3_SIZE: Optional[int] = None
_OCTREE_L3_MAX_NODES: Optional[int] = None
_OCTREE8_NODE_SIZE: Optional[int] = None


def _load_library(path: Optional[str] = None) -> ctypes.CDLL:
    if path is None:
        base = os.path.dirname(__file__)
        candidates = [
            os.path.join(base, f"{_LIB_NAME}.so"),
            os.path.join(base, f"lib{_LIB_NAME}.so"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                path = cand
                break
    if path is None:
        raise FileNotFoundError(
            "sbh_ffi shared library not found; build it and copy into python/"
        )
    return ctypes.CDLL(path)


def _get_octree_level3_layout(path: Optional[str] = None) -> tuple[int, int, int]:
    global _LEAF3_SIZE, _NODE3_SIZE, _OCTREE_L3_MAX_NODES
    if _LEAF3_SIZE is not None and _NODE3_SIZE is not None and _OCTREE_L3_MAX_NODES is not None:
        return _LEAF3_SIZE, _NODE3_SIZE, _OCTREE_L3_MAX_NODES

    lib = _load_library(path)
    lib.sbh_leaf3_size.restype = ctypes.c_size_t
    lib.sbh_node3_size.restype = ctypes.c_size_t
    lib.sbh_octree_level3_max_nodes.restype = ctypes.c_size_t
    _LEAF3_SIZE = int(lib.sbh_leaf3_size())
    _NODE3_SIZE = int(lib.sbh_node3_size())
    _OCTREE_L3_MAX_NODES = int(lib.sbh_octree_level3_max_nodes())
    return _LEAF3_SIZE, _NODE3_SIZE, _OCTREE_L3_MAX_NODES


def _get_octree8_layout(path: Optional[str] = None) -> tuple[int, int]:
    global _LEAF3_SIZE, _OCTREE8_NODE_SIZE
    if _LEAF3_SIZE is not None and _OCTREE8_NODE_SIZE is not None:
        return _LEAF3_SIZE, _OCTREE8_NODE_SIZE

    lib = _load_library(path)
    lib.sbh_leaf3_size.restype = ctypes.c_size_t
    lib.sbh_octree8_node_size.restype = ctypes.c_size_t
    _LEAF3_SIZE = int(lib.sbh_leaf3_size())
    _OCTREE8_NODE_SIZE = int(lib.sbh_octree8_node_size())
    return _LEAF3_SIZE, _OCTREE8_NODE_SIZE


def _octree8_total_nodes(max_depth: int) -> int:
    total = 0
    level = 1
    for _ in range(max_depth + 1):
        total += level
        level *= 8
    return total


def register_bruteforce_gravity(path: Optional[str] = None) -> None:
    lib = _load_library(path)
    fn = lib.sbh_bruteforce_gravity_ffi
    jax.ffi.register_ffi_target(
        _TARGET_BF,
        jax.ffi.pycapsule(fn),
        platform="CUDA",
        api_version=1,
    )


def register_build_octree_level3_buffers(path: Optional[str] = None) -> None:
    lib = _load_library(path)
    fn = lib.sbh_build_octree_level3_buffers_ffi
    jax.ffi.register_ffi_target(
        _TARGET_BUILD_OCTREE_L3,
        jax.ffi.pycapsule(fn),
        platform="CUDA",
        api_version=1,
    )


def register_build_octree8_buffers(path: Optional[str] = None) -> None:
    lib = _load_library(path)
    fn = lib.sbh_build_octree8_buffers_ffi
    jax.ffi.register_ffi_target(
        _TARGET_BUILD_OCTREE8,
        jax.ffi.pycapsule(fn),
        platform="CUDA",
        api_version=1,
    )


def register_barnes_hut_gravity(path: Optional[str] = None) -> None:
    lib = _load_library(path)
    fn = lib.sbh_barnes_hut_gravity_ffi
    jax.ffi.register_ffi_target(
        _TARGET_BH,
        jax.ffi.pycapsule(fn),
        platform="CUDA",
        api_version=1,
    )
    fn3 = lib.sbh_barnes_hut_gravity_ffi_3d
    jax.ffi.register_ffi_target(
        _TARGET_BH_3D,
        jax.ffi.pycapsule(fn3),
        platform="CUDA",
        api_version=1,
    )


def register_stochastic_barnes_hut_gravity(path: Optional[str] = None) -> None:
    lib = _load_library(path)
    fn2 = lib.sbh_stochastic_bh_gravity_ffi
    jax.ffi.register_ffi_target(
        _TARGET_SBH,
        jax.ffi.pycapsule(fn2),
        platform="CUDA",
        api_version=1,
    )
    fn3 = lib.sbh_stochastic_bh_gravity_ffi_3d
    jax.ffi.register_ffi_target(
        _TARGET_SBH_3D,
        jax.ffi.pycapsule(fn3),
        platform="CUDA",
        api_version=1,
    )


def register_stochastic_barnes_hut_force(path: Optional[str] = None) -> None:
    lib = _load_library(path)
    fn2 = lib.sbh_stochastic_bh_force_ffi
    jax.ffi.register_ffi_target(
        _TARGET_SBH_FORCE,
        jax.ffi.pycapsule(fn2),
        platform="CUDA",
        api_version=1,
    )
    fn3 = lib.sbh_stochastic_bh_force_ffi_3d
    jax.ffi.register_ffi_target(
        _TARGET_SBH_FORCE_3D,
        jax.ffi.pycapsule(fn3),
        platform="CUDA",
        api_version=1,
    )


def register_softened_stochastic_barnes_hut_force(path: Optional[str] = None) -> None:
    lib = _load_library(path)
    fn2 = lib.sbh_stochastic_bh_force_soft_ffi
    jax.ffi.register_ffi_target(
        _TARGET_SBH_FORCE_SOFT,
        jax.ffi.pycapsule(fn2),
        platform="CUDA",
        api_version=1,
    )
    fn3 = lib.sbh_stochastic_bh_force_soft_ffi_3d
    jax.ffi.register_ffi_target(
        _TARGET_SBH_FORCE_SOFT_3D,
        jax.ffi.pycapsule(fn3),
        platform="CUDA",
        api_version=1,
    )


def register_barnes_hut_force(path: Optional[str] = None) -> None:
    lib = _load_library(path)
    fn = lib.sbh_barnes_hut_force_ffi
    jax.ffi.register_ffi_target(
        _TARGET_BH_FORCE,
        jax.ffi.pycapsule(fn),
        platform="CUDA",
        api_version=1,
    )
    fn3 = lib.sbh_barnes_hut_force_ffi_3d
    jax.ffi.register_ffi_target(
        _TARGET_BH_FORCE_3D,
        jax.ffi.pycapsule(fn3),
        platform="CUDA",
        api_version=1,
    )


def register_softened_barnes_hut_force(path: Optional[str] = None) -> None:
    lib = _load_library(path)
    fn = lib.sbh_barnes_hut_force_soft_ffi
    jax.ffi.register_ffi_target(
        _TARGET_BH_FORCE_SOFT,
        jax.ffi.pycapsule(fn),
        platform="CUDA",
        api_version=1,
    )
    fn3 = lib.sbh_barnes_hut_force_soft_ffi_3d
    jax.ffi.register_ffi_target(
        _TARGET_BH_FORCE_SOFT_3D,
        jax.ffi.pycapsule(fn3),
        platform="CUDA",
        api_version=1,
    )


def register_barnes_hut_force_octree8(path: Optional[str] = None) -> None:
    lib = _load_library(path)
    fn = lib.sbh_barnes_hut_force_octree8_ffi
    jax.ffi.register_ffi_target(
        _TARGET_BH_FORCE_OCTREE8,
        jax.ffi.pycapsule(fn),
        platform="CUDA",
        api_version=1,
    )


def register_softened_barnes_hut_force_octree8(path: Optional[str] = None) -> None:
    lib = _load_library(path)
    fn = lib.sbh_barnes_hut_force_octree8_soft_ffi
    jax.ffi.register_ffi_target(
        _TARGET_BH_FORCE_OCTREE8_SOFT,
        jax.ffi.pycapsule(fn),
        platform="CUDA",
        api_version=1,
    )


def register_softened_barnes_hut_force_octree8_vjp(path: Optional[str] = None) -> None:
    lib = _load_library(path)
    fn = lib.sbh_barnes_hut_force_octree8_soft_vjp_ffi
    jax.ffi.register_ffi_target(
        _TARGET_BH_FORCE_OCTREE8_SOFT_VJP,
        jax.ffi.pycapsule(fn),
        platform="CUDA",
        api_version=1,
    )


def register_stochastic_barnes_hut_force_octree8(path: Optional[str] = None) -> None:
    lib = _load_library(path)
    fn = lib.sbh_stochastic_bh_force_octree8_ffi
    jax.ffi.register_ffi_target(
        _TARGET_SBH_FORCE_OCTREE8,
        jax.ffi.pycapsule(fn),
        platform="CUDA",
        api_version=1,
    )


def register_softened_stochastic_barnes_hut_force_octree8(path: Optional[str] = None) -> None:
    lib = _load_library(path)
    fn = lib.sbh_stochastic_bh_force_octree8_soft_ffi
    jax.ffi.register_ffi_target(
        _TARGET_SBH_FORCE_OCTREE8_SOFT,
        jax.ffi.pycapsule(fn),
        platform="CUDA",
        api_version=1,
    )


def brute_force_gravity(points: jnp.ndarray, masses: jnp.ndarray, queries: jnp.ndarray) -> jnp.ndarray:
    points = jnp.asarray(points, dtype=jnp.float32)
    masses = jnp.asarray(masses, dtype=jnp.float32)
    queries = jnp.asarray(queries, dtype=jnp.float32)

    if points.ndim != 2 or queries.ndim != 2:
        raise ValueError("points and queries must be rank-2")
    if masses.ndim != 1:
        raise ValueError("masses must be rank-1")
    if points.shape[0] != masses.shape[0]:
        raise ValueError("masses length mismatch")
    if points.shape[1] != queries.shape[1]:
        raise ValueError("dim mismatch")

    out_shape = jax.ShapeDtypeStruct((queries.shape[0],), jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_BF,
        result_shape_dtypes=out_shape,
    )(points, masses, queries)


def build_quadtree_buffers(
    points: np.ndarray,
    normals: np.ndarray,
    masses: np.ndarray,
    min_corner: Optional[np.ndarray] = None,
    max_corner: Optional[np.ndarray] = None,
    *,
    alpha: float = 1.0,
    level: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    import sbh

    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    masses = np.asarray(masses, dtype=np.float32)

    if min_corner is None:
        min_corner = points.min(axis=0) - 1e-3
    if max_corner is None:
        max_corner = points.max(axis=0) + 1e-3

    min_corner = np.asarray(min_corner, dtype=np.float32)
    max_corner = np.asarray(max_corner, dtype=np.float32)

    if level == 3:
        tree = sbh.Quadtree3(points, normals, masses, min_corner, max_corner, alpha)
    else:
        raise ValueError("level must be 3 (FFI expects Tree2DNode<1, 3>)")

    leaf_bytes = tree.leaf_data_bytes()
    node_bytes = tree.node_buffer_bytes()
    return leaf_bytes, node_bytes


def build_quadtree_buffers_with_contrib(
    points: np.ndarray,
    normals: np.ndarray,
    masses: np.ndarray,
    min_corner: Optional[np.ndarray] = None,
    max_corner: Optional[np.ndarray] = None,
    *,
    alpha: float = 1.0,
    level: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import sbh

    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    masses = np.asarray(masses, dtype=np.float32)

    if min_corner is None:
        min_corner = points.min(axis=0) - 1e-3
    if max_corner is None:
        max_corner = points.max(axis=0) + 1e-3

    min_corner = np.asarray(min_corner, dtype=np.float32)
    max_corner = np.asarray(max_corner, dtype=np.float32)

    if level == 3:
        tree = sbh.Quadtree3(points, normals, masses, min_corner, max_corner, alpha)
    else:
        raise ValueError("level must be 3 (FFI expects Tree2DNode<1, 3>)")

    leaf_bytes = tree.leaf_data_bytes()
    node_bytes = tree.node_buffer_bytes()
    contrib_bytes = tree.contrib_data_bytes()
    return leaf_bytes, node_bytes, contrib_bytes


def _build_octree_buffers_cpu(
    points: np.ndarray,
    normals: np.ndarray,
    masses: np.ndarray,
    min_corner: np.ndarray,
    max_corner: np.ndarray,
    alpha: float,
    level: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import sbh

    if level != 3:
        raise ValueError("level must be 3 (FFI expects Tree3DNode<1, 3>)")
    tree = sbh.Octree3(points, normals, masses, min_corner, max_corner, alpha)
    return tree.leaf_data_bytes(), tree.node_buffer_bytes(), tree.contrib_data_bytes()


def build_octree_buffers(
    points: jnp.ndarray | np.ndarray,
    normals: jnp.ndarray | np.ndarray,
    masses: jnp.ndarray | np.ndarray,
    min_corner: Optional[jnp.ndarray | np.ndarray] = None,
    max_corner: Optional[jnp.ndarray | np.ndarray] = None,
    *,
    alpha: float = 1.0,
    level: int = 3,
    backend: str = "ffi",
) -> tuple[jnp.ndarray | np.ndarray, jnp.ndarray | np.ndarray]:
    del alpha
    if level != 3:
        raise ValueError("level must be 3 (FFI expects Tree3DNode<1, 3>)")

    if backend == "cpu":
        p = np.asarray(points, dtype=np.float32)
        n = np.asarray(normals, dtype=np.float32)
        m = np.asarray(masses, dtype=np.float32)
        if min_corner is None:
            min_corner = p.min(axis=0) - 1e-3
        if max_corner is None:
            max_corner = p.max(axis=0) + 1e-3
        mn = np.asarray(min_corner, dtype=np.float32)
        mx = np.asarray(max_corner, dtype=np.float32)
        leaf, node, _ = _build_octree_buffers_cpu(p, n, m, mn, mx, 1.0, level)
        return leaf, node
    if backend != "ffi":
        raise ValueError("backend must be 'ffi' or 'cpu'")

    register_build_octree_level3_buffers()
    leaf_size, node_size, max_nodes = _get_octree_level3_layout()

    points = jnp.asarray(points, dtype=jnp.float32)
    normals = jnp.asarray(normals, dtype=jnp.float32)
    masses = jnp.asarray(masses, dtype=jnp.float32)
    if min_corner is None:
        min_corner = jnp.min(points, axis=0) - 1e-3
    if max_corner is None:
        max_corner = jnp.max(points, axis=0) + 1e-3
    min_corner = jnp.asarray(min_corner, dtype=jnp.float32)
    max_corner = jnp.asarray(max_corner, dtype=jnp.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if normals.shape != points.shape:
        raise ValueError("normals shape mismatch")
    if masses.ndim != 1 or masses.shape[0] != points.shape[0]:
        raise ValueError("masses must have shape (N,)")

    n_points = points.shape[0]
    leaf_shape = jax.ShapeDtypeStruct((n_points * leaf_size,), jnp.uint8)
    node_shape = jax.ShapeDtypeStruct((max_nodes * node_size,), jnp.uint8)
    contrib_shape = jax.ShapeDtypeStruct((max_nodes * leaf_size,), jnp.uint8)
    leaf_bytes, node_bytes, _ = jax.ffi.ffi_call(
        _TARGET_BUILD_OCTREE_L3,
        result_shape_dtypes=(leaf_shape, node_shape, contrib_shape),
    )(points, normals, masses, min_corner, max_corner)
    return leaf_bytes, node_bytes


def build_octree_buffers_with_contrib(
    points: jnp.ndarray | np.ndarray,
    normals: jnp.ndarray | np.ndarray,
    masses: jnp.ndarray | np.ndarray,
    min_corner: Optional[jnp.ndarray | np.ndarray] = None,
    max_corner: Optional[jnp.ndarray | np.ndarray] = None,
    *,
    alpha: float = 1.0,
    level: int = 3,
    backend: str = "ffi",
) -> tuple[jnp.ndarray | np.ndarray, jnp.ndarray | np.ndarray, jnp.ndarray | np.ndarray]:
    del alpha
    if level != 3:
        raise ValueError("level must be 3 (FFI expects Tree3DNode<1, 3>)")

    if backend == "cpu":
        p = np.asarray(points, dtype=np.float32)
        n = np.asarray(normals, dtype=np.float32)
        m = np.asarray(masses, dtype=np.float32)
        if min_corner is None:
            min_corner = p.min(axis=0) - 1e-3
        if max_corner is None:
            max_corner = p.max(axis=0) + 1e-3
        mn = np.asarray(min_corner, dtype=np.float32)
        mx = np.asarray(max_corner, dtype=np.float32)
        return _build_octree_buffers_cpu(p, n, m, mn, mx, 1.0, level)
    if backend != "ffi":
        raise ValueError("backend must be 'ffi' or 'cpu'")

    register_build_octree_level3_buffers()
    leaf_size, node_size, max_nodes = _get_octree_level3_layout()

    points = jnp.asarray(points, dtype=jnp.float32)
    normals = jnp.asarray(normals, dtype=jnp.float32)
    masses = jnp.asarray(masses, dtype=jnp.float32)
    if min_corner is None:
        min_corner = jnp.min(points, axis=0) - 1e-3
    if max_corner is None:
        max_corner = jnp.max(points, axis=0) + 1e-3
    min_corner = jnp.asarray(min_corner, dtype=jnp.float32)
    max_corner = jnp.asarray(max_corner, dtype=jnp.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if normals.shape != points.shape:
        raise ValueError("normals shape mismatch")
    if masses.ndim != 1 or masses.shape[0] != points.shape[0]:
        raise ValueError("masses must have shape (N,)")

    n_points = points.shape[0]
    leaf_shape = jax.ShapeDtypeStruct((n_points * leaf_size,), jnp.uint8)
    node_shape = jax.ShapeDtypeStruct((max_nodes * node_size,), jnp.uint8)
    contrib_shape = jax.ShapeDtypeStruct((max_nodes * leaf_size,), jnp.uint8)
    return jax.ffi.ffi_call(
        _TARGET_BUILD_OCTREE_L3,
        result_shape_dtypes=(leaf_shape, node_shape, contrib_shape),
    )(points, normals, masses, min_corner, max_corner)


def build_octree8_buffers(
    points: jnp.ndarray | np.ndarray,
    normals: jnp.ndarray | np.ndarray,
    masses: jnp.ndarray | np.ndarray,
    min_corner: Optional[jnp.ndarray | np.ndarray] = None,
    max_corner: Optional[jnp.ndarray | np.ndarray] = None,
    *,
    max_depth: int = 7,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    register_build_octree8_buffers()
    leaf_size, node_size = _get_octree8_layout()

    points = jnp.asarray(points, dtype=jnp.float32)
    normals = jnp.asarray(normals, dtype=jnp.float32)
    masses = jnp.asarray(masses, dtype=jnp.float32)
    if min_corner is None:
        min_corner = jnp.min(points, axis=0) - 1e-3
    if max_corner is None:
        max_corner = jnp.max(points, axis=0) + 1e-3
    min_corner = jnp.asarray(min_corner, dtype=jnp.float32)
    max_corner = jnp.asarray(max_corner, dtype=jnp.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if normals.shape != points.shape:
        raise ValueError("normals shape mismatch")
    if masses.ndim != 1 or masses.shape[0] != points.shape[0]:
        raise ValueError("masses must have shape (N,)")
    if max_depth < 1 or max_depth > 10:
        raise ValueError("max_depth must be in [1, 10]")

    n_points = points.shape[0]
    total_nodes = _octree8_total_nodes(int(max_depth))
    leaf_shape = jax.ShapeDtypeStruct((n_points * leaf_size,), jnp.uint8)
    node_shape = jax.ShapeDtypeStruct((total_nodes * node_size,), jnp.uint8)
    contrib_shape = jax.ShapeDtypeStruct((total_nodes * leaf_size,), jnp.uint8)
    leaf_bytes, node_bytes, _ = jax.ffi.ffi_call(
        _TARGET_BUILD_OCTREE8,
        result_shape_dtypes=(leaf_shape, node_shape, contrib_shape),
    )(
        points,
        normals,
        masses,
        min_corner,
        max_corner,
        max_depth=np.int32(max_depth),
    )
    return leaf_bytes, node_bytes


def build_octree8_buffers_with_contrib(
    points: jnp.ndarray | np.ndarray,
    normals: jnp.ndarray | np.ndarray,
    masses: jnp.ndarray | np.ndarray,
    min_corner: Optional[jnp.ndarray | np.ndarray] = None,
    max_corner: Optional[jnp.ndarray | np.ndarray] = None,
    *,
    max_depth: int = 7,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    register_build_octree8_buffers()
    leaf_size, node_size = _get_octree8_layout()

    points = jnp.asarray(points, dtype=jnp.float32)
    normals = jnp.asarray(normals, dtype=jnp.float32)
    masses = jnp.asarray(masses, dtype=jnp.float32)
    if min_corner is None:
        min_corner = jnp.min(points, axis=0) - 1e-3
    if max_corner is None:
        max_corner = jnp.max(points, axis=0) + 1e-3
    min_corner = jnp.asarray(min_corner, dtype=jnp.float32)
    max_corner = jnp.asarray(max_corner, dtype=jnp.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if normals.shape != points.shape:
        raise ValueError("normals shape mismatch")
    if masses.ndim != 1 or masses.shape[0] != points.shape[0]:
        raise ValueError("masses must have shape (N,)")
    if max_depth < 1 or max_depth > 10:
        raise ValueError("max_depth must be in [1, 10]")

    n_points = points.shape[0]
    total_nodes = _octree8_total_nodes(int(max_depth))
    leaf_shape = jax.ShapeDtypeStruct((n_points * leaf_size,), jnp.uint8)
    node_shape = jax.ShapeDtypeStruct((total_nodes * node_size,), jnp.uint8)
    contrib_shape = jax.ShapeDtypeStruct((total_nodes * leaf_size,), jnp.uint8)
    return jax.ffi.ffi_call(
        _TARGET_BUILD_OCTREE8,
        result_shape_dtypes=(leaf_shape, node_shape, contrib_shape),
    )(
        points,
        normals,
        masses,
        min_corner,
        max_corner,
        max_depth=np.int32(max_depth),
    )


def barnes_hut_gravity(
    leaf_bytes: jnp.ndarray, node_bytes: jnp.ndarray, queries: jnp.ndarray, *, beta: float = 2.0
) -> jnp.ndarray:
    leaf_bytes = jnp.asarray(leaf_bytes, dtype=jnp.uint8)
    node_bytes = jnp.asarray(node_bytes, dtype=jnp.uint8)
    queries = jnp.asarray(queries, dtype=jnp.float32)

    if queries.ndim != 2 or queries.shape[1] not in (2, 3):
        raise ValueError("queries must be (M, 2) or (M, 3)")

    out_shape = jax.ShapeDtypeStruct((queries.shape[0],), jnp.float32)
    target = _TARGET_BH if queries.shape[1] == 2 else _TARGET_BH_3D
    return jax.ffi.ffi_call(
        target,
        result_shape_dtypes=out_shape,
    )(leaf_bytes, node_bytes, queries, beta=np.float32(beta))


def barnes_hut_force(
    leaf_bytes: jnp.ndarray, node_bytes: jnp.ndarray, queries: jnp.ndarray, *, beta: float = 2.0
) -> jnp.ndarray:
    leaf_bytes = jnp.asarray(leaf_bytes, dtype=jnp.uint8)
    node_bytes = jnp.asarray(node_bytes, dtype=jnp.uint8)
    queries = jnp.asarray(queries, dtype=jnp.float32)

    if queries.ndim != 2 or queries.shape[1] not in (2, 3):
        raise ValueError("queries must be (M, 2) or (M, 3)")

    out_shape = jax.ShapeDtypeStruct((queries.shape[0], queries.shape[1]), jnp.float32)
    target = _TARGET_BH_FORCE if queries.shape[1] == 2 else _TARGET_BH_FORCE_3D
    return jax.ffi.ffi_call(
        target,
        result_shape_dtypes=out_shape,
    )(leaf_bytes, node_bytes, queries, beta=np.float32(beta))


def softened_barnes_hut_force(
    leaf_bytes: jnp.ndarray,
    node_bytes: jnp.ndarray,
    queries: jnp.ndarray,
    *,
    beta: float = 2.0,
    softening_scale: float = 1e-2,
    cutoff_scale: float = 1.0,
) -> jnp.ndarray:
    leaf_bytes = jnp.asarray(leaf_bytes, dtype=jnp.uint8)
    node_bytes = jnp.asarray(node_bytes, dtype=jnp.uint8)
    queries = jnp.asarray(queries, dtype=jnp.float32)

    if queries.ndim != 2 or queries.shape[1] not in (2, 3):
        raise ValueError("queries must be (M, 2) or (M, 3)")
    if softening_scale < 0:
        raise ValueError("softening_scale must be >= 0")
    if cutoff_scale <= 0:
        raise ValueError("cutoff_scale must be > 0")

    out_shape = jax.ShapeDtypeStruct((queries.shape[0], queries.shape[1]), jnp.float32)
    target = _TARGET_BH_FORCE_SOFT if queries.shape[1] == 2 else _TARGET_BH_FORCE_SOFT_3D
    return jax.ffi.ffi_call(
        target,
        result_shape_dtypes=out_shape,
    )(
        leaf_bytes,
        node_bytes,
        queries,
        beta=np.float32(beta),
        softening_scale=np.float32(softening_scale),
        cutoff_scale=np.float32(cutoff_scale),
    )


def stochastic_barnes_hut_gravity(
    leaf_bytes: jnp.ndarray,
    node_bytes: jnp.ndarray,
    contrib_bytes: jnp.ndarray,
    queries: jnp.ndarray,
    *,
    samples_per_subdomain: int = 1,
    seed: int = 0,
) -> jnp.ndarray:
    leaf_bytes = jnp.asarray(leaf_bytes, dtype=jnp.uint8)
    node_bytes = jnp.asarray(node_bytes, dtype=jnp.uint8)
    contrib_bytes = jnp.asarray(contrib_bytes, dtype=jnp.uint8)
    queries = jnp.asarray(queries, dtype=jnp.float32)

    if queries.ndim != 2 or queries.shape[1] not in (2, 3):
        raise ValueError("queries must be (M, 2) or (M, 3)")

    out_shape = jax.ShapeDtypeStruct((queries.shape[0],), jnp.float32)
    target = _TARGET_SBH if queries.shape[1] == 2 else _TARGET_SBH_3D
    return jax.ffi.ffi_call(
        target,
        result_shape_dtypes=out_shape,
    )(
        leaf_bytes,
        node_bytes,
        contrib_bytes,
        queries,
        samples_per_subdomain=np.int32(samples_per_subdomain),
        seed=np.int32(seed),
    )


def stochastic_barnes_hut_force(
    leaf_bytes: jnp.ndarray,
    node_bytes: jnp.ndarray,
    contrib_bytes: jnp.ndarray,
    queries: jnp.ndarray,
    *,
    samples_per_subdomain: int = 1,
    seed: int = 0,
) -> jnp.ndarray:
    leaf_bytes = jnp.asarray(leaf_bytes, dtype=jnp.uint8)
    node_bytes = jnp.asarray(node_bytes, dtype=jnp.uint8)
    contrib_bytes = jnp.asarray(contrib_bytes, dtype=jnp.uint8)
    queries = jnp.asarray(queries, dtype=jnp.float32)

    if queries.ndim != 2 or queries.shape[1] not in (2, 3):
        raise ValueError("queries must be (M, 2) or (M, 3)")

    out_shape = jax.ShapeDtypeStruct((queries.shape[0], queries.shape[1]), jnp.float32)
    target = _TARGET_SBH_FORCE if queries.shape[1] == 2 else _TARGET_SBH_FORCE_3D
    return jax.ffi.ffi_call(
        target,
        result_shape_dtypes=out_shape,
    )(
        leaf_bytes,
        node_bytes,
        contrib_bytes,
        queries,
        samples_per_subdomain=np.int32(samples_per_subdomain),
        seed=np.int32(seed),
    )


def softened_stochastic_barnes_hut_force(
    leaf_bytes: jnp.ndarray,
    node_bytes: jnp.ndarray,
    contrib_bytes: jnp.ndarray,
    queries: jnp.ndarray,
    *,
    samples_per_subdomain: int = 1,
    seed: int = 0,
    softening_scale: float = 1e-2,
    cutoff_scale: float = 1.0,
) -> jnp.ndarray:
    leaf_bytes = jnp.asarray(leaf_bytes, dtype=jnp.uint8)
    node_bytes = jnp.asarray(node_bytes, dtype=jnp.uint8)
    contrib_bytes = jnp.asarray(contrib_bytes, dtype=jnp.uint8)
    queries = jnp.asarray(queries, dtype=jnp.float32)

    if queries.ndim != 2 or queries.shape[1] not in (2, 3):
        raise ValueError("queries must be (M, 2) or (M, 3)")
    if softening_scale < 0:
        raise ValueError("softening_scale must be >= 0")
    if cutoff_scale <= 0:
        raise ValueError("cutoff_scale must be > 0")

    out_shape = jax.ShapeDtypeStruct((queries.shape[0], queries.shape[1]), jnp.float32)
    target = _TARGET_SBH_FORCE_SOFT if queries.shape[1] == 2 else _TARGET_SBH_FORCE_SOFT_3D
    return jax.ffi.ffi_call(
        target,
        result_shape_dtypes=out_shape,
    )(
        leaf_bytes,
        node_bytes,
        contrib_bytes,
        queries,
        samples_per_subdomain=np.int32(samples_per_subdomain),
        seed=np.int32(seed),
        softening_scale=np.float32(softening_scale),
        cutoff_scale=np.float32(cutoff_scale),
    )


def barnes_hut_force_octree8(
    leaf_bytes: jnp.ndarray,
    node_bytes: jnp.ndarray,
    queries: jnp.ndarray,
    *,
    beta: float = 2.0,
) -> jnp.ndarray:
    leaf_bytes = jnp.asarray(leaf_bytes, dtype=jnp.uint8)
    node_bytes = jnp.asarray(node_bytes, dtype=jnp.uint8)
    queries = jnp.asarray(queries, dtype=jnp.float32)

    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("queries must be (M, 3)")

    out_shape = jax.ShapeDtypeStruct((queries.shape[0], 3), jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_BH_FORCE_OCTREE8,
        result_shape_dtypes=out_shape,
    )(leaf_bytes, node_bytes, queries, beta=np.float32(beta))


def softened_barnes_hut_force_octree8(
    leaf_bytes: jnp.ndarray,
    node_bytes: jnp.ndarray,
    queries: jnp.ndarray,
    *,
    beta: float = 2.0,
    softening_scale: float = 1e-2,
    cutoff_scale: float = 1.0,
) -> jnp.ndarray:
    register_softened_barnes_hut_force_octree8()

    leaf_bytes = jnp.asarray(leaf_bytes, dtype=jnp.uint8)
    node_bytes = jnp.asarray(node_bytes, dtype=jnp.uint8)
    queries = jnp.asarray(queries, dtype=jnp.float32)

    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("queries must be (M, 3)")
    if softening_scale < 0:
        raise ValueError("softening_scale must be >= 0")
    if cutoff_scale <= 0:
        raise ValueError("cutoff_scale must be > 0")

    out_shape = jax.ShapeDtypeStruct((queries.shape[0], 3), jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_BH_FORCE_OCTREE8_SOFT,
        result_shape_dtypes=out_shape,
    )(
        leaf_bytes,
        node_bytes,
        queries,
        beta=np.float32(beta),
        softening_scale=np.float32(softening_scale),
        cutoff_scale=np.float32(cutoff_scale),
    )


def stochastic_barnes_hut_force_octree8(
    leaf_bytes: jnp.ndarray,
    node_bytes: jnp.ndarray,
    queries: jnp.ndarray,
    *,
    samples_per_subdomain: int = 1,
    seed: int = 0,
) -> jnp.ndarray:
    leaf_bytes = jnp.asarray(leaf_bytes, dtype=jnp.uint8)
    node_bytes = jnp.asarray(node_bytes, dtype=jnp.uint8)
    queries = jnp.asarray(queries, dtype=jnp.float32)

    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("queries must be (M, 3)")

    out_shape = jax.ShapeDtypeStruct((queries.shape[0], 3), jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_SBH_FORCE_OCTREE8,
        result_shape_dtypes=out_shape,
    )(
        leaf_bytes,
        node_bytes,
        queries,
        samples_per_subdomain=np.int32(samples_per_subdomain),
        seed=np.int32(seed),
    )


def softened_stochastic_barnes_hut_force_octree8(
    leaf_bytes: jnp.ndarray,
    node_bytes: jnp.ndarray,
    queries: jnp.ndarray,
    *,
    samples_per_subdomain: int = 1,
    seed: int = 0,
    softening_scale: float = 1e-2,
    cutoff_scale: float = 1.0,
) -> jnp.ndarray:
    leaf_bytes = jnp.asarray(leaf_bytes, dtype=jnp.uint8)
    node_bytes = jnp.asarray(node_bytes, dtype=jnp.uint8)
    queries = jnp.asarray(queries, dtype=jnp.float32)

    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("queries must be (M, 3)")
    if softening_scale < 0:
        raise ValueError("softening_scale must be >= 0")
    if cutoff_scale <= 0:
        raise ValueError("cutoff_scale must be > 0")

    out_shape = jax.ShapeDtypeStruct((queries.shape[0], 3), jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_SBH_FORCE_OCTREE8_SOFT,
        result_shape_dtypes=out_shape,
    )(
        leaf_bytes,
        node_bytes,
        queries,
        samples_per_subdomain=np.int32(samples_per_subdomain),
        seed=np.int32(seed),
        softening_scale=np.float32(softening_scale),
        cutoff_scale=np.float32(cutoff_scale),
    )


def softened_barnes_hut_force_octree8_vjp(
    leaf_bytes: jnp.ndarray,
    node_bytes: jnp.ndarray,
    queries: jnp.ndarray,
    cotangent: jnp.ndarray,
    *,
    beta: float = 2.0,
    softening_scale: float = 1e-2,
    cutoff_scale: float = 1.0,
) -> jnp.ndarray:
    register_softened_barnes_hut_force_octree8_vjp()

    leaf_bytes = jnp.asarray(leaf_bytes, dtype=jnp.uint8)
    node_bytes = jnp.asarray(node_bytes, dtype=jnp.uint8)
    queries = jnp.asarray(queries, dtype=jnp.float32)
    cotangent = jnp.asarray(cotangent, dtype=jnp.float32)

    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("queries must be (M, 3)")
    if cotangent.shape != queries.shape:
        raise ValueError("cotangent must have same shape as queries")
    if softening_scale < 0:
        raise ValueError("softening_scale must be >= 0")
    if cutoff_scale <= 0:
        raise ValueError("cutoff_scale must be > 0")

    out_shape = jax.ShapeDtypeStruct((queries.shape[0], 3), jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_BH_FORCE_OCTREE8_SOFT_VJP,
        result_shape_dtypes=out_shape,
    )(
        leaf_bytes,
        node_bytes,
        queries,
        cotangent,
        beta=np.float32(beta),
        softening_scale=np.float32(softening_scale),
        cutoff_scale=np.float32(cutoff_scale),
    )


@jax.custom_vjp
def _softened_barnes_hut_force_octree8_recompute_vjp_impl(
    points: jnp.ndarray | np.ndarray,
    masses: jnp.ndarray | np.ndarray,
    beta: float,
    softening_scale: float,
    cutoff_scale: float,
    max_depth: int,
) -> jnp.ndarray:
    points = jnp.asarray(points, dtype=jnp.float32)
    masses = jnp.asarray(masses, dtype=jnp.float32)
    normals = jnp.zeros_like(points)
    leaf_bytes, node_bytes = build_octree8_buffers(
        points,
        normals,
        masses,
        max_depth=max_depth,
    )
    return softened_barnes_hut_force_octree8(
        leaf_bytes,
        node_bytes,
        points,
        beta=beta,
        softening_scale=softening_scale,
        cutoff_scale=cutoff_scale,
    )


def softened_barnes_hut_force_octree8_recompute_vjp(
    points: jnp.ndarray | np.ndarray,
    masses: jnp.ndarray | np.ndarray,
    *,
    beta: float = 2.0,
    softening_scale: float = 1e-2,
    cutoff_scale: float = 1.0,
    max_depth: int = 7,
) -> jnp.ndarray:
    return _softened_barnes_hut_force_octree8_recompute_vjp_impl(
        points,
        masses,
        np.float32(beta),
        np.float32(softening_scale),
        np.float32(cutoff_scale),
        np.int32(max_depth),
    )


def _softened_barnes_hut_force_octree8_recompute_vjp_fwd(
    points: jnp.ndarray | np.ndarray,
    masses: jnp.ndarray | np.ndarray,
    beta: float,
    softening_scale: float,
    cutoff_scale: float,
    max_depth: int,
):
    points = jnp.asarray(points, dtype=jnp.float32)
    masses = jnp.asarray(masses, dtype=jnp.float32)
    normals = jnp.zeros_like(points)
    leaf_bytes, node_bytes = build_octree8_buffers(
        points,
        normals,
        masses,
        max_depth=max_depth,
    )
    out = softened_barnes_hut_force_octree8(
        leaf_bytes,
        node_bytes,
        points,
        beta=beta,
        softening_scale=softening_scale,
        cutoff_scale=cutoff_scale,
    )
    residual = (points, masses, float(beta), float(softening_scale), float(cutoff_scale), int(max_depth))
    return out, residual


def _softened_barnes_hut_force_octree8_recompute_vjp_bwd(residual, cotangent):
    points, masses, beta, softening_scale, cutoff_scale, max_depth = residual
    normals = jnp.zeros_like(points)
    leaf_bytes, node_bytes = build_octree8_buffers(
        points,
        normals,
        masses,
        max_depth=max_depth,
    )
    grad_points = softened_barnes_hut_force_octree8_vjp(
        leaf_bytes,
        node_bytes,
        points,
        cotangent,
        beta=beta,
        softening_scale=softening_scale,
        cutoff_scale=cutoff_scale,
    )
    grad_masses = jnp.zeros_like(masses)
    return (grad_points, grad_masses, None, None, None, None)


_softened_barnes_hut_force_octree8_recompute_vjp_impl.defvjp(
    _softened_barnes_hut_force_octree8_recompute_vjp_fwd,
    _softened_barnes_hut_force_octree8_recompute_vjp_bwd,
)
