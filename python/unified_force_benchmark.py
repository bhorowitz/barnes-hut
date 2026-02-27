#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special
import numpy as np

try:
    from scipy import special as sp_special
except Exception:  # pragma: no cover
    sp_special = None

import jax_sbh_ffi


EPS = 1e-12
SOFT_OFF_CUTOFF_FACTOR_DEFAULT = 1e6

METHOD_CPU_NUMPY_DIRECT = "cpu_numpy_direct"
METHOD_JAX_GPU_DIRECT = "jax_gpu_direct_no_custom"
METHOD_FFI_CUDA_DIRECT = "ffi_cuda_direct"
METHOD_CLASSIC_BH_L3 = "classic_bh_l3"
METHOD_CLASSIC_BH_OCTREE8 = "classic_bh_octree8"
METHOD_STOCHASTIC_BH_L3 = "stochastic_bh_l3"
METHOD_STOCHASTIC_BH_OCTREE8 = "stochastic_bh_octree8"

ALL_METHODS = (
    METHOD_CPU_NUMPY_DIRECT,
    METHOD_JAX_GPU_DIRECT,
    METHOD_FFI_CUDA_DIRECT,
    METHOD_CLASSIC_BH_L3,
    METHOD_CLASSIC_BH_OCTREE8,
    METHOD_STOCHASTIC_BH_L3,
    METHOD_STOCHASTIC_BH_OCTREE8,
)


ALL_GPU = (
    METHOD_JAX_GPU_DIRECT,
    METHOD_FFI_CUDA_DIRECT,
    METHOD_CLASSIC_BH_L3,
    METHOD_CLASSIC_BH_OCTREE8,
    METHOD_STOCHASTIC_BH_L3,
    METHOD_STOCHASTIC_BH_OCTREE8,
)

METHOD_ALIASES = {
    "cpu": METHOD_CPU_NUMPY_DIRECT,
    "cpu_numpy": METHOD_CPU_NUMPY_DIRECT,
    "cpu_numpy_direct": METHOD_CPU_NUMPY_DIRECT,
    "jax": METHOD_JAX_GPU_DIRECT,
    "jax_direct": METHOD_JAX_GPU_DIRECT,
    "jax_gpu_direct_no_custom": METHOD_JAX_GPU_DIRECT,
    "ffi": METHOD_FFI_CUDA_DIRECT,
    "ffi_cuda_direct": METHOD_FFI_CUDA_DIRECT,
    "cuda_direct": METHOD_FFI_CUDA_DIRECT,
    "classic_bh_l3": METHOD_CLASSIC_BH_L3,
    "bh_l3": METHOD_CLASSIC_BH_L3,
    "classic_bh_octree8": METHOD_CLASSIC_BH_OCTREE8,
    "bh_octree8": METHOD_CLASSIC_BH_OCTREE8,
    "stochastic_bh_l3": METHOD_STOCHASTIC_BH_L3,
    "sbh_l3": METHOD_STOCHASTIC_BH_L3,
    "stochastic_bh_octree8": METHOD_STOCHASTIC_BH_OCTREE8,
    "sbh_octree8": METHOD_STOCHASTIC_BH_OCTREE8,
}


@dataclass(frozen=True)
class ScanConfig:
    scan_id: int
    domain_length: float
    periodic: bool
    init_mode: str
    n_particles: int
    octree8_depth: int
    beta: float
    samples_per_subdomain: int
    use_softening: bool
    softening_scale: float
    cutoff_scale: float
    seed: int
    threads: int


def _block_until_ready(x):
    try:
        return x.block_until_ready()
    except AttributeError:
        return jax.block_until_ready(x)


def _parse_csv_tokens(text: str) -> list[str]:
    return [tok.strip() for tok in text.split(",") if tok.strip()]


def _parse_float_list(text: str) -> list[float]:
    vals = [float(v) for v in _parse_csv_tokens(text)]
    if not vals:
        raise ValueError("expected at least one float value")
    return vals


def _parse_int_list(text: str) -> list[int]:
    vals = [int(v) for v in _parse_csv_tokens(text)]
    if not vals:
        raise ValueError("expected at least one int value")
    return vals


def _parse_bool_list(text: str) -> list[bool]:
    mapping = {
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "t": True,
        "f": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }
    vals: list[bool] = []
    for tok in _parse_csv_tokens(text.lower()):
        if tok not in mapping:
            raise ValueError(f"invalid boolean token: {tok}")
        vals.append(mapping[tok])
    if not vals:
        raise ValueError("expected at least one bool value")
    return vals


def _parse_mode_list(text: str) -> list[str]:
    vals = [v.lower() for v in _parse_csv_tokens(text)]
    if not vals:
        raise ValueError("expected at least one init mode")
    allowed = {"uniform", "gaussian"}
    for v in vals:
        if v not in allowed:
            raise ValueError(f"unsupported init mode: {v}")
    return vals


def _parse_method_token(text: str) -> str:
    tok = text.strip().lower()
    if not tok:
        raise ValueError("expected method token")
    canonical = METHOD_ALIASES.get(tok)
    if canonical is None:
        supported = ", ".join(ALL_METHODS)
        raise ValueError(f"unknown method '{tok}'. Supported: {supported}")
    return canonical


def _parse_methods_list(text: str) -> tuple[str, ...]:
    tokens = [tok.lower() for tok in _parse_csv_tokens(text)]
    if not tokens:
        raise ValueError("expected at least one method token")
    if len(tokens) == 1 and tokens[0] == "all":
        return ALL_METHODS
    if len(tokens) == 1 and tokens[0] == "gpu":
        return ALL_GPU

    methods: list[str] = []
    for tok in tokens:
        if tok == "all":
            return ALL_METHODS
        if tok == "gpu":
            return ALL_GPU
        methods.append(_parse_method_token(tok))
    # Keep first occurrence ordering.
    return tuple(dict.fromkeys(methods).keys())


def _min_image_np(disp: np.ndarray, periodic: bool, box_length: float) -> np.ndarray:
    if not periodic:
        return disp
    return disp - box_length * np.round(disp / box_length)


def _min_image_jax(disp: jnp.ndarray, periodic: bool, box_length: float) -> jnp.ndarray:
    if not periodic:
        return disp
    return disp - box_length * jnp.round(disp / box_length)


def _attenuation_np(dist: np.ndarray, cutoff_scale: float) -> np.ndarray:
    if cutoff_scale <= 0.0:
        return np.ones_like(dist, dtype=np.float32)
    u = dist / (2.0 * cutoff_scale)
    if sp_special is not None:
        erfc_u = sp_special.erfc(u)
    else:  # pragma: no cover
        erfc_u = np.vectorize(math.erfc)(u)
    return erfc_u + (2.0 / np.sqrt(np.pi)) * u * np.exp(-(u * u))


def _force_softcut_np(
    points: np.ndarray,
    masses: np.ndarray,
    queries: np.ndarray,
    *,
    periodic: bool,
    box_length: float,
    softening_scale: float,
    cutoff_scale: float,
) -> np.ndarray:
    out = np.empty((queries.shape[0], 3), dtype=np.float32)
    eps2 = float(softening_scale) * float(softening_scale)
    for i in range(queries.shape[0]):
        disp = queries[i][None, :] - points
        disp = _min_image_np(disp, periodic, box_length)
        r2 = np.sum(disp * disp, axis=1)
        inv_r = 1.0 / np.sqrt(r2 + eps2 + 1e-12)
        inv_r3 = inv_r * inv_r * inv_r
        dist = np.sqrt(r2 + 1e-12)
        attenuation = _attenuation_np(dist, cutoff_scale).astype(np.float32)
        coeff = -masses * attenuation * inv_r3
        out[i] = np.sum(coeff[:, None] * disp, axis=0, dtype=np.float32)
    return out


def _force_softcut_jax(
    points: jnp.ndarray,
    masses: jnp.ndarray,
    queries: jnp.ndarray,
    *,
    periodic: bool,
    box_length: float,
    softening_scale: float,
    cutoff_scale: float,
) -> jnp.ndarray:
    eps2 = jnp.float32(softening_scale * softening_scale)
    cutoff = jnp.float32(cutoff_scale)

    def one_query(q: jnp.ndarray) -> jnp.ndarray:
        disp = q[None, :] - points
        disp = _min_image_jax(disp, periodic, box_length)
        r2 = jnp.sum(disp * disp, axis=1)
        inv_r = jax.lax.rsqrt(r2 + eps2 + 1e-12)
        inv_r3 = inv_r * inv_r * inv_r
        dist = jnp.sqrt(r2 + 1e-12)
        u = dist / (2.0 * cutoff)
        attenuation = jsp_special.erfc(u) + (2.0 / jnp.sqrt(jnp.pi)) * u * jnp.exp(
            -(u * u)
        )
        coeff = -masses * attenuation * inv_r3
        return jnp.sum(coeff[:, None] * disp, axis=0)

    return jax.vmap(one_query)(queries)


def _make_particles(
    *,
    n_particles: int,
    init_mode: str,
    domain_length: float,
    periodic: bool,
    gaussian_sigma_frac: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if periodic:
        lo = 0.0
        hi = domain_length
        center = domain_length * 0.5
    else:
        lo = -0.5 * domain_length
        hi = 0.5 * domain_length
        center = 0.0

    if init_mode == "uniform":
        points = rng.uniform(lo, hi, size=(n_particles, 3)).astype(np.float32)
    elif init_mode == "gaussian":
        sigma = gaussian_sigma_frac * domain_length
        points = rng.normal(center, sigma, size=(n_particles, 3)).astype(np.float32)
        if periodic:
            points = np.mod(points, domain_length).astype(np.float32)
        else:
            points = np.clip(points, lo, hi).astype(np.float32)
    else:  # pragma: no cover
        raise ValueError(f"invalid init_mode: {init_mode}")
    return points


def _time_callable(
    fn: Callable[[], object],
    *,
    repeats: int,
    warmup: int,
) -> tuple[float, object]:
    out = None
    for _ in range(max(0, warmup)):
        out = fn()
        _block_until_ready(out)

    times_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        _block_until_ready(out)
        times_ms.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(times_ms)), out


def _compute_error_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    outlier_rel_threshold: float,
) -> tuple[float, float]:
    diff = pred - ref
    rel_l2 = float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(ref.ravel()), EPS))
    ref_norm = np.linalg.norm(ref, axis=1)
    err_norm = np.linalg.norm(diff, axis=1)
    rel_err = err_norm / np.maximum(ref_norm, EPS)
    outlier_frac = float(np.mean(rel_err > outlier_rel_threshold))
    return rel_l2, outlier_frac


def _fmt_metric(value: float | None, *, sci: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    if sci:
        return f"{value:.3e}"
    return f"{value:.3f}"


def _print_table(rows: list[dict[str, object]], columns: list[str]) -> None:
    widths: dict[str, int] = {}
    for col in columns:
        widths[col] = len(col)
        for row in rows:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))

    header = " | ".join(col.ljust(widths[col]) for col in columns)
    sep = "-+-".join("-" * widths[col] for col in columns)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))


def _register_ffi_targets() -> None:
    jax_sbh_ffi.register_softened_direct_force()
    jax_sbh_ffi.register_build_octree_level3_buffers()
    jax_sbh_ffi.register_build_octree8_buffers()
    jax_sbh_ffi.register_softened_barnes_hut_force()
    jax_sbh_ffi.register_softened_barnes_hut_force_octree8()
    jax_sbh_ffi.register_softened_stochastic_barnes_hut_force()
    jax_sbh_ffi.register_softened_stochastic_barnes_hut_force_octree8()


def _effective_softening(
    *,
    use_softening: bool,
    domain_length: float,
    softening_scale: float,
    cutoff_scale: float,
    no_softening_cutoff_factor: float,
) -> tuple[float, float]:
    if use_softening:
        return float(softening_scale), float(cutoff_scale)
    return 0.0, float(domain_length * no_softening_cutoff_factor)


def _scan_configs(args: argparse.Namespace) -> Iterable[ScanConfig]:
    domain_lengths = _parse_float_list(args.domain_lengths)
    periodic_values = _parse_bool_list(args.periodic_values)
    init_modes = _parse_mode_list(args.init_modes)
    particle_counts = _parse_int_list(args.num_particles)
    octree8_depths = _parse_int_list(args.octree8_depths)
    betas = _parse_float_list(args.betas)
    samples_vals = _parse_int_list(args.samples_per_subdomain)
    use_soft_vals = _parse_bool_list(args.use_softening_values)
    soft_scales = _parse_float_list(args.softening_scales)
    cutoff_scales = _parse_float_list(args.cutoff_scales)

    scan_id = 0
    for (
        domain_length,
        periodic,
        init_mode,
        n_particles,
        octree8_depth,
        beta,
        samples_per_subdomain,
        use_softening,
    ) in itertools.product(
        domain_lengths,
        periodic_values,
        init_modes,
        particle_counts,
        octree8_depths,
        betas,
        samples_vals,
        use_soft_vals,
    ):
        if octree8_depth < 1 or octree8_depth > 10:
            raise ValueError(f"octree8 depth must be in [1, 10], got {octree8_depth}")
        if use_softening:
            soft_pairs = itertools.product(soft_scales, cutoff_scales)
        else:
            soft_pairs = [(0.0, 1.0)]
        for softening_scale, cutoff_scale in soft_pairs:
            scan_id += 1
            yield ScanConfig(
                scan_id=scan_id,
                domain_length=float(domain_length),
                periodic=bool(periodic),
                init_mode=init_mode,
                n_particles=int(n_particles),
                octree8_depth=int(octree8_depth),
                beta=float(beta),
                samples_per_subdomain=int(samples_per_subdomain),
                use_softening=bool(use_softening),
                softening_scale=float(softening_scale),
                cutoff_scale=float(cutoff_scale),
                seed=(
                    int(args.seed + scan_id)
                    if bool(getattr(args, "vary_seed_by_scan", False))
                    else int(args.seed)
                ),
                threads=int(args.threads),
            )


def run_scan(
    cfg: ScanConfig,
    *,
    methods: tuple[str, ...],
    reference_method: str,
    timing_repeats: int,
    warmup: int,
    outlier_rel_threshold: float,
    gaussian_sigma_frac: float,
    no_softening_cutoff_factor: float,
    prune_enabled: bool,
    prune_r_cut_mult: float,
) -> list[dict[str, object]]:
    points_np = _make_particles(
        n_particles=cfg.n_particles,
        init_mode=cfg.init_mode,
        domain_length=cfg.domain_length,
        periodic=cfg.periodic,
        gaussian_sigma_frac=gaussian_sigma_frac,
        seed=cfg.seed,
    )
    masses_np = np.full(cfg.n_particles, 1.0 / cfg.n_particles, dtype=np.float32)
    normals_np = np.zeros_like(points_np, dtype=np.float32)
    queries_np = points_np

    points_jax = jnp.asarray(points_np, dtype=jnp.float32)
    masses_jax = jnp.asarray(masses_np, dtype=jnp.float32)
    normals_jax = jnp.asarray(normals_np, dtype=jnp.float32)
    queries_jax = points_jax

    if cfg.periodic:
        min_corner = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
        max_corner = jnp.array(
            [cfg.domain_length, cfg.domain_length, cfg.domain_length], dtype=jnp.float32
        )
    else:
        lo = -0.5 * cfg.domain_length
        hi = 0.5 * cfg.domain_length
        min_corner = jnp.array([lo, lo, lo], dtype=jnp.float32)
        max_corner = jnp.array([hi, hi, hi], dtype=jnp.float32)

    eff_softening, eff_cutoff = _effective_softening(
        use_softening=cfg.use_softening,
        domain_length=cfg.domain_length,
        softening_scale=cfg.softening_scale,
        cutoff_scale=cfg.cutoff_scale,
        no_softening_cutoff_factor=no_softening_cutoff_factor,
    )

    rows: list[dict[str, object]] = []
    method_outputs: dict[str, np.ndarray] = {}
    methods_set = set(methods)

    def method_selected(name: str) -> bool:
        return name in methods_set

    def method_enabled(name: str) -> bool:
        return method_selected(name) or name == reference_method

    def add_row(
        *,
        method: str,
        status: str,
        compute_ms: float | None,
        compile_ms: float | None,
        rel_l2: float | None,
        outlier_frac: float | None,
        note: str = "",
    ) -> None:
        rows.append(
            {
                "scan_id": cfg.scan_id,
                "method": method,
                "reference_method": reference_method,
                "status": status,
                "n": cfg.n_particles,
                "init": cfg.init_mode,
                "periodic": int(cfg.periodic),
                "L": cfg.domain_length,
                "depth8": cfg.octree8_depth,
                "beta": cfg.beta,
                "samples": cfg.samples_per_subdomain,
                "use_soft": int(cfg.use_softening),
                "soft_scale": eff_softening,
                "cutoff_scale": eff_cutoff,
                "compute_ms": compute_ms,
                "compile_ms": compile_ms,
                "rel_l2": rel_l2,
                "outlier_frac": outlier_frac,
                "seed": cfg.seed,
                "threads": cfg.threads,
                "note": note,
            }
        )

    # 1) CPU direct
    if method_enabled(METHOD_CPU_NUMPY_DIRECT):
        try:
            compute_ms, pred = _time_callable(
                lambda: _force_softcut_np(
                    points_np,
                    masses_np,
                    queries_np,
                    periodic=cfg.periodic,
                    box_length=cfg.domain_length,
                    softening_scale=eff_softening,
                    cutoff_scale=eff_cutoff,
                ),
                repeats=timing_repeats,
                warmup=warmup,
            )
            method_outputs[METHOD_CPU_NUMPY_DIRECT] = np.asarray(pred, dtype=np.float32)
            if method_selected(METHOD_CPU_NUMPY_DIRECT):
                add_row(
                    method=METHOD_CPU_NUMPY_DIRECT,
                    status="ok",
                    compute_ms=compute_ms,
                    compile_ms=None,
                    rel_l2=None,
                    outlier_frac=None,
                )
        except Exception as exc:
            if method_selected(METHOD_CPU_NUMPY_DIRECT):
                add_row(
                    method=METHOD_CPU_NUMPY_DIRECT,
                    status="error",
                    compute_ms=None,
                    compile_ms=None,
                    rel_l2=None,
                    outlier_frac=None,
                    note=str(exc),
                )

    # 2) JAX GPU direct (no custom CUDA kernel)
    if method_enabled(METHOD_JAX_GPU_DIRECT):
        try:
            jax_direct = jax.jit(
                lambda p, m, q: _force_softcut_jax(
                    p,
                    m,
                    q,
                    periodic=cfg.periodic,
                    box_length=cfg.domain_length,
                    softening_scale=eff_softening,
                    cutoff_scale=eff_cutoff,
                )
            )
            t0 = time.perf_counter()
            compiled = jax_direct.lower(points_jax, masses_jax, queries_jax).compile()
            compile_ms = (time.perf_counter() - t0) * 1e3
            compute_ms, pred = _time_callable(
                lambda: compiled(points_jax, masses_jax, queries_jax),
                repeats=timing_repeats,
                warmup=warmup,
            )
            method_outputs[METHOD_JAX_GPU_DIRECT] = np.asarray(pred, dtype=np.float32)
            if method_selected(METHOD_JAX_GPU_DIRECT):
                add_row(
                    method=METHOD_JAX_GPU_DIRECT,
                    status="ok",
                    compute_ms=compute_ms,
                    compile_ms=compile_ms,
                    rel_l2=None,
                    outlier_frac=None,
                )
        except Exception as exc:
            if method_selected(METHOD_JAX_GPU_DIRECT):
                add_row(
                    method=METHOD_JAX_GPU_DIRECT,
                    status="error",
                    compute_ms=None,
                    compile_ms=None,
                    rel_l2=None,
                    outlier_frac=None,
                    note=str(exc),
                )

    # 3) CUDA direct force kernel (FFI)
    if method_enabled(METHOD_FFI_CUDA_DIRECT):
        try:
            compute_ms, pred = _time_callable(
                lambda: jax_sbh_ffi.softened_direct_force(
                    points_jax,
                    masses_jax,
                    queries_jax,
                    softening_scale=eff_softening,
                    cutoff_scale=eff_cutoff,
                    periodic=cfg.periodic,
                    box_length=cfg.domain_length,
                    threads=cfg.threads,
                ),
                repeats=timing_repeats,
                warmup=warmup,
            )
            method_outputs[METHOD_FFI_CUDA_DIRECT] = np.asarray(pred, dtype=np.float32)
            if method_selected(METHOD_FFI_CUDA_DIRECT):
                add_row(
                    method=METHOD_FFI_CUDA_DIRECT,
                    status="ok",
                    compute_ms=compute_ms,
                    compile_ms=None,
                    rel_l2=None,
                    outlier_frac=None,
                )
        except Exception as exc:
            if method_selected(METHOD_FFI_CUDA_DIRECT):
                add_row(
                    method=METHOD_FFI_CUDA_DIRECT,
                    status="error",
                    compute_ms=None,
                    compile_ms=None,
                    rel_l2=None,
                    outlier_frac=None,
                    note=str(exc),
                )

    # 4) classic BH (L3)
    if method_enabled(METHOD_CLASSIC_BH_L3):
        try:

            def bh_l3_fn():
                leaf, node = jax_sbh_ffi.build_octree_buffers(
                    points_jax,
                    normals_jax,
                    masses_jax,
                    min_corner=min_corner,
                    max_corner=max_corner,
                    level=3,
                    backend="ffi",
                )
                return jax_sbh_ffi.softened_barnes_hut_force(
                    leaf,
                    node,
                    queries_jax,
                    beta=cfg.beta,
                    softening_scale=eff_softening,
                    cutoff_scale=eff_cutoff,
                    periodic=cfg.periodic,
                    box_length=cfg.domain_length,
                    threads=cfg.threads,
                )

            compute_ms, pred = _time_callable(
                bh_l3_fn, repeats=timing_repeats, warmup=warmup
            )
            method_outputs[METHOD_CLASSIC_BH_L3] = np.asarray(pred, dtype=np.float32)
            if method_selected(METHOD_CLASSIC_BH_L3):
                add_row(
                    method=METHOD_CLASSIC_BH_L3,
                    status="ok",
                    compute_ms=compute_ms,
                    compile_ms=None,
                    rel_l2=None,
                    outlier_frac=None,
                )
        except Exception as exc:
            if method_selected(METHOD_CLASSIC_BH_L3):
                add_row(
                    method=METHOD_CLASSIC_BH_L3,
                    status="error",
                    compute_ms=None,
                    compile_ms=None,
                    rel_l2=None,
                    outlier_frac=None,
                    note=str(exc),
                )

    # 5) classic BH (octree8)
    if method_enabled(METHOD_CLASSIC_BH_OCTREE8):
        try:

            def bh_octree8_fn():
                leaf, node = jax_sbh_ffi.build_octree8_buffers(
                    points_jax,
                    normals_jax,
                    masses_jax,
                    min_corner=min_corner,
                    max_corner=max_corner,
                    max_depth=cfg.octree8_depth,
                )
                return jax_sbh_ffi.softened_barnes_hut_force_octree8(
                    leaf,
                    node,
                    queries_jax,
                    beta=cfg.beta,
                    softening_scale=eff_softening,
                    cutoff_scale=eff_cutoff,
                    prune_enabled=prune_enabled,
                    prune_r_cut_mult=prune_r_cut_mult,
                    periodic=cfg.periodic,
                    box_length=cfg.domain_length,
                    threads=cfg.threads,
                )

            compute_ms, pred = _time_callable(
                bh_octree8_fn, repeats=timing_repeats, warmup=warmup
            )
            method_outputs[METHOD_CLASSIC_BH_OCTREE8] = np.asarray(pred, dtype=np.float32)
            if method_selected(METHOD_CLASSIC_BH_OCTREE8):
                add_row(
                    method=METHOD_CLASSIC_BH_OCTREE8,
                    status="ok",
                    compute_ms=compute_ms,
                    compile_ms=None,
                    rel_l2=None,
                    outlier_frac=None,
                )
        except Exception as exc:
            if method_selected(METHOD_CLASSIC_BH_OCTREE8):
                add_row(
                    method=METHOD_CLASSIC_BH_OCTREE8,
                    status="error",
                    compute_ms=None,
                    compile_ms=None,
                    rel_l2=None,
                    outlier_frac=None,
                    note=str(exc),
                )

    # 6) stochastic BH (L3)
    if method_enabled(METHOD_STOCHASTIC_BH_L3):
        try:

            def sbh_l3_fn():
                leaf, node, contrib = jax_sbh_ffi.build_octree_buffers_with_contrib(
                    points_jax,
                    normals_jax,
                    masses_jax,
                    min_corner=min_corner,
                    max_corner=max_corner,
                    level=3,
                    backend="ffi",
                )
                return jax_sbh_ffi.softened_stochastic_barnes_hut_force(
                    leaf,
                    node,
                    contrib,
                    queries_jax,
                    samples_per_subdomain=cfg.samples_per_subdomain,
                    seed=cfg.seed,
                    softening_scale=eff_softening,
                    cutoff_scale=eff_cutoff,
                    periodic=cfg.periodic,
                    box_length=cfg.domain_length,
                    threads=cfg.threads,
                )

            compute_ms, pred = _time_callable(
                sbh_l3_fn, repeats=timing_repeats, warmup=warmup
            )
            method_outputs[METHOD_STOCHASTIC_BH_L3] = np.asarray(pred, dtype=np.float32)
            if method_selected(METHOD_STOCHASTIC_BH_L3):
                add_row(
                    method=METHOD_STOCHASTIC_BH_L3,
                    status="ok",
                    compute_ms=compute_ms,
                    compile_ms=None,
                    rel_l2=None,
                    outlier_frac=None,
                )
        except Exception as exc:
            if method_selected(METHOD_STOCHASTIC_BH_L3):
                add_row(
                    method=METHOD_STOCHASTIC_BH_L3,
                    status="error",
                    compute_ms=None,
                    compile_ms=None,
                    rel_l2=None,
                    outlier_frac=None,
                    note=str(exc),
                )

    # 7) stochastic BH (octree8)
    if method_enabled(METHOD_STOCHASTIC_BH_OCTREE8):
        try:

            def sbh_octree8_fn():
                leaf, node = jax_sbh_ffi.build_octree8_buffers(
                    points_jax,
                    normals_jax,
                    masses_jax,
                    min_corner=min_corner,
                    max_corner=max_corner,
                    max_depth=cfg.octree8_depth,
                )
                return jax_sbh_ffi.softened_stochastic_barnes_hut_force_octree8(
                    leaf,
                    node,
                    queries_jax,
                    beta=cfg.beta,
                    samples_per_subdomain=cfg.samples_per_subdomain,
                    seed=cfg.seed,
                    softening_scale=eff_softening,
                    cutoff_scale=eff_cutoff,
                    prune_enabled=prune_enabled,
                    prune_r_cut_mult=prune_r_cut_mult,
                    periodic=cfg.periodic,
                    box_length=cfg.domain_length,
                    threads=cfg.threads,
                )

            compute_ms, pred = _time_callable(
                sbh_octree8_fn, repeats=timing_repeats, warmup=warmup
            )
            method_outputs[METHOD_STOCHASTIC_BH_OCTREE8] = np.asarray(
                pred, dtype=np.float32
            )
            if method_selected(METHOD_STOCHASTIC_BH_OCTREE8):
                add_row(
                    method=METHOD_STOCHASTIC_BH_OCTREE8,
                    status="ok",
                    compute_ms=compute_ms,
                    compile_ms=None,
                    rel_l2=None,
                    outlier_frac=None,
                )
        except Exception as exc:
            if method_selected(METHOD_STOCHASTIC_BH_OCTREE8):
                add_row(
                    method=METHOD_STOCHASTIC_BH_OCTREE8,
                    status="error",
                    compute_ms=None,
                    compile_ms=None,
                    rel_l2=None,
                    outlier_frac=None,
                    note=str(exc),
                )

    ref_np = method_outputs.get(reference_method)
    if ref_np is None:
        ref_msg = f"reference '{reference_method}' unavailable"
        for row in rows:
            if row["status"] != "ok":
                continue
            prev_note = str(row.get("note", ""))
            row["note"] = f"{prev_note}; {ref_msg}" if prev_note else ref_msg
            row["rel_l2"] = None
            row["outlier_frac"] = None
    else:
        for row in rows:
            if row["status"] != "ok":
                continue
            method = str(row["method"])
            pred_np = method_outputs.get(method)
            if pred_np is None:
                continue
            if method == reference_method:
                row["rel_l2"] = 0.0
                row["outlier_frac"] = 0.0
            else:
                rel_l2, outlier_frac = _compute_error_metrics(
                    pred_np, ref_np, outlier_rel_threshold=outlier_rel_threshold
                )
                row["rel_l2"] = rel_l2
                row["outlier_frac"] = outlier_frac

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified force benchmark for JAX+FFI Barnes-Hut variants. "
            "Runs CPU reference, pure-JAX GPU direct force, and CUDA FFI methods."
        )
    )
    parser.add_argument("--domain-lengths", type=str, default="1.0")
    parser.add_argument("--periodic-values", type=str, default="false,true")
    parser.add_argument("--init-modes", type=str, default="uniform,gaussian")
    parser.add_argument("--num-particles", type=str, default="512")
    parser.add_argument("--octree8-depths", type=str, default="6")
    parser.add_argument("--betas", type=str, default="2.0")
    parser.add_argument("--samples-per-subdomain", type=str, default="2")
    parser.add_argument(
        "--use-softening-values",
        type=str,
        default="true",
        help=(
            "bool scan list. false means softening_scale=0 and very large cutoff "
            "(keeps method interfaces consistent with the CUDA direct-force kernel). "
            "For stochastic octree8, false can have very large estimator variance."
        ),
    )
    parser.add_argument("--softening-scales", type=str, default="0.05")
    parser.add_argument("--cutoff-scales", type=str, default="1.0")
    parser.add_argument(
        "--no-softening-cutoff-factor",
        type=float,
        default=SOFT_OFF_CUTOFF_FACTOR_DEFAULT,
        help=(
            "Used when softening is disabled: effective cutoff = "
            "domain_length * no_softening_cutoff_factor"
        ),
    )
    parser.add_argument("--gaussian-sigma-frac", type=float, default=0.15)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--timing-repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vary-seed-by-scan",
        action="store_true",
        help=(
            "offset seed by scan_id (seed + scan_id). "
            "Default keeps seed fixed across parameter scans for apples-to-apples comparisons."
        ),
    )
    parser.add_argument("--outlier-rel-threshold", type=float, default=0.05)
    parser.add_argument("--prune-enabled", action="store_true")
    parser.add_argument("--prune-r-cut-mult", type=float, default=4.0)
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help=(
            "Comma-separated methods to run (or 'all'). Canonical names: "
            f"{','.join(ALL_METHODS)}. Aliases: {','.join(sorted(METHOD_ALIASES))}"
        ),
    )
    parser.add_argument(
        "--reference-method",
        type=str,
        default="auto",
        help=(
            "Reference method used for rel_l2/outlier_frac. "
            "Accepts one canonical name or alias. "
            "Default 'auto' uses the first method from --methods."
        ),
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        default=0,
        help="optional cap on number of scan combinations; 0 means no cap",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="unified_force_benchmark.csv",
        help="output CSV path",
    )
    args = parser.parse_args()

    if args.timing_repeats < 1:
        raise ValueError("--timing-repeats must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    if args.outlier_rel_threshold <= 0:
        raise ValueError("--outlier-rel-threshold must be > 0")
    if args.prune_enabled and args.prune_r_cut_mult <= 0:
        raise ValueError("--prune-r-cut-mult must be > 0 when --prune-enabled is set")
    try:
        selected_methods = _parse_methods_list(args.methods)
    except ValueError as exc:
        parser.error(str(exc))
    reference_arg = args.reference_method.strip().lower()
    if reference_arg in {"", "auto"}:
        reference_method = selected_methods[0]
    else:
        try:
            reference_method = _parse_method_token(args.reference_method)
        except ValueError as exc:
            parser.error(str(exc))

    if not any(dev.platform == "gpu" for dev in jax.devices()):
        raise RuntimeError("No GPU detected by JAX; this benchmark requires a CUDA GPU")

    _register_ffi_targets()

    all_rows: list[dict[str, object]] = []
    scans = _scan_configs(args)
    for i, cfg in enumerate(scans, start=1):
        if args.max_scans > 0 and i > args.max_scans:
            break
        print(
            f"\n[scan {cfg.scan_id}] "
            f"L={cfg.domain_length} periodic={cfg.periodic} init={cfg.init_mode} "
            f"N={cfg.n_particles} depth8={cfg.octree8_depth} beta={cfg.beta} "
            f"samples={cfg.samples_per_subdomain} use_soft={cfg.use_softening} "
            f"methods={','.join(selected_methods)} ref={reference_method}"
        )
        if (
            not cfg.use_softening
            and (
                METHOD_STOCHASTIC_BH_L3 in selected_methods
                or METHOD_STOCHASTIC_BH_OCTREE8 in selected_methods
            )
        ):
            print(
                "warning: use_softening=false with stochastic BH can produce very large "
                "variance and inflated rel_l2."
            )
        rows = run_scan(
            cfg,
            methods=selected_methods,
            reference_method=reference_method,
            timing_repeats=args.timing_repeats,
            warmup=args.warmup,
            outlier_rel_threshold=args.outlier_rel_threshold,
            gaussian_sigma_frac=args.gaussian_sigma_frac,
            no_softening_cutoff_factor=args.no_softening_cutoff_factor,
            prune_enabled=args.prune_enabled,
            prune_r_cut_mult=args.prune_r_cut_mult,
        )

        # Print per-scan summary table.
        view_rows: list[dict[str, object]] = []
        for r in rows:
            view_rows.append(
                {
                    "method": r["method"],
                    "status": r["status"],
                    "rel_l2": _fmt_metric(
                        r["rel_l2"] if isinstance(r["rel_l2"], (int, float)) else None,
                        sci=True,
                    ),
                    "outlier_frac": _fmt_metric(
                        r["outlier_frac"]
                        if isinstance(r["outlier_frac"], (int, float))
                        else None,
                    ),
                    "compute_ms": _fmt_metric(
                        r["compute_ms"] if isinstance(r["compute_ms"], (int, float)) else None
                    ),
                    "compile_ms": _fmt_metric(
                        r["compile_ms"] if isinstance(r["compile_ms"], (int, float)) else None
                    ),
                }
            )
        _print_table(
            view_rows,
            ["method", "status", "rel_l2", "outlier_frac", "compute_ms", "compile_ms"],
        )
        all_rows.extend(rows)

    if not all_rows:
        print("No scans were run. Check scan arguments.")
        return

    csv_path = Path(args.csv_out).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scan_id",
        "method",
        "reference_method",
        "status",
        "n",
        "init",
        "periodic",
        "L",
        "depth8",
        "beta",
        "samples",
        "use_soft",
        "soft_scale",
        "cutoff_scale",
        "compute_ms",
        "compile_ms",
        "rel_l2",
        "outlier_frac",
        "seed",
        "threads",
        "note",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\nWrote {len(all_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
