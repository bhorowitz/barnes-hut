import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import jax_sbh_ffi


def _block_until_ready(x):
    try:
        return x.block_until_ready()
    except AttributeError:
        return jax.block_until_ready(x)


def _timeit(fn, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        _block_until_ready(fn())
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _block_until_ready(fn())
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.mean(times))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark JAX FFI Barnes-Hut octree builders")
    parser.add_argument("--n", type=int, default=100000, help="number of particles")
    parser.add_argument("--depth", type=int, default=7, help="octree8 max_depth (1-10)")
    parser.add_argument("--repeats", type=int, default=5, help="timing repeats")
    parser.add_argument("--warmup", type=int, default=1, help="warmup iterations")
    parser.add_argument("--stochastic", action="store_true", help="include stochastic octree8 force")
    parser.add_argument(
        "--softened",
        action="store_true",
        help="use softened octree8 force (softening + cutoff)",
    )
    parser.add_argument(
        "--softening-scale",
        type=float,
        default=1e-2,
        help="inner softening scale",
    )
    parser.add_argument(
        "--cutoff-scale",
        type=float,
        default=1.0,
        help="long-range exponential cutoff scale",
    )
    parser.add_argument("--samples", type=int, default=4, help="samples_per_subdomain for stochastic")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    n = args.n
    points = np.random.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    normals = np.zeros_like(points, dtype=np.float32)
    masses = np.full(n, 1.0 / n, dtype=np.float32)

    points_jax = jnp.asarray(points)
    normals_jax = jnp.asarray(normals)
    masses_jax = jnp.asarray(masses)

    jax_sbh_ffi.register_build_octree_level3_buffers()
    jax_sbh_ffi.register_build_octree8_buffers()
    jax_sbh_ffi.register_barnes_hut_force()
    jax_sbh_ffi.register_barnes_hut_force_octree8()
    if args.stochastic:
        jax_sbh_ffi.register_stochastic_barnes_hut_force_octree8()
    if args.softened:
        jax_sbh_ffi.register_softened_barnes_hut_force_octree8()
        if args.stochastic:
            jax_sbh_ffi.register_softened_stochastic_barnes_hut_force_octree8()

    def build_level3():
        leaf, node = jax_sbh_ffi.build_octree_buffers(
            points_jax, normals_jax, masses_jax, level=3, backend="ffi"
        )
        return node

    def build_octree8():
        leaf, node = jax_sbh_ffi.build_octree8_buffers(
            points_jax, normals_jax, masses_jax, max_depth=args.depth
        )
        return node

    def force_level3():
        leaf, node = jax_sbh_ffi.build_octree_buffers(
            points_jax, normals_jax, masses_jax, level=3, backend="ffi"
        )
        out = jax_sbh_ffi.barnes_hut_force(leaf, node, points_jax, beta=2.0)
        return out

    def force_octree8():
        leaf, node = jax_sbh_ffi.build_octree8_buffers(
            points_jax, normals_jax, masses_jax, max_depth=args.depth
        )
        if args.softened:
            out = jax_sbh_ffi.softened_barnes_hut_force_octree8(
                leaf,
                node,
                points_jax,
                beta=2.0,
                softening_scale=args.softening_scale,
                cutoff_scale=args.cutoff_scale,
            )
        else:
            out = jax_sbh_ffi.barnes_hut_force_octree8(leaf, node, points_jax, beta=2.0)
        return out

    def force_octree8_stochastic():
        leaf, node = jax_sbh_ffi.build_octree8_buffers(
            points_jax, normals_jax, masses_jax, max_depth=args.depth
        )
        if args.softened:
            out = jax_sbh_ffi.softened_stochastic_barnes_hut_force_octree8(
                leaf,
                node,
                points_jax,
                samples_per_subdomain=args.samples,
                seed=args.seed,
                softening_scale=args.softening_scale,
                cutoff_scale=args.cutoff_scale,
            )
        else:
            out = jax_sbh_ffi.stochastic_barnes_hut_force_octree8(
                leaf, node, points_jax, samples_per_subdomain=args.samples, seed=args.seed
            )
        return out

    print(f"N={n} depth={args.depth} repeats={args.repeats} warmup={args.warmup}")
    if args.softened:
        print(
            f"softened: softening_scale={args.softening_scale} cutoff_scale={args.cutoff_scale}"
        )
    print("Benchmarking level3 build only ...")
    t_build_l3 = _timeit(build_level3, args.repeats, args.warmup)
    print(f"level3 build: {t_build_l3 * 1e3:.2f} ms")

    print("Benchmarking octree8 build only ...")
    t_build_o8 = _timeit(build_octree8, args.repeats, args.warmup)
    print(f"octree8 build: {t_build_o8 * 1e3:.2f} ms")

    print("Benchmarking level3 build+force ...")
    t_force_l3 = _timeit(force_level3, args.repeats, args.warmup)
    print(f"level3 build+force: {t_force_l3 * 1e3:.2f} ms")

    print("Benchmarking octree8 build+force ...")
    t_force_o8 = _timeit(force_octree8, args.repeats, args.warmup)
    print(f"octree8 build+force: {t_force_o8 * 1e3:.2f} ms")

    if args.stochastic:
        print("Benchmarking octree8 build+force (stochastic) ...")
        t_force_o8s = _timeit(force_octree8_stochastic, args.repeats, args.warmup)
        print(f"octree8 stochastic build+force: {t_force_o8s * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
