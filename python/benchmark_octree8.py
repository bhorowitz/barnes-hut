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


def _parse_float_list(values: str, fallback: float) -> list[float]:
    if values.strip() == "":
        return [float(fallback)]
    out = []
    for v in values.split(","):
        t = v.strip()
        if t:
            out.append(float(t))
    if not out:
        out.append(float(fallback))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark JAX FFI Barnes-Hut octree builders")
    parser.add_argument("--n", type=int, default=100000, help="number of particles")
    parser.add_argument("--depth", type=int, default=7, help="octree8 max_depth (1-10)")
    parser.add_argument("--repeats", type=int, default=5, help="timing repeats")
    parser.add_argument("--warmup", type=int, default=1, help="warmup iterations")
    parser.add_argument("--coord-min", type=float, default=-1.0, help="minimum coordinate value")
    parser.add_argument("--coord-max", type=float, default=1.0, help="maximum coordinate value")
    parser.add_argument("--periodic", action="store_true", help="enable periodic minimum-image force")
    parser.add_argument(
        "--box-length",
        type=float,
        default=1.0,
        help="periodic box length used when --periodic is enabled",
    )
    parser.add_argument("--beta", type=float, default=2.0, help="BH acceptance beta")
    parser.add_argument(
        "--beta-values",
        type=str,
        default="",
        help="optional comma-separated beta sweep (overrides --beta)",
    )
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
    parser.add_argument(
        "--force-only",
        action="store_true",
        help="time force kernels only (build once, then reuse leaf/node buffers)",
    )
    parser.add_argument(
        "--prune-enabled",
        action="store_true",
        help="enable AABB tree pruning for softened deterministic octree8 force",
    )
    parser.add_argument(
        "--prune-r-cut-mult",
        type=float,
        default=4.0,
        help="pruning radius multiplier, r_cut = prune_r_cut_mult * cutoff_scale",
    )
    parser.add_argument(
        "--prune-r-cut-values",
        type=str,
        default="",
        help="optional comma-separated prune_r_cut_mult sweep",
    )
    parser.add_argument(
        "--compare-prune",
        action="store_true",
        help="for softened deterministic force-only benchmarks, compare prune on vs off",
    )
    parser.add_argument("--samples", type=int, default=4, help="samples_per_subdomain for stochastic")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    n = args.n
    points = np.random.uniform(args.coord_min, args.coord_max, size=(n, 3)).astype(np.float32)
    normals = np.zeros_like(points, dtype=np.float32)
    masses = np.full(n, 1.0 / n, dtype=np.float32)

    points_jax = jnp.asarray(points)
    normals_jax = jnp.asarray(normals)
    masses_jax = jnp.asarray(masses)
    betas = _parse_float_list(args.beta_values, args.beta)
    prune_rcuts = _parse_float_list(args.prune_r_cut_values, args.prune_r_cut_mult)
    if args.periodic and args.box_length <= 0:
        raise ValueError("--box-length must be > 0 when --periodic is set")

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

    def build_octree8_once():
        if args.periodic:
            min_corner = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
            max_corner = jnp.array(
                [args.box_length, args.box_length, args.box_length],
                dtype=jnp.float32,
            )
            return jax_sbh_ffi.build_octree8_buffers(
                points_jax,
                normals_jax,
                masses_jax,
                min_corner=min_corner,
                max_corner=max_corner,
                max_depth=args.depth,
            )
        return jax_sbh_ffi.build_octree8_buffers(
            points_jax,
            normals_jax,
            masses_jax,
            max_depth=args.depth,
        )

    def build_level3():
        leaf, node = jax_sbh_ffi.build_octree_buffers(
            points_jax, normals_jax, masses_jax, level=3, backend="ffi"
        )
        return node

    def build_octree8():
        leaf, node = build_octree8_once()
        return node

    def force_level3():
        leaf, node = jax_sbh_ffi.build_octree_buffers(
            points_jax, normals_jax, masses_jax, level=3, backend="ffi"
        )
        out = jax_sbh_ffi.barnes_hut_force(leaf, node, points_jax, beta=betas[0])
        return out

    def force_octree8_with(
        leaf,
        node,
        *,
        beta: float,
        prune_enabled: bool,
        prune_r_cut_mult: float,
    ):
        if args.softened:
            out = jax_sbh_ffi.softened_barnes_hut_force_octree8(
                leaf,
                node,
                points_jax,
                beta=beta,
                softening_scale=args.softening_scale,
                cutoff_scale=args.cutoff_scale,
                prune_enabled=prune_enabled,
                prune_r_cut_mult=prune_r_cut_mult,
                periodic=args.periodic,
                box_length=args.box_length,
            )
        else:
            out = jax_sbh_ffi.barnes_hut_force_octree8(
                leaf,
                node,
                points_jax,
                beta=beta,
                periodic=args.periodic,
                box_length=args.box_length,
            )
        return out

    def force_octree8():
        leaf, node = build_octree8_once()
        return force_octree8_with(
            leaf,
            node,
            beta=betas[0],
            prune_enabled=args.prune_enabled,
            prune_r_cut_mult=args.prune_r_cut_mult,
        )

    def force_octree8_stochastic():
        leaf, node = build_octree8_once()
        return force_octree8_stochastic_with(
            leaf,
            node,
            prune_enabled=args.prune_enabled,
            prune_r_cut_mult=args.prune_r_cut_mult,
        )

    def force_octree8_stochastic_with(
        leaf,
        node,
        *,
        prune_enabled: bool,
        prune_r_cut_mult: float,
    ):
        if args.softened:
            out = jax_sbh_ffi.softened_stochastic_barnes_hut_force_octree8(
                leaf,
                node,
                points_jax,
                samples_per_subdomain=args.samples,
                seed=args.seed,
                softening_scale=args.softening_scale,
                cutoff_scale=args.cutoff_scale,
                prune_enabled=prune_enabled,
                prune_r_cut_mult=prune_r_cut_mult,
                periodic=args.periodic,
                box_length=args.box_length,
            )
        else:
            out = jax_sbh_ffi.stochastic_barnes_hut_force_octree8(
                leaf,
                node,
                points_jax,
                samples_per_subdomain=args.samples,
                seed=args.seed,
                periodic=args.periodic,
                box_length=args.box_length,
            )
        return out

    print(f"N={n} depth={args.depth} repeats={args.repeats} warmup={args.warmup}")
    print(
        f"coords=[{args.coord_min},{args.coord_max}] periodic={args.periodic} box_length={args.box_length}"
    )
    if args.softened:
        print(
            f"softened: softening_scale={args.softening_scale} cutoff_scale={args.cutoff_scale}"
        )
        print(
            f"prune: enabled={args.prune_enabled} prune_r_cut_mult={args.prune_r_cut_mult}"
        )
    print(f"beta sweep: {betas}")
    if args.softened:
        print(f"prune_r_cut_mult sweep: {prune_rcuts}")

    if args.force_only:
        print("Benchmarking octree8 force only (tree build excluded) ...")
        leaf_cached, node_cached = build_octree8_once()

        if args.stochastic:
            if args.softened and args.compare_prune:
                print("samples,rcut_mult,ms_no_prune,ms_prune,speedup")
                t_off = _timeit(
                    lambda: force_octree8_stochastic_with(
                        leaf_cached,
                        node_cached,
                        prune_enabled=False,
                        prune_r_cut_mult=1.0,
                    ),
                    args.repeats,
                    args.warmup,
                )
                for rcut in prune_rcuts:
                    t_on = _timeit(
                        lambda r=rcut: force_octree8_stochastic_with(
                            leaf_cached,
                            node_cached,
                            prune_enabled=True,
                            prune_r_cut_mult=r,
                        ),
                        args.repeats,
                        args.warmup,
                    )
                    speedup = t_off / max(t_on, 1e-12)
                    print(
                        f"{args.samples},{rcut:.3f},{t_off*1e3:.3f},{t_on*1e3:.3f},{speedup:.3f}"
                    )
            else:
                print("samples,rcut_mult,prune_enabled,ms")
                for rcut in prune_rcuts:
                    t = _timeit(
                        lambda r=rcut: force_octree8_stochastic_with(
                            leaf_cached,
                            node_cached,
                            prune_enabled=args.prune_enabled,
                            prune_r_cut_mult=r,
                        ),
                        args.repeats,
                        args.warmup,
                    )
                    print(
                        f"{args.samples},{rcut:.3f},{1 if args.prune_enabled else 0},{t*1e3:.3f}"
                    )
            return

        if args.softened and args.compare_prune:
            print("beta,rcut_mult,ms_no_prune,ms_prune,speedup")
            for beta in betas:
                t_off = _timeit(
                    lambda b=beta: force_octree8_with(
                        leaf_cached,
                        node_cached,
                        beta=b,
                        prune_enabled=False,
                        prune_r_cut_mult=1.0,
                    ),
                    args.repeats,
                    args.warmup,
                )
                for rcut in prune_rcuts:
                    t_on = _timeit(
                        lambda b=beta, r=rcut: force_octree8_with(
                            leaf_cached,
                            node_cached,
                            beta=b,
                            prune_enabled=True,
                            prune_r_cut_mult=r,
                        ),
                        args.repeats,
                        args.warmup,
                    )
                    speedup = t_off / max(t_on, 1e-12)
                    print(
                        f"{beta:.3f},{rcut:.3f},{t_off*1e3:.3f},{t_on*1e3:.3f},{speedup:.3f}"
                    )
        else:
            print("beta,rcut_mult,prune_enabled,ms")
            for beta in betas:
                for rcut in prune_rcuts:
                    t = _timeit(
                        lambda b=beta, r=rcut: force_octree8_with(
                            leaf_cached,
                            node_cached,
                            beta=b,
                            prune_enabled=args.prune_enabled,
                            prune_r_cut_mult=r,
                        ),
                        args.repeats,
                        args.warmup,
                    )
                    print(
                        f"{beta:.3f},{rcut:.3f},{1 if args.prune_enabled else 0},{t*1e3:.3f}"
                    )
        return

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
