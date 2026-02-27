import argparse
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jaxpm.pm as jpm
import matplotlib.pyplot as plt
import numpy as np
from jaxpm.painting import cic_paint
from jaxpm.utils import cross_correlation_coefficients, power_spectrum

import jax_sbh_ffi

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
import readgadget
from treepm_jaxpm_demo import Config, _fit_tree_short_scale, _wrap_periodic_mesh


def _load_cv0_particles(path: str, mesh_n: int):
    header = readgadget.header(path)
    redshift = float(header.redshift)
    box_size = float(header.boxsize) / 1e3  # Mpc/h

    ptype = [1]
    ids = np.argsort(readgadget.read_block(path, "ID  ", ptype) - 1)
    pos = readgadget.read_block(path, "POS ", ptype)[ids] / 1e3
    vel = readgadget.read_block(path, "VEL ", ptype)[ids]

    n_part = pos.shape[0]
    n_side = int(round(n_part ** (1.0 / 3.0)))
    if n_side**3 != n_part:
        raise ValueError(f"Particle count is not cubic: {n_part}")
    if n_side != 256:
        raise ValueError(f"This loader expects 256^3 CV0 ordering, got n_side={n_side}")
    if 256 % mesh_n != 0:
        raise ValueError(f"mesh_n={mesh_n} must divide 256")

    stride = 256 // mesh_n
    pos = pos.reshape(4, 4, 4, 64, 64, 64, 3).transpose(0, 3, 1, 4, 2, 5, 6).reshape(-1, 3)
    vel = vel.reshape(4, 4, 4, 64, 64, 64, 3).transpose(0, 3, 1, 4, 2, 5, 6).reshape(-1, 3)

    pos_sub = pos.reshape(256, 256, 256, 3)[::stride, ::stride, ::stride, :].reshape(-1, 3)
    vel_sub = vel.reshape(256, 256, 256, 3)[::stride, ::stride, ::stride, :].reshape(-1, 3)
    pos_mesh = pos_sub / box_size * float(mesh_n)
    vel_mesh = vel_sub / 100.0 * (1.0 / (1.0 + redshift)) / box_size * float(mesh_n)
    a = 1.0 / (1.0 + redshift)

    return jnp.asarray(pos_mesh, dtype=jnp.float32), jnp.asarray(vel_mesh, dtype=jnp.float32), a, header


def _find_snapshot_closest_to_z(snapshot_dir: Path, target_z: float):
    snaps = sorted(snapshot_dir.glob("snapshot_*.hdf5"))
    if not snaps:
        raise FileNotFoundError(f"No snapshots found in {snapshot_dir}")
    best = None
    for snap in snaps:
        z = float(readgadget.header(str(snap)).redshift)
        dz = abs(z - target_z)
        if best is None or dz < best[0]:
            best = (dz, z, snap)
    return best[1], best[2]


def _density(pos, mesh_n: int):
    return cic_paint(jnp.zeros((mesh_n, mesh_n, mesh_n), dtype=jnp.float32), _wrap_periodic_mesh(pos, mesh_n))


def _a_to_E(cosmo, a):
    return jnp.sqrt(jc.background.Esqr(cosmo, a))


def _integrate_kdk(pos0, vel0, a0, a1, steps, accel_fn, cosmo, mesh_n):
    if steps < 1:
        raise ValueError("kdk steps must be >= 1")

    a0 = jnp.asarray(a0, jnp.float32)
    a1 = jnp.asarray(a1, jnp.float32)
    da = (a1 - a0) / jnp.asarray(steps, jnp.float32)

    i = jnp.arange(steps, dtype=jnp.float32)
    a_mid = a0 + (i + 0.5) * da
    a_next = a0 + (i + 1.0) * da
    drift_coeff = 1.0 / (a_mid**3 * _a_to_E(cosmo, a_mid))

    acc0 = accel_fn(pos0, a0, cosmo)

    def body(carry, inputs):
        pos, vel, acc = carry
        a_next_i, drift_i = inputs

        vel_half = vel + 0.5 * da * acc
        pos_next = _wrap_periodic_mesh(pos + da * drift_i * vel_half, mesh_n)

        acc_next = accel_fn(pos_next, a_next_i, cosmo)
        vel_next = vel_half + 0.5 * da * acc_next

        return (pos_next, vel_next, acc_next), None

    (posf, velf, _), _ = jax.lax.scan(
        body,
        init=(pos0, vel0, acc0),
        xs=(a_next, drift_coeff),
    )
    return posf, velf


def _pm_accel_fn(mesh_n):
    mesh_shape = (mesh_n, mesh_n, mesh_n)

    def fn(pos, a, cosmo):
        force = jpm.pm_forces(pos, mesh_shape=mesh_shape, r_split=0.0) * (1.5 * cosmo.Omega_m)
        return force / (a**2 * _a_to_E(cosmo, a))

    return fn


def _make_treepm_cv_stochastic_accel_fn(
    mesh_n,
    masses,
    cfg,
    short_scale,
    refresh_every,
    samples_per_subdomain,
    base_seed,
):
    mesh_shape = (mesh_n, mesh_n, mesh_n)
    min_corner = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    max_corner = jnp.array([float(mesh_n), float(mesh_n), float(mesh_n)], dtype=jnp.float32)
    refresh_every = int(max(1, refresh_every))
    samples_per_subdomain = int(max(1, samples_per_subdomain))
    base_seed = int(base_seed)

    def fn(pos, a, cosmo, step_i, cache):
        delta_prev, step_last_refresh, refresh_count = cache
        step_i = int(step_i)
        step_last_refresh = int(step_last_refresh)
        refresh_count = int(refresh_count)

        pm_full = jpm.pm_forces(pos, mesh_shape=mesh_shape, r_split=0.0)

        if (step_i - step_last_refresh) >= refresh_every:
            pm_long = jpm.pm_forces(pos, mesh_shape=mesh_shape, r_split=cfg.pm_r_split_cells)
            pm_sr = pm_full - pm_long

            leaf_bytes, node_bytes = jax_sbh_ffi.build_octree8_buffers(
                pos,
                jnp.zeros_like(pos),
                masses,
                min_corner=min_corner,
                max_corner=max_corner,
                max_depth=cfg.bh_max_depth,
            )
            tree_sr_stoch = jax_sbh_ffi.softened_stochastic_barnes_hut_force_octree8(
                leaf_bytes,
                node_bytes,
                pos,
                samples_per_subdomain=samples_per_subdomain,
                seed=base_seed + step_i,
                softening_scale=cfg.bh_softening,
                cutoff_scale=cfg.pm_r_split_cells,
                prune_enabled=cfg.bh_prune_enabled,
                prune_r_cut_mult=cfg.bh_prune_r_cut_mult,
                periodic=True,
                box_length=float(mesh_n),
            )
            delta = short_scale * tree_sr_stoch - pm_sr
            step_last_refresh_new = step_i
            refresh_count_new = refresh_count + 1
        else:
            delta = delta_prev
            step_last_refresh_new = step_last_refresh
            refresh_count_new = refresh_count

        force = (pm_full + delta) * (1.5 * cosmo.Omega_m)
        acc = force / (a**2 * _a_to_E(cosmo, a))
        return acc, (delta, step_last_refresh_new, refresh_count_new)

    return fn


def _integrate_kdk_treepm_cv(
    pos0,
    vel0,
    a0,
    a1,
    steps,
    accel_cv_fn,
    cosmo,
    mesh_n,
    refresh_every,
):
    if steps < 1:
        raise ValueError("kdk steps must be >= 1")

    pos = pos0
    vel = vel0
    a = jnp.asarray(a0, dtype=jnp.float32)
    a1 = jnp.asarray(a1, dtype=jnp.float32)
    da = (a1 - a) / float(steps)

    cache = (
        jnp.zeros_like(pos0),
        -int(max(1, refresh_every)),
        0,
    )
    acc, cache = accel_cv_fn(pos, a, cosmo, 0, cache)

    for i in range(steps):
        a_mid = a + 0.5 * da
        a_next = a + da
        drift = 1.0 / (a_mid**3 * _a_to_E(cosmo, a_mid))

        vel_half = vel + 0.5 * da * acc
        pos_next = _wrap_periodic_mesh(pos + da * drift * vel_half, mesh_n)

        acc_next, cache = accel_cv_fn(pos_next, a_next, cosmo, i + 1, cache)
        vel_next = vel_half + 0.5 * da * acc_next

        pos, vel, acc, a = pos_next, vel_next, acc_next, a_next

    return pos, vel, cache


def _plot_outputs(
    out_prefix: Path,
    k_ref,
    pk_ref,
    k_pm,
    pk_pm,
    k_tree,
    pk_tree,
    k_cross_pm,
    cross_pm,
    k_cross_tree,
    cross_tree,
    rho_ref,
    rho_pm,
    rho_tree,
    rho_ic,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].loglog(k_ref, pk_ref, "k--", lw=2, label="CV0 snapshot (ref)")
    axes[0].loglog(k_pm, pk_pm, "r", lw=2, label="JaxPM")
    axes[0].loglog(k_tree, pk_tree, "g", lw=2, label="TreePM stochastic-CV")
    axes[0].set_xlabel("k [h/Mpc]")
    axes[0].set_ylabel("P(k)")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend()
    axes[1].semilogx(k_cross_pm, cross_pm, "r", lw=2, label="r(k): ref vs JaxPM")
    axes[1].semilogx(k_cross_tree, cross_tree, "g", lw=2, label="r(k): ref vs TreePM stochastic-CV")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1].set_xlabel("k [h/Mpc]")
    axes[1].set_ylabel("Cross-correlation coefficient")
    axes[1].set_ylim(0.9 * cross_pm.min(), 1.05)
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend()
    fig.savefig(out_prefix.with_name(out_prefix.name + "_pk_cross.png"), dpi=170)
    plt.close(fig)

    ref_proj = np.asarray(rho_ref.sum(axis=2))
    pm_proj = np.asarray(rho_pm.sum(axis=2))
    tree_proj = np.asarray(rho_tree.sum(axis=2))
    vmax = max(float(ref_proj.max()), float(pm_proj.max()), float(tree_proj.max()))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(ref_proj, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
    axes[0].set_title("CV0 ref z~2")
    axes[1].imshow(pm_proj, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
    axes[1].set_title("JaxPM z~2")
    axes[2].imshow(tree_proj, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
    axes[2].set_title("TreePM stochastic-CV z~2")
    fig.savefig(out_prefix.with_name(out_prefix.name + "_projected_density.png"), dpi=170)
    plt.close(fig)

    ic_proj = np.asarray(rho_ic.sum(axis=2))
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 4), constrained_layout=True)
    ax.imshow(ic_proj, origin="lower", cmap="magma")
    ax.set_title("IC projected density")
    fig.savefig(out_prefix.with_name(out_prefix.name + "_ic_projected_density.png"), dpi=170)
    plt.close(fig)


def run(args):
    t_start = time.perf_counter()
    if not any(dev.platform == "gpu" for dev in jax.devices()):
        raise RuntimeError("GPU backend is required.")
    if args.integrator != "kdk":
        raise ValueError("This stochastic control-variate TreePM implementation currently supports --integrator kdk only.")

    jax_sbh_ffi.register_build_octree8_buffers()
    jax_sbh_ffi.register_softened_stochastic_barnes_hut_force_octree8()
    jax_sbh_ffi.register_softened_barnes_hut_force_octree8()
    jax_sbh_ffi.register_softened_barnes_hut_force_octree8_vjp()

    z_ref, snapshot_path = _find_snapshot_closest_to_z(Path(args.snapshot_dir), args.target_z)
    print(f"Using target snapshot: {snapshot_path} (z={z_ref:.5f})")

    pos_i, vel_i, a_i, _ = _load_cv0_particles(args.init_cond, args.mesh_n)
    pos_ref, _, a_ref, _ = _load_cv0_particles(str(snapshot_path), args.mesh_n)
    print(f"a_init={a_i:.6f}, a_target={a_ref:.6f}, mesh_n={args.mesh_n}, npart={pos_i.shape[0]}")

    cosmo = jc.Planck15(Omega_c=0.3 - 0.049, Omega_b=0.049, n_s=0.9624, h=0.6711, sigma8=0.8)
    masses = jnp.ones((pos_i.shape[0],), dtype=jnp.float32)
    cfg = Config(
        box_length=float(args.mesh_n),
        n_particles_1d=args.mesh_n,
        mesh_n=args.mesh_n,
        z_init=(1.0 / a_i) - 1.0,
        z_targets=(2.0, 1.0, 0.0),
        omega_c=0.3 - 0.049,
        sigma8=0.8,
        bh_beta=args.bh_beta,
        bh_softening=args.bh_softening,
        bh_max_depth=args.bh_max_depth,
        pm_r_split_cells=args.pm_r_split_cells,
        tree_short_scale=1.0 / (4.0 * math.pi),
        seed=args.cv_base_seed,
        bh_prune_enabled=bool(args.bh_prune_enabled),
        bh_prune_r_cut_mult=args.bh_prune_r_cut_mult,
    )

    if args.tree_short_scale == "auto":
        short_scale = _fit_tree_short_scale(pos_i, masses, cfg)
    elif args.tree_short_scale == "1_over_4pi":
        short_scale = float(1.0 / (4.0 * math.pi))
    else:
        short_scale = float(args.tree_short_scale)
    print(f"tree_short_scale={short_scale:.8g}")

    pm_accel = _pm_accel_fn(args.mesh_n)
    tree_accel_cv = _make_treepm_cv_stochastic_accel_fn(
        mesh_n=args.mesh_n,
        masses=masses,
        cfg=cfg,
        short_scale=short_scale,
        refresh_every=args.cv_refresh_every,
        samples_per_subdomain=args.cv_samples_per_subdomain,
        base_seed=args.cv_base_seed,
    )

    print(f"Integrating PM with KDK, steps={args.kdk_steps}")
    t_pm = time.perf_counter()
    pos_pm, _ = _integrate_kdk(pos_i, vel_i, float(a_i), float(a_ref), args.kdk_steps, pm_accel, cosmo, args.mesh_n)
    pm_main_time = time.perf_counter() - t_pm
    print(f"PM kdk wall time: {pm_main_time:.3f} s")

    print(
        "Integrating TreePM stochastic control-variate with KDK, "
        f"steps={args.kdk_steps}, refresh_every={args.cv_refresh_every}, "
        f"samples_per_subdomain={args.cv_samples_per_subdomain}"
    )
    t_tree = time.perf_counter()
    pos_tree, _, cachef = _integrate_kdk_treepm_cv(
        pos_i,
        vel_i,
        float(a_i),
        float(a_ref),
        args.kdk_steps,
        tree_accel_cv,
        cosmo,
        args.mesh_n,
        args.cv_refresh_every,
    )
    tree_main_time = time.perf_counter() - t_tree
    refresh_count = int(np.asarray(cachef[2]))
    print(f"TreePM stochastic-CV kdk wall time: {tree_main_time:.3f} s")
    print(f"Residual refresh count: {refresh_count}")

    rho_ref = _density(pos_ref, args.mesh_n)
    rho_pm = _density(pos_pm, args.mesh_n)
    rho_tree = _density(pos_tree, args.mesh_n)
    rho_ic = _density(pos_i, args.mesh_n)

    boxsize = np.array([args.box_size_mpc_h] * 3)
    kmin = np.pi / args.box_size_mpc_h
    dk = 2.0 * np.pi / args.box_size_mpc_h
    k_ref, pk_ref = power_spectrum(rho_ref, boxsize=boxsize, kmin=kmin, dk=dk)
    k_pm, pk_pm = power_spectrum(rho_pm, boxsize=boxsize, kmin=kmin, dk=dk)
    k_tree, pk_tree = power_spectrum(rho_tree, boxsize=boxsize, kmin=kmin, dk=dk)
    k_cross_pm, cross_pm = cross_correlation_coefficients(rho_ref, rho_pm, boxsize=boxsize, kmin=kmin, dk=dk)
    k_cross_tree, cross_tree = cross_correlation_coefficients(rho_ref, rho_tree, boxsize=boxsize, kmin=kmin, dk=dk)

    cross_pm /= jnp.sqrt(pk_pm * pk_ref)
    cross_tree /= jnp.sqrt(pk_tree * pk_ref)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    _plot_outputs(
        out_prefix,
        np.asarray(k_ref),
        np.asarray(pk_ref),
        np.asarray(k_pm),
        np.asarray(pk_pm),
        np.asarray(k_tree),
        np.asarray(pk_tree),
        np.asarray(k_cross_pm),
        np.asarray(cross_pm),
        np.asarray(k_cross_tree),
        np.asarray(cross_tree),
        rho_ref,
        rho_pm,
        rho_tree,
        rho_ic,
    )

    np.savez(
        out_prefix.with_name(out_prefix.name + "_fields.npz"),
        rho_ref=np.asarray(rho_ref),
        rho_pm=np.asarray(rho_pm),
        rho_tree=np.asarray(rho_tree),
        rho_ic=np.asarray(rho_ic),
    )

    np.savez(
        out_prefix.with_name(out_prefix.name + ".npz"),
        z_ref=np.float32(z_ref),
        a_ref=np.float32(a_ref),
        k_ref=np.asarray(k_ref),
        pk_ref=np.asarray(pk_ref),
        k_pm=np.asarray(k_pm),
        pk_pm=np.asarray(pk_pm),
        k_tree=np.asarray(k_tree),
        pk_tree=np.asarray(pk_tree),
        k_cross_pm=np.asarray(k_cross_pm),
        cross_pm=np.asarray(cross_pm),
        k_cross_tree=np.asarray(k_cross_tree),
        cross_tree=np.asarray(cross_tree),
        time_pm_main_s=np.float64(pm_main_time),
        time_tree_main_s=np.float64(tree_main_time),
        cv_refresh_every=np.int32(args.cv_refresh_every),
        cv_refresh_count=np.int32(refresh_count),
        cv_samples_per_subdomain=np.int32(args.cv_samples_per_subdomain),
        cv_base_seed=np.int32(args.cv_base_seed),
        tree_short_scale=np.float64(short_scale),
    )

    print(f"Saved: {out_prefix.with_name(out_prefix.name + '.npz')}")
    print(f"Saved: {out_prefix.with_name(out_prefix.name + '_fields.npz')}")
    print(f"Saved: {out_prefix.with_name(out_prefix.name + '_pk_cross.png')}")
    print(f"Saved: {out_prefix.with_name(out_prefix.name + '_projected_density.png')}")
    print(f"Saved: {out_prefix.with_name(out_prefix.name + '_ic_projected_density.png')}")
    print(f"Median r(k) PM={float(np.median(np.asarray(cross_pm))):.4f}")
    print(f"Median r(k) TreePM stochastic-CV={float(np.median(np.asarray(cross_tree))):.4f}")
    print(f"Total wall time: {time.perf_counter() - t_start:.3f} s")


def build_parser():
    p = argparse.ArgumentParser(
        description="CV0 z~2 comparison: PM vs TreePM (stochastic octree8 control-variate residual)."
    )
    p.add_argument("--init-cond", type=str, default="/gpfs02/work/diffusion/neural_ode/ic/ics")
    p.add_argument("--snapshot-dir", type=str, default="/gpfs02/work/diffusion/neural_ode/CV0")
    p.add_argument("--target-z", type=float, default=2.0)
    p.add_argument("--mesh-n", type=int, default=128)
    p.add_argument("--box-size-mpc-h", type=float, default=25.0)
    p.add_argument("--integrator", choices=["kdk"], default="kdk")
    p.add_argument("--kdk-steps", type=int, default=256)
    p.add_argument("--pm-r-split-cells", type=float, default=2.0)
    p.add_argument("--bh-softening", type=float, default=0.05)
    p.add_argument("--bh-beta", type=float, default=1.7)
    p.add_argument("--bh-max-depth", type=int, default=4)
    p.add_argument(
        "--bh-prune-enabled",
        type=int,
        default=1,
        help="Enable AABB tree pruning for softened BH short-range (1/0).",
    )
    p.add_argument(
        "--bh-prune-r-cut-mult",
        type=float,
        default=4.0,
        help="Prune radius multiplier: r_cut = bh_prune_r_cut_mult * pm_r_split_cells.",
    )
    p.add_argument("--tree-short-scale", type=str, default="1_over_4pi")
    p.add_argument("--cv-refresh-every", type=int, default=1)
    p.add_argument("--cv-samples-per-subdomain", type=int, default=4)
    p.add_argument("--cv-base-seed", type=int, default=0)
    p.add_argument(
        "--out-prefix",
        type=str,
        default="/home/ben.horowitz/apm/barnes-hutt/stochastic-barnes-hut/python/treepm_debug/cv0_z2_pm_vs_treepm_stochastic_cv",
    )
    return p


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
