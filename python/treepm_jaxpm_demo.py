import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jaxpm.pm as jpm
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental.ode import odeint
from jaxpm.painting import cic_paint

import jax_sbh_ffi


@dataclass(frozen=True)
class Config:
    box_length: float
    n_particles_1d: int
    mesh_n: int
    z_init: float
    z_targets: tuple[float, float, float]
    omega_c: float
    sigma8: float
    bh_beta: float
    bh_softening: float
    bh_max_depth: int
    pm_r_split_cells: float
    tree_short_scale: float
    seed: int


def _make_lattice_positions(n: int) -> jnp.ndarray:
    grid = jnp.arange(n, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(grid, grid, grid, indexing="ij")
    return jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)


def _make_pk_fn(cosmo: jc.Cosmology):
    k = jnp.logspace(-4, 1, 256)
    pk = jc.power.linear_matter_power(cosmo, k)
    return lambda x: jnp.interp(x.reshape([-1]), k, pk).reshape(x.shape)


def _wrap_periodic_mesh(x: jnp.ndarray, mesh_n: int) -> jnp.ndarray:
    return jnp.mod(x, float(mesh_n))


def _bh_short_force_mesh(
    positions_mesh: jnp.ndarray,
    masses: jnp.ndarray,
    cfg: Config,
) -> jnp.ndarray:
    normals = jnp.zeros_like(positions_mesh)
    min_corner = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    max_corner = jnp.array([float(cfg.mesh_n), float(cfg.mesh_n), float(cfg.mesh_n)], dtype=jnp.float32)
    leaf_bytes, node_bytes = jax_sbh_ffi.build_octree8_buffers(
        positions_mesh,
        normals,
        masses,
        min_corner=min_corner,
        max_corner=max_corner,
        max_depth=cfg.bh_max_depth,
    )
    return jax_sbh_ffi.softened_barnes_hut_force_octree8(
        leaf_bytes,
        node_bytes,
        positions_mesh,
        beta=cfg.bh_beta,
        softening_scale=cfg.bh_softening,
        cutoff_scale=cfg.pm_r_split_cells,
        periodic=True,
        box_length=float(cfg.mesh_n),
    )


def _make_treepm_ode_fn(
    mesh_shape: tuple[int, int, int], masses: jnp.ndarray, cfg: Config, short_scale: float
):
    def nbody_ode(state, a, cosmo):
        pos, vel = state
        pm_long = jpm.pm_forces(pos, mesh_shape=mesh_shape, r_split=cfg.pm_r_split_cells)
        short_tree = _bh_short_force_mesh(pos, masses, cfg) * short_scale
        forces = (pm_long + short_tree) * (1.5 * cosmo.Omega_m)

        e = jnp.sqrt(jc.background.Esqr(cosmo, a))
        dpos = vel / (a**3 * e)
        dvel = forces / (a**2 * e)
        return dpos, dvel

    return nbody_ode


def _fit_tree_short_scale(pos0: jnp.ndarray, masses: jnp.ndarray, cfg: Config) -> float:
    mesh_shape = (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n)
    full_pm = jpm.pm_forces(pos0, mesh_shape=mesh_shape, r_split=0.0)
    long_pm = jpm.pm_forces(pos0, mesh_shape=mesh_shape, r_split=cfg.pm_r_split_cells)
    target = full_pm - long_pm
    bh_short = _bh_short_force_mesh(pos0, masses, cfg)
    num = jnp.sum(target * bh_short)
    den = jnp.sum(bh_short * bh_short) + 1e-12
    return float(jnp.maximum(num / den, 0.0))


def _evolve_with_odeint(
    pos0: jnp.ndarray,
    vel0: jnp.ndarray,
    cosmo: jc.Cosmology,
    a_eval: jnp.ndarray,
    ode_fn,
    mesh_n: int,
) -> jnp.ndarray:
    res = odeint(
        ode_fn,
        (pos0, vel0),
        a_eval,
        cosmo,
        rtol=1e-3,
        atol=1e-3,
    )
    return _wrap_periodic_mesh(res[0], mesh_n)


def _projected_density(positions_mesh: jnp.ndarray, cfg: Config) -> np.ndarray:
    mesh = cic_paint(
        jnp.zeros((cfg.mesh_n, cfg.mesh_n, cfg.mesh_n), dtype=jnp.float32),
        _wrap_periodic_mesh(positions_mesh, cfg.mesh_n),
    )
    proj = jnp.sum(mesh, axis=2)
    proj = proj / (jnp.sum(proj) + 1e-8)
    return np.asarray(proj)


def _density_3d(positions_mesh: jnp.ndarray, cfg: Config) -> np.ndarray:
    mesh = cic_paint(
        jnp.zeros((cfg.mesh_n, cfg.mesh_n, cfg.mesh_n), dtype=jnp.float32),
        _wrap_periodic_mesh(positions_mesh, cfg.mesh_n),
    )
    return np.asarray(mesh)


def _power_spectrum_3d(density: np.ndarray, box_length: float, n_bins: int = 24) -> tuple[np.ndarray, np.ndarray]:
    n = density.shape[0]
    delta = density / np.mean(density) - 1.0
    dk = np.fft.rfftn(delta)

    d = box_length / n
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=d).reshape(n, 1, 1)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=d).reshape(1, n, 1)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=d).reshape(1, 1, n // 2 + 1)
    kmag = np.sqrt(kx**2 + ky**2 + kz**2).ravel()

    p3d = (np.abs(dk) ** 2).ravel() * (box_length**3) / (n**6)
    mask = kmag > 0
    kmag = kmag[mask]
    p3d = p3d[mask]

    bins = np.logspace(np.log10(kmag.min()), np.log10(kmag.max()), n_bins + 1)
    ids = np.digitize(kmag, bins) - 1
    k_out = []
    p_out = []
    for i in range(n_bins):
        m = ids == i
        if np.any(m):
            k_out.append(np.exp(np.mean(np.log(kmag[m]))))
            p_out.append(np.mean(p3d[m]))
    return np.asarray(k_out), np.asarray(p_out)


def _plot_projected_fields(
    pm_proj: dict[float, np.ndarray],
    treepm_proj: dict[float, np.ndarray],
    cfg: Config,
    path: str,
) -> None:
    zs = cfg.z_targets
    fig, axes = plt.subplots(2, len(zs), figsize=(4 * len(zs), 7), constrained_layout=True)
    for j, z in enumerate(zs):
        vmax = max(float(pm_proj[z].max()), float(treepm_proj[z].max()), 1e-8)
        im0 = axes[0, j].imshow(pm_proj[z], origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
        axes[0, j].set_title(f"PM z={z:.1f}")
        im1 = axes[1, j].imshow(treepm_proj[z], origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
        axes[1, j].set_title(f"TreePM z={z:.1f}")
        fig.colorbar(im1, ax=[axes[0, j], axes[1, j]], shrink=0.82)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_power_spectra(
    pm_ps: dict[float, tuple[np.ndarray, np.ndarray]],
    treepm_ps: dict[float, tuple[np.ndarray, np.ndarray]],
    cfg: Config,
    path: str,
) -> None:
    zs = cfg.z_targets
    fig, axes = plt.subplots(1, len(zs), figsize=(5 * len(zs), 4), constrained_layout=True)
    if len(zs) == 1:
        axes = [axes]
    for j, z in enumerate(zs):
        k_pm, p_pm = pm_ps[z]
        k_tp, p_tp = treepm_ps[z]
        axes[j].loglog(k_pm, p_pm, label="PM", lw=2)
        axes[j].loglog(k_tp, p_tp, label="TreePM", lw=2)
        axes[j].set_title(f"Power Spectrum z={z:.1f}")
        axes[j].set_xlabel("k")
        axes[j].set_ylabel("P(k)")
        axes[j].grid(True, alpha=0.3, which="both")
        axes[j].legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(cfg: Config, out_npz: str, out_proj_png: str, out_pk_png: str) -> None:
    if not any(dev.platform == "gpu" for dev in jax.devices()):
        raise RuntimeError("GPU backend is required.")

    jax_sbh_ffi.register_build_octree8_buffers()
    jax_sbh_ffi.register_softened_barnes_hut_force_octree8()
    jax_sbh_ffi.register_softened_barnes_hut_force_octree8_vjp()

    n = cfg.mesh_n if cfg.n_particles_1d <= 0 else cfg.n_particles_1d
    npart = n**3
    masses = jnp.ones((npart,), dtype=jnp.float32)
    grid_pos = _make_lattice_positions(n)

    key = jax.random.PRNGKey(cfg.seed)
    cosmo_pk = jc.Planck15(Omega_c=cfg.omega_c, sigma8=cfg.sigma8)
    cosmo_dyn = jc.Planck15(Omega_c=cfg.omega_c, sigma8=cfg.sigma8)
    pk_fn = _make_pk_fn(cosmo_pk)
    init_mesh = jpm.linear_field(
        (cfg.mesh_n, cfg.mesh_n, cfg.mesh_n),
        [cfg.box_length, cfg.box_length, cfg.box_length],
        pk_fn,
        key,
    )
    a_init = 1.0 / (1.0 + cfg.z_init)
    dx_mesh, p_mesh, _ = jpm.lpt(cosmo_dyn, init_mesh, grid_pos, a_init, order=1)

    pos0 = _wrap_periodic_mesh(grid_pos + dx_mesh, cfg.mesh_n)
    vel0 = p_mesh

    target_pairs = sorted(((1.0 / (1.0 + z), z) for z in cfg.z_targets), key=lambda t: t[0])
    if any(a < a_init for a, _ in target_pairs):
        raise ValueError("All target redshifts must be <= z_init so scale factor increases during integration.")
    a_eval = jnp.asarray([a_init] + [a for a, _ in target_pairs], dtype=jnp.float32)

    pm_ode = jpm.make_ode_fn((cfg.mesh_n, cfg.mesh_n, cfg.mesh_n))
    short_scale = cfg.tree_short_scale
    print(f"tree-short-scale (input): {short_scale:.6g}")
    if short_scale <= 0.0:
        short_scale = _fit_tree_short_scale(pos0, masses, cfg)
        print(f"auto tree-short-scale: {short_scale:.6g}")

    treepm_ode = _make_treepm_ode_fn((cfg.mesh_n, cfg.mesh_n, cfg.mesh_n), masses, cfg, short_scale)
    print("pm integration...")
    pm_positions = _evolve_with_odeint(pos0, vel0, cosmo_dyn, a_eval, pm_ode, cfg.mesh_n)
    print("treepm integration...")
    treepm_positions = _evolve_with_odeint(pos0, vel0, cosmo_dyn, a_eval, treepm_ode, cfg.mesh_n)

    pm_snaps = {z: pm_positions[i + 1] for i, (_, z) in enumerate(target_pairs)}
    treepm_snaps = {z: treepm_positions[i + 1] for i, (_, z) in enumerate(target_pairs)}

    pm_proj = {z: _projected_density(pm_snaps[z], cfg) for z in cfg.z_targets}
    treepm_proj = {z: _projected_density(treepm_snaps[z], cfg) for z in cfg.z_targets}

    pm_ps = {z: _power_spectrum_3d(_density_3d(pm_snaps[z], cfg), cfg.box_length) for z in cfg.z_targets}
    treepm_ps = {
        z: _power_spectrum_3d(_density_3d(treepm_snaps[z], cfg), cfg.box_length) for z in cfg.z_targets
    }

    _plot_projected_fields(pm_proj, treepm_proj, cfg, out_proj_png)
    _plot_power_spectra(pm_ps, treepm_ps, cfg, out_pk_png)

    np.savez(
        out_npz,
        z_targets=np.asarray(cfg.z_targets, dtype=np.float32),
        pm_proj_z2=pm_proj[cfg.z_targets[0]],
        pm_proj_z1=pm_proj[cfg.z_targets[1]],
        pm_proj_z0=pm_proj[cfg.z_targets[2]],
        treepm_proj_z2=treepm_proj[cfg.z_targets[0]],
        treepm_proj_z1=treepm_proj[cfg.z_targets[1]],
        treepm_proj_z0=treepm_proj[cfg.z_targets[2]],
        pm_k_z2=pm_ps[cfg.z_targets[0]][0],
        pm_p_z2=pm_ps[cfg.z_targets[0]][1],
        pm_k_z1=pm_ps[cfg.z_targets[1]][0],
        pm_p_z1=pm_ps[cfg.z_targets[1]][1],
        pm_k_z0=pm_ps[cfg.z_targets[2]][0],
        pm_p_z0=pm_ps[cfg.z_targets[2]][1],
        treepm_k_z2=treepm_ps[cfg.z_targets[0]][0],
        treepm_p_z2=treepm_ps[cfg.z_targets[0]][1],
        treepm_k_z1=treepm_ps[cfg.z_targets[1]][0],
        treepm_p_z1=treepm_ps[cfg.z_targets[1]][1],
        treepm_k_z0=treepm_ps[cfg.z_targets[2]][0],
        treepm_p_z0=treepm_ps[cfg.z_targets[2]][1],
    )

    print(f"Saved projected-density plot: {out_proj_png}")
    print(f"Saved power-spectrum plot: {out_pk_png}")
    print(f"Saved arrays: {out_npz}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PM vs TreePM demo with JaxPM ICs and BH short-range.")
    p.add_argument("--box-length", type=float, default=200.0)
    p.add_argument("--n-particles-1d", type=int, default=-1)
    p.add_argument("--mesh-n", type=int, default=24)
    p.add_argument("--z-init", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--omega-c", type=float, default=0.25)
    p.add_argument("--sigma8", type=float, default=0.8)
    p.add_argument("--pm-r-split-cells", type=float, default=1.0)
    p.add_argument("--tree-short-scale", type=float, default=0.10)
    p.add_argument("--bh-beta", type=float, default=1.7)
    p.add_argument("--bh-softening", type=float, default=0.1)
    p.add_argument("--bh-max-depth", type=int, default=6)
    p.add_argument("--out-npz", type=str, default="treepm_jaxpm_demo.npz")
    p.add_argument("--out-proj-png", type=str, default="treepm_jaxpm_projected_density.png")
    p.add_argument("--out-pk-png", type=str, default="treepm_jaxpm_power_spectra.png")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = Config(
        box_length=args.box_length,
        n_particles_1d=args.n_particles_1d,
        mesh_n=args.mesh_n,
        z_init=args.z_init,
        z_targets=(2.0, 1.0, 0.0),
        omega_c=args.omega_c,
        sigma8=args.sigma8,
        bh_beta=args.bh_beta,
        bh_softening=args.bh_softening,
        bh_max_depth=args.bh_max_depth,
        pm_r_split_cells=args.pm_r_split_cells,
        tree_short_scale=args.tree_short_scale,
        seed=args.seed,
    )
    run(cfg, args.out_npz, args.out_proj_png, args.out_pk_png)


if __name__ == "__main__":
    main()
