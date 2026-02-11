import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxpm.painting import cic_paint

import jax_sbh_ffi


@dataclass(frozen=True)
class SimConfig:
    n_particles: int
    n_steps: int
    dt: float
    g_const: float
    clip_radius: float
    obs_noise: float
    proj_grid_size: int
    pm_grid_size: int
    bh_beta: float
    bh_softening: float
    bh_cutoff: float
    bh_max_depth: int


_CIC_CONNECTION = jnp.array(
    [[[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0],
      [1.0, 1.0, 0], [1.0, 0, 1.0], [0, 1.0, 1.0], [1.0, 1.0, 1.0]]],
    dtype=jnp.float32,
)


def _cic_readout(mesh: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
    nxyz = jnp.array(mesh.shape, dtype=jnp.int32)
    p = jnp.expand_dims(positions, axis=1)
    floor = jnp.floor(p)
    neigh = floor + _CIC_CONNECTION
    kernel = 1.0 - jnp.abs(p - neigh)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]
    coords = jnp.mod(neigh.astype(jnp.int32), nxyz)
    vals = mesh[coords[..., 0], coords[..., 1], coords[..., 2]]
    return jnp.sum(vals * kernel, axis=-1)


def _project_density_cic_2d(
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    grid_size: int,
    clip_radius: float,
) -> jnp.ndarray:
    r = jnp.linalg.norm(positions, axis=-1)
    keep = (r <= clip_radius).astype(jnp.float32)
    p = jnp.clip(positions, -clip_radius, clip_radius)
    w = masses * keep
    xy = (p[:, :2] / (2.0 * clip_radius) + 0.5) * grid_size
    z = jnp.full((xy.shape[0], 1), 0.5, dtype=xy.dtype)
    pos_grid = jnp.concatenate([xy, z], axis=1)
    mesh = jnp.zeros((grid_size, grid_size, 1), dtype=jnp.float32)
    mesh = cic_paint(mesh, pos_grid, weight=w)
    density = jnp.squeeze(mesh, axis=2)
    return density / (jnp.sum(density) + 1e-8)


def _initial_conditions(
    sigma: jnp.ndarray,
    base_pos_noise: jnp.ndarray,
    base_vel_noise: jnp.ndarray,
    masses: jnp.ndarray,
    g_const: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    sigma = jnp.asarray(sigma, dtype=jnp.float32)
    pos = sigma * base_pos_noise
    total_mass = jnp.sum(masses)
    vel_std = jnp.sqrt(g_const * total_mass / (6.0 * jnp.maximum(sigma, 1e-3)))
    vel = vel_std * base_vel_noise
    pos = pos - jnp.mean(pos, axis=0, keepdims=True)
    vel = vel - jnp.mean(vel, axis=0, keepdims=True)
    return pos, vel


def _pm_accel(
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    g_const: float,
    grid_size: int,
    box_radius: float,
) -> jnp.ndarray:
    # Map [-R, R] -> [0, N)
    n = float(grid_size)
    box_size = 2.0 * box_radius
    dx = box_size / n
    pos_grid = (positions / box_size + 0.5) * n

    rho = cic_paint(jnp.zeros((grid_size, grid_size, grid_size), dtype=jnp.float32), pos_grid, weight=masses)
    delta = rho - jnp.mean(rho)
    delta_k = jnp.fft.rfftn(delta)

    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(grid_size, d=dx).reshape(grid_size, 1, 1)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(grid_size, d=dx).reshape(1, grid_size, 1)
    kz = 2.0 * jnp.pi * jnp.fft.rfftfreq(grid_size, d=dx).reshape(1, 1, grid_size // 2 + 1)
    k2 = kx**2 + ky**2 + kz**2
    k2 = jnp.where(k2 == 0.0, 1.0, k2)

    phi_k = -4.0 * jnp.pi * g_const * delta_k / k2
    phi_k = jnp.where((kx == 0) & (ky == 0) & (kz == 0), 0.0 + 0.0j, phi_k)

    ax = jnp.fft.irfftn(-1j * kx * phi_k, s=(grid_size, grid_size, grid_size)).real
    ay = jnp.fft.irfftn(-1j * ky * phi_k, s=(grid_size, grid_size, grid_size)).real
    az = jnp.fft.irfftn(-1j * kz * phi_k, s=(grid_size, grid_size, grid_size)).real

    acc_x = _cic_readout(ax, pos_grid)
    acc_y = _cic_readout(ay, pos_grid)
    acc_z = _cic_readout(az, pos_grid)
    return jnp.stack([acc_x, acc_y, acc_z], axis=-1)


def _simulate_bh(
    sigma: jnp.ndarray,
    base_pos_noise: jnp.ndarray,
    base_vel_noise: jnp.ndarray,
    masses: jnp.ndarray,
    cfg: SimConfig,
) -> jnp.ndarray:
    pos, vel = _initial_conditions(sigma, base_pos_noise, base_vel_noise, masses, cfg.g_const)

    def _accel_impl(x):
        normals = jnp.zeros_like(x)
        leaf_bytes, node_bytes = jax_sbh_ffi.build_octree8_buffers(
            x, normals, masses, max_depth=cfg.bh_max_depth
        )
        force = jax_sbh_ffi.softened_barnes_hut_force_octree8(
            leaf_bytes,
            node_bytes,
            x,
            beta=cfg.bh_beta,
            softening_scale=cfg.bh_softening,
            cutoff_scale=cfg.bh_cutoff,
        )
        return cfg.g_const * force

    @jax.custom_vjp
    def accel(x):
        return _accel_impl(x)

    def accel_fwd(x):
        y = _accel_impl(x)
        return y, x

    def accel_bwd(x, g):
        normals = jnp.zeros_like(x)
        leaf_bytes, node_bytes = jax_sbh_ffi.build_octree8_buffers(
            x, normals, masses, max_depth=cfg.bh_max_depth
        )
        grad_x = cfg.g_const * jax_sbh_ffi.softened_barnes_hut_force_octree8_vjp(
            leaf_bytes,
            node_bytes,
            x,
            g,
            beta=cfg.bh_beta,
            softening_scale=cfg.bh_softening,
            cutoff_scale=cfg.bh_cutoff,
        )
        return (grad_x,)

    accel.defvjp(accel_fwd, accel_bwd)

    def step(state, _):
        x, v = state
        a0 = accel(x)
        v_half = v + 0.5 * cfg.dt * a0
        x_next = x + cfg.dt * v_half
        a1 = accel(x_next)
        v_next = v_half + 0.5 * cfg.dt * a1
        return (x_next, v_next), None

    (x_final, _), _ = jax.lax.scan(step, (pos, vel), xs=None, length=cfg.n_steps)
    return x_final


def _simulate_pm(
    sigma: jnp.ndarray,
    base_pos_noise: jnp.ndarray,
    base_vel_noise: jnp.ndarray,
    masses: jnp.ndarray,
    cfg: SimConfig,
) -> jnp.ndarray:
    pos, vel = _initial_conditions(sigma, base_pos_noise, base_vel_noise, masses, cfg.g_const)

    def step(state, _):
        x, v = state
        a0 = _pm_accel(x, masses, cfg.g_const, cfg.pm_grid_size, cfg.clip_radius)
        v_half = v + 0.5 * cfg.dt * a0
        x_next = x + cfg.dt * v_half
        a1 = _pm_accel(x_next, masses, cfg.g_const, cfg.pm_grid_size, cfg.clip_radius)
        v_next = v_half + 0.5 * cfg.dt * a1
        return (x_next, v_next), None

    (x_final, _), _ = jax.lax.scan(step, (pos, vel), xs=None, length=cfg.n_steps)
    return x_final


def _pm_density_from_sigma(
    sigma: jnp.ndarray,
    base_pos_noise: jnp.ndarray,
    base_vel_noise: jnp.ndarray,
    masses: jnp.ndarray,
    cfg: SimConfig,
) -> jnp.ndarray:
    pm_final = _simulate_pm(sigma, base_pos_noise, base_vel_noise, masses, cfg)
    return _project_density_cic_2d(pm_final, masses, cfg.proj_grid_size, cfg.clip_radius)


def _save_plots(
    bh_density: np.ndarray,
    pm_density_true_sigma: np.ndarray,
    pm_density_opt: np.ndarray,
    sigma_true: float,
    sigma_opt: float,
    late_compare_path: str,
    recon_compare_path: str,
) -> None:
    # Plot 1: BH late-time vs PM late-time at true sigma.
    vmax1 = max(float(bh_density.max()), float(pm_density_true_sigma.max()), 1e-8)
    fig1, axes1 = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    im0 = axes1[0].imshow(bh_density, origin="lower", cmap="magma", vmin=0.0, vmax=vmax1)
    axes1[0].set_title(f"BH Late-Time (sigma={sigma_true:.3f})")
    im1 = axes1[1].imshow(pm_density_true_sigma, origin="lower", cmap="magma", vmin=0.0, vmax=vmax1)
    axes1[1].set_title("PM Late-Time (same sigma)")
    diff1 = pm_density_true_sigma - bh_density
    im2 = axes1[2].imshow(diff1, origin="lower", cmap="coolwarm")
    axes1[2].set_title("PM - BH")
    fig1.colorbar(im0, ax=[axes1[0], axes1[1]], shrink=0.9)
    fig1.colorbar(im2, ax=axes1[2], shrink=0.9)
    fig1.savefig(late_compare_path, dpi=150)
    plt.close(fig1)

    # Plot 2: PM reconstruction against BH observable.
    vmax2 = max(float(bh_density.max()), float(pm_density_opt.max()), 1e-8)
    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    im3 = axes2[0].imshow(bh_density, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax2)
    axes2[0].set_title("BH Observable (target)")
    im4 = axes2[1].imshow(pm_density_opt, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax2)
    axes2[1].set_title(f"PM Reconstruct (sigma={sigma_opt:.3f})")
    diff2 = pm_density_opt - bh_density
    im5 = axes2[2].imshow(diff2, origin="lower", cmap="coolwarm")
    axes2[2].set_title("Reconstruct - Target")
    fig2.colorbar(im3, ax=[axes2[0], axes2[1]], shrink=0.9)
    fig2.colorbar(im5, ax=axes2[2], shrink=0.9)
    fig2.savefig(recon_compare_path, dpi=150)
    plt.close(fig2)


def run_pipeline(args: argparse.Namespace) -> dict[str, np.ndarray | float]:
    if not any(dev.platform == "gpu" for dev in jax.devices()):
        raise RuntimeError("GPU backend is required.")

    jax_sbh_ffi.register_build_octree8_buffers()
    jax_sbh_ffi.register_softened_barnes_hut_force_octree8()
    jax_sbh_ffi.register_softened_barnes_hut_force_octree8_vjp()

    cfg = SimConfig(
        n_particles=args.n_particles,
        n_steps=args.n_steps,
        dt=args.dt,
        g_const=args.g_const,
        clip_radius=args.clip_radius,
        obs_noise=args.obs_noise,
        proj_grid_size=args.proj_grid_size,
        pm_grid_size=args.pm_grid_size,
        bh_beta=args.bh_beta,
        bh_softening=args.bh_softening,
        bh_cutoff=args.bh_cutoff,
        bh_max_depth=args.bh_max_depth,
    )

    key = jax.random.PRNGKey(args.seed)
    key_pos, key_vel = jax.random.split(key)
    base_pos_noise = jax.random.normal(key_pos, (cfg.n_particles, 3), dtype=jnp.float32)
    base_vel_noise = jax.random.normal(key_vel, (cfg.n_particles, 3), dtype=jnp.float32)
    masses = jnp.ones((cfg.n_particles,), dtype=jnp.float32) / cfg.n_particles

    sigma_true = jnp.array(args.sigma_true, dtype=jnp.float32)
    bh_final = _simulate_bh(sigma_true, base_pos_noise, base_vel_noise, masses, cfg)
    pm_final_true_sigma = _simulate_pm(sigma_true, base_pos_noise, base_vel_noise, masses, cfg)
    bh_density = jax.lax.stop_gradient(
        _project_density_cic_2d(bh_final, masses, cfg.proj_grid_size, cfg.clip_radius)
    )
    pm_density_true_sigma = _project_density_cic_2d(
        pm_final_true_sigma, masses, cfg.proj_grid_size, cfg.clip_radius
    )

    def ll_pm(s):
        pred = _pm_density_from_sigma(s, base_pos_noise, base_vel_noise, masses, cfg)
        resid = pred - bh_density
        return -0.5 * jnp.sum((resid / cfg.obs_noise) ** 2)

    sigma_grid = jnp.linspace(args.sigma_min, args.sigma_max, args.sigma_points, dtype=jnp.float32)
    ll_grid = jax.vmap(ll_pm)(sigma_grid)
    sigma_eval = jnp.array(args.sigma_eval, dtype=jnp.float32)
    ll_eval = ll_pm(sigma_eval)
    grad_eval = jax.grad(ll_pm)(sigma_eval)

    sigma_path = [float(sigma_eval)]
    sigma_cur = sigma_eval
    for _ in range(args.grad_steps):
        g = jax.grad(ll_pm)(sigma_cur)
        sigma_cur = jnp.clip(sigma_cur + args.grad_lr * g, args.sigma_min, args.sigma_max)
        sigma_path.append(float(sigma_cur))

    sigma_opt = sigma_cur
    pm_density_opt = _pm_density_from_sigma(sigma_opt, base_pos_noise, base_vel_noise, masses, cfg)

    return {
        "sigma_true": float(sigma_true),
        "sigma_eval": float(sigma_eval),
        "sigma_opt": float(sigma_opt),
        "ll_eval": float(ll_eval),
        "grad_eval": float(grad_eval),
        "sigma_grid": np.asarray(sigma_grid),
        "ll_grid": np.asarray(ll_grid),
        "sigma_path": np.asarray(sigma_path, dtype=np.float32),
        "bh_density": np.asarray(bh_density),
        "pm_density_true_sigma": np.asarray(pm_density_true_sigma),
        "pm_density_opt": np.asarray(pm_density_opt),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare BH vs PM late-time fields and reconstruct BH target with PM by sigma optimization."
    )
    p.add_argument("--n-particles", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=80)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--g-const", type=float, default=1.0)
    p.add_argument("--clip-radius", type=float, default=1.5)
    p.add_argument("--obs-noise", type=float, default=2e-3)
    p.add_argument("--proj-grid-size", type=int, default=32)
    p.add_argument("--pm-grid-size", type=int, default=64)

    p.add_argument("--bh-beta", type=float, default=1.7)
    p.add_argument("--bh-softening", type=float, default=0.04)
    p.add_argument("--bh-cutoff", type=float, default=3.0)
    p.add_argument("--bh-max-depth", type=int, default=6)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma-true", type=float, default=0.22)
    p.add_argument("--sigma-eval", type=float, default=0.35)
    p.add_argument("--sigma-min", type=float, default=0.12)
    p.add_argument("--sigma-max", type=float, default=0.45)
    p.add_argument("--sigma-points", type=int, default=25)
    p.add_argument("--grad-lr", type=float, default=1e-5)
    p.add_argument("--grad-steps", type=int, default=8)

    p.add_argument("--out-npz", type=str, default="gaussian_sigma_pm_vs_bh.npz")
    p.add_argument("--out-plot-late-compare", type=str, default="gaussian_sigma_pm_vs_bh_late_compare.png")
    p.add_argument("--out-plot-recon-compare", type=str, default="gaussian_sigma_pm_reconstruct_bh.png")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    out = run_pipeline(args)
    np.savez(
        args.out_npz,
        sigma_true=out["sigma_true"],
        sigma_eval=out["sigma_eval"],
        sigma_opt=out["sigma_opt"],
        ll_eval=out["ll_eval"],
        grad_eval=out["grad_eval"],
        sigma_grid=out["sigma_grid"],
        ll_grid=out["ll_grid"],
        sigma_path=out["sigma_path"],
        bh_density=out["bh_density"],
        pm_density_true_sigma=out["pm_density_true_sigma"],
        pm_density_opt=out["pm_density_opt"],
    )
    _save_plots(
        bh_density=out["bh_density"],
        pm_density_true_sigma=out["pm_density_true_sigma"],
        pm_density_opt=out["pm_density_opt"],
        sigma_true=out["sigma_true"],
        sigma_opt=out["sigma_opt"],
        late_compare_path=args.out_plot_late_compare,
        recon_compare_path=args.out_plot_recon_compare,
    )

    print(f"sigma_true={out['sigma_true']:.6f}")
    print(f"sigma_eval={out['sigma_eval']:.6f}")
    print(f"sigma_opt={out['sigma_opt']:.6f}")
    print(f"loglike_pm(sigma_eval)={out['ll_eval']:.6f}")
    print(f"dloglike_pm/dsigma at sigma_eval={out['grad_eval']:.6f}")
    print(f"sigma grid max-likelihood ~= {out['sigma_grid'][np.argmax(out['ll_grid'])]:.6f}")
    print(f"gradient-ascent sigma path: {out['sigma_path']}")
    print(f"saved results to {args.out_npz}")
    print(f"saved plot to {args.out_plot_late_compare}")
    print(f"saved plot to {args.out_plot_recon_compare}")


if __name__ == "__main__":
    main()
