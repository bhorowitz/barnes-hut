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
    beta: float
    softening_scale: float
    cutoff_scale: float
    max_depth: int
    grid_size: int
    clip_radius: float
    obs_noise: float
    g_const: float


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

    # Map physical coordinates in [-clip_radius, clip_radius] to grid coordinates.
    xy = (p[:, :2] / (2.0 * clip_radius) + 0.5) * grid_size
    z = jnp.full((xy.shape[0], 1), 0.5, dtype=xy.dtype)
    pos_grid = jnp.concatenate([xy, z], axis=1)
    mesh = jnp.zeros((grid_size, grid_size, 1), dtype=jnp.float32)
    mesh = cic_paint(mesh, pos_grid, weight=w)
    density = jnp.squeeze(mesh, axis=2)
    # Normalize to a probability-like map for stable likelihood scaling.
    return density / (jnp.sum(density) + 1e-8)


def _simulate_final_positions(
    sigma: jnp.ndarray,
    base_pos_noise: jnp.ndarray,
    base_vel_noise: jnp.ndarray,
    masses: jnp.ndarray,
    cfg: SimConfig,
) -> jnp.ndarray:
    sigma = jnp.asarray(sigma, dtype=jnp.float32)
    pos = sigma * base_pos_noise
    total_mass = jnp.sum(masses)
    vel_std = jnp.sqrt(cfg.g_const * total_mass / (6.0 * jnp.maximum(sigma, 1e-3)))
    vel = vel_std * base_vel_noise

    # Remove bulk translation and drift.
    pos = pos - jnp.mean(pos, axis=0, keepdims=True)
    vel = vel - jnp.mean(vel, axis=0, keepdims=True)

    def _accel_impl(x):
        normals = jnp.zeros_like(x)
        leaf_bytes, node_bytes = jax_sbh_ffi.build_octree8_buffers(
            x,
            normals,
            masses,
            max_depth=cfg.max_depth,
        )
        force = jax_sbh_ffi.softened_barnes_hut_force_octree8(
            leaf_bytes,
            node_bytes,
            x,
            beta=cfg.beta,
            softening_scale=cfg.softening_scale,
            cutoff_scale=cfg.cutoff_scale,
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
            x,
            normals,
            masses,
            max_depth=cfg.max_depth,
        )
        grad_x = cfg.g_const * jax_sbh_ffi.softened_barnes_hut_force_octree8_vjp(
            leaf_bytes,
            node_bytes,
            x,
            g,
            beta=cfg.beta,
            softening_scale=cfg.softening_scale,
            cutoff_scale=cfg.cutoff_scale,
        )
        return (grad_x,)

    accel.defvjp(accel_fwd, accel_bwd)

    def leapfrog_step(state, _):
        x, v = state
        a0 = accel(x)
        v_half = v + 0.5 * cfg.dt * a0
        x_next = x + cfg.dt * v_half
        a1 = accel(x_next)
        v_next = v_half + 0.5 * cfg.dt * a1
        return (x_next, v_next), None

    (final_pos, _), _ = jax.lax.scan(leapfrog_step, (pos, vel), xs=None, length=cfg.n_steps)
    return final_pos


def _initial_positions(
    sigma: jnp.ndarray,
    base_pos_noise: jnp.ndarray,
) -> jnp.ndarray:
    sigma = jnp.asarray(sigma, dtype=jnp.float32)
    pos = sigma * base_pos_noise
    return pos - jnp.mean(pos, axis=0, keepdims=True)


def _model_density(
    sigma: jnp.ndarray,
    base_pos_noise: jnp.ndarray,
    base_vel_noise: jnp.ndarray,
    masses: jnp.ndarray,
    cfg: SimConfig,
) -> jnp.ndarray:
    final_pos = _simulate_final_positions(sigma, base_pos_noise, base_vel_noise, masses, cfg)
    return _project_density_cic_2d(final_pos, masses, cfg.grid_size, cfg.clip_radius)


def _log_likelihood(
    sigma: jnp.ndarray,
    observed_density: jnp.ndarray,
    base_pos_noise: jnp.ndarray,
    base_vel_noise: jnp.ndarray,
    masses: jnp.ndarray,
    cfg: SimConfig,
) -> jnp.ndarray:
    pred = _model_density(sigma, base_pos_noise, base_vel_noise, masses, cfg)
    # Gaussian pixel-noise likelihood.
    resid = pred - observed_density
    return -0.5 * jnp.sum((resid / cfg.obs_noise) ** 2)


def run_pipeline(args: argparse.Namespace) -> dict[str, np.ndarray | float]:
    if not any(dev.platform == "gpu" for dev in jax.devices()):
        raise RuntimeError("GPU backend is required for this script.")

    jax_sbh_ffi.register_build_octree8_buffers()
    jax_sbh_ffi.register_softened_barnes_hut_force_octree8()
    jax_sbh_ffi.register_softened_barnes_hut_force_octree8_vjp()

    cfg = SimConfig(
        n_particles=args.n_particles,
        n_steps=args.n_steps,
        dt=args.dt,
        beta=args.beta,
        softening_scale=args.softening,
        cutoff_scale=args.cutoff,
        max_depth=args.max_depth,
        grid_size=args.grid_size,
        clip_radius=args.clip_radius,
        obs_noise=args.obs_noise,
        g_const=args.g_const,
    )

    key = jax.random.PRNGKey(args.seed)
    key_pos, key_vel = jax.random.split(key)
    base_pos_noise = jax.random.normal(key_pos, (cfg.n_particles, 3), dtype=jnp.float32)
    base_vel_noise = jax.random.normal(key_vel, (cfg.n_particles, 3), dtype=jnp.float32)
    masses = jnp.ones((cfg.n_particles,), dtype=jnp.float32) / cfg.n_particles
    print("initializing...")
    sigma_true = jnp.array(args.sigma_true, dtype=jnp.float32)
    initial_true_density = _project_density_cic_2d(
        _initial_positions(sigma_true, base_pos_noise),
        masses,
        cfg.grid_size,
        cfg.clip_radius,
    )
    print("simulating......")

    observed_density = jax.lax.stop_gradient(
        _model_density(sigma_true, base_pos_noise, base_vel_noise, masses, cfg)
    )
    print("gradienting......")

    sigma_grid = jnp.linspace(args.sigma_min, args.sigma_max, args.sigma_points, dtype=jnp.float32)
    ll_fn = lambda s: _log_likelihood(
        s, observed_density, base_pos_noise, base_vel_noise, masses, cfg
    )
    ll_values = jax.vmap(ll_fn)(sigma_grid)

    sigma_eval = jnp.array(args.sigma_eval, dtype=jnp.float32)
    ll_eval = ll_fn(sigma_eval)
    grad_eval = jax.grad(ll_fn)(sigma_eval)
    print("iterating......")

    # Demonstrate backprop-through-pipeline via a few gradient ascent steps.
    sigma_path = [float(sigma_eval)]
    sigma_cur = sigma_eval
    for _ in range(args.grad_steps):
        g = jax.grad(ll_fn)(sigma_cur)
        print("Current sigma:", float(sigma_cur), "g:", float(g))
        sigma_cur = jnp.clip(sigma_cur + args.grad_lr * g, args.sigma_min, args.sigma_max)
        sigma_path.append(float(sigma_cur))

    sigma_opt = sigma_cur
    opt_final_density = _model_density(
        sigma_opt, base_pos_noise, base_vel_noise, masses, cfg
    )

    return {
        "sigma_true": float(sigma_true),
        "sigma_eval": float(sigma_eval),
        "sigma_opt": float(sigma_opt),
        "ll_eval": float(ll_eval),
        "grad_eval": float(grad_eval),
        "sigma_grid": np.asarray(sigma_grid),
        "ll_grid": np.asarray(ll_values),
        "sigma_path": np.asarray(sigma_path, dtype=np.float32),
        "initial_true_density": np.asarray(initial_true_density),
        "opt_final_density": np.asarray(opt_final_density),
        "observed_density": np.asarray(observed_density),
    }


def _save_density_plots(
    initial_true_density: np.ndarray,
    observed_density: np.ndarray,
    opt_final_density: np.ndarray,
    sigma_true: float,
    sigma_opt: float,
    ic_final_plot_path: str,
    opt_vs_obs_plot_path: str,
) -> None:
    vmax_1 = max(float(initial_true_density.max()), float(observed_density.max()), 1e-8)
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im0 = axes1[0].imshow(initial_true_density, origin="lower", cmap="magma", vmin=0.0, vmax=vmax_1)
    axes1[0].set_title(f"IC Projected Density (sigma={sigma_true:.3f})")
    axes1[0].set_xlabel("x")
    axes1[0].set_ylabel("y")
    im1 = axes1[1].imshow(observed_density, origin="lower", cmap="magma", vmin=0.0, vmax=vmax_1)
    axes1[1].set_title("Final Projected Density (Observable)")
    axes1[1].set_xlabel("x")
    axes1[1].set_ylabel("y")
    fig1.colorbar(im1, ax=axes1.ravel().tolist(), shrink=0.9)
    fig1.savefig(ic_final_plot_path, dpi=140)
    plt.close(fig1)

    vmax_2 = max(float(observed_density.max()), float(opt_final_density.max()), 1e-8)
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im2 = axes2[0].imshow(observed_density, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax_2)
    axes2[0].set_title("Modeled Observable (Target)")
    axes2[0].set_xlabel("x")
    axes2[0].set_ylabel("y")
    im3 = axes2[1].imshow(opt_final_density, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax_2)
    axes2[1].set_title(f"Optimization Final (sigma={sigma_opt:.3f})")
    axes2[1].set_xlabel("x")
    axes2[1].set_ylabel("y")
    fig2.colorbar(im3, ax=axes2.ravel().tolist(), shrink=0.9)
    fig2.savefig(opt_vs_obs_plot_path, dpi=140)
    plt.close(fig2)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Trace likelihood vs initial Gaussian radius (sigma) for a 3D particle sphere "
            "evolved with softened Barnes-Hut, and compute dL/dsigma through custom VJP."
        )
    )
    p.add_argument("--n-particles", type=int, default=10000)
    p.add_argument("--n-steps", type=int, default=160)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--beta", type=float, default=1.7)
    p.add_argument("--softening", type=float, default=0.04)
    p.add_argument("--cutoff", type=float, default=3.0)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--grid-size", type=int, default=32)
    p.add_argument("--clip-radius", type=float, default=1.5)
    p.add_argument("--obs-noise", type=float, default=2e-3)
    p.add_argument("--g-const", type=float, default=.10)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--sigma-true", type=float, default=0.22)
    p.add_argument("--sigma-eval", type=float, default=0.26)
    p.add_argument("--sigma-min", type=float, default=0.10)
    p.add_argument("--sigma-max", type=float, default=0.45)
    p.add_argument("--sigma-points", type=int, default=25)

    p.add_argument("--grad-lr", type=float, default=3.5e-4)
    p.add_argument("--grad-steps", type=int, default=100)
    p.add_argument("--out-npz", type=str, default="gaussian_sigma_likelihood_bh.npz")
    p.add_argument(
        "--out-plot-ic-final",
        type=str,
        default="gaussian_sigma_likelihood_ic_vs_final.png",
    )
    p.add_argument(
        "--out-plot-opt-vs-obs",
        type=str,
        default="gaussian_sigma_likelihood_opt_vs_observable.png",
    )
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
        initial_true_density=out["initial_true_density"],
        opt_final_density=out["opt_final_density"],
        observed_density=out["observed_density"],
    )
    _save_density_plots(
        initial_true_density=out["initial_true_density"],
        observed_density=out["observed_density"],
        opt_final_density=out["opt_final_density"],
        sigma_true=out["sigma_true"],
        sigma_opt=out["sigma_opt"],
        ic_final_plot_path=args.out_plot_ic_final,
        opt_vs_obs_plot_path=args.out_plot_opt_vs_obs,
    )

    print(f"sigma_true={out['sigma_true']:.6f}")
    print(f"sigma_eval={out['sigma_eval']:.6f}")
    print(f"sigma_opt={out['sigma_opt']:.6f}")
    print(f"loglike(sigma_eval)={out['ll_eval']:.6f}")
    print(f"dloglike/dsigma at sigma_eval={out['grad_eval']:.6f}")
    print(f"sigma grid max-likelihood ~= {out['sigma_grid'][np.argmax(out['ll_grid'])]:.6f}")
    print(f"gradient-ascent sigma path: {out['sigma_path']}")
    print(f"saved results to {args.out_npz}")
    print(f"saved plot to {args.out_plot_ic_final}")
    print(f"saved plot to {args.out_plot_opt_vs_obs}")


if __name__ == "__main__":
    main()
