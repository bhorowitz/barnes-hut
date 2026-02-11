import os
import numpy as np
import jax.numpy as jnp

import jax_sbh_ffi


def brute_force_gravity(points: np.ndarray, masses: np.ndarray, queries: np.ndarray) -> np.ndarray:
    out = np.zeros((queries.shape[0],), dtype=np.float32)
    for qi, q in enumerate(queries):
        acc = 0.0
        for p, m in zip(points, masses):
            dist = np.linalg.norm(p - q)
            acc += -m / (dist + 1e-1)
        out[qi] = acc
    return out


def main():
    np.random.seed(0)

    n = 300
    points = np.random.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)
    normals = np.zeros_like(points, dtype=np.float32)
    masses = np.full(n, 1.0 / n, dtype=np.float32)

    leaf, node, contrib = jax_sbh_ffi.build_quadtree_buffers_with_contrib(
        points, normals, masses, level=3
    )

    jax_sbh_ffi.register_stochastic_barnes_hut_gravity()

    queries = points
    sbh = jax_sbh_ffi.stochastic_barnes_hut_gravity(
        jnp.asarray(leaf),
        jnp.asarray(node),
        jnp.asarray(contrib),
        jnp.asarray(queries),
        samples_per_subdomain=2,
        seed=123,
    )
    sbh = np.asarray(sbh)

    bf = brute_force_gravity(points, masses, queries)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    ax.scatter(bf, sbh, s=8, alpha=0.7)
    ax.set_title("2D Stochastic BH vs Brute Force")
    ax.set_xlabel("Brute force")
    ax.set_ylabel("Stochastic BH")
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(os.getcwd(), "sbh_stochastic_compare_2d.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
