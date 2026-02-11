import os
import numpy as np
import jax
import jax.numpy as jnp

import jax_sbh_ffi


def brute_force_force_3d(points: np.ndarray, masses: np.ndarray, queries: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    m = queries.shape[0]
    forces = np.zeros((m, 3), dtype=np.float32)
    for qi in range(m):
        q = queries[qi]
        acc = np.zeros(3, dtype=np.float32)
        for i in range(n):
            dvec = q - points[i]
            dist = np.linalg.norm(dvec)
            acc += -masses[i] / (dist + 1e-1) ** 3 * dvec
        forces[qi] = acc
    return forces


def main():
    np.random.seed(0)

    n = 300
    points = np.random.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    normals = np.zeros_like(points, dtype=np.float32)
    masses = np.full(n, 1.0 / n, dtype=np.float32)

    # Build octree buffers
    leaf_bytes, node_bytes = jax_sbh_ffi.build_octree_buffers(points, normals, masses, level=3)

    # Register FFI targets and run Barnes-Hut force
    jax_sbh_ffi.register_barnes_hut_force()
    forces_bh = jax_sbh_ffi.barnes_hut_force(
        jnp.asarray(leaf_bytes), jnp.asarray(node_bytes), jnp.asarray(points), beta=2.0
    )
    forces_bh = np.asarray(forces_bh)

    # Brute-force CPU reference
    forces_bf = brute_force_force_3d(points, masses, points)

    # Plot comparison
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    labels = ["Fx", "Fy", "Fz"]
    for i in range(3):
        axs[i].scatter(forces_bf[:, i], forces_bh[:, i], s=8, alpha=0.7)
        axs[i].set_title(f"{labels[i]}: brute force vs Barnes–Hut")
        axs[i].set_xlabel("Brute force")
        axs[i].set_ylabel("Barnes–Hut")
        axs[i].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "bh_force_compare_3d.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
