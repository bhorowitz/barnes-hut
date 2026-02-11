import numpy as np
import jax
import jax.numpy as jnp

import sbh_gpu

import jax_sbh


def _circle_points(n: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    points = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    normals = points.copy()
    masses = np.full(n, 1.0 / n, dtype=np.float32)
    return points, normals, masses


def test_gravity_2d_bruteforce_wrapper():
    points, normals, masses = _circle_points(64)
    min_corner = points.min(axis=0) - 0.01
    max_corner = points.max(axis=0) + 0.01

    tree = jax_sbh.build_gpu_worker(points, normals, masses, min_corner, max_corner, level=1)

    queries = np.array([[0.0, 0.0], [0.1, 0.2], [0.5, -0.1]], dtype=np.float32)
    try:
        gt, _ = sbh_gpu.eval_brute_force_gravity(jax_sbh.build_gpu_tree(points, normals, masses, min_corner, max_corner).gpu_tree, queries)
        got = jax_sbh.eval_brute_force_gravity(tree, jnp.asarray(queries))
        np.testing.assert_allclose(np.asarray(got), gt, rtol=1e-5, atol=1e-6)
    finally:
        tree.close()


def test_gravity_2d_barnes_hut_wrapper():
    points, normals, masses = _circle_points(128)
    min_corner = points.min(axis=0) - 0.01
    max_corner = points.max(axis=0) + 0.01

    tree = jax_sbh.build_gpu_worker(points, normals, masses, min_corner, max_corner, level=1)

    queries = np.array([[0.0, 0.0], [0.25, 0.1], [0.5, -0.25]], dtype=np.float32)
    beta = 2.0

    try:
        gt_tree = jax_sbh.build_gpu_tree(points, normals, masses, min_corner, max_corner, level=1)
        gt, _ = sbh_gpu.eval_barnes_hut_gravity(gt_tree.gpu_tree, queries, beta)
        got = jax_sbh.eval_barnes_hut_gravity(tree, jnp.asarray(queries), beta=beta)
        np.testing.assert_allclose(np.asarray(got), gt, rtol=1e-5, atol=1e-6)
    finally:
        tree.close()


if __name__ == "__main__":
    test_gravity_2d_bruteforce_wrapper()
    test_gravity_2d_barnes_hut_wrapper()
    print("OK")
