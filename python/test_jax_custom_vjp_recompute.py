import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_sbh_ffi


def _require_gpu() -> None:
    if not any(dev.platform == "gpu" for dev in jax.devices()):
        pytest.skip("CUDA device required for sbh_ffi tests")


def _make_inputs(n: int = 32):
    key_p, key_m, key_g = jax.random.split(jax.random.PRNGKey(123), 3)
    points = jax.random.uniform(
        key_p, (n, 3), minval=-0.7, maxval=0.7, dtype=jnp.float32
    )
    masses = jax.random.uniform(
        key_m, (n,), minval=0.3, maxval=1.2, dtype=jnp.float32
    )
    cotangent = jax.random.normal(key_g, (n, 3), dtype=jnp.float32)
    normals = jnp.zeros_like(points)
    return points, masses, normals, cotangent


def test_softened_octree8_vjp_matches_finite_difference():
    _require_gpu()
    jax_sbh_ffi.register_build_octree8_buffers()
    jax_sbh_ffi.register_softened_barnes_hut_force_octree8()
    jax_sbh_ffi.register_softened_barnes_hut_force_octree8_vjp()
    points, masses, normals, cotangent = _make_inputs(n=20)

    beta = 1.8
    softening_scale = 0.05
    cutoff_scale = 1.3
    max_depth = 5

    leaf_bytes, node_bytes = jax_sbh_ffi.build_octree8_buffers(
        points, normals, masses, max_depth=max_depth
    )

    grad_vjp = jax_sbh_ffi.softened_barnes_hut_force_octree8_vjp(
        leaf_bytes,
        node_bytes,
        points,
        cotangent,
        beta=beta,
        softening_scale=softening_scale,
        cutoff_scale=cutoff_scale,
    )

    def scalar_loss(q):
        force = jax_sbh_ffi.softened_barnes_hut_force_octree8(
            leaf_bytes,
            node_bytes,
            q,
            beta=beta,
            softening_scale=softening_scale,
            cutoff_scale=cutoff_scale,
        )
        return jnp.sum(force * cotangent)

    eps = 2e-3
    q_np = np.asarray(points)
    fd = np.zeros_like(q_np)
    for i in range(q_np.shape[0]):
        for d in range(3):
            qp = q_np.copy()
            qm = q_np.copy()
            qp[i, d] += eps
            qm[i, d] -= eps
            lp = float(scalar_loss(jnp.asarray(qp, dtype=jnp.float32)))
            lm = float(scalar_loss(jnp.asarray(qm, dtype=jnp.float32)))
            fd[i, d] = (lp - lm) / (2.0 * eps)

    np.testing.assert_allclose(
        np.asarray(grad_vjp),
        fd,
        rtol=4e-2,
        atol=7e-2,
    )


def test_recompute_custom_vjp_matches_fixed_tree_vjp():
    _require_gpu()
    jax_sbh_ffi.register_build_octree8_buffers()
    jax_sbh_ffi.register_softened_barnes_hut_force_octree8()
    jax_sbh_ffi.register_softened_barnes_hut_force_octree8_vjp()
    points, masses, normals, cotangent = _make_inputs(n=104)

    beta = 1.6
    softening_scale = 0.05
    cutoff_scale = 1.4
    max_depth = 5

    def loss_fn(p):
        force = jax_sbh_ffi.softened_barnes_hut_force_octree8_recompute_vjp(
            p,
            masses,
            beta=beta,
            softening_scale=softening_scale,
            cutoff_scale=cutoff_scale,
            max_depth=max_depth,
        )
        return jnp.sum(force * cotangent)

    grad_custom = jax.grad(loss_fn)(points)
    grad_custom_jit = jax.jit(jax.grad(loss_fn))(points)

    leaf_bytes, node_bytes = jax_sbh_ffi.build_octree8_buffers(
        points, normals, masses, max_depth=max_depth
    )
    grad_ref = jax_sbh_ffi.softened_barnes_hut_force_octree8_vjp(
        leaf_bytes,
        node_bytes,
        points,
        cotangent,
        beta=beta,
        softening_scale=softening_scale,
        cutoff_scale=cutoff_scale,
    )

    print("grad_custom",grad_custom)
    print("grad_ref",grad_ref)

    np.testing.assert_allclose(np.asarray(grad_custom), np.asarray(grad_ref), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(grad_custom_jit), np.asarray(grad_ref), rtol=1e-5, atol=1e-6
    )


if __name__ == "__main__":
    test_softened_octree8_vjp_matches_finite_difference()
    test_recompute_custom_vjp_matches_fixed_tree_vjp()
    print("OK")
