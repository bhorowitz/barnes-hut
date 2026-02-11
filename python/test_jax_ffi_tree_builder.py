import jax
import jax.numpy as jnp
import numpy as np

import jax_sbh_ffi


def test_octree_builder_ffi_chain_3d():
    jax_sbh_ffi.register_build_octree_level3_buffers()
    jax_sbh_ffi.register_barnes_hut_force()
    jax_sbh_ffi.register_stochastic_barnes_hut_force()
    jax_sbh_ffi.register_softened_barnes_hut_force()
    jax_sbh_ffi.register_softened_stochastic_barnes_hut_force()

    key = jax.random.PRNGKey(0)
    n = 256
    points = jax.random.uniform(
        key, (n, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32
    )
    normals = jnp.zeros_like(points)
    masses = jnp.ones((n,), dtype=jnp.float32)

    leaf_bytes, node_bytes, contrib_bytes = jax_sbh_ffi.build_octree_buffers_with_contrib(
        points, normals, masses, backend="ffi"
    )
    assert leaf_bytes.dtype == jnp.uint8
    assert node_bytes.dtype == jnp.uint8
    assert contrib_bytes.dtype == jnp.uint8

    force_bh = jax_sbh_ffi.barnes_hut_force(leaf_bytes, node_bytes, points, beta=2.0)
    force_sbh = jax_sbh_ffi.stochastic_barnes_hut_force(
        leaf_bytes,
        node_bytes,
        contrib_bytes,
        points,
        samples_per_subdomain=2,
        seed=7,
    )
    assert force_bh.shape == (n, 3)
    assert force_sbh.shape == (n, 3)
    np.testing.assert_allclose(np.isfinite(np.asarray(force_bh)).all(), True)
    np.testing.assert_allclose(np.isfinite(np.asarray(force_sbh)).all(), True)

    force_bh_soft = jax_sbh_ffi.softened_barnes_hut_force(
        leaf_bytes,
        node_bytes,
        points,
        beta=2.0,
        softening_scale=5e-2,
        cutoff_scale=1.5,
    )
    force_sbh_soft = jax_sbh_ffi.softened_stochastic_barnes_hut_force(
        leaf_bytes,
        node_bytes,
        contrib_bytes,
        points,
        samples_per_subdomain=2,
        seed=11,
        softening_scale=5e-2,
        cutoff_scale=1.5,
    )
    assert force_bh_soft.shape == (n, 3)
    assert force_sbh_soft.shape == (n, 3)
    np.testing.assert_allclose(np.isfinite(np.asarray(force_bh_soft)).all(), True)
    np.testing.assert_allclose(np.isfinite(np.asarray(force_sbh_soft)).all(), True)

    @jax.jit
    def step(pos):
        nrm = jnp.zeros_like(pos)
        m = jnp.ones((pos.shape[0],), dtype=pos.dtype)
        leaf_b, node_b, contrib_b = jax_sbh_ffi.build_octree_buffers_with_contrib(
            pos, nrm, m, backend="ffi"
        )
        return jax_sbh_ffi.stochastic_barnes_hut_force(
            leaf_b,
            node_b,
            contrib_b,
            pos,
            samples_per_subdomain=2,
            seed=5,
        )

    force_jit = step(points)
    assert force_jit.shape == (n, 3)
    np.testing.assert_allclose(np.isfinite(np.asarray(force_jit)).all(), True)


def test_octree8_builder_ffi_chain_3d():
    jax_sbh_ffi.register_build_octree8_buffers()
    jax_sbh_ffi.register_barnes_hut_force_octree8()
    jax_sbh_ffi.register_stochastic_barnes_hut_force_octree8()
    jax_sbh_ffi.register_softened_barnes_hut_force_octree8()
    jax_sbh_ffi.register_softened_stochastic_barnes_hut_force_octree8()

    key = jax.random.PRNGKey(1)
    n = 256
    points = jax.random.uniform(
        key, (n, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32
    )
    normals = jnp.zeros_like(points)
    masses = jnp.ones((n,), dtype=jnp.float32)

    leaf_bytes, node_bytes, contrib_bytes = jax_sbh_ffi.build_octree8_buffers_with_contrib(
        points, normals, masses, max_depth=5
    )
    assert leaf_bytes.dtype == jnp.uint8
    assert node_bytes.dtype == jnp.uint8
    assert contrib_bytes.dtype == jnp.uint8

    force_bh = jax_sbh_ffi.barnes_hut_force_octree8(leaf_bytes, node_bytes, points, beta=2.0)
    force_sbh = jax_sbh_ffi.stochastic_barnes_hut_force_octree8(
        leaf_bytes,
        node_bytes,
        points,
        samples_per_subdomain=2,
        seed=7,
    )
    assert force_bh.shape == (n, 3)
    assert force_sbh.shape == (n, 3)
    np.testing.assert_allclose(np.isfinite(np.asarray(force_bh)).all(), True)
    np.testing.assert_allclose(np.isfinite(np.asarray(force_sbh)).all(), True)

    force_bh_soft = jax_sbh_ffi.softened_barnes_hut_force_octree8(
        leaf_bytes,
        node_bytes,
        points,
        beta=2.0,
        softening_scale=5e-2,
        cutoff_scale=1.5,
    )
    force_sbh_soft = jax_sbh_ffi.softened_stochastic_barnes_hut_force_octree8(
        leaf_bytes,
        node_bytes,
        points,
        samples_per_subdomain=2,
        seed=11,
        softening_scale=5e-2,
        cutoff_scale=1.5,
    )
    assert force_bh_soft.shape == (n, 3)
    assert force_sbh_soft.shape == (n, 3)
    np.testing.assert_allclose(np.isfinite(np.asarray(force_bh_soft)).all(), True)
    np.testing.assert_allclose(np.isfinite(np.asarray(force_sbh_soft)).all(), True)


if __name__ == "__main__":
    test_octree_builder_ffi_chain_3d()
    test_octree8_builder_ffi_chain_3d()
    print("OK")
