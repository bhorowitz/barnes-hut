import jax
import jax.numpy as jnp
import numpy as np

import jax_sbh_ffi


def _ref_softened_direct_force(points, masses, queries, softening_scale, cutoff_scale):
    disp = queries[:, None, :] - points[None, :, :]
    r2 = jnp.sum(disp * disp, axis=-1)
    inv_r = jax.lax.rsqrt(r2 + softening_scale * softening_scale + 1e-12)
    inv_r3 = inv_r * inv_r * inv_r
    dist = jnp.sqrt(r2 + 1e-12)
    u = dist / (2.0 * cutoff_scale)
    attenuation = jax.scipy.special.erfc(u) + (2.0 / np.sqrt(np.pi)) * u * jnp.exp(-(u * u))
    coeff = -masses[None, :] * attenuation * inv_r3
    return jnp.sum(coeff[..., None] * disp, axis=1)


def test_softened_direct_force_ffi_matches_reference():
    jax_sbh_ffi.register_softened_direct_force()

    key = jax.random.PRNGKey(23)
    n = 96
    m = 31
    points = jax.random.uniform(key, (n, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32)
    masses = jax.random.uniform(
        jax.random.PRNGKey(24), (n,), minval=0.1, maxval=1.3, dtype=jnp.float32
    )
    queries = jax.random.uniform(
        jax.random.PRNGKey(25), (m, 3), minval=-0.7, maxval=0.7, dtype=jnp.float32
    )
    softening_scale = 5e-2
    cutoff_scale = 0.9

    force_ffi = jax_sbh_ffi.softened_direct_force(
        points,
        masses,
        queries,
        softening_scale=softening_scale,
        cutoff_scale=cutoff_scale,
    )
    force_ref = _ref_softened_direct_force(
        points, masses, queries, softening_scale=softening_scale, cutoff_scale=cutoff_scale
    )
    np.testing.assert_allclose(np.asarray(force_ffi), np.asarray(force_ref), rtol=5e-5, atol=5e-6)


def test_softened_direct_force_vjp_matches_reference_grad():
    jax_sbh_ffi.register_softened_direct_force_vjp()

    key = jax.random.PRNGKey(31)
    n = 64
    m = 17
    points = jax.random.uniform(key, (n, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32)
    masses = jax.random.uniform(
        jax.random.PRNGKey(32), (n,), minval=0.2, maxval=1.1, dtype=jnp.float32
    )
    queries = jax.random.uniform(
        jax.random.PRNGKey(33), (m, 3), minval=-0.5, maxval=0.5, dtype=jnp.float32
    )
    cotangent = jax.random.normal(jax.random.PRNGKey(34), (m, 3), dtype=jnp.float32)
    softening_scale = 7e-2
    cutoff_scale = 1.4

    grad_ffi = jax_sbh_ffi.softened_direct_force_vjp(
        points,
        masses,
        queries,
        cotangent,
        softening_scale=softening_scale,
        cutoff_scale=cutoff_scale,
    )

    def loss_fn(q):
        force = _ref_softened_direct_force(
            points,
            masses,
            q,
            softening_scale=softening_scale,
            cutoff_scale=cutoff_scale,
        )
        return jnp.sum(force * cotangent)

    grad_ref = jax.grad(loss_fn)(queries)
    np.testing.assert_allclose(np.asarray(grad_ffi), np.asarray(grad_ref), rtol=1e-4, atol=1e-5)


def test_softened_direct_force_recompute_vjp_grad_matches_explicit_vjp():
    key = jax.random.PRNGKey(41)
    n = 72
    points = jax.random.uniform(key, (n, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32)
    masses = jax.random.uniform(
        jax.random.PRNGKey(42), (n,), minval=0.2, maxval=1.1, dtype=jnp.float32
    )
    cotangent = jax.random.normal(jax.random.PRNGKey(43), (n, 3), dtype=jnp.float32)
    softening_scale = 6e-2
    cutoff_scale = 1.1

    def loss_fn(x):
        force = jax_sbh_ffi.softened_direct_force_recompute_vjp(
            x,
            masses,
            softening_scale=softening_scale,
            cutoff_scale=cutoff_scale,
        )
        return jnp.sum(force * cotangent)

    grad_recompute = jax.grad(loss_fn)(points)
    grad_explicit = jax_sbh_ffi.softened_direct_force_vjp(
        points,
        masses,
        points,
        cotangent,
        softening_scale=softening_scale,
        cutoff_scale=cutoff_scale,
    )
    np.testing.assert_allclose(
        np.asarray(grad_recompute),
        np.asarray(grad_explicit),
        rtol=1e-6,
        atol=1e-6,
    )


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


def test_octree8_periodic_minimum_image_shift_invariance():
    jax_sbh_ffi.register_build_octree8_buffers()
    jax_sbh_ffi.register_barnes_hut_force_octree8()

    key = jax.random.PRNGKey(3)
    n = 192
    box_length = 2.0
    points = jax.random.uniform(
        key, (n, 3), minval=0.0, maxval=box_length, dtype=jnp.float32
    )
    normals = jnp.zeros_like(points)
    masses = jnp.ones((n,), dtype=jnp.float32)
    min_corner = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    max_corner = jnp.array([box_length, box_length, box_length], dtype=jnp.float32)
    leaf_bytes, node_bytes = jax_sbh_ffi.build_octree8_buffers(
        points,
        normals,
        masses,
        min_corner=min_corner,
        max_corner=max_corner,
        max_depth=5,
    )

    queries = points[:32]
    queries_shift = queries + jnp.array([box_length, 0.0, 0.0], dtype=jnp.float32)
    force0 = jax_sbh_ffi.barnes_hut_force_octree8(
        leaf_bytes,
        node_bytes,
        queries,
        beta=1.8,
        periodic=True,
        box_length=box_length,
    )
    force1 = jax_sbh_ffi.barnes_hut_force_octree8(
        leaf_bytes,
        node_bytes,
        queries_shift,
        beta=1.8,
        periodic=True,
        box_length=box_length,
    )
    np.testing.assert_allclose(np.asarray(force0), np.asarray(force1), rtol=1e-2, atol=5e-3)


def test_octree8_softened_prune_large_cutoff_matches_disabled():
    jax_sbh_ffi.register_build_octree8_buffers()
    jax_sbh_ffi.register_softened_barnes_hut_force_octree8()

    key = jax.random.PRNGKey(11)
    n = 320
    box_length = 2.0
    points = jax.random.uniform(
        key, (n, 3), minval=0.0, maxval=box_length, dtype=jnp.float32
    )
    normals = jnp.zeros_like(points)
    masses = jnp.ones((n,), dtype=jnp.float32)
    min_corner = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    max_corner = jnp.array([box_length, box_length, box_length], dtype=jnp.float32)
    leaf_bytes, node_bytes = jax_sbh_ffi.build_octree8_buffers(
        points,
        normals,
        masses,
        min_corner=min_corner,
        max_corner=max_corner,
        max_depth=6,
    )

    queries = points[:96]
    force_no_prune = jax_sbh_ffi.softened_barnes_hut_force_octree8(
        leaf_bytes,
        node_bytes,
        queries,
        beta=1.8,
        softening_scale=5e-2,
        cutoff_scale=0.8,
        prune_enabled=False,
        periodic=True,
        box_length=box_length,
    )
    force_huge_cut = jax_sbh_ffi.softened_barnes_hut_force_octree8(
        leaf_bytes,
        node_bytes,
        queries,
        beta=1.8,
        softening_scale=5e-2,
        cutoff_scale=0.8,
        prune_enabled=True,
        prune_r_cut_mult=1e6,
        periodic=True,
        box_length=box_length,
    )
    np.testing.assert_allclose(
        np.asarray(force_no_prune),
        np.asarray(force_huge_cut),
        rtol=1e-6,
        atol=1e-6,
    )


def test_octree8_softened_prune_periodic_shift_invariance():
    jax_sbh_ffi.register_build_octree8_buffers()
    jax_sbh_ffi.register_softened_barnes_hut_force_octree8()

    key = jax.random.PRNGKey(13)
    n = 256
    box_length = 3.0
    points = jax.random.uniform(
        key, (n, 3), minval=0.0, maxval=box_length, dtype=jnp.float32
    )
    normals = jnp.zeros_like(points)
    masses = jnp.ones((n,), dtype=jnp.float32)
    min_corner = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    max_corner = jnp.array([box_length, box_length, box_length], dtype=jnp.float32)
    leaf_bytes, node_bytes = jax_sbh_ffi.build_octree8_buffers(
        points,
        normals,
        masses,
        min_corner=min_corner,
        max_corner=max_corner,
        max_depth=6,
    )

    queries = points[:48]
    queries_shift = queries + jnp.array([box_length, 0.0, 0.0], dtype=jnp.float32)
    force0 = jax_sbh_ffi.softened_barnes_hut_force_octree8(
        leaf_bytes,
        node_bytes,
        queries,
        beta=1.9,
        softening_scale=4e-2,
        cutoff_scale=0.7,
        prune_enabled=True,
        prune_r_cut_mult=3.0,
        periodic=True,
        box_length=box_length,
    )
    force1 = jax_sbh_ffi.softened_barnes_hut_force_octree8(
        leaf_bytes,
        node_bytes,
        queries_shift,
        beta=1.9,
        softening_scale=4e-2,
        cutoff_scale=0.7,
        prune_enabled=True,
        prune_r_cut_mult=3.0,
        periodic=True,
        box_length=box_length,
    )
    np.testing.assert_allclose(np.asarray(force0), np.asarray(force1), rtol=1e-2, atol=5e-3)


def test_octree8_softened_stochastic_prune_large_cutoff_matches_disabled():
    jax_sbh_ffi.register_build_octree8_buffers()
    jax_sbh_ffi.register_softened_stochastic_barnes_hut_force_octree8()

    key = jax.random.PRNGKey(17)
    n = 320
    box_length = 2.0
    points = jax.random.uniform(
        key, (n, 3), minval=0.0, maxval=box_length, dtype=jnp.float32
    )
    normals = jnp.zeros_like(points)
    masses = jnp.ones((n,), dtype=jnp.float32)
    min_corner = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    max_corner = jnp.array([box_length, box_length, box_length], dtype=jnp.float32)
    leaf_bytes, node_bytes = jax_sbh_ffi.build_octree8_buffers(
        points,
        normals,
        masses,
        min_corner=min_corner,
        max_corner=max_corner,
        max_depth=6,
    )

    queries = points[:96]
    force_no_prune = jax_sbh_ffi.softened_stochastic_barnes_hut_force_octree8(
        leaf_bytes,
        node_bytes,
        queries,
        samples_per_subdomain=3,
        seed=11,
        softening_scale=5e-2,
        cutoff_scale=0.8,
        prune_enabled=False,
        periodic=True,
        box_length=box_length,
    )
    force_huge_cut = jax_sbh_ffi.softened_stochastic_barnes_hut_force_octree8(
        leaf_bytes,
        node_bytes,
        queries,
        samples_per_subdomain=3,
        seed=11,
        softening_scale=5e-2,
        cutoff_scale=0.8,
        prune_enabled=True,
        prune_r_cut_mult=1e6,
        periodic=True,
        box_length=box_length,
    )
    np.testing.assert_allclose(
        np.asarray(force_no_prune),
        np.asarray(force_huge_cut),
        rtol=1e-6,
        atol=1e-6,
    )


def test_octree8_softened_stochastic_prune_periodic_shift_invariance():
    jax_sbh_ffi.register_build_octree8_buffers()
    jax_sbh_ffi.register_softened_stochastic_barnes_hut_force_octree8()

    key = jax.random.PRNGKey(19)
    n = 256
    box_length = 3.0
    points = jax.random.uniform(
        key, (n, 3), minval=0.0, maxval=box_length, dtype=jnp.float32
    )
    normals = jnp.zeros_like(points)
    masses = jnp.ones((n,), dtype=jnp.float32)
    min_corner = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    max_corner = jnp.array([box_length, box_length, box_length], dtype=jnp.float32)
    leaf_bytes, node_bytes = jax_sbh_ffi.build_octree8_buffers(
        points,
        normals,
        masses,
        min_corner=min_corner,
        max_corner=max_corner,
        max_depth=6,
    )

    queries = points[:48]
    queries_shift = queries + jnp.array([box_length, 0.0, 0.0], dtype=jnp.float32)
    force0 = jax_sbh_ffi.softened_stochastic_barnes_hut_force_octree8(
        leaf_bytes,
        node_bytes,
        queries,
        samples_per_subdomain=3,
        seed=7,
        softening_scale=4e-2,
        cutoff_scale=0.7,
        prune_enabled=True,
        prune_r_cut_mult=3.0,
        periodic=True,
        box_length=box_length,
    )
    force1 = jax_sbh_ffi.softened_stochastic_barnes_hut_force_octree8(
        leaf_bytes,
        node_bytes,
        queries_shift,
        samples_per_subdomain=3,
        seed=7,
        softening_scale=4e-2,
        cutoff_scale=0.7,
        prune_enabled=True,
        prune_r_cut_mult=3.0,
        periodic=True,
        box_length=box_length,
    )
    np.testing.assert_allclose(np.asarray(force0), np.asarray(force1), rtol=3e-2, atol=1e-2)


if __name__ == "__main__":
    test_softened_direct_force_ffi_matches_reference()
    test_softened_direct_force_vjp_matches_reference_grad()
    test_softened_direct_force_recompute_vjp_grad_matches_explicit_vjp()
    test_octree_builder_ffi_chain_3d()
    test_octree8_builder_ffi_chain_3d()
    test_octree8_periodic_minimum_image_shift_invariance()
    test_octree8_softened_prune_large_cutoff_matches_disabled()
    test_octree8_softened_prune_periodic_shift_invariance()
    test_octree8_softened_stochastic_prune_large_cutoff_matches_disabled()
    test_octree8_softened_stochastic_prune_periodic_shift_invariance()
    print("OK")
