import jax
import jax.numpy as jnp
import numpy as np

from constrain_cleaning.spectral import divergence, leray_project, wavenumbers

jax.config.update("jax_enable_x64", True)


def _random_field(n=64, seed=0):
    return jax.random.normal(jax.random.PRNGKey(seed), (2, n, n), dtype=jnp.float64)


def _rms(a):
    return float(jnp.sqrt(jnp.mean(a**2)))


def test_projection_is_divergence_free():
    u = _random_field()
    assert _rms(divergence(u)) > 1.0
    assert _rms(divergence(leray_project(u))) < 1e-10


def test_projection_is_idempotent():
    p = leray_project(_random_field())
    np.testing.assert_allclose(leray_project(p), p, atol=1e-12)


def test_solenoidal_field_unchanged_and_gradient_removed():
    n = 64
    kx, ky = wavenumbers(n, n)
    phi_h = jnp.fft.rfft2(_random_field(n)[0])
    curl = jnp.fft.irfft2(jnp.stack([1j * ky * phi_h, -1j * kx * phi_h]), s=(n, n))
    grad = jnp.fft.irfft2(jnp.stack([1j * kx * phi_h, 1j * ky * phi_h]), s=(n, n))
    np.testing.assert_allclose(leray_project(curl), curl, atol=1e-12)
    np.testing.assert_allclose(leray_project(grad), 0.0, atol=1e-12)


def test_projection_works_on_batches_and_odd_grids():
    u = jax.random.normal(jax.random.PRNGKey(1), (3, 2, 33, 40), dtype=jnp.float64)
    assert _rms(divergence(leray_project(u))) < 1e-10
