"""Spectral operators on the periodic box [0, 2pi)^2.

Fields are channels-first, ``(..., C, Nx, Ny)``, with axis -2 = x and axis -1 = y.
Spectral quantities use the ``rfft2`` layout ``(..., Nx, Ny // 2 + 1)``.
"""

import jax.numpy as jnp


def wavenumbers(nx, ny, zero_nyquist=True):
    """Integer wavenumbers (kx, ky) broadcastable to the rfft2 grid.

    With ``zero_nyquist`` the Nyquist wavenumbers are set to 0, the usual
    convention for odd spectral derivatives. This makes ``i k`` a real operator
    (it maps Hermitian spectra to Hermitian spectra), so the Leray projection
    below returns a real field that is exactly divergence-free under the same
    spectral divergence. This is the role of the conjugate-symmetry / real-Nyquist
    constraints in Appendix B of Duruisseaux et al. (2024).
    """
    kx = jnp.fft.fftfreq(nx, 1.0 / nx)
    ky = jnp.fft.rfftfreq(ny, 1.0 / ny)
    if zero_nyquist:
        if nx % 2 == 0:
            kx = kx.at[nx // 2].set(0.0)
        if ny % 2 == 0:
            ky = ky.at[ny // 2].set(0.0)
    return kx[:, None], ky[None, :]


def divergence(u):
    """Spectral divergence of a velocity field ``u`` with shape (..., 2, Nx, Ny)."""
    nx, ny = u.shape[-2:]
    kx, ky = wavenumbers(nx, ny)
    uh = jnp.fft.rfft2(u)
    div_h = 1j * (kx * uh[..., 0, :, :] + ky * uh[..., 1, :, :])
    return jnp.fft.irfft2(div_h, s=(nx, ny))


def leray_project(u):
    """L2-orthogonal projection of ``u`` onto divergence-free fields.

    Closed-form solution of the least-norm problem (12)-(13) in Duruisseaux et
    al. (2024): the constraint xi . u_hat(xi) = 0 decouples over modes, so
    C^+ C acts per mode as xi xi^T / |xi|^2.
    """
    nx, ny = u.shape[-2:]
    kx, ky = wavenumbers(nx, ny)
    k2 = kx**2 + ky**2
    k2 = jnp.where(k2 == 0, 1.0, k2)  # mean (and pure-Nyquist) modes are left as is
    uh = jnp.fft.rfft2(u)
    div_h = (kx * uh[..., 0, :, :] + ky * uh[..., 1, :, :]) / k2
    uh = jnp.stack([uh[..., 0, :, :] - kx * div_h, uh[..., 1, :, :] - ky * div_h], axis=-3)
    return jnp.fft.irfft2(uh, s=(nx, ny))
