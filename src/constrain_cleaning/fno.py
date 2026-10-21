"""2D Fourier Neural Operator (Li et al., 2020) in Equinox, channels-first (C, Nx, Ny)."""

import equinox as eqx
import jax
import jax.numpy as jnp

from constrain_cleaning.spectral import leray_project


class SpectralConv2d(eqx.Module):
    # Complex weights stored as real/imag parts so optimizers see real parameters.
    w_re: jax.Array
    w_im: jax.Array
    modes: int = eqx.field(static=True)

    def __init__(self, width, modes, key):
        scale = 1.0 / (width * width)
        k1, k2 = jax.random.split(key)
        # Two blocks: positive and negative kx frequencies.
        shape = (2, width, width, modes, modes)
        self.w_re = scale * jax.random.uniform(k1, shape)
        self.w_im = scale * jax.random.uniform(k2, shape)
        self.modes = modes

    def __call__(self, x):
        _, nx, ny = x.shape
        m = self.modes
        w = self.w_re + 1j * self.w_im
        x_h = jnp.fft.rfft2(x)
        out = jnp.zeros((w.shape[2], nx, ny // 2 + 1), dtype=x_h.dtype)
        out = out.at[:, :m, :m].set(jnp.einsum("ixy,ioxy->oxy", x_h[:, :m, :m], w[0]))
        out = out.at[:, -m:, :m].set(jnp.einsum("ixy,ioxy->oxy", x_h[:, -m:, :m], w[1]))
        return jnp.fft.irfft2(out, s=(nx, ny))


class FNO2d(eqx.Module):
    lift: eqx.nn.Conv2d
    spectral: list
    pointwise: list
    proj1: eqx.nn.Conv2d
    proj2: eqx.nn.Conv2d

    def __init__(self, in_channels, out_channels, width=32, modes=12, n_layers=4, *, key):
        keys = jax.random.split(key, 2 * n_layers + 3)
        # +2 input channels for the (x, y) grid coordinates
        self.lift = eqx.nn.Conv2d(in_channels + 2, width, 1, key=keys[0])
        self.spectral = [SpectralConv2d(width, modes, k) for k in keys[1 : n_layers + 1]]
        self.pointwise = [eqx.nn.Conv2d(width, width, 1, key=k) for k in keys[n_layers + 1 : 2 * n_layers + 1]]
        self.proj1 = eqx.nn.Conv2d(width, 4 * width, 1, key=keys[-2])
        self.proj2 = eqx.nn.Conv2d(4 * width, out_channels, 1, key=keys[-1])

    def __call__(self, x):
        _, nx, ny = x.shape
        gx, gy = jnp.meshgrid(jnp.linspace(0, 1, nx, endpoint=False), jnp.linspace(0, 1, ny, endpoint=False), indexing="ij")
        x = self.lift(jnp.concatenate([x, gx[None], gy[None]]))
        n_layers = len(self.spectral)
        for i, (k, w) in enumerate(zip(self.spectral, self.pointwise)):
            x = k(x) + w(x)
            if i < n_layers - 1:
                x = jax.nn.gelu(x)
        return self.proj2(jax.nn.gelu(self.proj1(x)))


class Projected(eqx.Module):
    """G* = pr_C o G~ : wraps any surrogate with the divergence-free projection (eq. 5)."""

    model: eqx.Module

    def __call__(self, x):
        return leray_project(self.model(x))
