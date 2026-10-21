"""Pseudo-spectral solver for 2D Kolmogorov flow (vorticity form).

    d_t w + u . grad w = (1/Re) lap w - 4 cos(4y),     u = (d_y psi, -d_x psi),  -lap psi = w

i.e. the curl of  d_t u + u . grad u = -grad p + (1/Re) lap u + sin(4y) x_hat  (eq. 15).
Time stepping: integrating-factor RK4 (viscosity exact), 2/3-rule dealiasing.
"""

import jax
import jax.numpy as jnp

from constrain_cleaning.spectral import wavenumbers


def _grid(n):
    x = jnp.arange(n) * 2 * jnp.pi / n
    return jnp.meshgrid(x, x, indexing="ij")


def _operators(n):
    kx, ky = wavenumbers(n, n, zero_nyquist=False)
    k2 = kx**2 + ky**2
    inv_k2 = jnp.where(k2 == 0, 0.0, 1.0 / jnp.where(k2 == 0, 1.0, k2))
    kmax = n // 3
    dealias = (jnp.abs(kx) <= kmax) & (jnp.abs(ky) <= kmax)
    return kx, ky, k2, inv_k2, dealias


def velocity_from_vorticity(w_h, n):
    """Velocity (2, n, n) from the rfft2 of vorticity."""
    kx, ky, _, inv_k2, _ = _operators(n)
    psi_h = w_h * inv_k2
    u_h = jnp.stack([1j * ky * psi_h, -1j * kx * psi_h])
    return jnp.fft.irfft2(u_h, s=(n, n))


def make_stepper(n, re, dt):
    kx, ky, k2, inv_k2, dealias = _operators(n)
    _, y = _grid(n)
    forcing_h = jnp.fft.rfft2(-4.0 * jnp.cos(4.0 * y))
    e_half = jnp.exp(-k2 / re * dt / 2)
    e_full = e_half**2

    def rhs(w_h):
        psi_h = w_h * inv_k2
        u = jnp.fft.irfft2(1j * ky * psi_h, s=(n, n))
        v = jnp.fft.irfft2(-1j * kx * psi_h, s=(n, n))
        wx = jnp.fft.irfft2(1j * kx * w_h, s=(n, n))
        wy = jnp.fft.irfft2(1j * ky * w_h, s=(n, n))
        return -jnp.fft.rfft2(u * wx + v * wy) * dealias + forcing_h

    def step(w_h):
        k1 = dt * rhs(w_h)
        k2_ = dt * rhs(e_half * (w_h + k1 / 2))
        k3 = dt * rhs(e_half * w_h + k2_ / 2)
        k4 = dt * rhs(e_full * w_h + e_half * k3)
        return e_full * w_h + (e_full * k1 + 2 * e_half * (k2_ + k3) + k4) / 6

    return step


def random_vorticity(key, n, alpha=2.5, tau=7.0):
    """Gaussian random field with spectrum ~ (|k|^2 + tau^2)^(-alpha), unit std."""
    kx, ky, k2, _, _ = _operators(n)
    noise_h = jnp.fft.rfft2(jax.random.normal(key, (n, n)))
    w_h = noise_h * (k2 + tau**2) ** (-alpha / 2)
    w_h = w_h.at[0, 0].set(0.0)
    w = jnp.fft.irfft2(w_h, s=(n, n))
    return jnp.fft.rfft2(w / jnp.std(w))


def simulate(key, n_traj, n=128, re=500.0, dt=2e-3, t_spinup=5.0, tau=0.25, n_snapshots=100, out_n=64):
    """Velocity trajectories of shape (n_traj, n_snapshots + 1, 2, out_n, out_n).

    Simulated at resolution ``n`` and spatially subsampled to ``out_n`` (as in the
    paper, which makes the data only approximately divergence-free on the coarse grid).
    """
    step = make_stepper(n, re, dt)
    stride = n // out_n

    def advance(w_h, n_steps):
        return jax.lax.fori_loop(0, n_steps, lambda _, w: step(w), w_h)

    def snapshot(w_h):
        return velocity_from_vorticity(w_h, n)[:, ::stride, ::stride]

    @jax.jit
    def run(keys):
        w_h = jax.vmap(lambda k: random_vorticity(k, n))(keys)
        w_h = jax.vmap(lambda w: advance(w, round(t_spinup / dt)))(w_h)

        def body(w_h, _):
            w_next = jax.vmap(lambda w: advance(w, round(tau / dt)))(w_h)
            return w_next, jax.vmap(snapshot)(w_next)

        _, traj = jax.lax.scan(body, w_h, None, length=n_snapshots)
        first = jax.vmap(snapshot)(w_h)[None]
        return jnp.concatenate([first, traj]).swapaxes(0, 1)

    return run(jax.random.split(key, n_traj))
