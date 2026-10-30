"""Official discrete operators: Pencil Code sixth-order central differences on a periodic grid.

Fields are tensors [batch, component, x, y, z] (scalars [batch, 1, x, y, z]) on the box [0, 2pi)^3 with
spacing h = 2 pi / N. Every operator is built from the one first-derivative stencil

    D f_i = (45 (f_{i+1} - f_{i-1}) - 9 (f_{i+2} - f_{i-2}) + (f_{i+3} - f_{i-3})) / (60 h),

so the discrete identities hold to round-off: div(curl A) = 0 and lap = div(grad).
In Fourier space D acts as i s(k), with

    s(k) = (45 sin kh - 9 sin 2kh + sin 3kh) / (30 h),

which vanishes at k = 0 and at the Nyquist wavenumber kh = pi. The official Laplacian D.D therefore
also annihilates Nyquist modes (decision 2026-10-30: kept, with sources generated as f = D.g).
"""
import math

import torch

SPATIAL = (-3, -2, -1)   # x, y, z


def spacing(n):
    return 2 * math.pi / n


def ddx(f, axis):
    """Official first derivative of f along axis 0, 1, 2 (= x, y, z)."""
    dim = SPATIAL[axis]
    h = spacing(f.shape[dim])
    r = lambda m: torch.roll(f, shifts=-m, dims=dim)        # f_{i+m}
    return (45 * (r(1) - r(-1)) - 9 * (r(2) - r(-2)) + (r(3) - r(-3))) / (60 * h)


def grad(phi):
    """[B, 1, ...] -> [B, 3, ...]."""
    return torch.cat([ddx(phi, a) for a in range(3)], dim=1)


def div(u):
    """[B, 3, ...] -> [B, 1, ...]."""
    return ddx(u[:, 0:1], 0) + ddx(u[:, 1:2], 1) + ddx(u[:, 2:3], 2)


def curl(u):
    """[B, 3, ...] -> [B, 3, ...]."""
    ux, uy, uz = u[:, 0:1], u[:, 1:2], u[:, 2:3]
    return torch.cat([ddx(uz, 1) - ddx(uy, 2), ddx(ux, 2) - ddx(uz, 0), ddx(uy, 0) - ddx(ux, 1)], dim=1)


def lap(phi):
    """Official Laplacian, D.D (= div of grad)."""
    return div(grad(phi))


def jacobian(u):
    """Full gradient D u of a vector field, [B, 3, ...] -> [B, 9, ...] (component-major)."""
    return torch.cat([ddx(u[:, c:c + 1], a) for c in range(3) for a in range(3)], dim=1)


def wavenumbers(n, device=None, dtype=torch.float64):
    """Integer wavenumbers in FFT order for the 2 pi box."""
    return torch.fft.fftfreq(n, d=1.0 / n, device=device, dtype=dtype)


def symbol(n, device=None, dtype=torch.float64):
    """s(k) on the FFT grid, with D -> i s(k)."""
    kh = wavenumbers(n, device, dtype) * spacing(n)
    return (45 * torch.sin(kh) - 9 * torch.sin(2 * kh) + torch.sin(3 * kh)) / (30 * spacing(n))
