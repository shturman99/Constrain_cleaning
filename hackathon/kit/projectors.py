"""Exact reference projectors, exact under the official operator D (not just spectrally).

Leray: the L2-nearest field with D.u' = 0. With D -> i s(k) per direction,
    u_hat' = u_hat - s (s . u_hat) / |s|^2,
i.e. u' = u - D phi with D.D phi = D.u. Modes with |s| = 0 (the mean and modes made only of k = 0 and
Nyquist components) are invisible to D.u and are left unchanged, which also preserves the mean.
"""
import torch

from .operators import SPATIAL, symbol


def _symbols(u):
    n = u.shape[-1]
    if u.shape[-3:] != (n, n, n):
        raise ValueError(f"cubic grids only, got {tuple(u.shape[-3:])}")
    s = symbol(n, u.device, torch.float64 if u.dtype == torch.float64 else torch.float32)
    return s.view(n, 1, 1), s.view(1, n, 1), s.view(1, 1, n)


def leray(u):
    """Exact discrete Leray projection of u [B, 3, N, N, N] (uses an FFT: organizer side only)."""
    sx, sy, sz = _symbols(u)
    uh = torch.fft.fftn(u, dim=SPATIAL)
    s2 = sx ** 2 + sy ** 2 + sz ** 2
    sdotu = (sx * uh[:, 0] + sy * uh[:, 1] + sz * uh[:, 2]) / torch.where(s2 > 0, s2, torch.ones_like(s2))
    sdotu = torch.where(s2 > 0, sdotu, torch.zeros_like(sdotu))
    out = torch.stack([uh[:, 0] - sx * sdotu, uh[:, 1] - sy * sdotu, uh[:, 2] - sz * sdotu], dim=1)
    return torch.fft.ifftn(out, dim=SPATIAL).real.to(u.dtype)
