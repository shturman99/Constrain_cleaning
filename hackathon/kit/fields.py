"""Level 1 field generator.

`slope` is the exponent of the per-mode power, |u_hat(k)|^2 ~ |k|^-slope, so slope 0 is white noise as
in the brief. (Open point: a shell spectrum E(k) ~ k^-5/3 corresponds to a per-mode slope 11/3, so the
brief's "5/3, Kolmogorov-like" family is only Kolmogorov-like if slope means the shell spectrum.)
"""
import torch

from .operators import SPATIAL, wavenumbers

FAMILIES = {"white": 0.0, "kolmogorov": 5.0 / 3.0, "smooth": 3.0}


def make_field(n, slope, seed, batch=1, device=None, dtype=torch.float32):
    """Random periodic 3-component field [batch, 3, n, n, n]: per-mode power ~ |k|^-slope, zero mean,
    unit RMS per sample. Deterministic in (n, slope, seed, batch); generated in float64 on the CPU."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    w = torch.randn(batch, 3, n, n, n, generator=g, dtype=torch.float64)
    k = wavenumbers(n)
    k2 = k.view(n, 1, 1) ** 2 + k.view(1, n, 1) ** 2 + k.view(1, 1, n) ** 2
    filt = torch.where(k2 > 0, k2.clamp(min=1.0) ** (-slope / 4.0), torch.zeros_like(k2))
    u = torch.fft.ifftn(torch.fft.fftn(w, dim=SPATIAL) * filt, dim=SPATIAL).real
    u = u / u.pow(2).mean(dim=(1, 2, 3, 4), keepdim=True).sqrt()
    return u.to(device=device, dtype=dtype)
