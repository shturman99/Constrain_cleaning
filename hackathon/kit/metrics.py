"""Level 1 metrics, one function each. Inputs are tensors [B, 3, N, N, N]; outputs are per sample [B].

Computed in float64 whatever the model's precision, so the scores measure the model, not the metric.
Residuals are normalised by the full gradient norm, so a value means the same at every resolution.
"""
import torch

from .operators import SPATIAL, curl, div, jacobian, wavenumbers


def _f64(x):
    return x.detach().to(torch.float64)


def _norm(x):
    return x.flatten(1).norm(dim=1)


def relative_error(pred, target):
    """||T u - P u|| / ||P u||."""
    pred, target = _f64(pred), _f64(target)
    return _norm(pred - target) / _norm(target)


def divergence_residual(v):
    """||D.v|| / ||D v||: 0 for a divergence-free field, ~1 for a pure gradient."""
    v = _f64(v)
    return _norm(div(v)) / _norm(jacobian(v))


def idempotence(tu, ttu):
    """||T(T u) - T u|| / ||T u||."""
    tu, ttu = _f64(tu), _f64(ttu)
    return _norm(ttu - tu) / _norm(tu)


def range_invariance(valid, t_valid):
    """||T v - v|| / ||v|| for a valid (divergence-free) input v."""
    valid, t_valid = _f64(valid), _f64(t_valid)
    return _norm(t_valid - valid) / _norm(valid)


def expansion(u, tu):
    """||T u|| / ||u||; at most 1 for a non-expansive map (the exact projection is orthogonal)."""
    return _norm(_f64(tu)) / _norm(_f64(u))


def minimal_correction(u, tu):
    """||curl(T u - u)|| / ||D (T u - u)||: 0 when the correction is a pure gradient."""
    c = _f64(tu) - _f64(u)
    return _norm(curl(c)) / _norm(jacobian(c)).clamp(min=1e-300)


def mean_shift(u, tu):
    """|<T u> - <u>| / rms(u): the mean mode must be preserved."""
    u, tu = _f64(u), _f64(tu)
    dm = (tu.mean(dim=SPATIAL) - u.mean(dim=SPATIAL)).norm(dim=1)
    return dm / u.pow(2).mean(dim=(1, 2, 3, 4)).sqrt()


def shell_error(pred, target):
    """Relative error per |k| shell, [B, n_shells]: sum |FFT(T u - P u)|^2 / sum |FFT(P u)|^2 per shell."""
    pred, target = _f64(pred), _f64(target)
    n = pred.shape[-1]
    k = wavenumbers(n, pred.device)
    shell = torch.sqrt(k.view(n, 1, 1) ** 2 + k.view(1, n, 1) ** 2 + k.view(1, 1, n) ** 2).round().long().flatten()
    e = torch.fft.fftn(pred - target, dim=SPATIAL).abs().pow(2).sum(dim=1).flatten(1)
    t = torch.fft.fftn(target, dim=SPATIAL).abs().pow(2).sum(dim=1).flatten(1)
    ns = int(shell.max()) + 1
    es = torch.zeros(pred.shape[0], ns, dtype=torch.float64, device=pred.device).index_add_(1, shell, e)
    ts = torch.zeros(pred.shape[0], ns, dtype=torch.float64, device=pred.device).index_add_(1, shell, t)
    return (es / ts.clamp(min=1e-300)).sqrt()
