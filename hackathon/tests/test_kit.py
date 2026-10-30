"""Validation of the Level 1 kit in float64: what must be exact is exact, and the identity map fails."""
import math

import pytest
import torch

from kit import metrics as M
from kit.fields import FAMILIES, make_field
from kit.operators import curl, ddx, div, grad, jacobian, lap, symbol
from kit.projectors import leray

TOL = 1e-12


def rand(shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


@pytest.mark.parametrize("n", [16, 32])
def test_div_curl_is_round_off(n):
    a = rand((2, 3, n, n, n))
    assert (M._norm(div(curl(a))) / M._norm(jacobian(curl(a)))).max() < TOL


def test_derivative_matches_symbol():
    n = 32
    x = torch.arange(n, dtype=torch.float64) * 2 * math.pi / n
    s = symbol(n)
    for k in (1, 5, 11, 16):
        f = torch.sin(k * x).view(1, 1, n, 1, 1).expand(1, 1, n, n, n)
        exact = s[k] * torch.cos(k * x).view(1, 1, n, 1, 1)
        assert (ddx(f, 0) - exact).abs().max() < TOL
    assert abs(float(s[n // 2])) < TOL            # D cannot see the Nyquist mode


def test_laplacian_is_div_grad_with_symbol_minus_s2():
    n = 16
    phi = rand((1, 1, n, n, n), 1)
    assert torch.equal(lap(phi), div(grad(phi)))
    sx = symbol(n).view(n, 1, 1)
    s2 = sx ** 2 + symbol(n).view(1, n, 1) ** 2 + symbol(n).view(1, 1, n) ** 2
    ref = torch.fft.ifftn(-s2 * torch.fft.fftn(phi, dim=(-3, -2, -1)), dim=(-3, -2, -1)).real
    assert (lap(phi) - ref).abs().max() < TOL * phi.abs().max() * float(s2.max())


@pytest.mark.parametrize("slope", list(FAMILIES.values()))
def test_exact_projector_scores_zero(slope):
    u = make_field(32, slope, seed=3, batch=2, dtype=torch.float64)
    p = leray(u)
    assert M.divergence_residual(p).max() < TOL
    assert M.idempotence(p, leray(p)).max() < TOL
    assert M.range_invariance(p, leray(p)).max() < TOL
    assert M.expansion(u, p).max() <= 1 + TOL
    assert M.minimal_correction(u, p).max() < TOL
    assert M.mean_shift(u, p).max() < TOL


def test_projector_keeps_curls_and_removes_gradients_and_mean():
    n = 16
    a, phi = rand((1, 3, n, n, n), 4), rand((1, 1, n, n, n), 5)
    c = curl(a)
    assert (leray(c) - c).abs().max() < TOL * c.abs().max()
    assert leray(grad(phi)).abs().max() < TOL * grad(phi).abs().max()
    mean = torch.ones(1, 3, n, n, n, dtype=torch.float64) * torch.tensor([1.0, -2.0, 0.5]).view(1, 3, 1, 1, 1)
    assert (leray(mean) - mean).abs().max() < TOL


def test_identity_fails_every_constraint_check():
    u = make_field(32, 0.0, seed=6, batch=2, dtype=torch.float64)
    p = leray(u)
    assert M.divergence_residual(u).min() > 0.1
    assert M.relative_error(u, p).min() > 0.1


def test_field_generator_is_deterministic_normalised_and_mean_free():
    a, b = make_field(16, 5 / 3, seed=9, batch=2), make_field(16, 5 / 3, seed=9, batch=2)
    assert torch.equal(a, b)
    assert torch.allclose(a.double().pow(2).mean(dim=(1, 2, 3, 4)), torch.ones(2, dtype=torch.float64), atol=1e-6)
    assert a.double().mean(dim=(-3, -2, -1)).abs().max() < 1e-6


# ---------------------------------------------------------------- Level 2
from kit import level2 as L2  # noqa: E402


def clean_b(n=16, seed=11, batch=2):
    return curl(rand((batch, 3, n, n, n), seed))


def test_symmetries_keep_divergence_free():
    b = clean_b()
    for i in range(48):
        assert M.divergence_residual(L2.symmetry(b.clone(), i)).max() < TOL


def test_gradient_noise_projects_back_to_clean_b():
    b = clean_b()
    for amp in (0.01, 0.1, 1.0):
        bs = L2.gradient_noise(b, amp, seed=1)
        assert torch.allclose((bs - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1),
                              torch.full((2,), amp, dtype=torch.float64), rtol=1e-12)
        assert (leray(bs) - b).abs().max() < TOL * b.abs().max()


def test_regrid_and_truncation_create_divergence():
    b = clean_b()
    assert torch.allclose(L2._sample(b, [0.0, 0.0, 0.0]), b)
    assert M.divergence_residual(L2.regrid(b, seed=2)).min() > 1e-4
    assert M.divergence_residual(L2.truncation(b, seed=3)).min() > 1e-6


def test_spectrum_parseval_and_metrics_zero_on_exact():
    b = clean_b()
    assert torch.allclose(L2.spectrum(b).sum(dim=1), 0.5 * b.pow(2).sum(dim=1).mean(dim=(-3, -2, -1)), rtol=1e-12)
    assert L2.spectrum_error(b, b).max() == 0
    assert L2.current_sheet_error(b, b).max() == 0


def test_level2_evaluator_end_to_end(tmp_path):
    import numpy as np
    from kit.evaluate import evaluate
    b = clean_b(n=16, batch=2).float().numpy()
    for i in range(2):
        np.savez(tmp_path / f"white_forced_VAR{i}_N016.npz", B=b[i], run="white_forced")
    r = evaluate("examples/sub_identity", level=2, n=16, device="cpu", data=tmp_path)
    assert r["status"] == "FAIL"
    assert r["families"]["gradient 1"]["rel_error"] > 0.3
    assert r["families"]["gradient 0.01"]["exact_vs_clean"] < 1e-6
