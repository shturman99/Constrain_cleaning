"""Checks of the coarse/reference error decomposition (src/constrain_cleaning/pencil_error.py)."""
import numpy as np
import pytest

from constrain_cleaning import pencil_error as pe

L = 2 * np.pi


def grid(n):
    """Cell-centred Pencil grid on [-pi, pi): first point and (z, y, x) mesh."""
    x = -np.pi + (np.arange(n) + 0.5) * L / n
    return x[0], np.meshgrid(x, x, x, indexing="ij")


def bandlimited(n, modes):
    """Real field sum_m a_m cos(k_m.x + phi_m) sampled on an n^3 Pencil grid."""
    _, (Z, Y, X) = grid(n)
    f = np.zeros((n,) * 3)
    for a, (kx, ky, kz), phi in modes:
        f += a * np.cos(kx * X + ky * Y + kz * Z + phi)
    return f


MODES = [(1.0, (1, 0, 0), 0.3), (0.5, (2, -3, 1), 1.1), (0.2, (5, 4, -6), -0.7), (0.1, (7, 7, 7), 2.0)]


def test_restriction_is_exact_for_bandlimited_fields():
    x0c, _ = grid(32)
    x0f, _ = grid(64)
    cc = pe.drop_nyquist(pe.coeffs(bandlimited(32, MODES), (x0c,) * 3))
    cf = pe.restrict(pe.coeffs(bandlimited(64, MODES), (x0f,) * 3), 32)
    assert np.max(np.abs(cc - cf)) < 1e-12
    # Parseval: volume average of f^2 = sum a^2/2
    assert np.isclose(pe.norm2(cc), sum(a ** 2 / 2 for a, _, _ in MODES))


def random_state(n, rng, kmax=6, gauss_exact=True):
    """Coefficient-space state with random low-k E, A, Gamma; J0 = div E (and Gamma = div A) if exact."""
    k = pe.kgrid(n)
    kz, ky, kx = k
    mask = (np.abs(kx) <= kmax) & (np.abs(ky) <= kmax) & (np.abs(kz) <= kmax)
    mask[0, 0, 0] = False   # mean-free: a uniform J0 (k=0) cannot be produced by any E
    def rnd():
        c = (rng.normal(size=(n,) * 3) + 1j * rng.normal(size=(n,) * 3)) * mask
        return pe.coeffs(np.real(np.fft.ifftn(c)), (0.0,) * 3)   # real field
    s = {f: rnd() for f in pe.FIELDS}
    if gauss_exact:
        s["rhoe"] = pe.div(s, pe.VEC["E"], k)
        s["Gamma"] = pe.div(s, pe.VEC["A"], k)
    return s


def to_state(c, x0):
    """Coefficients (origin-referenced) back to a real Pencil-grid snapshot dict."""
    n = c["ex"].shape[0]
    kz, ky, kx = pe.kgrid(n, drop_nyquist=False)
    phase = np.exp(1j * (kz * x0 + ky * x0 + kx * x0))
    s = {f: np.real(np.fft.ifftn(c[f] * phase * c[f].size)) for f in pe.FIELDS}
    s["t"], s["x0"] = 0.0, (x0,) * 3
    return s


@pytest.fixture
def pair():
    rng = np.random.default_rng(1)
    n = 32
    x0, _ = grid(n)
    ref = random_state(n, rng)                     # satisfies both constraints exactly
    err = random_state(n, rng, gauss_exact=False)  # generic error, violates them
    coarse = {f: ref[f] + 1e-3 * err[f] for f in pe.FIELDS}
    return to_state(coarse, x0), to_state(ref, x0)


def test_reference_satisfies_constraints(pair):
    coarse, ref = pair
    out = pe.decompose(coarse, ref)
    assert out["vG2_ref"] < 1e-24 * max(out["vG2"], 1e-300) + 1e-28
    assert out["vA2_ref"] < 1e-24 * max(out["vA2"], 1e-300) + 1e-28


def test_cleaning_removes_violation(pair):
    out = pe.decompose(*pair)
    for lam, r in out["gauss"].items():
        assert r["vG2_after"] < 1e-20 * out["vG2"] + 1e-30, f"lam={lam}"


def test_pythagoras_for_finite_weights(pair):
    out = pe.decompose(*pair)
    for group in ("gauss", "gamma"):
        for w, r in out[group].items():
            if "eperp2" in r:
                assert np.isclose(r["metric_after"], r["metric_before"] - r["eperp2"], rtol=1e-10), (group, w)
