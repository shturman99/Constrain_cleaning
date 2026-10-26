"""Coarse-vs-reference error of Pencil snapshots, and what constraint cleaning can do to it.

Coarse state x_N, reference x_R (finer grid, same time), spectral restriction P (keep |k_i| < N/2):
    e = x_N - P x_R.
Linear constraints, in Fourier space (2 pi box, integer k):
    Gauss:  C_G = i k.E - J0          Gamma:  C_A = Gamma - i k.A
The reference satisfies them to its (much smaller) truncation error, so the coarse violation
v = C(x_N) ~ C(e). A cleaning step computed from v alone,
    min |dE|^2 + lam^2 |dJ0|^2   s.t.  C_G(x_N + dx) = 0,
gives per mode, with D = k^2 + lam^-2,
    dE = -v (ik)^*/D,     dJ0 = v lam^-2/D.
For finite lam this is the orthogonal projection onto {C_G = 0} in that metric, and since P x_R lies
there, |e_after|^2 = |e|^2 - |e_perp|^2 with |e_perp|^2 = sum |v|^2/D (checked numerically).
The one-sided limits (lam = inf: change E only; lam = 0: change J0 only, J0 <- div E) are not
orthogonal projections of e, so the error can also grow; they are evaluated explicitly.
The k = 0 mode of v (mean charge, -<J0>) cannot be removed by changing E (lam = inf); it is zero
for charge-neutral states.
Same for C_A with  min |dGamma|^2 + mu^2 |dA|^2:  dGamma = -v/D', dA = -v (ik)/(mu^2 D'), D' = 1 + k^2/mu^2.
All norms are volume averages (Parseval), in code units.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path.home() / "pencil-code" / "python"))

FIELDS = ["ux", "uy", "uz", "lnrho", "ax", "ay", "az", "ex", "ey", "ez", "rhoe", "Gamma"]
VEC = {"u": ("ux", "uy", "uz"), "A": ("ax", "ay", "az"), "E": ("ex", "ey", "ez")}


def read_state(run, ivar):
    """Snapshot VAR<ivar> of a Pencil run: dict of (nz, ny, nx) arrays, t, and first grid point (z, y, x)."""
    import pencil as pc
    v = pc.read.var(datadir=str(Path(run) / "data"), ivar=ivar, trimall=True, quiet=True)
    s = {name: np.asarray(getattr(v, name)) for name in FIELDS}
    s["t"] = float(v.t)
    s["x0"] = (float(v.z[0]), float(v.y[0]), float(v.x[0]))
    return s


def int_wavenumbers(n):
    """Integer wavenumbers in FFT order for a 2 pi periodic grid of n points."""
    return np.fft.fftfreq(n, d=1.0 / n)


def kgrid(n, drop_nyquist=True):
    """(kz, ky, kx) integer wavenumber arrays on an n^3 grid; Nyquist set to 0 if dropped."""
    k = int_wavenumbers(n)
    if drop_nyquist:
        k = np.where(np.abs(k) >= n / 2, 0.0, k)
    return np.meshgrid(k, k, k, indexing="ij")


def coeffs(f, x0):
    """Coefficients c_k of a real periodic field, f(x) = sum c_k exp(i k.x).

    Pencil grids are cell centred (x_j = x0 + j dx, x0 = -pi + dx/2): the DFT is phase-corrected
    to the physical origin, so c_k agrees between grids of different N.
    """
    kz, ky, kx = kgrid(f.shape[0], drop_nyquist=False)
    phase = np.exp(-1j * (kz * x0[0] + ky * x0[1] + kx * x0[2]))
    return np.fft.fftn(f) * phase / f.size


def restrict(c, nc):
    """Coefficients restricted to an nc^3 grid (FFT order), modes with |k_i| >= nc/2 dropped."""
    kf = int_wavenumbers(c.shape[0])
    idx = np.where(np.abs(kf) < nc / 2)[0]
    kc = int_wavenumbers(nc)
    tgt = np.array([np.where(kc == kf[i])[0][0] for i in idx])
    out = np.zeros((nc,) * 3, dtype=complex)
    out[np.ix_(tgt, tgt, tgt)] = c[np.ix_(idx, idx, idx)]
    return out


def drop_nyquist(c):
    k = int_wavenumbers(c.shape[0])
    m = np.abs(k) < c.shape[0] / 2
    out = np.zeros_like(c)
    out[np.ix_(m, m, m)] = c[np.ix_(m, m, m)]
    return out


def norm2(c):
    """Volume average of the squared real field (Parseval)."""
    return float(np.sum(np.abs(c) ** 2))


def shell_sum(w):
    """Sum of w over unit-width shells around integer |k|."""
    kz, ky, kx = kgrid(w.shape[0], drop_nyquist=False)
    ks = np.rint(np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)).astype(int)
    return np.bincount(ks.ravel(), weights=w.ravel())


def div(c, v, k):
    kz, ky, kx = k
    return 1j * (kx * c[v[0]] + ky * c[v[1]] + kz * c[v[2]])     # v = (x, y, z) names


def gauss_correction(v, k, lam):
    """Minimal-norm correction (dE, dJ0) with C_G(x + dx) = 0, from the violation v alone."""
    kz, ky, kx = k
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    if lam == 0:                                   # J0 only: J0 <- div E
        return (np.zeros_like(v),) * 3, v.copy()
    D = k2 + (0.0 if np.isinf(lam) else lam ** -2.0)
    s = np.divide(v, D, out=np.zeros_like(v), where=D > 0)
    dE = tuple(-s * np.conj(1j * kk) for kk in (kx, ky, kz))
    dJ0 = np.zeros_like(v) if np.isinf(lam) else lam ** -2.0 * s
    return dE, dJ0


def gamma_correction(v, k, mu):
    """Minimal-norm correction (dGamma, dA) with C_A(x + dx) = 0."""
    kz, ky, kx = k
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    if np.isinf(mu):                               # Gamma only: Gamma <- div A
        return -v, (np.zeros_like(v),) * 3
    if mu == 0:                                    # A only
        s = np.divide(v, k2, out=np.zeros_like(v), where=k2 > 0)
        return np.zeros_like(v), tuple(-s * 1j * kk for kk in (kx, ky, kz))
    D = 1.0 + k2 / mu ** 2
    return -v / D, tuple(-v * 1j * kk / (mu ** 2 * D) for kk in (kx, ky, kz))


def decompose(coarse, ref, lams=(np.inf, 1.0, 0.0), mus=(0.0, 1.0, np.inf)):
    """Error of the coarse snapshot against the restricted reference, before and after cleaning.

    Returns per-field |e|^2 and |x_ref|^2, violations, and for each lam (Gauss) / mu (Gamma) the
    errors after the cleaning step, plus |e_perp|^2 and the Pythagoras check for finite weights.
    """
    nc = coarse["ex"].shape[0]
    cN = {f: drop_nyquist(coeffs(coarse[f], coarse["x0"])) for f in FIELDS}
    cR = {f: restrict(coeffs(ref[f], ref["x0"]), nc) for f in FIELDS}
    e = {f: cN[f] - cR[f] for f in FIELDS}
    k = kgrid(nc)
    k2 = sum(kk ** 2 for kk in k)

    vG = div(cN, VEC["E"], k) - cN["rhoe"]         # coarse violation (seen by the coarse run)
    vG_ref = div(cR, VEC["E"], k) - cR["rhoe"]     # reference violation, should be << vG
    vA = cN["Gamma"] - div(cN, VEC["A"], k)
    vA_ref = cR["Gamma"] - div(cR, VEC["A"], k)

    E2 = lambda d: sum(norm2(d[i]) for i in range(3))
    eE = tuple(e[f] for f in VEC["E"]); eA = tuple(e[f] for f in VEC["A"])
    out = {"t": coarse["t"], "N": nc,
           "e2": {f: norm2(e[f]) for f in FIELDS}, "x2": {f: norm2(cR[f]) for f in FIELDS},
           "vG2": norm2(vG), "vG2_ref": norm2(vG_ref), "vA2": norm2(vA), "vA2_ref": norm2(vA_ref),
           "gauss": {}, "gamma": {}}
    eE2, eJ2 = E2(eE), out["e2"]["rhoe"]
    for lam in lams:
        dE, dJ = gauss_correction(vG, k, lam)
        aE = tuple(eE[i] + dE[i] for i in range(3)); aJ = e["rhoe"] + dJ
        r = {"eE2": E2(aE), "eJ2": norm2(aJ), "vG2_after": norm2(div(dict(zip(VEC["E"], aE)), VEC["E"], k) - aJ),
             "spec_E": shell_sum(sum(np.abs(a) ** 2 for a in aE)), "spec_J": shell_sum(np.abs(aJ) ** 2)}
        if np.isfinite(lam) and lam > 0:
            D = k2 + lam ** -2.0
            r["eperp2"] = float(np.sum(np.abs(vG) ** 2 / D))
            r["metric_before"] = eE2 + lam ** 2 * eJ2
            r["metric_after"] = r["eE2"] + lam ** 2 * r["eJ2"]
        out["gauss"][lam] = r
    eA2, eG2 = E2(eA), out["e2"]["Gamma"]
    for mu in mus:
        dG, dA = gamma_correction(vA, k, mu)
        aA = tuple(eA[i] + dA[i] for i in range(3)); aG = e["Gamma"] + dG
        r = {"eA2": E2(aA), "eG2": norm2(aG)}
        if np.isfinite(mu) and mu > 0:
            D = 1.0 + k2 / mu ** 2
            r["eperp2"] = float(np.sum(np.abs(vA) ** 2 / D))
            r["metric_before"] = eG2 + mu ** 2 * eA2
            r["metric_after"] = r["eG2"] + mu ** 2 * r["eA2"]
        out["gamma"][mu] = r
    out["spec_e"] = {name: shell_sum(sum(np.abs(e[f]) ** 2 for f in comps)) for name, comps in
                     {"u": VEC["u"], "A": VEC["A"], "E": VEC["E"], "J0": ("rhoe",), "Gamma": ("Gamma",)}.items()}
    out["spec_vG"] = shell_sum(np.abs(vG) ** 2)
    return out
