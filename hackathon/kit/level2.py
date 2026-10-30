"""Level 2: MHD magnetic fields, augmentations, the three corruption families and the extra metrics.

Clean fields come from Pencil snapshots restricted to N^3 with B = D x A (data_prep/extract_level2.py),
so D.B = 0 to round-off under the official operator. The target for every corruption is the nearest
valid field, leray(B*); for gradient noise this is exactly the clean B.
"""
import glob
import math
from pathlib import Path

import numpy as np
import torch

from .fields import make_field
from .operators import SPATIAL, curl, ddx, grad, wavenumbers

# ---------------------------------------------------------------- data


def load_snapshots(data_dir, grid, runs=None, dtype=torch.float32):
    """Clean B fields [S, 3, N, N, N] from data_dir/<run>_<var>_N<grid>.npz, optionally only some runs."""
    files = sorted(glob.glob(str(Path(data_dir) / f"*_N{grid:03d}.npz")))
    out, names = [], []
    for f in files:
        d = np.load(f, allow_pickle=False)
        if runs is not None and str(d["run"]) not in runs:
            continue
        out.append(torch.from_numpy(d["B"]))
        names.append(Path(f).stem)
    if not out:
        raise FileNotFoundError(f"no N{grid} snapshots in {data_dir} for runs {runs}")
    return torch.stack(out).to(dtype), names


def symmetry(b, index):
    """One of the 48 cube symmetries (axis permutation + flips) applied to a vector field [B, 3, x, y, z].

    The grid is cell centred on [-pi, pi), so a flip maps it onto itself; the components move with the
    axes and change sign with the flips, so D.B = 0 is preserved exactly.
    """
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    perm, flips = perms[index // 8], [(index >> k) & 1 for k in range(3)]
    b = b[:, list(perm)].permute(0, 1, *[2 + p for p in perm])
    for ax in range(3):
        if flips[ax]:
            b = torch.flip(b, dims=[2 + ax])
            b[:, ax] = -b[:, ax]
    return b


def augment(b, gen):
    """Random cube symmetry and random periodic shift per sample."""
    out = []
    for i in range(b.shape[0]):
        bi = symmetry(b[i:i + 1], int(torch.randint(48, (1,), generator=gen)))
        shifts = [int(s) for s in torch.randint(b.shape[-1], (3,), generator=gen)]
        out.append(torch.roll(bi, shifts=shifts, dims=SPATIAL))
    return torch.cat(out)

# ---------------------------------------------------------------- corruptions


def gradient_noise(b, amplitude, seed, slope=4.0):
    """B* = B + D psi with a random smooth psi scaled so ||D psi|| / ||B|| = amplitude."""
    psi = make_field(b.shape[-1], slope, seed, batch=b.shape[0], device=b.device, dtype=torch.float64)[:, :1]
    g = grad(psi)
    g = g * (amplitude * b.double().flatten(1).norm(dim=1) / g.flatten(1).norm(dim=1)).view(-1, 1, 1, 1, 1)
    return (b.double() + g).to(b.dtype)


def _sample(b, offset):
    """Trilinear periodic interpolation of b at points shifted by `offset` (in grid cells, per axis)."""
    n = b.shape[-1]
    idx = [torch.arange(n, device=b.device, dtype=b.dtype) + o for o in offset]   # sample points, in cells
    i0 = [torch.floor(t).long() % n for t in idx]
    w = [t - torch.floor(t) for t in idx]
    out = torch.zeros_like(b)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wx = w[0] if dx else 1 - w[0]
                wy = w[1] if dy else 1 - w[1]
                wz = w[2] if dz else 1 - w[2]
                vx = b.index_select(2, (i0[0] + dx) % n)
                vxy = vx.index_select(3, (i0[1] + dy) % n)
                v = vxy.index_select(4, (i0[2] + dz) % n)
                out = out + v * (wx.view(-1, 1, 1) * wy.view(1, -1, 1) * wz.view(1, 1, -1))
    return out


def regrid(b, seed, max_offset=0.5):
    """Interpolation error: each component remapped to its own sub-cell-shifted grid and back.

    One common shift for all components would be a convolution, which commutes with D and keeps
    D.B = 0; components living on different (e.g. staggered) grids are what creates divergence.
    """
    gen = torch.Generator().manual_seed(int(seed))
    out = []
    for c in range(3):
        off = [float(o) for o in (torch.rand(3, generator=gen, dtype=torch.float64) * 2 - 1) * max_offset]
        there = _sample(b[:, c:c + 1].double(), off)
        out.append(_sample(there, [-o for o in off]))
    return torch.cat(out, dim=1).to(b.dtype)


def induction_rhs(b, u):
    """Expanded induction term (B.grad)u - (u.grad)B - B div u: drops u (div B), so it does not keep D.B = 0."""
    out = []
    du = [[ddx(u[:, c:c + 1], a) for a in range(3)] for c in range(3)]
    db = [[ddx(b[:, c:c + 1], a) for a in range(3)] for c in range(3)]
    divu = du[0][0] + du[1][1] + du[2][2]
    for c in range(3):
        bgu = sum(b[:, a:a + 1] * du[c][a] for a in range(3))
        ugb = sum(u[:, a:a + 1] * db[c][a] for a in range(3))
        out.append(bgu - ugb - b[:, c:c + 1] * divu)
    return torch.cat(out, dim=1)


def truncation(b, seed, steps=5, cells=0.03):
    """Truncation error: RK3 steps of the expanded induction equation with a random smooth unit-RMS flow.

    The size is set by the total displacement of the RMS flow, `cells` grid cells: on near-white 32^3
    fields 0.03 cells gives |B* - B| ~ 5% with a divergent part ~ 1.5%; one cell rewrites B entirely.
    """
    u = make_field(b.shape[-1], 3.0, seed, batch=b.shape[0], device=b.device, dtype=torch.float64)
    dt = cells * (2 * math.pi / b.shape[-1]) / steps
    x = b.double()
    for _ in range(steps):   # Williamson low-storage RK3, as in the Pencil Code
        k = torch.zeros_like(x)
        for a_, b_ in ((0.0, 1 / 3), (-5 / 9, 15 / 16), (-153 / 128, 8 / 15)):
            k = a_ * k + dt * induction_rhs(x, u)
            x = x + b_ * k
    return x.to(b.dtype)

# ---------------------------------------------------------------- metrics


def spectrum(b):
    """Magnetic energy per |k| shell, [B, n_shells]."""
    b = b.detach().double()
    n = b.shape[-1]
    k = wavenumbers(n, b.device)
    shell = torch.sqrt(k.view(n, 1, 1) ** 2 + k.view(1, n, 1) ** 2 + k.view(1, 1, n) ** 2).round().long().flatten()
    e = (torch.fft.fftn(b, dim=SPATIAL).abs() ** 2).sum(dim=1).flatten(1) / n ** 6 / 2
    return torch.zeros(b.shape[0], int(shell.max()) + 1, dtype=torch.float64, device=b.device).index_add_(1, shell, e)


def spectrum_error(pred, target, kmax=None):
    """Max over shells 1..kmax of |E_pred(k) - E_target(k)| / E_target(k), per sample."""
    ep, et = spectrum(pred), spectrum(target)
    kmax = kmax or pred.shape[-1] // 3
    return ((ep[:, 1:kmax + 1] - et[:, 1:kmax + 1]).abs() / et[:, 1:kmax + 1].clamp(min=1e-300)).max(dim=1).values


def current_sheet_error(pred, target, quantile=0.99):
    """Relative error of J = D x B in the cells where |J_target| is in its top 1%, per sample."""
    jp, jt = curl(pred.detach().double()), curl(target.detach().double())
    mag = jt.norm(dim=1).flatten(1)
    thr = torch.quantile(mag, quantile, dim=1, keepdim=True)
    mask = (mag >= thr).unsqueeze(1)
    d = ((jp - jt).flatten(2) ** 2 * mask).sum(dim=(1, 2))
    t = (jt.flatten(2) ** 2 * mask).sum(dim=(1, 2))
    return (d / t).sqrt()
