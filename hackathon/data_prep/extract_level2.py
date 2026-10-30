"""Level 2 data from existing 512^3 Pencil snapshots (runs on Vera; Python 3.7 + numpy + pencil).

For each snapshot: read A, restrict it spectrally to N^3 (modes |k_i| < N/2, phase-corrected for the
cell-centred grid), compute B = D x A with the official sixth-order stencil (so D.B = 0 to round-off
under the official operator), normalise to unit RMS of B, and save A, B (float32), the scale factor and
the snapshot time. Usage:
    python extract_level2.py <out_dir> <run_dir>:<var_file>[,<var_file>...] [...] --grids 32 64 128
"""
import argparse
import os
import sys

import numpy as np

for _p in (os.path.join(os.environ.get("PENCIL_HOME", ""), "python"),
           "/hildafs/projects/phy210003p/murman/pencil-code/python", os.path.expanduser("~/pencil-code/python")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break
import pencil as pc  # noqa: E402

L = 2 * np.pi


def kint(n):
    return np.fft.fftfreq(n, d=1.0 / n)


def coeffs(f, x0):
    """c_k with f(x) = sum c_k exp(i k.x); arrays (z, y, x), x0 = (z0, y0, x0) first grid point."""
    n = f.shape[0]
    kz, ky, kx = np.meshgrid(kint(n), kint(n), kint(n), indexing="ij")
    return np.fft.fftn(f) * np.exp(-1j * (kz * x0[0] + ky * x0[1] + kx * x0[2])) / f.size


def restrict_to_grid(c, nc):
    """Field on the cell-centred nc^3 grid from fine coefficients c (modes |k_i| < nc/2 kept)."""
    kf = kint(c.shape[0])
    idx = np.where(np.abs(kf) < nc / 2)[0]
    kc = kint(nc)
    tgt = np.array([np.where(kc == kf[i])[0][0] for i in idx])
    cc = np.zeros((nc,) * 3, dtype=complex)
    cc[np.ix_(tgt, tgt, tgt)] = c[np.ix_(idx, idx, idx)]
    x0 = -np.pi + 0.5 * L / nc
    kz, ky, kx = np.meshgrid(kc, kc, kc, indexing="ij")
    return np.real(np.fft.ifftn(cc * np.exp(1j * (kz + ky + kx) * x0)) * nc ** 3)


def ddx(f, axis):
    """Official first derivative; arrays (z, y, x): axis 0, 1, 2 = x, y, z -> numpy axis 2, 1, 0."""
    ax = 2 - axis
    h = L / f.shape[ax]
    r = lambda m: np.roll(f, -m, axis=ax)
    return (45 * (r(1) - r(-1)) - 9 * (r(2) - r(-2)) + (r(3) - r(-3))) / (60 * h)


def curl(ax_, ay_, az_):
    return (ddx(az_, 1) - ddx(ay_, 2), ddx(ax_, 2) - ddx(az_, 0), ddx(ay_, 0) - ddx(ax_, 1))


def div_residual(bx, by, bz):
    d = ddx(bx, 0) + ddx(by, 1) + ddx(bz, 2)
    jac = sum(np.sum(ddx(b, a) ** 2) for b in (bx, by, bz) for a in range(3))
    return float(np.sqrt(np.sum(d ** 2) / jac))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("sources", nargs="+", help="<run_dir>:<var_file>[,<var_file>...]")
    ap.add_argument("--grids", type=int, nargs="+", default=[32, 64, 128])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for src in a.sources:
        run, files = src.split(":")
        name = os.path.basename(run.rstrip("/"))
        for vf in files.split(","):
            v = pc.read.var(datadir=os.path.join(run, "data"), var_file=vf, trimall=True, quiet=True)
            x0 = (float(v.z[0]), float(v.y[0]), float(v.x[0]))
            assert abs(x0[2] - (-np.pi + 0.5 * L / v.ax.shape[2])) < 1e-10, "expected a cell-centred grid on [-pi, pi)"
            c = [coeffs(np.asarray(f, dtype=np.float64), x0) for f in (v.ax, v.ay, v.az)]
            t = float(v.t)
            del v
            for n in a.grids:
                A = [restrict_to_grid(ci, n) for ci in c]
                B = curl(*A)
                scale = float(np.sqrt(np.mean(sum(b ** 2 for b in B))))
                A = np.stack(A) / scale
                B = np.stack(B) / scale
                r64 = div_residual(*B)
                b32 = B.astype(np.float32)
                r32 = div_residual(*b32.astype(np.float64))
                tag = "%s_%s_N%03d" % (name, vf, n)
                # store as [component, x, y, z]: arrays are (z, y, x) from pencil
                np.savez(os.path.join(a.out, tag + ".npz"),
                         A=np.transpose(A, (0, 3, 2, 1)).astype(np.float32),
                         B=np.transpose(B, (0, 3, 2, 1)).astype(np.float32),
                         scale=scale, t=t, run=name, var_file=vf, grid=n)
                print("%-28s t=%7.3f  B_rms(raw)=%.4e  |D.B|/|DB| float64 %.1e float32 %.1e"
                      % (tag, t, scale, r64, r32), flush=True)


if __name__ == "__main__":
    main()
