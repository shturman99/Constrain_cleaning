"""Evaluate a submission, as the evaluation server would. From hackathon/:

    python -m kit.evaluate examples/sub_stencil --level 1 [--n 32]
    python -m kit.evaluate examples/sub_stencil --level 2 --data <level2 data dir> [--n 32]

Steps: load the package, reject it if one forward pass runs an FFT, score it on the level's hidden test
sets with every metric, compare with the pass thresholds, write a JSON report.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from . import level2 as L2
from . import metrics as M
from .fields import FAMILIES, make_field
from .projectors import leray
from .submission import fft_ops, load

HIDDEN_SEED = 1_000_000_007   # test seeds exist only on the evaluation server; teams train on their own
TEST_RUNS = ["white_forced"]  # Level 2: held-out run (teams train on white_decay)
# Pass checks. Level 1 values are the Student brief's placeholders; None = not set yet (reported, not
# enforced). Level 2 is relative to doing nothing (identity): *_ratio = submission / identity.
THRESHOLDS = {
    1: {"rel_error[white]": 1e-2, "beats_stencil_r8": None},
    2: {"rel_error_ratio[all]": 0.1, "div_residual_ratio[all]": 0.1, "spectrum_error[all]": 2e-2},
}
COMMON = ["rel_error", "div_residual", "idempotence", "range_invariance", "expansion_max",
          "minimal_correction", "mean_shift", "shell_error_k1_4", "shell_error_high_k"]
LEVEL2_EXTRA = ["rel_error_ratio", "div_residual_ratio", "spectrum_error", "current_sheet_error", "rel_error_vs_clean"]


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def level1_sets(n, batch, device):
    for i, (fam, slope) in enumerate(FAMILIES.items()):
        yield fam, make_field(n, slope, HIDDEN_SEED + i, batch, device), None


def level2_sets(data_dir, n, device):
    clean, _ = L2.load_snapshots(data_dir, n, runs=TEST_RUNS)
    clean = clean.to(device)
    for amp in (0.01, 0.1, 1.0):
        yield f"gradient {amp:g}", L2.gradient_noise(clean, amp, HIDDEN_SEED + int(1000 * amp)), clean
    yield "regrid", L2.regrid(clean, HIDDEN_SEED + 7), clean
    yield "truncation", L2.truncation(clean, HIDDEN_SEED + 8), clean


def score(model, u, clean, device, level):
    target = leray(u.double())                      # nearest valid field, float64
    for _ in range(2):                              # warm-up
        model(u)
    _sync(device)
    t0 = time.perf_counter()
    tu = model(u)
    _sync(device)
    dt = time.perf_counter() - t0
    ttu, t_valid = model(tu), model(target.float())
    shells = M.shell_error(tu, target).mean(0)
    n = u.shape[-1]
    r = {"input_div_residual": float(M.divergence_residual(u).mean()),
         "rel_error": float(M.relative_error(tu, target).mean()),
         "div_residual": float(M.divergence_residual(tu).mean()),
         "idempotence": float(M.idempotence(tu, ttu).mean()),
         "range_invariance": float(M.range_invariance(target, t_valid).mean()),
         "expansion_max": float(M.expansion(u, tu).max()),
         "minimal_correction": float(M.minimal_correction(u, tu).mean()),
         "mean_shift": float(M.mean_shift(u, tu).mean()),
         "shell_error_k1_4": float(shells[1:5].mean()),
         "shell_error_high_k": float(shells[n // 4: n // 2].mean()),
         "shell_error": [float(x) for x in shells]}
    if level == 2:
        r["rel_error_ratio"] = r["rel_error"] / float(M.relative_error(u, target).mean())
        r["div_residual_ratio"] = r["div_residual"] / r["input_div_residual"]
        r["spectrum_error"] = float(L2.spectrum_error(tu, target).max())
        r["current_sheet_error"] = float(L2.current_sheet_error(tu, target).mean())
        r["rel_error_vs_clean"] = float(M.relative_error(tu, clean).mean())
        r["exact_vs_clean"] = float(M.relative_error(target, clean).mean())
    return r, dt


def evaluate(path, level=1, n=32, batch=4, device=None, data=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, man = load(path, device)
    banned = fft_ops(model, make_field(n, 0.0, HIDDEN_SEED - 1, 1, device))
    sets = level1_sets(n, batch, device) if level == 1 else level2_sets(data, n, device)
    families, times = {}, []
    with torch.no_grad():
        for fam, u, clean in sets:
            families[fam], dt = score(model, u, clean, device, level)
            times.append(dt)

    checks = {"fft_free": not banned}
    for key, thr in THRESHOLDS[level].items():
        if thr is None:
            continue
        metric, which = key[:-1].split("[")
        vals = [f[metric] for name, f in families.items() if which == "all" or name == which]
        checks[key] = max(vals) < thr
    status = "REJECTED (FFT in forward pass)" if banned else ("PASS" if all(checks.values()) else "FAIL")
    return {"team": man["team"], "level": level, "grid": n, "device": device, "status": status,
            "checks": checks, "fft_ops": banned, "thresholds": THRESHOLDS[level],
            "time_per_call_s": statistics.median(times), "families": families}


def print_report(r):
    print(f"\n=== {r['team']} | Level {r['level']} | {r['grid']}^3 | {r['device']} ===")
    print(f"status: {r['status']}")
    for k, v in r["checks"].items():
        print(f"  check {k:22s} {'pass' if v else 'FAIL'}")
    for k, v in r["thresholds"].items():
        if v is None:
            print(f"  check {k:22s} not set yet (threshold from baseline scores)")
    if r["fft_ops"]:
        print(f"  FFT operators found: {', '.join(r['fft_ops'][:4])}{' ...' if len(r['fft_ops']) > 4 else ''}")
    fams = list(r["families"])
    rows = ["input_div_residual"] + COMMON + (LEVEL2_EXTRA + ["exact_vs_clean"] if r["level"] == 2 else [])
    print(f"\n{'metric':22s}" + "".join(f"{f:>14s}" for f in fams))
    for m in rows:
        print(f"{m:22s}" + "".join(f"{r['families'][f][m]:14.3e}" for f in fams))
    print(f"\ntime per call (informal, not the benchmark node): {r['time_per_call_s'] * 1e3:.2f} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("submission")
    ap.add_argument("--level", type=int, default=1, choices=[1, 2])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4, help="Level 1 samples per family")
    ap.add_argument("--data", default=None, help="Level 2 data directory")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="JSON report (default: <submission>/report_level<L>.json)")
    a = ap.parse_args()
    if a.level == 2 and not a.data:
        ap.error("--level 2 needs --data")
    r = evaluate(a.submission, a.level, a.n, a.batch, a.device, a.data)
    print_report(r)
    out = Path(a.out or Path(a.submission) / f"report_level{a.level}.json")
    out.write_text(json.dumps(r, indent=1))
    print(f"report written to {out}")


if __name__ == "__main__":
    main()
