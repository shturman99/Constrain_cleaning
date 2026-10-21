#!/usr/bin/env python3
"""lsolve_J0 experiments: definition and scheduler.

    ./plan.py list            show all experiments and their status
    ./plan.py run [PATTERN]   create (if needed) and run experiments, skipping finished ones

Runs share executables with templates/T032 (2 cores), T064 (4 cores), T128 (8 cores) and are
scheduled so that at most MAX_CORES cores are busy. Each run directory has an OVERRIDES file
listing its parameters on top of base/start.in and base/run.in.

Common to all runs: nu=2e-2, t = 0..3 (600 steps of dt=5e-3), diagnostics every 10 steps
(9 significant digits), snapshots every 0.5. At 64^3 this keeps the Gauss residual below ~1e-7
(see v1/probe/). The first pass (E=0 initially in A/D, 6-digit output) is kept in v1/.

Suites
  A  relativistic corrections (question 1): fast flow ampluu=0.2 (|u|max ~ 0.49, gamma_max ~ 1.15),
     E from Ohm's law initially (initee='ohmic', no start-up transient). Ohm's law model relativistic / gamma1 / galilean, with and without the
     relativistic fluid prefactors (hydro lrelativistic), and a flow-speed scan.
  D  displacement current (question 2): same setup, with vs without displacement current
     (lJ0_no_disp) for sigma = 1/eta = 1 ... 100.
  G  Gauss-law enforcement (question 3): resolution-study initial condition (ampluu=0.1,
     E with a longitudinal part), Gamma weight w = 2 (default), 1 (Maxwell), 0 (Gamma blind to J0),
     with/without the J0-for-div(E) substitution in div J, each without and with an initial
     Gauss violation J0 += 1e-3 sin(2y); a sigma scan of the cleaning rate.
  R  resolution checks of key A/D/G runs at 32^3 and 128^3.
"""
import os, subprocess, sys, time
from fnmatch import fnmatch
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_CORES = 8
CORES = {"T032": 2, "T064": 4, "T128": 8}

COMMON = ["run:run_pars:nt=600", "run:viscosity_run_pars:nu=2e-2"]
# fast flow, E initially from Ohm's law with the run's eta (avoids the sigma-independent start-up
# kick of a non-Ohmic E, seen in v1/ where E=0 initially)
FAST = lambda e="8.4e-2": ["start:hydro_init_pars:ampluu=0.2",
                           f"start:special_init_pars:initee='ohmic', eta_init_ohmic={e}"]
REL = {"rel": [], "gamma1": ["run:special_run_pars:J0_model='gamma1'"],
       "galilean": ["run:special_run_pars:J0_model='galilean'"]}
NRHYDRO = ["run:hydro_run_pars:lrelativistic=F"]
NODISP = ["run:special_run_pars:lJ0_no_disp=T"]
PERT = ["start:special_init_pars:ampl_J0pert=1e-3, k_J0pert=2."]
eta = lambda e: [f"run:magnetic_run_pars:eta={e}"]
weight = lambda w: [f"run:special_run_pars:weight_longitudinalE={w}"]
DIVE = ["run:special_run_pars:lJ0_divE_in_divJ=T"]
# Without displacement current the induction equation is diffusive (parabolic), so explicit
# stepping needs eta*dt/dx^2 small; with dt=5e-3 it is unstable for eta*dt/dx^2 >~ 0.15.
# These runs use a smaller dt (diffusion number <= 0.1), keeping output times identical (it1*dt=0.05).
smalldt = lambda dt: [f"run:run_pars:dt={dt}, nt={round(3 / dt)}, it1={round(0.05 / dt)}"]
NODISP_DT = {"1.": smalldt(5e-4), "0.3": smalldt(1e-3)}

EXP = {}   # name -> (template, overrides)
def add(name, tmpl, *ovs):
    EXP[name] = (tmpl, COMMON + [o for ov in ovs for o in ov])

# A: relativistic corrections
for m, ov in REL.items():
    add(f"A/{m}", "T064", FAST(), ov)
add("A/rel_nrhydro", "T064", FAST(), NRHYDRO)
add("A/galilean_nrhydro", "T064", FAST(), REL["galilean"], NRHYDRO)
for u in ["0.05", "0.1"]:
    for m in ["rel", "galilean"]:
        add(f"A/u{u}_{m}", "T064", FAST(), [f"start:hydro_init_pars:ampluu={u}"], REL[m])

# D: displacement current vs sigma (eta=8.4e-2 is the base value, A/rel is its 'with' run)
for e in ["1.", "0.3", "0.03", "1e-2"]:
    add(f"D/eta{e}_disp", "T064", FAST(e), eta(e))
for e in ["1.", "0.3", "8.4e-2", "0.03", "1e-2"]:
    add(f"D/eta{e}_nodisp", "T064", FAST(e), eta(e), NODISP, NODISP_DT.get(e, []))

# G: Gauss-law enforcement
GVAR = {"w2": [], "w1": weight("1."), "w0": weight("0."), "w2_divE": DIVE, "w0_divE": weight("0.") + DIVE}
for v, ov in GVAR.items():
    add(f"G/{v}", "T064", ov)
    add(f"G/{v}_pert", "T064", ov, PERT)
for e in ["1.", "1e-2"]:
    add(f"G/w2_pert_eta{e}", "T064", PERT, eta(e))

# R: resolution checks
for N in ["T032", "T128"]:
    n = N[1:]
    add(f"R/A_rel_N{n}", N, FAST())
    add(f"R/A_galilean_N{n}", N, FAST(), REL["galilean"])
    add(f"R/D_eta8.4e-2_nodisp_N{n}", N, FAST(), NODISP, smalldt(1e-3) if n == "128" else [])
    add(f"R/G_w2_N{n}", N)
    add(f"R/G_w0_divE_N{n}", N, GVAR["w0_divE"])
    add(f"R/G_w2_pert_N{n}", N, PERT)


def status(name):
    d = HERE / name
    if (d / "COMPLETED").exists():
        return "done"
    if (d / "data" / "time_series.dat").exists():
        return "started"
    return "new"


def launch(name):
    tmpl, ovs = EXP[name]
    d = HERE / name
    if not d.exists():
        subprocess.run([str(HERE / "mkexp.py"), str(HERE / "templates" / tmpl), str(d), *ovs],
                       check=True, stdout=subprocess.DEVNULL)
    cmd = "pc_start > start.log 2>&1 && pc_run > run.log 2>&1"
    return subprocess.Popen(cmd, shell=True, cwd=d)


def run(pattern):
    todo = [n for n in EXP if fnmatch(n, pattern) and status(n) != "done"]
    # big runs first, so the 8-core runs do not end up alone at the end
    todo.sort(key=lambda n: -CORES[EXP[n][0]])
    running = {}
    t0 = time.time()
    while todo or running:
        for n, p in list(running.items()):
            if p.poll() is not None:
                ok = status(n) == "done"
                print(f"[{(time.time()-t0)/60:6.1f} min] {'done  ' if ok else 'FAILED'} {n}", flush=True)
                del running[n]
        busy = sum(CORES[EXP[n][0]] for n in running)
        for n in list(todo):
            c = CORES[EXP[n][0]]
            if busy + c <= MAX_CORES:
                running[n] = launch(n)
                busy += c
                todo.remove(n)
                print(f"[{(time.time()-t0)/60:6.1f} min] start  {n} ({c} cores)", flush=True)
        time.sleep(5)
    print(f"all done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    pattern = sys.argv[2] if len(sys.argv) > 2 else "*"
    if cmd == "list":
        for n, (t, ovs) in EXP.items():
            print(f"{status(n):8s} {t}  {n:28s} {' '.join(o.split(':', 2)[2] for o in ovs[len(COMMON):])}")
        print(len(EXP), "experiments")
    elif cmd == "run":
        run(pattern)
