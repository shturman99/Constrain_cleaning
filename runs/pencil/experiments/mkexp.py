#!/usr/bin/env python3
"""Create a Pencil experiment run from a built template run, sharing its executables.

    mkexp.py TEMPLATE NAME [start:NAMELIST:key=value ...] [run:NAMELIST:key=value ...]

e.g.  mkexp.py templates/T064 A/rel_u02 run:special_run_pars:J0_model="'galilean'"

The run directory is created with `pc_newrun -s` (inputs copied, start.x/run.x linked from
the template), then each override is appended as a new line at the end of the given namelist
block of start.in or run.in. Fortran namelist input takes the last assignment of a variable,
so appended lines override earlier ones. A file OVERRIDES in the run lists what was changed.
"""
import subprocess, sys
from pathlib import Path


def set_param(path, nml, assignment):
    lines = path.read_text().splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines) if l.strip().lower() == f"&{nml.lower()}"), None)
    if start is None:
        raise SystemExit(f"{path}: namelist &{nml} not found")
    end = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "/")
    lines.insert(end, f"  {assignment}\n")
    path.write_text("".join(lines))


def main():
    template, name, *overrides = sys.argv[1:]
    template, run = Path(template).resolve(), Path(name).resolve()
    if run.exists():
        raise SystemExit(f"{run} exists")
    run.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pc_newrun", "-s", str(template), str(run)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for ov in overrides:
        which, nml, assignment = ov.split(":", 2)
        set_param(run / f"{which}.in", nml, assignment)
    (run / "OVERRIDES").write_text("\n".join(overrides) + "\n")
    for x in ("start.x", "run.x"):
        if not (run / "src" / x).exists():
            raise SystemExit(f"{run}/src/{x} missing: build the template first")
    print(run)


if __name__ == "__main__":
    main()
