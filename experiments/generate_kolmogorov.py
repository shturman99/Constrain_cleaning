"""Generate Kolmogorov-flow velocity trajectories for operator learning."""

import argparse
from pathlib import Path

import jax
import numpy as np

from constrain_cleaning.kolmogorov import simulate

p = argparse.ArgumentParser()
p.add_argument("--n-traj", type=int, default=100)
p.add_argument("--n-snapshots", type=int, default=100)
p.add_argument("--re", type=float, default=500.0)
p.add_argument("--tau", type=float, default=0.25, help="time between snapshots (operator time-step)")
p.add_argument("--sim-n", type=int, default=128)
p.add_argument("--out-n", type=int, default=64)
p.add_argument("--chunk", type=int, default=25, help="trajectories simulated at once")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", type=Path, default=Path("data/kolmogorov_re500.npz"))
args = p.parse_args()

keys = jax.random.split(jax.random.PRNGKey(args.seed), -(-args.n_traj // args.chunk))
chunks = []
for i, key in enumerate(keys):
    n = min(args.chunk, args.n_traj - i * args.chunk)
    chunks.append(np.asarray(simulate(key, n, n=args.sim_n, re=args.re, tau=args.tau,
                                      n_snapshots=args.n_snapshots, out_n=args.out_n)))
    print(f"chunk {i + 1}/{len(keys)} done")
u = np.concatenate(chunks).astype(np.float32)

args.out.parent.mkdir(parents=True, exist_ok=True)
np.savez(args.out, u=u, re=args.re, tau=args.tau)
print(f"saved {u.shape} to {args.out}")
