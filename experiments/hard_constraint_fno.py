"""Reproduce Duruisseaux et al. (2024), "Towards Enforcing Hard Physics Constraints in
Operator Learning Frameworks": FNO for u(t) -> u(t + tau) on Kolmogorov flow, comparing

  Baseline           FNO trained N epochs
  B+Projection       baseline, projected at inference only
  B+Finetuning       FNO trained N - N' epochs, then G* = pr_C o G~ finetuned N' epochs
  Fully Constrained  G* trained N epochs

Reports test relative L2 error and RMS spectral divergence (Table 2 of the paper).
"""

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from constrain_cleaning.fno import FNO2d, Projected
from constrain_cleaning.spectral import divergence

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, default=Path("data/kolmogorov_re500.npz"))
p.add_argument("--epochs", type=int, default=100, help="N (paper: 800)")
p.add_argument("--finetune-epochs", type=int, default=10, help="N' (paper: 50)")
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--width", type=int, default=32, help="paper: 64")
p.add_argument("--modes", type=int, default=12, help="paper: 20")
p.add_argument("--layers", type=int, default=4, help="paper: 6")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", type=Path, default=Path("runs/hard_constraint_fno"))
args = p.parse_args()

# ---- data: pairs (u(t), u(t+tau)), split 90/10 by trajectory ----
u = jnp.asarray(np.load(args.data)["u"])  # (traj, time, 2, n, n)
n_train = int(0.9 * u.shape[0])


def pairs(traj):
    return traj[:, :-1].reshape(-1, *traj.shape[2:]), traj[:, 1:].reshape(-1, *traj.shape[2:])


x_train, y_train = pairs(u[:n_train])
x_test, y_test = pairs(u[n_train:])
print(f"train {x_train.shape[0]} pairs, test {x_test.shape[0]} pairs, grid {x_train.shape[-1]}")


def rel_l2(pred, y):
    return jnp.linalg.norm((pred - y).reshape(len(y), -1), axis=1) / jnp.linalg.norm(y.reshape(len(y), -1), axis=1)


def rms_div(v):
    return jnp.sqrt(jnp.mean(divergence(v) ** 2, axis=(-2, -1)))


def evaluate(model, bs=256):
    errs, divs = [], []
    for i in range(0, len(x_test), bs):
        pred = jax.jit(jax.vmap(model))(x_test[i : i + bs])
        errs.append(rel_l2(pred, y_test[i : i + bs]))
        divs.append(rms_div(pred))
    return float(jnp.concatenate(errs).mean()), float(jnp.concatenate(divs).mean())


def train(model, epochs, key, label, snapshot_epoch=None):
    """Returns the trained model and a copy of it taken at the start of `snapshot_epoch`."""
    # torch Adam(weight_decay=1e-6) + ReduceLROnPlateau(factor=0.8, patience=20) stepped per epoch
    opt = optax.chain(optax.add_decayed_weights(1e-6), optax.adam(args.lr))
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    plateau = optax.contrib.reduce_on_plateau(factor=0.8, patience=20)
    plateau_state = plateau.init(None)

    @eqx.filter_jit
    def step(model, opt_state, x, y, scale):
        loss, grads = eqx.filter_value_and_grad(lambda m: rel_l2(jax.vmap(m)(x), y).mean())(model)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        updates = jax.tree.map(lambda g: scale * g, updates)
        return eqx.apply_updates(model, updates), opt_state, loss

    n_batches = len(x_train) // args.batch_size
    snapshot = None
    for epoch in range(epochs):
        if epoch == snapshot_epoch:
            snapshot = model
        key, sub = jax.random.split(key)
        perm = jax.random.permutation(sub, len(x_train))
        losses = []
        for b in range(n_batches):
            idx = perm[b * args.batch_size : (b + 1) * args.batch_size]
            model, opt_state, loss = step(model, opt_state, x_train[idx], y_train[idx], plateau_state.scale)
            losses.append(loss)
        epoch_loss = jnp.mean(jnp.stack(losses))
        _, plateau_state = plateau.update(None, plateau_state, value=epoch_loss)
        if epoch % 10 == 0 or epoch == epochs - 1:
            lr = args.lr * float(plateau_state.scale)
            print(f"[{label}] epoch {epoch:4d}  train rel L2 {float(epoch_loss):.4e}  lr {lr:.2e}")
    return model, snapshot


key_model, key_train = jax.random.split(jax.random.PRNGKey(args.seed))
fno = FNO2d(2, 2, args.width, args.modes, args.layers, key=key_model)
print(f"FNO parameters: {sum(a.size for a in jax.tree.leaves(eqx.filter(fno, eqx.is_array)))}")

baseline, pre = train(fno, args.epochs, key_train, "baseline", snapshot_epoch=args.epochs - args.finetune_epochs)
finetuned, _ = train(Projected(pre), args.finetune_epochs, key_train, "finetune")
constrained, _ = train(Projected(fno), args.epochs, key_train, "constrained")

results = {
    "Baseline": evaluate(baseline),
    "B+Projection": evaluate(Projected(baseline)),
    "B+Finetuning": evaluate(finetuned),
    "Fully Constrained": evaluate(constrained),
}
print(f"\ndata (test targets): RMS divergence {float(rms_div(y_test).mean()):.3e}")
print(f"{'Method':<20}{'Test rel L2 (x1e-2)':>22}{'RMS divergence':>18}")
for name, (err, div) in results.items():
    print(f"{name:<20}{100 * err:>22.4f}{div:>18.3e}")

args.out.mkdir(parents=True, exist_ok=True)
(args.out / f"results_seed{args.seed}.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "results": results}, indent=2))
