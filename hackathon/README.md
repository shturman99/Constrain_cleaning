# Hackathon kit (Levels 1-2)

Reference infrastructure for the learned-projection hackathon (design doc:
https://claude.ai/artifact/C7qN3gsS81S5gMd3XuunWD). Tensors are `[batch, component, x, y, z]` on the
periodic box `[0, 2pi)^3`; all derivatives use the Pencil Code sixth-order first-derivative stencil.

| Path | Contents |
|---|---|
| `kit/operators.py` | official `ddx`, `div`, `curl`, `grad`, `lap` (= D.D), `jacobian`; FD6 symbol `s(k)` |
| `kit/projectors.py` | exact discrete Leray projection (FFT with `s(k)`; organizer side only) |
| `kit/fields.py` | `make_field(n, slope, seed, batch)`: per-mode power ~ `|k|^-slope`, zero mean, unit RMS |
| `kit/metrics.py` | one function per metric, computed in float64 |
| `kit/submission.py` | submission format, loader, FFT ban (PyTorch profiler) |
| `kit/level2.py` | Level 2: snapshot loader, 48 cube symmetries + shifts, corruptions (gradient noise, regrid, truncation), spectrum and current-sheet metrics |
| `kit/evaluate.py` | evaluator: `python -m kit.evaluate <submission> --level 1|2 [--data DIR] [--n 32]` |
| `data_prep/extract_level2.py` | Pencil 512^3 snapshots -> N^3 npz (spectral restriction of A, B = D x A); runs on Vera |
| `examples/` | `sub_stencil` (a real entry), `sub_identity` (must fail), `sub_fft_cheat` (must be rejected) |
| `tests/` | float64 validation: `python -m pytest hackathon/tests` from the repo root |

## A submission

```
my_team/
  submission.toml   team, level, entry = "model.py:Projector", weights, train, seed, [model_args]
  model.py          class Projector(torch.nn.Module): forward(u, f=None, a=None) -> same shape as u
  weights.pt        state_dict
  train.py          reproduces weights.pt from the seed
```

Run from `hackathon/` with the project environment (`uv sync --group hackathon` installs PyTorch):

```bash
../.venv/bin/python examples/sub_stencil/train.py          # reproduce the weights
../.venv/bin/python -m kit.evaluate examples/sub_stencil   # score it; writes report_level1.json
../.venv/bin/python -m kit.evaluate examples/sub_stencil --level 2 --data data/level2   # held-out run white_forced
```

## Level 2 data

`data/level2/<run>_<var>_N<grid>.npz` with `B` (float32, `[3, x, y, z]`, unit RMS, D.B = 0 to ~1e-7),
`A`, `scale`, `t`. Teams train on `white_decay` (VAR1-3, var.dat); `white_forced` is held out for scoring.
