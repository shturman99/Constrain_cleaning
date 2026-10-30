"""Reproduce weights.pt from seed 0: train the stencil to imitate the exact projection.

Training data come from the kit generator with the team's own seeds (the hidden test seeds are not
available to teams), mixing the three spectral families. Run from hackathon/:
    python examples/sub_stencil/train.py
"""
import sys
import tomllib
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))      # hackathon/ -> `kit`
sys.path.insert(0, str(HERE))

from kit.fields import FAMILIES, make_field   # noqa: E402
from kit.projectors import leray              # noqa: E402
from model import Projector                   # noqa: E402

man = tomllib.loads((HERE / "submission.toml").read_text())
torch.manual_seed(man["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Projector(**man.get("model_args", {})).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
n, batch, steps = 32, 8, 1500
slopes = list(FAMILIES.values())

for step in range(steps):
    u = make_field(n, slopes[step % 3], seed=man["seed"] * 10**6 + step, batch=batch, device=device)
    loss = torch.mean((model(u) - leray(u)) ** 2) / torch.mean(leray(u) ** 2)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 300 == 0 or step == steps - 1:
        print(f"step {step:5d}  relative MSE {loss.item():.4e}")

torch.save(model.state_dict(), HERE / man["weights"])
print("saved", HERE / man["weights"])
