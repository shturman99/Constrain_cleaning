"""Negative control: calls the exact FFT projection inside the model. Must be rejected by the FFT ban."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kit.projectors import leray  # noqa: E402


class Projector(torch.nn.Module):
    def forward(self, u, f=None, a=None):
        return leray(u)
