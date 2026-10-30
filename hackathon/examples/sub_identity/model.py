"""Negative control: returns its input unchanged. Must fail every constraint check."""
import torch


class Projector(torch.nn.Module):
    def forward(self, u, f=None, a=None):
        return u
