"""Example Level 1 submission: a local linear stencil (baseline 2 of the brief).

A 3 -> 3 channel periodic convolution of radius R. It is local, so it cannot reproduce the nonlocal
projection kernel on smooth fields; that is the point of the baseline.
"""
import torch


class Projector(torch.nn.Module):
    def __init__(self, radius=3):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 3, 2 * radius + 1, padding=radius, padding_mode="circular", bias=False)
        torch.nn.init.zeros_(self.conv.weight)
        with torch.no_grad():                    # start from the identity map
            for c in range(3):
                self.conv.weight[c, c, radius, radius, radius] = 1.0

    def forward(self, u, f=None, a=None):
        return self.conv(u)
