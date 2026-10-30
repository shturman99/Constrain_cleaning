"""Submission format, loader and the FFT ban.

A submission is a directory with:
    submission.toml   team, level, entry ("model.py:Projector"), weights (optional), train, seed, [model_args]
    model.py          defines the Projector: forward(u, f=None, a=None) -> same shape as u
    weights.pt        state_dict (optional, e.g. for a parameter-free model)
    train.py          reproduces weights.pt from the fixed seed
"""
import importlib.util
import sys
import tomllib
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

REQUIRED_KEYS = ("team", "level", "entry", "train", "seed")


def read_manifest(path):
    path = Path(path)
    with open(path / "submission.toml", "rb") as fh:
        man = tomllib.load(fh)
    missing = [k for k in REQUIRED_KEYS if k not in man]
    if missing:
        raise ValueError(f"submission.toml is missing {missing}")
    if not (path / man["train"]).exists():
        raise ValueError(f"training script {man['train']} not found")
    return man


def load(path, device="cpu"):
    """Instantiate the team's Projector from a submission directory, weights loaded, in eval mode."""
    path = Path(path).resolve()
    man = read_manifest(path)
    file, cls = man["entry"].split(":")
    spec = importlib.util.spec_from_file_location(f"submission_{man['team']}", path / file)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path))
    model = getattr(mod, cls)(**man.get("model_args", {}))
    if man.get("weights"):
        model.load_state_dict(torch.load(path / man["weights"], map_location="cpu", weights_only=True))
    return model.to(device).eval(), man


def fft_ops(model, sample):
    """Names of FFT operators executed in one forward pass (empty list = allowed)."""
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if sample.is_cuda else [])
    with profile(activities=acts) as prof, torch.no_grad():
        model(sample)
    return sorted({e.name for e in prof.events() if "fft" in e.name.lower()})
