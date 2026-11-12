import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def deep_update(base, other):
    out = dict(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_update(load_config(base), cfg)
    return cfg


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(name="auto"):
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Timer:
    def __init__(self):
        self.times = {}

    @contextmanager
    def __call__(self, name):
        t = time.perf_counter()
        yield
        self.times[name] = time.perf_counter() - t
