"""Generate a small synthetic dataset so the pipeline can be tried without real data.

The clips are NOT real cries: each class is a harmonic tone with a different
pitch range and rhythm, just enough to check that training works end to end.

Usage:
    python -m infant_cry.synthetic --out data_demo --per-class 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from .features import SAMPLE_RATE

# label: (base pitch range in Hz, bursts per second)
CLASSES = {
    "hungry": ((350, 450), 1.5),
    "tired": ((250, 320), 0.7),
    "discomfort": ((500, 650), 2.5),
}


def make_clip(f_range, rate, rng, seconds=3.0, sr=SAMPLE_RATE):
    t = np.arange(int(seconds * sr)) / sr
    f0 = rng.uniform(*f_range) * (1 + 0.05 * np.sin(2 * np.pi * rng.uniform(2, 5) * t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    tone = sum(np.sin(k * phase) / k for k in range(1, 6))
    envelope = 0.5 * (1 + np.sin(2 * np.pi * rate * t - np.pi / 2))
    noise = 0.05 * rng.standard_normal(t.size)
    return (0.4 * tone * envelope + noise).astype(np.float32)


def generate(out_dir: str | Path, per_class: int = 20, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    for label, (f_range, rate) in CLASSES.items():
        (out_dir / label).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            sf.write(out_dir / label / f"{label}_{i:03d}.wav", make_clip(f_range, rate, rng), SAMPLE_RATE)
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data_demo")
    p.add_argument("--per-class", type=int, default=20)
    a = p.parse_args()
    print(f"Synthetic dataset written to {generate(a.out, a.per_class)}")
