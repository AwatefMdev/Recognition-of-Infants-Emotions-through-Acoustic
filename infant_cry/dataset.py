"""Load a dataset laid out as one folder per emotion / cry reason:

    data/
    ├── hungry/      *.wav
    ├── tired/       *.wav
    ├── discomfort/  *.wav
    └── ...
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import features_from_file

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".3gp", ".caf"}


def list_files(data_dir: str | Path) -> list[tuple[Path, str]]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    items = [
        (path, label_dir.name)
        for label_dir in sorted(p for p in data_dir.iterdir() if p.is_dir())
        for path in sorted(label_dir.rglob("*"))
        if path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not items:
        raise ValueError(f"No audio files found under {data_dir}")
    return items


def load_dataset(data_dir: str | Path, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    files = list_files(data_dir)
    X, y = [], []
    for i, (path, label) in enumerate(files, 1):
        try:
            X.append(features_from_file(str(path)))
            y.append(label)
        except Exception as exc:  # corrupted file: skip it but keep going
            print(f"  skipped {path.name}: {exc}")
        if verbose and i % 50 == 0:
            print(f"  processed {i}/{len(files)} files")
    return np.vstack(X), np.array(y)
