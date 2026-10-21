"""Acoustic feature extraction for infant cry recordings.

Each recording is turned into a fixed-length vector that summarises
its spectral and prosodic content:

- 20 MFCCs + their deltas (mean and std over time)
- spectral centroid, bandwidth, roll-off, contrast
- zero-crossing rate and RMS energy
- fundamental frequency (pitch) statistics, which change a lot between cry types
"""
from __future__ import annotations

import numpy as np
import librosa

SAMPLE_RATE = 16_000
N_MFCC = 20


def load_audio(path: str, sr: int = SAMPLE_RATE, duration: float | None = 7.0) -> np.ndarray:
    """Load a mono clip, trim leading/trailing silence and normalise its volume."""
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    y, _ = librosa.effects.trim(y, top_db=30)
    peak = np.max(np.abs(y)) if y.size else 0.0
    return y / peak if peak > 0 else y


def _stats(matrix: np.ndarray) -> np.ndarray:
    """Mean and standard deviation of every row of a (features x frames) matrix."""
    return np.concatenate([matrix.mean(axis=1), matrix.std(axis=1)])


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a 1-D feature vector for an audio signal."""
    if y.size < sr // 10:  # pad very short clips to 100 ms
        y = np.pad(y, (0, sr // 10 - y.size))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)

    spectral = np.vstack([
        librosa.feature.spectral_centroid(y=y, sr=sr),
        librosa.feature.spectral_bandwidth(y=y, sr=sr),
        librosa.feature.spectral_rolloff(y=y, sr=sr),
        librosa.feature.zero_crossing_rate(y),
        librosa.feature.rms(y=y),
    ])
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Infant cries usually sit between ~250 and ~1000 Hz
    f0 = librosa.yin(y, fmin=150, fmax=1200, sr=sr)
    pitch = np.array([f0.mean(), f0.std(), f0.min(), f0.max()])

    return np.concatenate([_stats(mfcc), _stats(delta), _stats(spectral), _stats(contrast), pitch])


def features_from_file(path: str) -> np.ndarray:
    return extract_features(load_audio(path))
