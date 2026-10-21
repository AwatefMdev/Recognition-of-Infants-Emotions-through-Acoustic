"""Predict the emotional state behind one or more cry recordings.

Usage:
    python -m infant_cry.predict path/to/cry.wav [more.wav ...]
"""
from __future__ import annotations

import argparse

import joblib

from .features import features_from_file


def predict(paths: list[str], model_path: str = "models/cry_model.joblib") -> list[tuple[str, str, float]]:
    model = joblib.load(model_path)
    results = []
    for path in paths:
        probs = model.predict_proba([features_from_file(path)])[0]
        best = probs.argmax()
        results.append((path, model.classes_[best], float(probs[best])))
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--model", default="models/cry_model.joblib")
    args = parser.parse_args(argv)
    for path, label, confidence in predict(args.files, args.model):
        print(f"{path}: {label} ({confidence:.0%})")


if __name__ == "__main__":
    main()
