"""Train and evaluate the classifier.

Usage:
    python -m infant_cry.train --data data --out models/cry_model.joblib
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .dataset import load_dataset
from .model import build_model


def main(argv: list[str] | None = None) -> float:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="folder with one sub-folder per class")
    parser.add_argument("--out", default="models/cry_model.joblib", help="where to save the trained model")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    print(f"Extracting features from {args.data} ...")
    X, y = load_dataset(args.data)
    print(f"{len(y)} clips, {X.shape[1]} features, classes: {dict(Counter(y))}")

    model = build_model()

    n_splits = min(5, min(Counter(y).values()))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
        print(f"{n_splits}-fold cross-validated macro F1: {scores.mean():.3f} ± {scores.std():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nHeld-out test set:")
    print(classification_report(y_test, y_pred, zero_division=0))
    labels = np.unique(y)
    print("Confusion matrix (rows = true, cols = predicted):", list(labels))
    print(confusion_matrix(y_test, y_pred, labels=labels))

    # Refit on all data before saving
    model.fit(X, y)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"\nModel saved to {args.out}")
    return float((y_pred == y_test).mean())


if __name__ == "__main__":
    main()
