# 👶 Recognition of Infants' Emotions through Acoustic Analysis

Classify the emotional state or need behind an infant's cry (for example *hungry*, *tired*,
*discomfort*) from the audio alone, using acoustic features and a machine-learning classifier.

## Pipeline

```
cry.wav ──► preprocessing ──► acoustic features ──► scaler + SVM ──► "hungry" (87%)
            (16 kHz mono,      (MFCC + Δ, spectral,
             trim silence,      ZCR, RMS, pitch / F0)
             normalise)
```

| Step | Details |
| --- | --- |
| Preprocessing | Resample to 16 kHz mono, trim silence, peak-normalise, keep the first 7 s |
| Features (≈ 140 values per clip) | 20 MFCCs and their deltas, spectral centroid / bandwidth / roll-off / contrast, zero-crossing rate, RMS energy (mean and std over time), and pitch (F0) statistics, since cry types differ a lot in pitch |
| Model | `StandardScaler` → RBF `SVC` with `class_weight="balanced"` (cry datasets are usually unbalanced) |
| Evaluation | Stratified k-fold macro-F1, plus a held-out test set with a classification report and confusion matrix |

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Try it with synthetic data (no download needed)

```bash
python -m infant_cry.synthetic --out data_demo
python -m infant_cry.train --data data_demo
```

The synthetic clips are generated tones, not real cries. They only check that the pipeline works.

### 2. Train on real recordings

Put your audio files in one folder per class:

```
data/
├── hungry/       001.wav, 002.wav, ...
├── tired/
├── discomfort/
├── belly_pain/
└── burping/
```

A public option is the *Donate-a-Cry* corpus, which uses these five classes.
Check its license before you use it.

```bash
python -m infant_cry.train --data data --out models/cry_model.joblib
```

### 3. Predict

```bash
python -m infant_cry.predict my_recording.wav
# my_recording.wav: hungry (87%)
```

## Project structure

```
infant_cry/
├── features.py   # audio loading + feature extraction
├── dataset.py    # folder-per-class dataset loader
├── model.py      # scikit-learn pipeline
├── train.py      # cross-validation, evaluation, saving
├── predict.py    # inference on new clips
└── synthetic.py  # demo data generator
tests/
└── test_pipeline.py
```

## Tests

```bash
python -m pytest -q
```

## Ideas for next steps

- Data augmentation (pitch shift, time stretch, background noise)
- A CNN on log-mel spectrograms, or pretrained audio embeddings (YAMNet, wav2vec 2.0)
- Segment long recordings into individual cry bursts before classifying

## Disclaimer

This is a research and learning project, not a medical device.
