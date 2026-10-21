import numpy as np

from infant_cry.features import SAMPLE_RATE, extract_features
from infant_cry.synthetic import generate
from infant_cry import train, predict


def test_feature_vector_has_fixed_length():
    short = np.random.default_rng(0).standard_normal(SAMPLE_RATE // 2).astype(np.float32)
    long = np.random.default_rng(1).standard_normal(SAMPLE_RATE * 3).astype(np.float32)
    a, b = extract_features(short), extract_features(long)
    assert a.shape == b.shape
    assert np.all(np.isfinite(a))


def test_end_to_end_on_synthetic_data(tmp_path):
    data = generate(tmp_path / "data", per_class=8)
    model_path = tmp_path / "model.joblib"
    accuracy = train.main(["--data", str(data), "--out", str(model_path), "--test-size", "0.25"])
    assert accuracy >= 0.8

    clip = next((data / "tired").glob("*.wav"))
    [(_, label, confidence)] = predict.predict([str(clip)], str(model_path))
    assert label == "tired"
    assert 0 <= confidence <= 1
