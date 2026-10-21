"""Model definition: standardised features -> class-balanced SVM."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model(C: float = 10.0) -> Pipeline:
    # class_weight="balanced" matters: cry datasets are usually very unbalanced
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(C=C, kernel="rbf", gamma="scale", class_weight="balanced", probability=True)),
    ])
