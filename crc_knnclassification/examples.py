"""Runnable examples for the CRC weighted KNN workflow."""

import numpy as np
from sklearn.preprocessing import LabelEncoder

from .pipeline import run_knn_pipeline


def demo_knn_pipeline():
    rng = np.random.default_rng(42)
    n_per_class = 40

    class_0 = rng.normal(
        loc=[2.0, 3.0, 1.0],
        scale=[0.3, 0.4, 0.2],
        size=(n_per_class, 3),
    )
    class_1 = rng.normal(
        loc=[5.0, 4.0, 3.5],
        scale=[0.4, 0.3, 0.3],
        size=(n_per_class, 3),
    )
    class_2 = rng.normal(
        loc=[8.0, 7.0, 6.0],
        scale=[0.5, 0.4, 0.4],
        size=(n_per_class, 3),
    )

    X = np.vstack([class_0, class_1, class_2])
    y = np.array(
        ["Class_1"] * n_per_class
        + ["Class_2"] * n_per_class
        + ["Class_3"] * n_per_class
    )
    y_encoded = LabelEncoder().fit_transform(y)

    results = run_knn_pipeline(
        X=X,
        y=y_encoded,
        k=5,
        p=1.5,
        n_splits=5,
        random_state=42,
        noise_levels=np.linspace(0.0, 0.10, 6),
        M=50,
    )

    print("\n=== Cross-Validation Results ===")
    cv = results["cv_results"]
    print(f"Score_CV: {cv['Score_CV']:.4f}   CI90: {cv['Score_CI90']}")
    print(f"F1_CV   : {cv['F1_CV']:.4f}   CI90: {cv['F1_CI90']}")
    print(f"MDM_CV  : {cv['MDM_CV']:.4f}   CI90: {cv['MDM_CI90']}")

    print("\n=== Performance Threshold Check ===")
    for metric, status in results["performance"].items():
        print(f"{metric}: {status}")

    print("\n=== Class Medoids ===")
    for cls, med in results["medoids"].items():
        print(f"Class {cls}: {med}")

    print("\n=== Global Sensitivity (SRCC) ===")
    print(results["sensitivity"])

    print("\n=== Noise Reliability ===")
    print(results["reliability"]["noise_results"])
    print(
        f"Reliability Index (RI): {results['reliability']['RI']:.4f} "
        f"-> {results['reliability']['RI_level']}"
    )
    print(
        f"Reliability slope (lambda): {results['reliability']['lambda']:.4f} "
        f"-> {results['reliability']['lambda_level']}"
    )
