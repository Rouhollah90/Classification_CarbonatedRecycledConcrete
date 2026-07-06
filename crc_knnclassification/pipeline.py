"""End-to-end weighted KNN classification pipeline."""

import numpy as np

from .knn import WeightedKNN
from .robustness import noise_reliability_assessment
from .sensitivity import global_sensitivity_srcc
from .validation import (
    compute_class_medoids,
    cross_validate_knn,
    interpret_classifier_performance,
)


def filter_outliers_by_feature_variation(X, threshold_ratio=0.02):
    X = np.asarray(X, dtype=float)
    row_mean = np.mean(np.abs(X), axis=1)
    row_std = np.std(X, axis=1, ddof=1)
    variation = np.divide(row_std, row_mean, out=np.zeros_like(row_std), where=row_mean > 0)
    keep_mask = variation <= threshold_ratio
    return keep_mask, variation


def run_knn_pipeline(
    X,
    y,
    k=5,
    p=1.5,
    n_splits=5,
    random_state=42,
    noise_levels=None,
    M=50,
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    model = WeightedKNN(k=k, p=p).fit(X, y)
    cv_results = cross_validate_knn(
        X=X,
        y=y,
        k=k,
        p=p,
        n_splits=n_splits,
        random_state=random_state,
    )
    medoids = compute_class_medoids(X, y, p=p)
    sensitivity_df = global_sensitivity_srcc(X, y, model)
    reliability = noise_reliability_assessment(
        X=X,
        y=y,
        k=k,
        p=p,
        noise_levels=noise_levels,
        M=M,
        random_state=random_state,
    )
    f1_ci_width = cv_results["F1_CI90"][1] - cv_results["F1_CI90"][0]
    performance = interpret_classifier_performance(
        f1_cv=cv_results["F1_CV"],
        mdm_cv=cv_results["MDM_CV"],
        ci90_width=f1_ci_width,
    )

    return {
        "model": model,
        "medoids": medoids,
        "cv_results": cv_results,
        "sensitivity": sensitivity_df,
        "reliability": reliability,
        "performance": performance,
    }
