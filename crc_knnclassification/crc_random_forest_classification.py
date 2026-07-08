"""Random Forest classifier demo for the CRC classification workflow."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def generate_synthetic_crc_data(random_state=42, n_per_class=40):
    rng = np.random.default_rng(random_state)
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
    return X, LabelEncoder().fit_transform(y)


def confidence_interval_90(values):
    values = np.asarray(values, dtype=float)
    mean_value = float(np.mean(values))

    if len(values) < 2:
        return mean_value, mean_value, mean_value, 0.0

    sigma = float(np.std(values, ddof=1))
    margin = 1.645 * sigma / np.sqrt(len(values))
    return mean_value, max(0.0, float(mean_value - margin)), min(1.0, float(mean_value + margin)), sigma


def cross_validate_classifier(model, X, y, n_splits=5, random_state=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_accuracy = []
    fold_f1 = []

    for train_idx, test_idx in skf.split(X, y):
        fold_model = clone(model)
        fold_model.fit(X[train_idx], y[train_idx])
        y_pred = fold_model.predict(X[test_idx])
        fold_accuracy.append(np.mean(y_pred == y[test_idx]))
        fold_f1.append(f1_score(y[test_idx], y_pred, average="weighted", zero_division=0))

    acc_mean, acc_low, acc_high, acc_sigma = confidence_interval_90(fold_accuracy)
    f1_mean, f1_low, f1_high, f1_sigma = confidence_interval_90(fold_f1)
    return {
        "fold_accuracy": np.array(fold_accuracy),
        "fold_f1": np.array(fold_f1),
        "Accuracy_CV": acc_mean,
        "Accuracy_CI90": (acc_low, acc_high),
        "Accuracy_sigma": acc_sigma,
        "F1_CV": f1_mean,
        "F1_CI90": (f1_low, f1_high),
        "F1_sigma": f1_sigma,
    }


def global_sensitivity_srcc(model, X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    fitted_model = clone(model).fit(X, y)
    confidence = np.max(fitted_model.predict_proba(X), axis=1)
    records = []

    for class_id in np.unique(y):
        class_mask = y == class_id
        X_class = X[class_mask]
        confidence_class = confidence[class_mask]

        for feature_index in range(X.shape[1]):
            feature_values = X_class[:, feature_index]
            if len(np.unique(feature_values)) < 2 or len(np.unique(confidence_class)) < 2:
                rho = np.nan
                p_value = np.nan
            else:
                rho, p_value = spearmanr(feature_values, confidence_class)

            abs_rho = np.abs(rho) if not np.isnan(rho) else np.nan
            if np.isnan(abs_rho):
                level = "Undefined"
            elif abs_rho >= 0.7:
                level = "High"
            elif abs_rho >= 0.4:
                level = "Moderate"
            else:
                level = "Low"

            records.append(
                {
                    "class": class_id,
                    "feature_index": feature_index,
                    "SRCC": rho,
                    "abs_SRCC": abs_rho,
                    "sensitivity_level": level,
                    "p_value": p_value,
                }
            )

    return pd.DataFrame(records)


def add_multiplicative_gaussian_noise(X, sigma, random_state=None):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    return X * (1.0 + rng.normal(loc=0.0, scale=sigma, size=X.shape))


def noise_reliability_assessment(
    model,
    X,
    y,
    noise_levels=None,
    M=50,
    random_state=42,
):
    noise_levels = np.linspace(0.0, 0.10, 6) if noise_levels is None else noise_levels
    fitted_model = clone(model).fit(X, y)
    baseline_pred = fitted_model.predict(X)
    rng = np.random.default_rng(random_state)
    rows = []

    for sigma in noise_levels:
        reliability_values = []
        for _ in range(M):
            seed = int(rng.integers(0, 1_000_000_000))
            X_noisy = add_multiplicative_gaussian_noise(X, sigma=sigma, random_state=seed)
            reliability_values.append(np.mean(fitted_model.predict(X_noisy) == baseline_pred))

        reliability_values = np.asarray(reliability_values)
        mean_reliability = float(np.mean(reliability_values))
        sd_reliability = (
            float(np.std(reliability_values, ddof=1))
            if len(reliability_values) > 1
            else 0.0
        )
        level = "High" if mean_reliability >= 0.95 else "Moderate" if mean_reliability >= 0.90 else "Low"
        rows.append(
            {
                "sigma": sigma,
                "R_bar": mean_reliability,
                "SD_R": sd_reliability,
                "consistency_level": level,
            }
        )

    return pd.DataFrame(rows)


def forest_vote_summary(model, X, y):
    fitted_model = clone(model).fit(X, y)
    tree_predictions = np.asarray([tree.predict(X) for tree in fitted_model.estimators_])
    rows = []

    for sample_index in range(X.shape[0]):
        votes, counts = np.unique(tree_predictions[:, sample_index], return_counts=True)
        majority_class = votes[np.argmax(counts)]
        vote_fraction = np.max(counts) / len(fitted_model.estimators_)
        rows.append(
            {
                "sample_index": sample_index,
                "majority_vote_class": int(majority_class),
                "vote_fraction": float(vote_fraction),
                "representative_features": X[sample_index],
            }
        )

    return pd.DataFrame(rows)


def main():
    X, y = generate_synthetic_crc_data()
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )

    cv_results = cross_validate_classifier(model, X, y)
    sensitivity = global_sensitivity_srcc(model, X, y)
    reliability = noise_reliability_assessment(model, X, y)
    vote_summary = forest_vote_summary(model, X, y)

    print("\n=== Random Forest Cross-Validation Results ===")
    print(f"Accuracy_CV: {cv_results['Accuracy_CV']:.4f}   CI90: {cv_results['Accuracy_CI90']}")
    print(f"F1_CV      : {cv_results['F1_CV']:.4f}   CI90: {cv_results['F1_CI90']}")

    fitted_model = clone(model).fit(X, y)
    print("\n=== Feature Importances ===")
    print(pd.DataFrame({"feature_index": range(X.shape[1]), "importance": fitted_model.feature_importances_}))

    print("\n=== Majority-Vote Representatives ===")
    print(vote_summary.head(12))

    print("\n=== Global Sensitivity (SRCC) ===")
    print(sensitivity)

    print("\n=== Noise Reliability ===")
    print(reliability)


if __name__ == "__main__":
    main()
