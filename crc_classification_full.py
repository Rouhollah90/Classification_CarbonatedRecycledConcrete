from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


# Data Cleaning Functions
# ==================================

def remove_duplicate_rows(x):
    x = np.asarray(x)
    return np.unique(x)


def trim_percentiles(x, lower=0.025, upper=0.975):
    x = np.asarray(x, dtype=float)

    q_low = np.quantile(x, lower)
    q_high = np.quantile(x, upper)

    keep_mask = (x >= q_low) & (x <= q_high)
    trimmed_x = x[keep_mask]

    return trimmed_x, keep_mask, q_low, q_high


def identify_extreme_percentiles(x, lower=0.05, upper=0.95):
    x = np.asarray(x, dtype=float)

    q_low = np.quantile(x, lower)
    q_high = np.quantile(x, upper)

    flag_mask = (x < q_low) | (x > q_high)

    return flag_mask, q_low, q_high


def winsorize_percentiles(x, lower=0.05, upper=0.95):
    x = np.asarray(x, dtype=float)

    q_low = np.quantile(x, lower)
    q_high = np.quantile(x, upper)

    x_w = np.clip(x, q_low, q_high)

    return x_w, q_low, q_high


def clean_distribution_data(
        x,
        trim_lower=0.025,
        trim_upper=0.975,
        winsor_lower=0.05,
        winsor_upper=0.95
):
    # Remove duplicates
    x = remove_duplicate_rows(x)

    # Two-sided trimming
    trimmed_x, keep_mask, q025, q975 = trim_percentiles(
        x,
        lower=trim_lower,
        upper=trim_upper
    )

    # Flag extreme observations
    flagged_mask, q05, q95 = identify_extreme_percentiles(
        trimmed_x,
        lower=winsor_lower,
        upper=winsor_upper
    )

    # Winsorisation
    winsorized_x, _, _ = winsorize_percentiles(
        trimmed_x,
        lower=winsor_lower,
        upper=winsor_upper
    )

    report = {
        "Q025": q025,
        "Q975": q975,
        "Q05": q05,
        "Q95": q95,
        "n_original": len(x),
        "n_trimmed": len(trimmed_x),
        "n_flagged": int(np.sum(flagged_mask)),
        "trimmed_percent":
            100 * (len(x) - len(trimmed_x)) / len(x)
    }

    return winsorized_x, report


# ==================================
# Example Usage
# ==================================

def demo_data_cleaning():
    """Run a small data-cleaning example."""

    data = np.array([
        12.1, 11.8, 12.5, 13.0, 12.2,
        11.9, 12.8, 12.4, 12.0, 12.6,
        100.0
    ])

    cleaned_data, report = clean_distribution_data(data)

    print("\n=== Data Cleaning Report ===")

    print(f"Original observations : {report['n_original']}")
    print(f"After trimming        : {report['n_trimmed']}")
    print(f"Flagged observations  : {report['n_flagged']}")
    print(f"Trimmed percentage    : {report['trimmed_percent']:.2f}%")

    print(f"\nQ2.5  = {report['Q025']:.4f}")
    print(f"Q97.5 = {report['Q975']:.4f}")

    print(f"\nQ5    = {report['Q05']:.4f}")
    print(f"Q95   = {report['Q95']:.4f}")

    print("\nCleaned data:")
    print(cleaned_data)

#GOF
# ==================================

def standardized_residuals(x, mu_hat=None, sigma_hat=None):
    x = np.asarray(x, dtype=float)

    if mu_hat is None:
        mu_hat = np.mean(x)

    if sigma_hat is None:
        sigma_hat = np.std(x, ddof=1)

    if sigma_hat <= 0:
        raise ValueError("Standard deviation must be positive.")

    z = (x - mu_hat) / sigma_hat
    return z, mu_hat, sigma_hat

def remove_outliers_by_zscore(x, threshold=3.0):
    x = np.asarray(x, dtype=float)
    z, _, _ = standardized_residuals(x)
    outlier_mask = np.abs(z) > threshold
    filtered_x = x[~outlier_mask]
    return filtered_x, z, outlier_mask

def anderson_darling_statistic(x, cdf_function):
    x = np.asarray(x, dtype=float)
    x = np.sort(x)
    n = len(x)

    if n < 2:
        raise ValueError("At least two observations are required.")

    Fx = np.asarray(cdf_function(x), dtype=float)
    eps = 1e-12
    Fx = np.clip(Fx, eps, 1 - eps)

    i = np.arange(1, n + 1)

    A2 = -n - (1.0 / n) * np.sum(
        (2 * i - 1) * (np.log(Fx) + np.log(1.0 - Fx[::-1]))
    )
    return A2

#Distributions
def ad_normal(x):
    x = np.asarray(x, dtype=float)
    mu_hat = np.mean(x)
    sigma_hat = np.std(x, ddof=1)

    if sigma_hat <= 0:
        raise ValueError("Normal fit failed because standard deviation is zero.")

    return anderson_darling_statistic(
        x,
        lambda v: stats.norm.cdf(v, loc=mu_hat, scale=sigma_hat)
    )

def ad_lognormal(x):
    x = np.asarray(x, dtype=float)

    if np.any(x <= 0):
        raise ValueError("Log-normal distribution requires all data > 0.")

    shape, loc, scale = stats.lognorm.fit(x, floc=0)

    return anderson_darling_statistic(
        x,
        lambda v: stats.lognorm.cdf(v, s=shape, loc=loc, scale=scale)
    )

## TLN: Truncated Log-Normal to Long tail
def truncated_lognormal_cdf(x, shape, loc, scale, xmin, xmax):
    x = np.asarray(x, dtype=float)

    base_cdf = lambda t: stats.lognorm.cdf(t, s=shape, loc=loc, scale=scale)

    Fmin = base_cdf(xmin)
    Fmax = base_cdf(xmax)

    denom = Fmax - Fmin
    if denom <= 0:
        raise ValueError("Invalid truncation interval for TLN: denominator <= 0.")

    Fx = np.empty_like(x, dtype=float)

    lower_mask = x < xmin
    upper_mask = x > xmax
    middle_mask = (~lower_mask) & (~upper_mask)

    Fx[lower_mask] = 0.0
    Fx[upper_mask] = 1.0
    Fx[middle_mask] = (base_cdf(x[middle_mask]) - Fmin) / denom

    return Fx

def ad_tln(x, xmin=None, xmax=None):
    x = np.asarray(x, dtype=float)

    if np.any(x <= 0):
        raise ValueError("TLN requires all data > 0.")

    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)

    if xmin <= 0:
        raise ValueError("TLN requires xmin > 0.")

    if xmax <= xmin:
        raise ValueError("TLN requires xmax > xmin.")

    shape, loc, scale = stats.lognorm.fit(x, floc=0)

    return anderson_darling_statistic(
        x,
        lambda v: truncated_lognormal_cdf(v, shape, loc, scale, xmin, xmax)
    )

def ad_weibull(x):
    x = np.asarray(x, dtype=float)

    c, loc, scale = stats.weibull_min.fit(x)

    return anderson_darling_statistic(
        x,
        lambda v: stats.weibull_min.cdf(v, c=c, loc=loc, scale=scale)
    )

def ad_gamma(x):
    x = np.asarray(x, dtype=float)

    a, loc, scale = stats.gamma.fit(x)

    return anderson_darling_statistic(
        x,
        lambda v: stats.gamma.cdf(v, a=a, loc=loc, scale=scale)
    )

def compare_distributions(x, remove_outliers=False, z_threshold=3.0, xmin=None, xmax=None):
    x = np.asarray(x, dtype=float)

    if remove_outliers:
        used_data, z, outlier_mask = remove_outliers_by_zscore(x, threshold=z_threshold)
    else:
        used_data = x
        z, _, _ = standardized_residuals(x)
        outlier_mask = np.zeros_like(x, dtype=bool)

    results = {}

    results["Normal"] = ad_normal(used_data)
    # Positive-only distributions
    if np.all(used_data > 0):
        results["Log-normal"] = ad_lognormal(used_data)
        results["TLN"] = ad_tln(used_data, xmin=xmin, xmax=xmax)
        results["Weibull"] = ad_weibull(used_data)
        results["Gamma"] = ad_gamma(used_data)

    results = dict(sorted(results.items(), key=lambda item: item[1]))

    return results, used_data
def print_interpretation(results):
    
    print("\nDistribution ranking by Anderson-Darling statistic:")
    for name, value in results.items():
        fit_quality = "good fit" if value < 1.0 else "poorer fit"
        print(f"{name:12s} A^2 = {value:.6f}   -> {fit_quality}")


# -------------------------
# Example usage
# -------------------------
def demo_distribution_fitting():
    """Run a small Anderson-Darling distribution comparison example."""
    
    data = np.array([12.1, 11.8, 12.5, 13.0, 12.2, 11.9, 12.8, 12.4, 12.0, 12.6])

    # Step 1: standardized residuals
    z, mu_hat, sigma_hat = standardized_residuals(data)

    print("Mean estimate (mu_hat):", mu_hat)
    print("Standard deviation estimate (sigma_hat):", sigma_hat)
    print("Standardized residuals (z_i):")
    print(z)

    # Optional outlier check
    filtered_data, z_all, outlier_mask = remove_outliers_by_zscore(data, threshold=3.0)
    outlier_percent = 100.0 * np.sum(outlier_mask) / len(data)

    print("\nOutlier analysis using |z| > 3:")
    print("Outlier mask:", outlier_mask)
    print(f"Outlier percentage: {outlier_percent:.2f}%")
    print("Data used after outlier removal:")
    print(filtered_data)

    # Step 2: compare distributions
    results, used_data = compare_distributions(
        data,
        remove_outliers=True,
        z_threshold=3.0,
        xmin=None,
        xmax=None
    )

    print_interpretation(results)

    # Best Fitted distribution
    best_name = next(iter(results))
    best_value = results[best_name]
    print(f"\nBest-fitting distribution: {best_name} with A^2 = {best_value:.6f}")

# Class Labelling
# =====================================================

def assign_act_classes(duration, pressure):

    duration = np.asarray(duration, dtype=float)
    pressure = np.asarray(pressure, dtype=float)

    labels = []

    for d, p in zip(duration, pressure):

        if (p <= 1.0) and (d <= 10.0):
            labels.append("ACT-1")

        elif (p <= 1.0) and (d > 10.0):
            labels.append("ACT-2")

        elif (p > 1.0) and (p < 3.0) and (d <= 10.0):
            labels.append("ACT-3")

        elif (p >= 3.0) and (d <= 10.0):
            labels.append("ACT-4")

        else:
            labels.append("Unclassified")

    return np.asarray(labels)


def make_labels_for_k_classes(duration, pressure, k):
    
    duration = np.asarray(duration, dtype=float)
    pressure = np.asarray(pressure, dtype=float)

    labels = []

    for d, p in zip(duration, pressure):

        if k == 2:
            if p <= 1.0:
                labels.append("Atmospheric")
            else:
                labels.append("Pressurised")

        elif k == 3:
            if p <= 1.0:
                labels.append("Atmospheric")
            elif 1.0 < p < 3.0:
                labels.append("Medium-pressure")
            else:
                labels.append("High-pressure")

        elif k == 4:
            if (p <= 1.0) and (d <= 10.0):
                labels.append("ACT-1")
            elif (p <= 1.0) and (d > 10.0):
                labels.append("ACT-2")
            elif (1.0 < p < 3.0) and (d <= 10.0):
                labels.append("ACT-3")
            elif (p >= 3.0) and (d <= 10.0):
                labels.append("ACT-4")
            else:
                labels.append("Unclassified")

        elif k == 5:
            if (p <= 1.0) and (d <= 10.0):
                labels.append("Atmospheric-short")
            elif (p <= 1.0) and (d > 10.0):
                labels.append("Atmospheric-long")
            elif (1.0 < p < 3.0) and (d <= 10.0):
                labels.append("Medium-short")
            elif (1.0 < p < 3.0) and (d > 10.0):
                labels.append("Medium-long")
            elif (p >= 3.0) and (d <= 10.0):
                labels.append("High-short")
            else:
                labels.append("Unclassified")

        elif k == 6:
            if (p <= 1.0) and (d <= 10.0):
                labels.append("Atmospheric-short")
            elif (p <= 1.0) and (d > 10.0):
                labels.append("Atmospheric-long")
            elif (1.0 < p < 3.0) and (d <= 10.0):
                labels.append("Medium-short")
            elif (1.0 < p < 3.0) and (d > 10.0):
                labels.append("Medium-long")
            elif (p >= 3.0) and (d <= 10.0):
                labels.append("High-short")
            elif (p >= 3.0) and (d > 10.0):
                labels.append("High-long")
            else:
                labels.append("Unclassified")

        else:
            raise ValueError("k must be between 2 and 6.")

    return np.asarray(labels)


def evaluate_label_structure(X, labels):
  
    labels = np.asarray(labels)

    valid_mask = labels != "Unclassified"
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    if len(np.unique(labels_valid)) < 2:
        raise ValueError("At least two classes are required.")

    encoded = LabelEncoder().fit_transform(labels_valid)

    silhouette = silhouette_score(X_valid, encoded)
    dbi = davies_bouldin_score(X_valid, encoded)
    chi = calinski_harabasz_score(X_valid, encoded)

    return silhouette, dbi, chi, X_valid, labels_valid


def evaluate_all_class_structures(duration, pressure):
   
    X = np.column_stack([pressure, duration])

    structure_names = {
        2: "Atmospheric-pressure; Pressurised",
        3: "Atmospheric-pressure; Medium-pressure; High-pressure",
        4: "Atmospheric–short-duration; Atmospheric–long-duration; Medium-pressure–short-duration; High-pressure–short-duration",
        5: "Atmospheric-pressure–short-duration; Atmospheric-pressure–long-duration; Medium-pressure–short-duration; Medium-pressure–long-duration; High-pressure–short-duration",
        6: "Full pressure and duration combination"
    }

    rows = []

    for k in [2, 3, 4, 5, 6]:

        labels = make_labels_for_k_classes(duration, pressure, k)

        silhouette, dbi, chi, _, _ = evaluate_label_structure(X, labels)

        rows.append({
            "ACT Classes": k,
            "Structure assumption": structure_names[k],
            "Silhouette": silhouette,
            "DBI": dbi,
            "CHI": chi
        })

    return pd.DataFrame(rows)


def match_kmeans_clusters_to_act(y_true, y_cluster):
 
    cm = confusion_matrix(y_true, y_cluster)

    row_ind, col_ind = linear_sum_assignment(-cm)

    mapping = {}

    for r, c in zip(row_ind, col_ind):
        mapping[c] = r

    mapped_clusters = np.array([mapping[c] for c in y_cluster])

    return mapped_clusters

# K-mean Validation
#==============================

def validate_kmeans_against_act(duration, pressure):
    
    X = np.column_stack([pressure, duration])

    act_labels = assign_act_classes(duration, pressure)

    valid_mask = act_labels != "Unclassified"

    X_valid = X[valid_mask]
    act_labels_valid = act_labels[valid_mask]

    encoder = LabelEncoder()
    y_true = encoder.fit_transform(act_labels_valid)

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=20
    )

    y_cluster = kmeans.fit_predict(X_valid)

    y_cluster_mapped = match_kmeans_clusters_to_act(
        y_true,
        y_cluster
    )

    overall_agreement = 100.0 * np.mean(y_true == y_cluster_mapped)

    ari = adjusted_rand_score(y_true, y_cluster)
    nmi = normalized_mutual_info_score(y_true, y_cluster)

    rows = []

    for class_id in np.unique(y_true):

        mask = y_true == class_id

        class_agreement = 100.0 * np.mean(
            y_true[mask] == y_cluster_mapped[mask]
        )

        rows.append({
            "Labeled ACT": encoder.inverse_transform([class_id])[0],
            "K-means cluster": f"Cluster-{class_id + 1}",
            "Agreement (%)": class_agreement
        })

    table_s5 = pd.DataFrame(rows)

    validation_metrics = pd.DataFrame({
        "Metric": [
            "Overall agreement (%)",
            "ARI",
            "NMI"
        ],
        "Value": [
            overall_agreement,
            ari,
            nmi
        ]
    })

    return table_s5, validation_metrics


# =====================================================
# Example Usage
# =====================================================

def demo_act_validation():
    """Run a synthetic ACT label and K-means validation example."""

    np.random.seed(42)

    n = 250

    # ACT-1: atmospheric pressure, short duration
    duration_1 = np.random.uniform(1, 10, n)
    pressure_1 = np.random.normal(1.0, 0.08, n)

    # ACT-2: atmospheric pressure, long duration
    duration_2 = np.random.uniform(11, 50, n)
    pressure_2 = np.random.normal(1.0, 0.08, n)

    # ACT-3: medium pressure, short duration
    duration_3 = np.random.uniform(1, 10, n)
    pressure_3 = np.random.uniform(1.2, 2.8, n)

    # ACT-4: high pressure, short duration
    duration_4 = np.random.uniform(1, 10, n)
    pressure_4 = np.random.uniform(3.0, 5.0, n)

    duration = np.concatenate([
        duration_1,
        duration_2,
        duration_3,
        duration_4
    ])

    pressure = np.concatenate([
        pressure_1,
        pressure_2,
        pressure_3,
        pressure_4
    ])

    pressure = np.clip(pressure, 0, 6)

    table_s4 = evaluate_all_class_structures(
        duration,
        pressure
    )

    table_s5, validation_metrics = validate_kmeans_against_act(
        duration,
        pressure
    )

    print("\nTable S4. Labelling class evaluation considering non-constant variables")
    print(table_s4.round({
        "Silhouette": 2,
        "DBI": 2,
        "CHI": 0
    }))

    print("\nTable S5. Compared K-means clusters and 4-ACT labels")
    print(table_s5.round({
        "Agreement (%)": 0
    }))

    print("\nValidation metrics")
    print(validation_metrics.round({
        "Value": 2
    }))

# KNN Hyperparameter Tuning
# =====================================================

def lp_distance_matrix(XA, XB=None, p=1.5):
    
    XA = np.asarray(XA, dtype=float)
    XB = XA if XB is None else np.asarray(XB, dtype=float)

    diff = np.abs(XA[:, None, :] - XB[None, :, :]) ** p
    D = np.sum(diff, axis=2) ** (1.0 / p)

    return D


def silverman_bandwidth(X):
    
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    if n < 2:
        raise ValueError("At least two samples are required.")

    s_hat = np.mean(np.std(X, axis=0, ddof=1))
    phi = 1.6 * s_hat * (n ** (-1.0 / 5.0))

    return max(phi, 1e-12)


def compute_knn_weights(distances, weighting="gaussian", phi=None, epsilon=1e-12):
   
    distances = np.asarray(distances, dtype=float)

    if weighting == "uniform":

        weights = np.ones_like(distances)

    elif weighting == "inverse-distance":

        weights = 1.0 / (distances + epsilon)

    elif weighting == "gaussian":

        if phi is None:
            raise ValueError("phi is required for Gaussian weighting.")

        weights = np.exp(
            -(distances ** 2) / (2.0 * (phi ** 2))
        )

    else:

        raise ValueError(
            "weighting must be 'uniform', 'inverse-distance', or 'gaussian'."
        )

    return weights


@dataclass
class TunableWeightedKNN:

    k: int = 5
    p: float = 1.5
    weighting: str = "gaussian"
    phi: float = None
    epsilon: float = 1e-12

    def fit(self, X, y):

        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

        if self.X_train.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if len(self.X_train) != len(self.y_train):
            raise ValueError("X and y must have the same number of samples.")

        if self.k < 1:
            raise ValueError("k must be at least 1.")

        if self.k > len(self.X_train):
            raise ValueError("k cannot be larger than the number of training samples.")

        self.classes_ = np.unique(self.y_train)

        if self.phi is None:
            self.phi_ = silverman_bandwidth(self.X_train)
        else:
            self.phi_ = float(self.phi)

        return self

    def _predict_one(self, x):

        x = np.asarray(x, dtype=float).reshape(1, -1)

        distances = lp_distance_matrix(
            x,
            self.X_train,
            p=self.p
        ).ravel()

        nn_idx = np.argsort(distances)[:self.k]

        nn_distances = distances[nn_idx]
        nn_labels = self.y_train[nn_idx]

        nn_weights = compute_knn_weights(
            nn_distances,
            weighting=self.weighting,
            phi=self.phi_,
            epsilon=self.epsilon
        )

        class_scores = {}

        for c in self.classes_:
            class_scores[c] = np.sum(
                nn_weights[nn_labels == c]
            )

        predicted_class = max(
            class_scores,
            key=class_scores.get
        )

        return predicted_class

    def predict(self, X):

        X = np.atleast_2d(np.asarray(X, dtype=float))

        predictions = [
            self._predict_one(x)
            for x in X
        ]

        return np.asarray(predictions)


def confidence_interval_90(values):
    
    values = np.asarray(values, dtype=float)
    n = len(values)

    mean_value = float(np.mean(values))

    if n < 2:
        return mean_value, mean_value, mean_value, 0.0

    sigma = float(np.std(values, ddof=1))
    margin = 1.645 * sigma / np.sqrt(n)

    lower = float(mean_value - margin)
    upper = float(mean_value + margin)

    return mean_value, lower, upper, sigma


def cross_validate_tunable_knn(
        X,
        y,
        k=5,
        p=1.5,
        weighting="gaussian",
        n_splits=5,
        random_state=42
):

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    fold_f1 = []
    fold_accuracy = []

    for train_idx, test_idx in skf.split(X, y):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model = TunableWeightedKNN(
            k=k,
            p=p,
            weighting=weighting
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = np.mean(y_pred == y_test)

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        fold_accuracy.append(acc)
        fold_f1.append(f1)

    acc_mean, acc_low, acc_high, acc_sigma = confidence_interval_90(
        fold_accuracy
    )

    f1_mean, f1_low, f1_high, f1_sigma = confidence_interval_90(
        fold_f1
    )

    return {
        "k": k,
        "p": p,
        "weighting": weighting,
        "fold_accuracy": fold_accuracy,
        "fold_f1": fold_f1,
        "Accuracy_CV": acc_mean,
        "Accuracy_CI90_low": acc_low,
        "Accuracy_CI90_high": acc_high,
        "Accuracy_sigma": acc_sigma,
        "F1_CV": f1_mean,
        "F1_CI90_low": f1_low,
        "F1_CI90_high": f1_high,
        "F1_sigma": f1_sigma
    }


def run_knn_hyperparameter_tuning(
        X,
        y,
        k_values=None,
        p_values=None,
        weighting_functions=None,
        n_splits=5,
        random_state=42
):
   
    if k_values is None:
        k_values = list(range(3, 16, 2))

    if p_values is None:
        p_values = [1.0, 1.5, 2.0]

    if weighting_functions is None:
        weighting_functions = [
            "uniform",
            "inverse-distance",
            "gaussian"
        ]

    rows = []

    for k in k_values:

        for p in p_values:

            for weighting in weighting_functions:

                result = cross_validate_tunable_knn(
                    X=X,
                    y=y,
                    k=k,
                    p=p,
                    weighting=weighting,
                    n_splits=n_splits,
                    random_state=random_state
                )

                rows.append(result)

    results_df = pd.DataFrame(rows)

    results_df = results_df.sort_values(
        by=["F1_CV", "Accuracy_CV"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return results_df


# =====================================================
# Example Usage
# =====================================================

def demo_hyperparameter_tuning():
    """Run weighted KNN hyperparameter tuning on synthetic data."""

    np.random.seed(42)

    n = 250

    # ACT-1: atmospheric pressure, short duration
    duration_1 = np.random.uniform(1, 10, n)
    pressure_1 = np.random.normal(1.0, 0.08, n)
    co2_1 = np.random.normal(20, 3, n)

    # ACT-2: atmospheric pressure, long duration
    duration_2 = np.random.uniform(11, 50, n)
    pressure_2 = np.random.normal(1.0, 0.08, n)
    co2_2 = np.random.normal(35, 4, n)

    # ACT-3: medium pressure, short duration
    duration_3 = np.random.uniform(1, 10, n)
    pressure_3 = np.random.uniform(1.2, 2.8, n)
    co2_3 = np.random.normal(45, 5, n)

    # ACT-4: high pressure, short duration
    duration_4 = np.random.uniform(1, 10, n)
    pressure_4 = np.random.uniform(3.0, 5.0, n)
    co2_4 = np.random.normal(60, 6, n)

    X = np.vstack([
        np.column_stack([duration_1, pressure_1, co2_1]),
        np.column_stack([duration_2, pressure_2, co2_2]),
        np.column_stack([duration_3, pressure_3, co2_3]),
        np.column_stack([duration_4, pressure_4, co2_4])
    ])

    y = np.array(
        ["ACT-1"] * n +
        ["ACT-2"] * n +
        ["ACT-3"] * n +
        ["ACT-4"] * n
    )

    results = run_knn_hyperparameter_tuning(
        X=X,
        y=y,
        k_values=list(range(3, 16, 2)),
        p_values=[1.0, 1.5, 2.0],
        weighting_functions=[
            "uniform",
            "inverse-distance",
            "gaussian"
        ],
        n_splits=5,
        random_state=42
    )

    print("\nKNN Hyperparameter Tuning Results")
    print(results[
        [
            "k",
            "p",
            "weighting",
            "F1_CV",
            "F1_CI90_low",
            "F1_CI90_high",
            "Accuracy_CV"
        ]
    ].round(4))

    best = results.iloc[0]

#Classification
# ==================================

# Distance matrix
def lp_distance_matrix(XA, XB=None, p=1.5):
    XA = np.asarray(XA, dtype=float)
    XB = XA if XB is None else np.asarray(XB, dtype=float)

    diff = np.abs(XA[:, None, :] - XB[None, :, :]) ** p
    D = np.sum(diff, axis=2) ** (1.0 / p)
    return D

def silverman_bandwidth(X):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    if n < 2:
        raise ValueError("At least two samples are required to compute bandwidth.")

    s_hat = np.mean(np.std(X, axis=0, ddof=1))
    phi = 1.6 * s_hat * (n ** (-1.0 / 5.0))

    return max(phi, 1e-12)

def gaussian_weights(distances, phi):
    distances = np.asarray(distances, dtype=float)
    return np.exp(-(distances ** 2) / (2.0 * (phi ** 2)))

def confidence_interval_90(values):
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < 2:
        mean_val = float(np.mean(values))
        return mean_val, mean_val, mean_val, 0.0

    mean_val = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))
    margin = 1.645 * sigma / np.sqrt(n)

    lower = float(mean_val - margin)
    upper = float(mean_val + margin)
    return mean_val, lower, upper, sigma


# Weighted KNN Classifier
#----------------------------------
@dataclass
class WeightedKNN:
    k: int = 5
    p: float = 1.5
    phi: float = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

        if self.X_train.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if len(self.X_train) != len(self.y_train):
            raise ValueError("X and y must have the same number of samples.")

        if self.k < 1:
            raise ValueError("k must be at least 1.")

        if self.k > len(self.X_train):
            raise ValueError("k cannot be larger than the number of training samples.")

        self.classes_ = np.unique(self.y_train)

        if self.phi is None:
            self.phi_ = silverman_bandwidth(self.X_train)
        else:
            self.phi_ = float(self.phi)

        return self

    def _predict_one_with_details(self, x):
        x = np.asarray(x, dtype=float).reshape(1, -1)

        distances = lp_distance_matrix(x, self.X_train, p=self.p).ravel()
        nn_idx = np.argsort(distances)[:self.k]

        nn_distances = distances[nn_idx]
        nn_labels = self.y_train[nn_idx]
        nn_weights = gaussian_weights(nn_distances, self.phi_)

        class_scores = {}
        for c in self.classes_:
            class_scores[c] = np.sum(nn_weights[nn_labels == c])

        total_score = np.sum(list(class_scores.values()))
        if total_score <= 0:
            class_probs = {c: 0.0 for c in self.classes_}
        else:
            class_probs = {c: class_scores[c] / total_score for c in self.classes_}

        predicted_class = max(class_scores, key=class_scores.get)
        mean_neighbor_distance = float(np.mean(nn_distances))

        return predicted_class, class_probs, mean_neighbor_distance, nn_idx

    def predict(self, X):
        # Class labeling
        X = np.atleast_2d(np.asarray(X, dtype=float))
        preds = [self._predict_one_with_details(x)[0] for x in X]
        return np.asarray(preds)

    def predict_confidence(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        conf = []

        for x in X:
            _, _, mean_dist, _ = self._predict_one_with_details(x)
            c_i = 1.0 / (1.0 + mean_dist)
            conf.append(c_i)

        return np.asarray(conf)

    def predict_posterior_confidence(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        conf = []

        for x in X:
            _, class_probs, _, _ = self._predict_one_with_details(x)
            conf.append(max(class_probs.values()))

        return np.asarray(conf)


# Medoid selection and MDM
# ------------------------------------
def class_medoid(X_class, p=1.5):
    X_class = np.asarray(X_class, dtype=float)

    if len(X_class) == 0:
        raise ValueError("Cannot compute medoid for an empty class.")

    if len(X_class) == 1:
        return X_class[0]

    D = lp_distance_matrix(X_class, X_class, p=p)
    total_dist = np.sum(D, axis=1)
    idx = np.argmin(total_dist)
    return X_class[idx]

def compute_class_medoids(X, y, p=1.5):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    medoids = {}
    for c in np.unique(y):
        Xc = X[y == c]
        medoids[c] = class_medoid(Xc, p=p)
    return medoids

def mean_distance_of_medoid(X, y, medoids, p=1.5):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    distances = []
    for xi, yi in zip(X, y):
        medoid = medoids[yi]
        d = lp_distance_matrix(np.array([xi]), np.array([medoid]), p=p)[0, 0]
        distances.append(d)

    return float(np.mean(distances))


# Cross validation and performance metrics
def cross_validate_knn(X, y, k=5, p=1.5, n_splits=5, random_state=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_scores = []
    fold_f1 = []
    fold_mdm = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = WeightedKNN(k=k, p=p)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        score_i = np.mean(y_pred == y_test)
        fold_scores.append(score_i)

        f1_i = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        fold_f1.append(f1_i)

        medoids = compute_class_medoids(X_train, y_train, p=p)

        valid_mask = np.array([yi in medoids for yi in y_test])
        mdm_i = mean_distance_of_medoid(X_test[valid_mask], y_test[valid_mask], medoids, p=p)
        fold_mdm.append(mdm_i)

    score_mean, score_ci_low, score_ci_high, score_sigma = confidence_interval_90(fold_scores)
    f1_mean, f1_ci_low, f1_ci_high, f1_sigma = confidence_interval_90(fold_f1)
    mdm_mean, mdm_ci_low, mdm_ci_high, mdm_sigma = confidence_interval_90(fold_mdm)

    results = {
        "fold_scores": np.array(fold_scores),
        "fold_f1": np.array(fold_f1),
        "fold_mdm": np.array(fold_mdm),

        "Score_CV": score_mean,
        "F1_CV": f1_mean,
        "MDM_CV": mdm_mean,

        "Score_CI90": (score_ci_low, score_ci_high),
        "F1_CI90": (f1_ci_low, f1_ci_high),
        "MDM_CI90": (mdm_ci_low, mdm_ci_high),

        "Score_sigma": score_sigma,
        "F1_sigma": f1_sigma,
        "MDM_sigma": mdm_sigma
    }

    return results

# Threshold interpretation
def interpret_classifier_performance(f1_cv, mdm_cv, ci90_width):
    
    return {
        "Prediction": "Pass" if f1_cv > 0.75 else "Fail",
        "Cohesion": "Pass" if mdm_cv < 0.2 else "Fail",
        "Stability": "Pass" if ci90_width < 0.15 else "Fail"
    }

# Global Sensitivity Analysis (SRCC)
# -------------------------------------
def global_sensitivity_srcc(X, y, model):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    classes = np.unique(y)
    n_features = X.shape[1]
    records = []
    confidences = model.predict_posterior_confidence(X)

    for c in classes:
        class_mask = (y == c)
        Xc = X[class_mask]
        Cc = confidences[class_mask]

        for m in range(n_features):
            feature_values = Xc[:, m]

            if len(np.unique(feature_values)) < 2 or len(np.unique(Cc)) < 2:
                rho = np.nan
                pval = np.nan
            else:
                rho, pval = spearmanr(feature_values, Cc)

            abs_rho = np.abs(rho) if not np.isnan(rho) else np.nan

            if np.isnan(abs_rho):
                level = "Undefined"
            elif abs_rho >= 0.7:
                level = "High"
            elif abs_rho >= 0.4:
                level = "Moderate"
            else:
                level = "Low"

            records.append({
                "class": c,
                "feature_index": m,
                "SRCC": rho,
                "abs_SRCC": abs_rho,
                "sensitivity_level": level,
                "p_value": pval
            })

    return pd.DataFrame(records)

# Noise robustness reliability assessment
# ----------------------------------------
def add_multiplicative_gaussian_noise(X, sigma, random_state=None):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)

    noise = rng.normal(loc=0.0, scale=sigma, size=X.shape)
    X_noisy = X * (1.0 + noise)
    return X_noisy

def noise_reliability_assessment(
    X, y, k=5, p=1.5, noise_levels=None, M=50, random_state=42
):
    
    if noise_levels is None:
        noise_levels = np.linspace(0.0, 0.10, 6)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    model = WeightedKNN(k=k, p=p).fit(X, y)
    baseline_pred = model.predict(X)

    rng = np.random.default_rng(random_state)

    results = []

    for sigma in noise_levels:
        R_values = []

        for _ in range(M):
            seed = int(rng.integers(0, 1_000_000_000))
            X_noisy = add_multiplicative_gaussian_noise(X, sigma=sigma, random_state=seed)
            pred_noisy = model.predict(X_noisy)

            R_sigma = np.mean(pred_noisy == baseline_pred)
            R_values.append(R_sigma)

        R_values = np.asarray(R_values)
        R_bar = float(np.mean(R_values))              
        SD_R = float(np.std(R_values, ddof=1)) if len(R_values) > 1 else 0.0   

        if R_bar >= 0.95:
            consistency_level = "High"
        elif R_bar >= 0.90:
            consistency_level = "Moderate"
        else:
            consistency_level = "Low"

        results.append({
            "sigma": sigma,
            "R_values": R_values,
            "R_bar": R_bar,
            "SD_R": SD_R,
            "consistency_level": consistency_level
        })

    results_df = pd.DataFrame(results)

    # RI
    mean_R = results_df["R_bar"].mean()
    sd_over_sigma = results_df["R_bar"].std(ddof=1) if len(results_df) > 1 else 0.0

    if mean_R == 0:
        RI = np.nan
    else:
        RI = 1.0 - (sd_over_sigma / mean_R)

    if np.isnan(RI):
        ri_level = "Undefined"
    elif RI >= 0.95:
        ri_level = "High"
    elif RI >= 0.90:
        ri_level = "Moderate"
    else:
        ri_level = "Low"

    # Reliability slope
    sigma0 = results_df["sigma"].iloc[0]
    sigma_max = results_df["sigma"].iloc[-1]
    R0 = results_df["R_bar"].iloc[0]
    Rmax = results_df["R_bar"].iloc[-1]

    if sigma_max == sigma0:
        lambda_slope = np.nan
    else:
        lambda_slope = (Rmax - R0) / (sigma_max - sigma0)

    abs_lambda = np.abs(lambda_slope) if not np.isnan(lambda_slope) else np.nan

    if np.isnan(abs_lambda):
        lambda_level = "Undefined"
    elif abs_lambda < 0.7:
        lambda_level = "High"
    elif abs_lambda < 1.0:
        lambda_level = "Moderate"
    else:
        lambda_level = "Low"

    summary = {
        "noise_results": results_df,
        "RI": RI,
        "RI_level": ri_level,
        "lambda": lambda_slope,
        "lambda_level": lambda_level
    }

    return summary

# Outlier filtering
# -----------------------------------
def filter_outliers_by_feature_variation(X, threshold_ratio=0.02):
    X = np.asarray(X, dtype=float)

    row_mean = np.mean(np.abs(X), axis=1)
    row_std = np.std(X, axis=1, ddof=1)
    variation = np.divide(row_std, row_mean, out=np.zeros_like(row_std), where=row_mean > 0)

    keep_mask = variation <= threshold_ratio
    return keep_mask, variation

# Classifier pipeline
# -----------------------------------
def run_knn_pipeline(
    X, y,
    k=5,
    p=1.5,
    n_splits=5,
    random_state=42,
    noise_levels=None,
    M=50
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
        random_state=random_state
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
        random_state=random_state
    )

    f1_ci_width = cv_results["F1_CI90"][1] - cv_results["F1_CI90"][0]
    performance = interpret_classifier_performance(
        f1_cv=cv_results["F1_CV"],
        mdm_cv=cv_results["MDM_CV"],
        ci90_width=f1_ci_width
    )

    return {
        "model": model,
        "medoids": medoids,
        "cv_results": cv_results,
        "sensitivity": sensitivity_df,
        "reliability": reliability,
        "performance": performance
    }

# -------------------------
# Example usage
# -------------------------
def demo_knn_pipeline():
    """Run the end-to-end weighted KNN classification pipeline demo."""

    rng = np.random.default_rng(42)

    n_per_class = 40

    class_0 = rng.normal(loc=[2.0, 3.0, 1.0], scale=[0.3, 0.4, 0.2], size=(n_per_class, 3))
    class_1 = rng.normal(loc=[5.0, 4.0, 3.5], scale=[0.4, 0.3, 0.3], size=(n_per_class, 3))
    class_2 = rng.normal(loc=[8.0, 7.0, 6.0], scale=[0.5, 0.4, 0.4], size=(n_per_class, 3))

    X = np.vstack([class_0, class_1, class_2])
    y = np.array(["Class_1"] * n_per_class + ["Class_2"] * n_per_class + ["Class_3"] * n_per_class)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Pipeline running
    results = run_knn_pipeline(
        X=X,
        y=y_encoded,
        k=5,
        p=1.5,
        n_splits=5,
        random_state=42,
        noise_levels=np.linspace(0.0, 0.10, 6),
        M=50
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
    print(f"Reliability Index (RI): {results['reliability']['RI']:.4f} "
          f"-> {results['reliability']['RI_level']}")
    print(f"Reliability slope (lambda): {results['reliability']['lambda']:.4f} "
          f"-> {results['reliability']['lambda_level']}")


def main():
    """Run the default demonstration for command-line use."""
    demo_knn_pipeline()


if __name__ == "__main__":
    main()
