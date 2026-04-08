import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import spearmanr
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

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
if __name__ == "__main__":
    
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

    lower = mean_val - margin
    upper = mean_val + margin
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
        X = np.asarray(X, dtype=float)
        preds = [self._predict_one_with_details(x)[0] for x in X]
        return np.asarray(preds)

    def predict_confidence(self, X):
        X = np.asarray(X, dtype=float)
        conf = []

        for x in X:
            _, _, mean_dist, _ = self._predict_one_with_details(x)
            c_i = 1.0 / (1.0 + mean_dist)
            conf.append(c_i)

        return np.asarray(conf)

    def predict_posterior_confidence(self, X):
        X = np.asarray(X, dtype=float)
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

        # Eq. 7a: classification score
        score_i = np.mean(y_pred == y_test)
        fold_scores.append(score_i)

        # Eq. 9a and 9b: weighted F1 for multi-class classification
        f1_i = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        fold_f1.append(f1_i)

        # Eq. 8a and 8b: MDM on test set using training medoids
        medoids = compute_class_medoids(X_train, y_train, p=p)

        # Only evaluate MDM for test samples whose class exists in training
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

            # Eq. 16
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
if __name__ == "__main__":

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
