"""Noise robustness and reliability assessment."""

import numpy as np
import pandas as pd

from .knn import WeightedKNN


def add_multiplicative_gaussian_noise(X, sigma, random_state=None):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    noise = rng.normal(loc=0.0, scale=sigma, size=X.shape)
    return X * (1.0 + noise)


def noise_reliability_assessment(
    X,
    y,
    k=5,
    p=1.5,
    noise_levels=None,
    M=50,
    random_state=42,
):
    noise_levels = np.linspace(0.0, 0.10, 6) if noise_levels is None else noise_levels
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
            R_values.append(np.mean(model.predict(X_noisy) == baseline_pred))

        R_values = np.asarray(R_values)
        R_bar = float(np.mean(R_values))
        SD_R = float(np.std(R_values, ddof=1)) if len(R_values) > 1 else 0.0

        if R_bar >= 0.95:
            consistency_level = "High"
        elif R_bar >= 0.90:
            consistency_level = "Moderate"
        else:
            consistency_level = "Low"

        results.append(
            {
                "sigma": sigma,
                "R_values": R_values,
                "R_bar": R_bar,
                "SD_R": SD_R,
                "consistency_level": consistency_level,
            }
        )

    results_df = pd.DataFrame(results)
    mean_R = results_df["R_bar"].mean()
    sd_over_sigma = results_df["R_bar"].std(ddof=1) if len(results_df) > 1 else 0.0
    RI = np.nan if mean_R == 0 else 1.0 - (sd_over_sigma / mean_R)

    if np.isnan(RI):
        ri_level = "Undefined"
    elif RI >= 0.95:
        ri_level = "High"
    elif RI >= 0.90:
        ri_level = "Moderate"
    else:
        ri_level = "Low"

    sigma0 = results_df["sigma"].iloc[0]
    sigma_max = results_df["sigma"].iloc[-1]
    R0 = results_df["R_bar"].iloc[0]
    Rmax = results_df["R_bar"].iloc[-1]
    lambda_slope = np.nan if sigma_max == sigma0 else (Rmax - R0) / (sigma_max - sigma0)
    abs_lambda = np.abs(lambda_slope) if not np.isnan(lambda_slope) else np.nan

    if np.isnan(abs_lambda):
        lambda_level = "Undefined"
    elif abs_lambda < 0.7:
        lambda_level = "High"
    elif abs_lambda < 1.0:
        lambda_level = "Moderate"
    else:
        lambda_level = "Low"

    return {
        "noise_results": results_df,
        "RI": RI,
        "RI_level": ri_level,
        "lambda": lambda_slope,
        "lambda_level": lambda_level,
    }
