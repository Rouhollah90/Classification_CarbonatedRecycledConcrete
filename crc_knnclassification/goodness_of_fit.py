"""Distribution goodness-of-fit checks."""

import numpy as np
from scipy import stats


def standardized_residuals(x, mu_hat=None, sigma_hat=None):
    x = np.asarray(x, dtype=float)
    mu_hat = np.mean(x) if mu_hat is None else mu_hat
    sigma_hat = np.std(x, ddof=1) if sigma_hat is None else sigma_hat

    if sigma_hat <= 0:
        raise ValueError("Standard deviation must be positive.")

    return (x - mu_hat) / sigma_hat, mu_hat, sigma_hat


def remove_outliers_by_zscore(x, threshold=3.0):
    x = np.asarray(x, dtype=float)
    z, _, _ = standardized_residuals(x)
    outlier_mask = np.abs(z) > threshold
    return x[~outlier_mask], z, outlier_mask


def anderson_darling_statistic(x, cdf_function):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)

    if n < 2:
        raise ValueError("At least two observations are required.")

    fx = np.clip(np.asarray(cdf_function(x), dtype=float), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    return -n - (1.0 / n) * np.sum(
        (2 * i - 1) * (np.log(fx) + np.log(1.0 - fx[::-1]))
    )


def ad_normal(x):
    x = np.asarray(x, dtype=float)
    mu_hat = np.mean(x)
    sigma_hat = np.std(x, ddof=1)

    if sigma_hat <= 0:
        raise ValueError("Normal fit failed because standard deviation is zero.")

    return anderson_darling_statistic(
        x,
        lambda v: stats.norm.cdf(v, loc=mu_hat, scale=sigma_hat),
    )


def ad_lognormal(x):
    x = np.asarray(x, dtype=float)

    if np.any(x <= 0):
        raise ValueError("Log-normal distribution requires all data > 0.")

    shape, loc, scale = stats.lognorm.fit(x, floc=0)
    return anderson_darling_statistic(
        x,
        lambda v: stats.lognorm.cdf(v, s=shape, loc=loc, scale=scale),
    )


def truncated_lognormal_cdf(x, shape, loc, scale, xmin, xmax):
    x = np.asarray(x, dtype=float)
    base_cdf = lambda t: stats.lognorm.cdf(t, s=shape, loc=loc, scale=scale)
    fmin = base_cdf(xmin)
    fmax = base_cdf(xmax)
    denom = fmax - fmin

    if denom <= 0:
        raise ValueError("Invalid truncation interval for TLN.")

    fx = np.empty_like(x, dtype=float)
    lower_mask = x < xmin
    upper_mask = x > xmax
    middle_mask = (~lower_mask) & (~upper_mask)
    fx[lower_mask] = 0.0
    fx[upper_mask] = 1.0
    fx[middle_mask] = (base_cdf(x[middle_mask]) - fmin) / denom
    return fx


def ad_tln(x, xmin=None, xmax=None):
    x = np.asarray(x, dtype=float)

    if np.any(x <= 0):
        raise ValueError("TLN requires all data > 0.")

    xmin = np.min(x) if xmin is None else xmin
    xmax = np.max(x) if xmax is None else xmax

    if xmin <= 0:
        raise ValueError("TLN requires xmin > 0.")
    if xmax <= xmin:
        raise ValueError("TLN requires xmax > xmin.")

    shape, loc, scale = stats.lognorm.fit(x, floc=0)
    return anderson_darling_statistic(
        x,
        lambda v: truncated_lognormal_cdf(v, shape, loc, scale, xmin, xmax),
    )


def ad_weibull(x):
    x = np.asarray(x, dtype=float)
    c, loc, scale = stats.weibull_min.fit(x)
    return anderson_darling_statistic(
        x,
        lambda v: stats.weibull_min.cdf(v, c=c, loc=loc, scale=scale),
    )


def ad_gamma(x):
    x = np.asarray(x, dtype=float)
    a, loc, scale = stats.gamma.fit(x)
    return anderson_darling_statistic(
        x,
        lambda v: stats.gamma.cdf(v, a=a, loc=loc, scale=scale),
    )


def compare_distributions(x, remove_outliers=False, z_threshold=3.0, xmin=None, xmax=None):
    x = np.asarray(x, dtype=float)

    if remove_outliers:
        used_data, _, _ = remove_outliers_by_zscore(x, threshold=z_threshold)
    else:
        used_data = x

    results = {"Normal": ad_normal(used_data)}

    if np.all(used_data > 0):
        results["Log-normal"] = ad_lognormal(used_data)
        results["TLN"] = ad_tln(used_data, xmin=xmin, xmax=xmax)
        results["Weibull"] = ad_weibull(used_data)
        results["Gamma"] = ad_gamma(used_data)

    return dict(sorted(results.items(), key=lambda item: item[1])), used_data


def print_interpretation(results):
    print("\nDistribution ranking by Anderson-Darling statistic:")
    for name, value in results.items():
        fit_quality = "good fit" if value < 1.0 else "poorer fit"
        print(f"{name:12s} A^2 = {value:.6f}   -> {fit_quality}")
