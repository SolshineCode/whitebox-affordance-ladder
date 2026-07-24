"""Standardized probe configuration for all V3 experiments.

Provides a single source of truth for linear probe hyperparameters, CV strategy,
and statistical testing utilities. All V3 validation scripts should import from
here rather than defining their own LogisticRegression/cross_val_score configs.

Created: 2026-03-31 (Issues #10, #12 — standardization + statistical rigor)
"""
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import stats as sp_stats

STANDARD_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def make_standard_probe():
    """Standard probe pipeline used across all experiments."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=42)
    )


def run_standard_probe(X, y):
    """Run probe with standardized config. Returns dict with accuracy, std, ci_lower, ci_upper."""
    probe = make_standard_probe()
    scores = cross_val_score(probe, X, y, cv=STANDARD_CV, scoring='accuracy')
    mean = scores.mean()
    # Use sample std (ddof=1) for CI calculation per statistical convention
    std = scores.std(ddof=1)
    # 95% CI using t-distribution (n=5 folds)
    t_val = sp_stats.t.ppf(0.975, df=len(scores)-1)
    ci_half = t_val * std / np.sqrt(len(scores))
    return {
        'accuracy': float(mean),
        'std': float(std),
        'ci_lower': float(mean - ci_half),
        'ci_upper': float(mean + ci_half),
        'fold_scores': scores.tolist(),
    }


def make_pca_probe(n_components):
    """Probe with PCA dimensionality reduction applied INSIDE the CV loop.

    PCA is a pipeline step so it's fit only on training folds, preventing
    test-set information leakage.
    """
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=42),
        LogisticRegression(max_iter=5000, random_state=42)
    )


def run_pca_probe(X, y, n_components):
    """Run probe with PCA reduction. Returns dict matching run_standard_probe format."""
    probe = make_pca_probe(n_components)
    scores = cross_val_score(probe, X, y, cv=STANDARD_CV, scoring='accuracy')
    bal_scores = cross_val_score(probe, X, y, cv=STANDARD_CV, scoring='balanced_accuracy')
    mean = scores.mean()
    bal_mean = bal_scores.mean()
    std = scores.std(ddof=1)
    bal_std = bal_scores.std(ddof=1)
    t_val = sp_stats.t.ppf(0.975, df=len(scores)-1)
    ci_half = t_val * std / np.sqrt(len(scores))
    bal_ci_half = t_val * bal_std / np.sqrt(len(bal_scores))
    return {
        'accuracy': float(mean),
        'std': float(std),
        'ci_lower': float(mean - ci_half),
        'ci_upper': float(mean + ci_half),
        'balanced_accuracy': float(bal_mean),
        'balanced_std': float(bal_std),
        'balanced_ci_lower': float(bal_mean - bal_ci_half),
        'balanced_ci_upper': float(bal_mean + bal_ci_half),
        'n_components': n_components,
        'fold_scores': scores.tolist(),
        'balanced_fold_scores': bal_scores.tolist(),
    }


def bootstrap_ci(X, y, n_bootstrap=1000, random_state=42):
    """Bootstrap 95% confidence interval for probe accuracy.

    Uses out-of-bag (OOB) evaluation to avoid data leakage: for each bootstrap
    sample, the probe is trained on the bootstrap sample and evaluated on the
    samples NOT selected (out-of-bag). This avoids the problem of duplicate
    entries appearing in both train and validation folds of cross_val_score.
    """
    rng = np.random.RandomState(random_state)
    n = len(y)
    accuracies = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[idx] = False
        oob_idx = np.where(oob_mask)[0]

        # Skip if OOB set is too small or missing a class
        if len(oob_idx) < 5 or len(np.unique(y[oob_idx])) < 2:
            continue
        # Skip if bootstrap sample is missing a class
        if len(np.unique(y[idx])) < 2:
            continue

        probe = make_standard_probe()
        probe.fit(X[idx], y[idx])
        acc = probe.score(X[oob_idx], y[oob_idx])
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    return {
        'mean': float(accuracies.mean()),
        'ci_lower': float(np.percentile(accuracies, 2.5)),
        'ci_upper': float(np.percentile(accuracies, 97.5)),
        'n_bootstrap': len(accuracies),
    }


def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    corrected_alpha = alpha / n
    return {
        'corrected_alpha': corrected_alpha,
        'n_comparisons': n,
        'significant': [p < corrected_alpha for p in p_values],
        'original_p_values': p_values,
    }


def compute_power(accuracy, n_samples, chance=0.5, alpha=0.05):
    """Compute achieved statistical power for a probe accuracy result."""
    # Effect size (Cohen's h for proportions)
    h = 2 * np.arcsin(np.sqrt(accuracy)) - 2 * np.arcsin(np.sqrt(chance))
    # Standard error
    se = 1 / np.sqrt(n_samples)
    # Z-scores
    z_alpha = sp_stats.norm.ppf(1 - alpha/2)
    z_power = abs(h) / se - z_alpha
    power = sp_stats.norm.cdf(z_power)
    return {
        'power': float(power),
        'effect_size_h': float(h),
        'n_samples': n_samples,
        'alpha': alpha,
    }
