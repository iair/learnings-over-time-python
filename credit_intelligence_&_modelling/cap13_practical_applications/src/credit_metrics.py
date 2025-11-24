"""
Credit Scoring Metrics and Transformations

This module provides pure functional implementations of common credit scoring
metrics and transformations following SOLID principles adapted to functional programming.

Each function:
- Does one thing (Single Responsibility)
- Has no side effects (Immutability)
- Takes all dependencies as parameters (Dependency Inversion)
- Has minimal parameters (Interface Segregation)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from scipy import stats


# ============================================================================
# 1. FEATURE TRANSFORMATIONS
# ============================================================================

def min_max_scale(values: np.ndarray) -> np.ndarray:
    """
    Apply Min-Max scaling to transform values to [0,1] range.

    Formula: (X - X_min) / (X_max - X_min)

    Args:
        values: Array of numeric values to scale

    Returns:
        Scaled values in range [0,1]
    """
    min_val = np.min(values)
    max_val = np.max(values)

    if max_val == min_val:
        return np.zeros_like(values)

    return (values - min_val) / (max_val - min_val)


def z_score_standardize(values: np.ndarray) -> np.ndarray:
    """
    Apply Z-score standardization to transform values.

    Formula: (X - μ) / σ

    Args:
        values: Array of numeric values to standardize

    Returns:
        Standardized values with mean=0 and std=1
    """
    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        return np.zeros_like(values)

    return (values - mean) / std


def create_bins_equal_width(values: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Create bins with equal width intervals.

    Args:
        values: Array of numeric values to bin
        n_bins: Number of bins to create

    Returns:
        Array of bin indices (0 to n_bins-1)
    """
    min_val = np.min(values)
    max_val = np.max(values)

    edges = np.linspace(min_val, max_val, n_bins + 1)
    edges[-1] += 0.001  # Ensure max value is included

    return np.digitize(values, edges) - 1


def create_bins_equal_frequency(values: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Create bins with approximately equal number of observations.

    Args:
        values: Array of numeric values to bin
        n_bins: Number of bins to create

    Returns:
        Array of bin indices (0 to n_bins-1)
    """
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    edges[-1] += 0.001

    return np.digitize(values, edges) - 1


# ============================================================================
# 2. WEIGHT OF EVIDENCE (WOE) CALCULATIONS
# ============================================================================

def calculate_bin_statistics(
    bins: np.ndarray,
    target: np.ndarray
) -> pd.DataFrame:
    """
    Calculate statistics for each bin including counts and percentages.

    Args:
        bins: Array of bin assignments
        target: Binary target array (0=good, 1=bad)

    Returns:
        DataFrame with bin statistics
    """
    df = pd.DataFrame({'bin': bins, 'target': target})

    total_goods = np.sum(target == 0)
    total_bads = np.sum(target == 1)

    stats = df.groupby('bin').agg(
        total=('target', 'count'),
        goods=('target', lambda x: np.sum(x == 0)),
        bads=('target', lambda x: np.sum(x == 1))
    ).reset_index()

    stats['pct_goods'] = stats['goods'] / total_goods
    stats['pct_bads'] = stats['bads'] / total_bads
    stats['conversion_rate'] = stats['goods'] / stats['total']

    return stats


def calculate_woe(pct_goods: float, pct_bads: float) -> float:
    """
    Calculate Weight of Evidence for a single bin.

    Formula: ln(% Goods / % Bads)

    Args:
        pct_goods: Percentage of goods in bin
        pct_bads: Percentage of bads in bin

    Returns:
        WOE value
    """
    # Avoid division by zero and log(0)
    pct_goods = max(pct_goods, 0.0001)
    pct_bads = max(pct_bads, 0.0001)

    return np.log(pct_goods / pct_bads)


def calculate_woe_for_bins(bin_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate WOE for all bins in a DataFrame.

    Args:
        bin_stats: DataFrame with pct_goods and pct_bads columns

    Returns:
        DataFrame with WOE column added
    """
    result = bin_stats.copy()
    result['woe'] = result.apply(
        lambda row: calculate_woe(row['pct_goods'], row['pct_bads']),
        axis=1
    )
    return result


# ============================================================================
# 3. INFORMATION VALUE (IV)
# ============================================================================

def calculate_iv_component(pct_goods: float, pct_bads: float, woe: float) -> float:
    """
    Calculate IV component for a single bin.

    Formula: (% Goods - % Bads) × WOE

    Args:
        pct_goods: Percentage of goods in bin
        pct_bads: Percentage of bads in bin
        woe: Weight of Evidence for the bin

    Returns:
        IV component value
    """
    return (pct_goods - pct_bads) * woe


def calculate_information_value(bin_stats: pd.DataFrame) -> float:
    """
    Calculate total Information Value for a variable.

    Args:
        bin_stats: DataFrame with WOE calculations

    Returns:
        Total IV value
    """
    if 'woe' not in bin_stats.columns:
        bin_stats = calculate_woe_for_bins(bin_stats)

    bin_stats['iv_component'] = bin_stats.apply(
        lambda row: calculate_iv_component(
            row['pct_goods'],
            row['pct_bads'],
            row['woe']
        ),
        axis=1
    )

    return np.abs(bin_stats['iv_component']).sum()


def interpret_iv(iv: float) -> str:
    """
    Interpret Information Value according to standard thresholds.

    Args:
        iv: Information Value

    Returns:
        Interpretation string
    """
    if iv < 0.02:
        return "Not predictive"
    elif iv < 0.1:
        return "Weak"
    elif iv < 0.3:
        return "Medium"
    elif iv < 0.5:
        return "Strong"
    else:
        return "Suspicious (possible overfitting)"


# ============================================================================
# 4. POPULATION STABILITY INDEX (PSI)
# ============================================================================

def calculate_psi_component(actual_pct: float, expected_pct: float) -> float:
    """
    Calculate PSI component for a single bin.

    Formula: (Actual% - Expected%) × ln(Actual% / Expected%)

    Args:
        actual_pct: Actual percentage in bin
        expected_pct: Expected (baseline) percentage in bin

    Returns:
        PSI component value
    """
    # Avoid division by zero and log(0)
    actual_pct = max(actual_pct, 0.0001)
    expected_pct = max(expected_pct, 0.0001)

    return (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)


def calculate_psi(
    expected_dist: np.ndarray,
    actual_dist: np.ndarray
) -> float:
    """
    Calculate Population Stability Index between two distributions.

    Args:
        expected_dist: Expected (baseline) distribution percentages
        actual_dist: Actual (current) distribution percentages

    Returns:
        PSI value
    """
    psi = 0
    for exp_pct, act_pct in zip(expected_dist, actual_dist):
        psi += calculate_psi_component(act_pct, exp_pct)

    return psi


def interpret_psi(psi: float) -> str:
    """
    Interpret PSI according to standard thresholds.

    Args:
        psi: Population Stability Index

    Returns:
        Interpretation string
    """
    if psi < 0.10:
        return "No significant change"
    elif psi < 0.25:
        return "Moderate change"
    else:
        return "Significant change"


# ============================================================================
# 5. CHI-SQUARE TEST
# ============================================================================

def calculate_chi_square(
    observed: np.ndarray,
    expected: np.ndarray
) -> Tuple[float, float, int]:
    """
    Calculate Chi-Square statistic for observed vs expected frequencies.

    Formula: Σ((O - E)² / E)

    Args:
        observed: Observed frequencies
        expected: Expected frequencies

    Returns:
        Tuple of (chi_square_statistic, p_value, degrees_of_freedom)
    """
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    df = len(observed) - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    return chi2_stat, p_value, df


def chi_square_test_independence(
    contingency_table: np.ndarray
) -> Dict[str, float]:
    """
    Perform Chi-Square test of independence on contingency table.

    Args:
        contingency_table: 2D array of observed frequencies

    Returns:
        Dictionary with test results
    """
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    return {
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'is_significant': p_value < 0.05
    }


# ============================================================================
# 6. GINI COEFFICIENT AND LORENZ CURVE
# ============================================================================

def calculate_gini_from_arrays(
    actual: np.ndarray,
    predicted: np.ndarray
) -> float:
    """
    Calculate Gini coefficient from actual and predicted values.

    Formula: Gini = 2 × AUC - 1

    Args:
        actual: Actual binary outcomes (0 or 1)
        predicted: Predicted probabilities or scores

    Returns:
        Gini coefficient
    """
    from sklearn.metrics import roc_auc_score

    try:
        auc = roc_auc_score(actual, predicted)
        return 2 * auc - 1
    except ValueError:
        return 0.0


def calculate_lorenz_curve(
    actual: np.ndarray,
    predicted: np.ndarray,
    n_points: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Lorenz curve coordinates.

    Args:
        actual: Actual binary outcomes
        predicted: Predicted probabilities or scores
        n_points: Number of points to calculate (deciles)

    Returns:
        Tuple of (cumulative_population_pct, cumulative_bads_pct)
    """
    # Sort by predicted score (ascending)
    sorted_indices = np.argsort(predicted)
    sorted_actual = actual[sorted_indices]

    n = len(actual)
    points = np.linspace(0, n, n_points + 1, dtype=int)

    cum_pop_pct = []
    cum_bads_pct = []
    total_bads = np.sum(actual)

    for point in points:
        if point == 0:
            cum_pop_pct.append(0)
            cum_bads_pct.append(0)
        else:
            cum_pop_pct.append(point / n)
            cum_bads_pct.append(np.sum(sorted_actual[:point]) / total_bads)

    return np.array(cum_pop_pct), np.array(cum_bads_pct)


def interpret_gini(gini: float) -> str:
    """
    Interpret Gini coefficient according to standard thresholds.

    Args:
        gini: Gini coefficient

    Returns:
        Interpretation string
    """
    if gini < 0.20:
        return "Poor"
    elif gini < 0.40:
        return "Acceptable"
    elif gini < 0.60:
        return "Good"
    elif gini < 0.80:
        return "Very good"
    else:
        return "Excellent (verify overfitting)"


# ============================================================================
# 7. LIFT AND CAP CURVES
# ============================================================================

def calculate_lift_by_decile(
    actual: np.ndarray,
    predicted: np.ndarray,
    n_deciles: int = 10
) -> pd.DataFrame:
    """
    Calculate lift statistics by decile.

    Args:
        actual: Actual binary outcomes
        predicted: Predicted probabilities or scores
        n_deciles: Number of deciles to calculate

    Returns:
        DataFrame with lift statistics per decile
    """
    df = pd.DataFrame({
        'actual': actual,
        'predicted': predicted
    })

    # Sort by predicted score descending (best first)
    df = df.sort_values('predicted', ascending=False).reset_index(drop=True)

    # Create deciles
    df['decile'] = pd.qcut(df.index, n_deciles, labels=False, duplicates='drop') + 1

    # Calculate metrics
    overall_rate = df['actual'].mean()

    lift_stats = df.groupby('decile').agg(
        count=('actual', 'count'),
        events=('actual', 'sum'),
        event_rate=('actual', 'mean')
    ).reset_index()

    lift_stats['lift'] = lift_stats['event_rate'] / overall_rate
    lift_stats['cumulative_events'] = lift_stats['events'].cumsum()
    lift_stats['cumulative_count'] = lift_stats['count'].cumsum()
    lift_stats['cumulative_event_rate'] = (
        lift_stats['cumulative_events'] / lift_stats['cumulative_count']
    )
    lift_stats['cumulative_lift'] = (
        lift_stats['cumulative_event_rate'] / overall_rate
    )

    return lift_stats


# ============================================================================
# 8. DEVIANCE AND MODEL FIT
# ============================================================================

def calculate_deviance(
    actual: np.ndarray,
    predicted_probs: np.ndarray
) -> float:
    """
    Calculate deviance (negative log-likelihood) for logistic regression.

    Args:
        actual: Actual binary outcomes
        predicted_probs: Predicted probabilities

    Returns:
        Deviance value
    """
    # Clip probabilities to avoid log(0)
    predicted_probs = np.clip(predicted_probs, 1e-15, 1 - 1e-15)

    log_likelihood = np.sum(
        actual * np.log(predicted_probs) +
        (1 - actual) * np.log(1 - predicted_probs)
    )

    return -2 * log_likelihood


def calculate_mcfadden_r2(
    null_deviance: float,
    residual_deviance: float
) -> float:
    """
    Calculate McFadden's Pseudo R² for logistic regression.

    Formula: 1 - (Residual Deviance / Null Deviance)

    Args:
        null_deviance: Deviance of null model (intercept only)
        residual_deviance: Deviance of fitted model

    Returns:
        McFadden's R² value
    """
    return 1 - (residual_deviance / null_deviance)


# ============================================================================
# 9. CLUSTERING EVALUATION
# ============================================================================

def calculate_calinski_harabasz(
    X: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Calculate Calinski-Harabasz index for clustering quality.

    Formula: (BCSS/(k-1)) / (WCSS/(n-k))

    Args:
        X: Feature matrix
        labels: Cluster labels

    Returns:
        Calinski-Harabasz index
    """
    from sklearn.metrics import calinski_harabasz_score

    return calinski_harabasz_score(X, labels)


# ============================================================================
# 10. GINI VARIANCE AND CONFIDENCE INTERVALS
# ============================================================================

def calculate_gini_variance(
    auc: float,
    n_positives: int,
    n_negatives: int
) -> float:
    """
    Calculate variance of Gini coefficient using DeLong method.

    Args:
        auc: Area Under ROC Curve
        n_positives: Number of positive cases
        n_negatives: Number of negative cases

    Returns:
        Variance of Gini
    """
    q1 = auc / (2 - auc)
    q2 = (2 * auc ** 2) / (1 + auc)

    var_auc = (
        auc * (1 - auc) +
        (n_positives - 1) * (q1 - auc ** 2) +
        (n_negatives - 1) * (q2 - auc ** 2)
    ) / (n_positives * n_negatives)

    # Gini = 2*AUC - 1, so Var(Gini) = 4*Var(AUC)
    return 4 * var_auc


def calculate_gini_confidence_interval(
    gini: float,
    gini_variance: float,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for Gini coefficient.

    Args:
        gini: Gini coefficient
        gini_variance: Variance of Gini
        confidence_level: Confidence level (default 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    se = np.sqrt(gini_variance)

    lower = gini - z_score * se
    upper = gini + z_score * se

    return lower, upper


def test_gini_difference(
    gini1: float,
    var1: float,
    gini2: float,
    var2: float
) -> Dict[str, float]:
    """
    Test if two Gini coefficients are significantly different.

    Args:
        gini1: First Gini coefficient
        var1: Variance of first Gini
        gini2: Second Gini coefficient
        var2: Variance of second Gini

    Returns:
        Dictionary with test results
    """
    se_diff = np.sqrt(var1 + var2)
    z_stat = (gini2 - gini1) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        'difference': gini2 - gini1,
        'z_statistic': z_stat,
        'p_value': p_value,
        'is_significant': p_value < 0.05
    }
