"""
Credit Scoring Metrics and Transformations Module

This module provides functional implementations of credit scoring metrics
as described in Chapter 13 of "Credit Intelligence and Modelling" by Raymond Anderson.
"""

from .credit_metrics import (
    # Feature Transformations
    min_max_scale,
    z_score_standardize,
    create_bins_equal_width,
    create_bins_equal_frequency,

    # WOE Calculations
    calculate_bin_statistics,
    calculate_woe,
    calculate_woe_for_bins,

    # Information Value
    calculate_iv_component,
    calculate_information_value,
    interpret_iv,

    # Population Stability Index
    calculate_psi_component,
    calculate_psi,
    interpret_psi,

    # Chi-Square
    calculate_chi_square,
    chi_square_test_independence,

    # Gini and Lorenz
    calculate_gini_from_arrays,
    calculate_lorenz_curve,
    interpret_gini,

    # Lift and CAP
    calculate_lift_by_decile,

    # Deviance
    calculate_deviance,
    calculate_mcfadden_r2,

    # Clustering
    calculate_calinski_harabasz,

    # Gini Variance
    calculate_gini_variance,
    calculate_gini_confidence_interval,
    test_gini_difference
)

__version__ = '1.0.0'
__author__ = 'Credit Analytics Team'
__all__ = [
    'min_max_scale',
    'z_score_standardize',
    'create_bins_equal_width',
    'create_bins_equal_frequency',
    'calculate_bin_statistics',
    'calculate_woe',
    'calculate_woe_for_bins',
    'calculate_iv_component',
    'calculate_information_value',
    'interpret_iv',
    'calculate_psi_component',
    'calculate_psi',
    'interpret_psi',
    'calculate_chi_square',
    'chi_square_test_independence',
    'calculate_gini_from_arrays',
    'calculate_lorenz_curve',
    'interpret_gini',
    'calculate_lift_by_decile',
    'calculate_deviance',
    'calculate_mcfadden_r2',
    'calculate_calinski_harabasz',
    'calculate_gini_variance',
    'calculate_gini_confidence_interval',
    'test_gini_difference'
]
