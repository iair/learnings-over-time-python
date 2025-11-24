# Chapter 13: Practical Applications - Credit Intelligence and Modelling

This project implements all the credit scoring metrics and transformations described in Chapter 13 of "Credit Intelligence and Modelling" by Raymond Anderson.

## Project Structure

```
cap13_practical_applications/
├── src/
│   ├── __init__.py
│   └── credit_metrics.py       # Generalized functions for all metrics
├── notebooks/
│   ├── 01_generate_synthetic_dataset.ipynb
│   └── 02_credit_scoring_metrics_demonstration.ipynb
├── data/
│   ├── lead_conversion_development.csv
│   └── lead_conversion_production.csv
└── README.md
```

## Features

### 1. Feature Transformations
- **Min-Max Scaling**: Normalize values to [0,1] range
- **Z-Score Standardization**: Transform to mean=0, std=1
- **Discretization**: Create bins using equal width or equal frequency methods

### 2. Feature Evaluation Metrics
- **Information Value (IV)**: Measure predictive power of variables
- **Weight of Evidence (WOE)**: Analyze relationship between bins and target
- **Population Stability Index (PSI)**: Monitor distribution shifts over time
- **Chi-Square Test**: Test statistical significance of categorical associations

### 3. Model Evaluation Metrics
- **Gini Coefficient**: Measure model discrimination power
- **Lorenz Curve**: Visualize cumulative performance
- **CAP Curve**: Cumulative Accuracy Profile analysis
- **Lift Analysis**: Quantify improvement over random selection

### 4. Advanced Metrics
- **Deviance**: Measure model fit quality
- **McFadden R²**: Pseudo R² for logistic regression
- **Calinski-Harabasz Index**: Optimize customer segmentation
- **Gini Variance**: Statistical confidence in model performance

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter
```

### Usage

1. **Generate Synthetic Dataset**
   ```bash
   jupyter notebook notebooks/01_generate_synthetic_dataset.ipynb
   ```
   This creates two datasets:
   - `lead_conversion_development.csv` (5,000 records)
   - `lead_conversion_production.csv` (3,000 records)

2. **Explore Metrics and Transformations**
   ```bash
   jupyter notebook notebooks/02_credit_scoring_metrics_demonstration.ipynb
   ```
   This demonstrates all metrics on the synthetic data.

### Using the Credit Metrics Module

```python
import sys
sys.path.append('src')
import credit_metrics as cm
import numpy as np

# Example: Calculate Information Value
bins = cm.create_bins_equal_frequency(data, n_bins=5)
bin_stats = cm.calculate_bin_statistics(bins, target)
bin_stats = cm.calculate_woe_for_bins(bin_stats)
iv = cm.calculate_information_value(bin_stats)
print(f"IV: {iv:.4f} - {cm.interpret_iv(iv)}")

# Example: Calculate Gini Coefficient
gini = cm.calculate_gini_from_arrays(y_actual, y_predicted)
print(f"Gini: {gini:.4f} - {cm.interpret_gini(gini)}")

# Example: Calculate PSI
psi = cm.calculate_psi(expected_distribution, actual_distribution)
print(f"PSI: {psi:.4f} - {cm.interpret_psi(psi)}")
```

## Design Principles

This project follows functional programming principles as outlined in CLAUDE.md:

- **Pure Functions**: All functions are side-effect free
- **Single Responsibility**: Each function does one thing
- **Immutability**: No mutation of external variables
- **Composability**: Functions can be easily combined
- **Minimal Parameters**: Functions take only what they need

## Metrics Reference

### Information Value (IV) Interpretation
| IV Range | Predictive Power |
|----------|------------------|
| < 0.02 | Not predictive |
| 0.02 - 0.1 | Weak |
| 0.1 - 0.3 | Medium |
| 0.3 - 0.5 | Strong |
| > 0.5 | Suspicious |

### PSI Interpretation
| PSI Range | Interpretation | Action |
|-----------|----------------|--------|
| < 0.10 | No significant change | Continue monitoring |
| 0.10 - 0.25 | Moderate change | Investigate causes |
| > 0.25 | Significant change | Recalibrate model |

### Gini Interpretation
| Gini Range | Interpretation |
|------------|----------------|
| < 0.20 | Poor |
| 0.20 - 0.40 | Acceptable |
| 0.40 - 0.60 | Good |
| 0.60 - 0.80 | Very good |
| > 0.80 | Excellent |

## Synthetic Dataset Details

The synthetic dataset simulates lead conversion data with realistic distributions:

- **Sample Size**: 5,000 development, 3,000 production
- **Conversion Rate**: ~25-30%
- **Features**:
  - `monthly_income`: Log-normal distribution (3,000 - 50,000 MXN)
  - `employment_tenure`: Exponential distribution (0 - 360 months)
  - `age`: Normal distribution (18 - 70 years)
  - `employment_type`: Categorical (Salaried, Self-employed, Retired, Other)
  - `acquisition_channel`: Categorical (Digital, Branch, Referral, Phone)
  - `marital_status`: Categorical (Single, Married, Divorced, Widowed)
  - `gender`: Binary (M, F)
  - `converted`: Binary target (0, 1)

## References

- Anderson, R. (2019). *Credit Intelligence and Modelling: Many Paths Through the Forest*. Oxford University Press.
- Siddiqi, N. (2017). *Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards*. Wiley.

## License

This project is for educational purposes as part of the learnings-over-time repository.
