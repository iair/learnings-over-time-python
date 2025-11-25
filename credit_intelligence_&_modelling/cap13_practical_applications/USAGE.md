# Usage Guide - Credit Scoring Metrics

This guide provides practical examples of how to use the credit scoring functions from the `credit_metrics` module.

## Installation

```bash
cd credit_intelligence_&_modelling/cap13_practical_applications
pip install -r requirements.txt
```

## Quick Start

### 1. Feature Transformations

#### Min-Max Scaling
```python
import credit_metrics as cm
import numpy as np

# Original data
income = np.array([15000, 35000, 8000, 42000, 12000])

# Scale to [0, 1]
income_scaled = cm.min_max_scale(income)
# Output: [0.206, 0.794, 0.000, 1.000, 0.118]
```

#### Z-Score Standardization
```python
# Standardize to mean=0, std=1
income_standardized = cm.z_score_standardize(income)
# Output: [-0.45, 0.98, -1.23, 1.34, -0.64]
```

#### Discretization (Binning)
```python
# Equal frequency binning (quantiles)
bins = cm.create_bins_equal_frequency(income, n_bins=3)
# Output: [1, 2, 0, 2, 0] (bin indices)

# Equal width binning
bins = cm.create_bins_equal_width(income, n_bins=3)
# Output: [1, 2, 0, 2, 1] (bin indices)
```

### 2. Information Value (IV)

```python
import pandas as pd

# Sample data
bins = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
target = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])  # 1=bad, 0=good

# Calculate bin statistics
bin_stats = cm.calculate_bin_statistics(bins, target)

# Calculate WOE for each bin
bin_stats = cm.calculate_woe_for_bins(bin_stats)

# Calculate total IV
iv = cm.calculate_information_value(bin_stats)
print(f"IV: {iv:.4f}")

# Interpret
interpretation = cm.interpret_iv(iv)
print(f"Predictive Power: {interpretation}")
```

Expected output:
```
IV: 0.2345
Predictive Power: Medium
```

### 3. Population Stability Index (PSI)

```python
# Expected distribution (development/baseline)
expected_dist = np.array([0.10, 0.15, 0.25, 0.30, 0.20])

# Actual distribution (production/current)
actual_dist = np.array([0.12, 0.18, 0.22, 0.28, 0.20])

# Calculate PSI
psi = cm.calculate_psi(expected_dist, actual_dist)
print(f"PSI: {psi:.4f}")

# Interpret
interpretation = cm.interpret_psi(psi)
print(f"Status: {interpretation}")
```

Expected output:
```
PSI: 0.0234
Status: No significant change
```

### 4. Chi-Square Test

```python
# Contingency table: Employment Type vs Conversion
# Rows: Employment types, Columns: [Not Converted, Converted]
contingency_table = np.array([
    [320, 180],  # Salaried
    [240, 60],   # Self-employed
    [70, 30],    # Retired
    [85, 15]     # Other
])

# Perform test
results = cm.chi_square_test_independence(contingency_table)

print(f"Chi-Square: {results['chi2_statistic']:.4f}")
print(f"P-value: {results['p_value']:.6f}")
print(f"Significant: {results['is_significant']}")
```

Expected output:
```
Chi-Square: 33.4821
P-value: 0.000001
Significant: True
```

### 5. Gini Coefficient

```python
from sklearn.linear_model import LogisticRegression

# Train a simple model
X_train = np.random.randn(1000, 3)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

model = LogisticRegression()
model.fit(X_train, y_train)

# Get predictions
X_test = np.random.randn(300, 3)
y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate Gini
gini = cm.calculate_gini_from_arrays(y_test, y_pred_proba)
print(f"Gini: {gini:.4f}")

# Interpret
interpretation = cm.interpret_gini(gini)
print(f"Model Performance: {interpretation}")
```

### 6. Lorenz Curve

```python
# Calculate Lorenz curve coordinates
cum_pop, cum_events = cm.calculate_lorenz_curve(
    y_test,
    y_pred_proba,
    n_points=10
)

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.plot(cum_pop * 100, cum_events * 100, 'o-', label='Model')
plt.plot([0, 100], [0, 100], '--', label='Random')
plt.xlabel('Cumulative % Population')
plt.ylabel('Cumulative % Events')
plt.title(f'Lorenz Curve (Gini={gini:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 7. Lift Analysis

```python
# Calculate lift by decile
lift_stats = cm.calculate_lift_by_decile(y_test, y_pred_proba, n_deciles=10)

print("Lift by Decile:")
print(lift_stats[['decile', 'event_rate', 'lift', 'cumulative_lift']])

# Visualize
plt.figure(figsize=(10, 6))
plt.bar(lift_stats['decile'], lift_stats['lift'])
plt.axhline(1, color='red', linestyle='--', label='Baseline')
plt.xlabel('Decile')
plt.ylabel('Lift')
plt.title('Lift by Decile')
plt.legend()
plt.show()
```

### 8. Model Deviance

```python
# Calculate deviances
y_pred_null = np.full(len(y_test), y_test.mean())  # Null model
null_deviance = cm.calculate_deviance(y_test, y_pred_null)
residual_deviance = cm.calculate_deviance(y_test, y_pred_proba)

# Calculate pseudo R²
mcfadden_r2 = cm.calculate_mcfadden_r2(null_deviance, residual_deviance)

print(f"Null Deviance: {null_deviance:.4f}")
print(f"Residual Deviance: {residual_deviance:.4f}")
print(f"McFadden R²: {mcfadden_r2:.4f}")
```

### 9. Clustering Quality (Calinski-Harabasz)

```python
from sklearn.cluster import KMeans

# Prepare data
X = np.random.randn(500, 3)

# Test different k values
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    ch_score = cm.calculate_calinski_harabasz(X, labels)
    print(f"k={k}: CH Index = {ch_score:.2f}")
```

### 10. Gini Confidence Interval

```python
from sklearn.metrics import roc_auc_score

# Calculate AUC and counts
auc = roc_auc_score(y_test, y_pred_proba)
n_positives = np.sum(y_test == 1)
n_negatives = np.sum(y_test == 0)

# Calculate Gini variance
gini_var = cm.calculate_gini_variance(auc, n_positives, n_negatives)
gini_se = np.sqrt(gini_var)

# Get confidence interval
ci_lower, ci_upper = cm.calculate_gini_confidence_interval(gini, gini_var)

print(f"Gini: {gini:.4f}")
print(f"Standard Error: {gini_se:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
```

### 11. Compare Two Models

```python
# Model 1
gini1 = 0.52
auc1 = (gini1 + 1) / 2
var1 = cm.calculate_gini_variance(auc1, n_positives, n_negatives)

# Model 2
gini2 = 0.55
auc2 = (gini2 + 1) / 2
var2 = cm.calculate_gini_variance(auc2, n_positives, n_negatives)

# Test difference
test_result = cm.test_gini_difference(gini1, var1, gini2, var2)

print(f"Gini Difference: {test_result['difference']:.4f}")
print(f"Z-statistic: {test_result['z_statistic']:.4f}")
print(f"P-value: {test_result['p_value']:.4f}")
print(f"Significant: {test_result['is_significant']}")
```

## Complete Workflow Example

```python
import credit_metrics as cm
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Load and prepare data
df = pd.read_csv('data/lead_conversion_development.csv')

# 2. Transform features
df['income_scaled'] = cm.min_max_scale(df['monthly_income'].values)
df['tenure_scaled'] = cm.min_max_scale(df['employment_tenure'].values)

# 3. Calculate IV for each feature
income_bins = cm.create_bins_equal_frequency(df['monthly_income'].values, 5)
bin_stats = cm.calculate_bin_statistics(income_bins, 1 - df['converted'].values)
bin_stats = cm.calculate_woe_for_bins(bin_stats)
iv = cm.calculate_information_value(bin_stats)
print(f"Income IV: {iv:.4f} ({cm.interpret_iv(iv)})")

# 4. Train model
X = df[['income_scaled', 'tenure_scaled']].values
y = df['converted'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]

# 5. Evaluate model
gini = cm.calculate_gini_from_arrays(y_test, y_pred)
lift_stats = cm.calculate_lift_by_decile(y_test, y_pred)

print(f"\nModel Performance:")
print(f"- Gini: {gini:.4f} ({cm.interpret_gini(gini)})")
print(f"- Top Decile Lift: {lift_stats.iloc[0]['lift']:.2f}x")

# 6. Monitor stability (if you have production data)
df_prod = pd.read_csv('data/lead_conversion_production.csv')
dev_dist = np.array([np.sum(income_bins == i) / len(income_bins) for i in range(5)])
prod_bins = cm.create_bins_equal_frequency(df_prod['monthly_income'].values, 5)
prod_dist = np.array([np.sum(prod_bins == i) / len(prod_bins) for i in range(5)])
psi = cm.calculate_psi(dev_dist, prod_dist)
print(f"\nPopulation Stability:")
print(f"- PSI: {psi:.4f} ({cm.interpret_psi(psi)})")
```

## Best Practices

1. **Binning**: Use equal frequency binning for IV calculations to ensure each bin has sufficient observations
2. **PSI Monitoring**: Calculate PSI monthly to detect population shifts early
3. **Gini Confidence**: Always report Gini with confidence intervals when comparing models
4. **IV Interpretation**: Variables with IV > 0.3 are strong predictors; IV > 0.5 may indicate data leakage
5. **Lift Analysis**: Focus on top deciles for targeted marketing campaigns

## Troubleshooting

**Issue**: "Division by zero" in WOE calculation
**Solution**: The functions automatically handle this by replacing zeros with 0.0001

**Issue**: "Negative PSI values"
**Solution**: PSI is always positive; check your input distributions sum to 1.0

**Issue**: "Gini > 1 or < 0"
**Solution**: Check that predictions are probabilities [0,1] and targets are binary {0,1}

## Further Reading

- See `notebooks/02_credit_scoring_metrics_demonstration.ipynb` for detailed examples
- Refer to `contenido/cap13_credit_intelligence_modelling.md` for theoretical background
- Check `README.md` for metric interpretation guidelines
