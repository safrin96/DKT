# scripts/run_ab_testing.py
from src.ab_testing import ab_test_performance, is_statistically_significant
import numpy as np

# Load predictions from Group A and Group B (these would come from your trained models)
y_pred_a = np.load('../KT3/y_pred_a.npy')
y_pred_b = np.load('../KT3/y_pred_b.npy')

# Perform A/B testing
t_stat, p_value = ab_test_performance(y_pred_a, y_pred_b)
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# Check if the difference is statistically significant
if is_statistically_significant(p_value):
    print("The difference between Group A and Group B is statistically significant!")
else:
    print("No significant difference between Group A and Group B.")
