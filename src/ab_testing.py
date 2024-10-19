# src/ab_testing.py
import numpy as np
from scipy import stats

def ab_test_performance(y_pred_a, y_pred_b):
    """
    Perform a t-test to compare the performance of two groups (A/B test).
    """
    t_stat, p_value = stats.ttest_ind(y_pred_a.flatten(), y_pred_b.flatten())
    return t_stat, p_value

def is_statistically_significant(p_value, threshold=0.05):
    """
    Check if the p-value is below the threshold for statistical significance.
    """
    return p_value < threshold
