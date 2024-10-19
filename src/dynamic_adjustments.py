# src/dynamic_adjustments.py
import numpy as np

def dynamic_adjustments(X_test):
    """
    Analyze average time spent on explanations and lectures, and provide recommendations for dynamic adjustments.
    """
    explanation_times = np.mean(X_test[:, :, -2], axis=1)  # Average time spent on explanations per user
    lecture_times = np.mean(X_test[:, :, -1], axis=1)  # Average time spent on lectures per user
    return explanation_times, lecture_times