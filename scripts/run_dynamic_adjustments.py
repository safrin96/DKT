# scripts/run_dynamic_adjustments.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Users/sumaiyashrabony/Desktop/DataScience/src')))

from src.dynamic_adjustments import dynamic_adjustments # type: ignore
import numpy as np

# Load the test data (X_test)
X_test = np.load('/Users/sumaiyashrabony/Desktop/DataScience/KT3/X_test.npy')

# Run dynamic adjustments based on learning activity
explanation_times, lecture_times = dynamic_adjustments(X_test)

# Analyze the results (e.g., suggest review or advanced lessons based on time spent)
for i in range(len(explanation_times)):
    if explanation_times[i] < 30 and lecture_times[i] < 45:  # Example thresholds
        print(f"User {i}: Needs more time on explanations and lectures. Suggest review.")
    elif explanation_times[i] > 30 and lecture_times[i] > 45:
        print(f"User {i}: Ready for more advanced content.")
    else:
        print(f"User {i}: Continue current learning path.")