import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Users/sumaiyashrabony/Desktop/DataScience/src')))

# scripts/run_data_processing.py
from src.data_processing import load_data, preprocess_data, calculate_time_spent # type: ignore

# Path to your dataset
data_dir = '/Users/sumaiyashrabony/Desktop/DataScience/KT3'

# Load and preprocess the data
df = load_data(data_dir)
df = preprocess_data(df)
df_explanations, df_lectures = calculate_time_spent(df)

# Save the preprocessed data if needed
df.to_csv("/Users/sumaiyashrabony/Desktop/DataScience/KT3/preprocessed_u1.csv", index=False)