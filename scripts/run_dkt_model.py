import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Users/sumaiyashrabony/Desktop/DataScience/src')))

# scripts/run_dkt_model.py
from src.data_processing import load_data, preprocess_data # type: ignore
from src.dkt_model import build_dkt_model, train_dkt_model, generate_sequences, prepare_input_data # type: ignore
from sklearn.model_selection import train_test_split
import numpy as np

# Load and preprocess the data
data_path = '/Users/sumaiyashrabony/Desktop/DataScience/KT3/preprocessed_u1.csv'  # Update this to the correct path where the preprocessed data is saved
df = pd.read_csv(data_path)
df = preprocess_data(df)

# Generate sequences for the DKT model
sequences, labels = generate_sequences(df)

# Get the number of unique skills
num_skills = df['Skill_ID'].nunique()

# Prepare the input data
X, y = prepare_input_data(sequences, num_skills)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM-based DKT model
input_shape = (X_train.shape[1], X_train.shape[2])  # Shape: (max_seq_length, num_features)
model = build_dkt_model(input_shape)

# Train the DKT model
history = train_dkt_model(model, X_train, y_train, X_test, y_test, epochs=10)

# Save the model and history if needed
model.save('/Users/sumaiyashrabony/Desktop/DataScience/KT3/saved_model.h5')