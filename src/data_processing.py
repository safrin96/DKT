# src/data_preprocessing.py
import os
import pandas as pd

def load_data(data_dir):
    """
    Load the KT3 dataset from a CSV file.
    """
    # Step 1: Data Preparation - Upload your KT3 CSV files into Databricks FileStore
#data_dir = '/Users/sumaiyashrabony/Desktop/DataScience/KT3'  # Update this path to where your KT3 dataset files are located

    # Read all CSV files from the directory and concatenate them into one DataFrame
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]  # List all CSV files in the directory
    df_list = []

    # Loop over all files, read them, and append to the list
    for file in all_files:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        df_list.append(df)
        #df = pd.read_csv(filepath)
        df = pd.concat(df_list, ignore_index=True)
    return df

def preprocess_data(df):
    """
    Preprocess the KT3 dataset, converting timestamp, user actions, and creating time spent features.
    """
    # Convert timestamp to datetime format
    df['timestamp'] = pd.to_datetime(
    df['timestamp'], 
    errors='coerce', 
    unit='ms').fillna(pd.to_datetime(df['timestamp'], errors='coerce'))

    # Generate unique User_ID based on 'platform' column
    df['User_ID'] = df.groupby('platform').ngroup()

    # Treat 'item_id' as 'Skill_ID'
    df['Skill_ID'] = df['item_id'].astype('category').cat.codes

    # Create Correctness feature from 'user_answer'
    df['Correct'] = df['user_answer'].apply(lambda x: 1 if x in ['a', 'b', 'c', 'd'] else 0)

    return df

def calculate_time_spent(df):
    """
    Calculate the time spent on explanations and lectures by using 'enter' and 'quit' actions.
    """
    # Calculate time spent by subtracting 'enter' from 'quit' actions for explanations and lectures
    df['Time_Spent'] = df.groupby(['User_ID', 'item_id'])['timestamp'].diff().fillna(pd.Timedelta(seconds=0))

    # Filter for explanation and lecture activities
    df_explanations = df[(df['action_type'] == 'quit') & (df['source'].isin(['sprint', 'my_note']))].copy()
    df_lectures = df[(df['action_type'] == 'quit') & (df['source'].isin(['archive', 'adaptive_offer']))].copy()

    # Use .loc[] to assign values to slices of DataFrames to avoid the SettingWithCopyWarning
    df_explanations.loc[:, 'Explanation_Time_Spent'] = df_explanations['Time_Spent'].dt.total_seconds()
    df_lectures.loc[:, 'Lecture_Time_Spent'] = df_lectures['Time_Spent'].dt.total_seconds()

    return df_explanations, df_lectures