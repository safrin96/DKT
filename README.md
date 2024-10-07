# Dynamic Knowledge Tracing for Personalized Learning Pathways Using Deep Learning

In the era of digital education, adaptive learning has become a powerful tool to improve student engagement and outcomes. This project explores the use of Deep Knowledge Tracing (DKT), a machine learning technique that leverages sequential data modeling, to predict a learner's mastery of various skills over time. By analyzing sequences of student interactions with educational content, the DKT model estimates the probability that a learner has mastered specific topics and dynamically adjusts the learning path based on real-time performance.

The project utilizes a recent dataset of student interactions, such as the [EdNet KT3 (2023)](https://www.kaggle.com/datasets/anhtu96/ednet-kt34) dataset, which contains detailed records of learners' responses to educational exercises. By training a recurrent neural network (RNN) model on this dataset, the system predicts whether a learner will answer future questions correctly, identifies when a learner needs to review past material, and suggests new challenges when they are ready for more difficult content.

Through this approach, the project aims to provide personalized learning experiences by dynamically adjusting the difficulty of future lessons and recommending review sessions when necessary. The DKT model continuously adapts to the learner's progress, making it a valuable tool for improving learning efficiency and retention in online education platforms.

## Step-by-Step Guide:

1. Download and Explore the EdNet KT3 Dataset:
    - Download the dataset from the link above. It includes student interaction data over time, including the questions they answered, whether they got the answer right, and the knowledge component (skill) involved.
    
    - Data Overview:
      - User_ID: A unique identifier for each student.
      - Question_ID: The specific question the student answered.
      - Skill_ID: The skill or knowledge component being assessed.
      - Correct: Whether the student answered correctly (1 for correct, 0 for incorrect).
      - Timestamp: When the interaction occurred.

    Data Example:
  
    ```csv
    User_ID, Question_ID, Skill_ID, Correct, Timestamp
    1001, Q1001, S1, 1, 2023-01-01 12:00:00
    1001, Q1002, S2, 0, 2023-01-01 12:05:00
    1002, Q1003, S1, 1, 2023-01-01 12:10:00
    1003, Q1004, S3, 0, 2023-01-01 12:15:00
    ```

2. Preprocessing the Data:

    - Load the dataset into Python using pandas and preprocess it for analysis. Clean any missing values and ensure the data is correctly formatted for knowledge tracing.
    - Code to load and inspect the dataset:

      ```python
      import pandas as pd

      # Load the dataset
      df = pd.read_csv('ednet_kt3.csv')
  
      # Check the first few rows
      print(df.head())```

    - Create features such as the number of attempts for each skill and the time between attempts for a more in-depth understanding of user behavior.

      - Assumptions:
      
        The dataset contains columns for User_ID, Skill_ID (or Component_ID), Correct (whether the user answered correctly), and Timestamp (when the user interacted with the system).
      
        We will:

        - Count the number of attempts for each skill.
        - Calculate the time between attempts for each skill to understand how frequently users are interacting with the content.
      
        #### Step 1: Count the Number of Attempts for Each Skill
        
          First, we’ll calculate how many times each user has attempted questions related to each skill (including both correct and incorrect attempts).
        
          ```python
          
          import pandas as pd
          
          # Load dataset
          df = pd.read_csv('ednet_kt3.csv')
          
          # Ensure the Timestamp column is in datetime format
          df['Timestamp'] = pd.to_datetime(df['Timestamp'])
          
          # Sort the data by User_ID and Timestamp for easier time-based calculations
          df = df.sort_values(by=['User_ID', 'Timestamp'])
          
          # Create a column to count the number of attempts per skill for each user
          df['Attempt_Number'] = df.groupby(['User_ID', 'Skill_ID']).cumcount() + 1
          
          # Check the result
          print(df[['User_ID', 'Skill_ID', 'Attempt_Number']].head())
          ```
        This code creates an Attempt_Number column that shows how many times each user has attempted questions for each skill.
      
        #### Step 2: Calculate Time Between Attempts
          Next, we will calculate the time between attempts for each user on the same skill.
      
          ```python
          
          # Calculate the time difference between consecutive attempts for each skill by user
          df['Time_Diff'] = df.groupby(['User_ID', 'Skill_ID'])['Timestamp'].diff()
          
          # Check the result
          print(df[['User_ID', 'Skill_ID', 'Timestamp', 'Time_Diff']].head())
          ```
      
        This code calculates the time difference (in hours, minutes, or days) between consecutive attempts for the same skill by each user. The first attempt for each user and skill will have a NaT (Not a Time) value for Time_Diff because there's no previous attempt to compare it with.

        #### Step 3: Handling Missing Values (First Attempt)

          For the first attempt at each skill by each user, there won’t be any previous attempt to calculate the time difference. We can fill these missing values with a default time or leave them as missing for now.
      
          ```python
          
          # Fill the NaT values with 0 (or another value if needed)
          df['Time_Diff'] = df['Time_Diff'].fillna(pd.Timedelta(seconds=0))
          
          # Convert time differences to minutes (or leave them as timedeltas if preferred)
          df['Time_Diff_Minutes'] = df['Time_Diff'].dt.total_seconds() / 60
          
          # Check the result
          print(df[['User_ID', 'Skill_ID', 'Time_Diff_Minutes']].head())
          ```
      
        Here, we’ve filled the missing values in Time_Diff with 0 and converted the time differences into minutes. You can choose to leave the time differences in their original form as timedelta objects if you want more flexibility.
      
        #### Step 4: Aggregating Data for Analysis
        
          If you want to aggregate the data to analyze the average time between attempts or the total number of attempts per user and skill, you can use groupby:
        
          ```python
          
          # Calculate the total number of attempts and average time between attempts per user and skill
          user_skill_stats = df.groupby(['User_ID', 'Skill_ID']).agg(
              total_attempts=('Attempt_Number', 'max'),
              avg_time_between_attempts=('Time_Diff_Minutes', 'mean')
          ).reset_index()
          
          # Check the result
          print(user_skill_stats.head())
          ```
        
          This code aggregates the data to give:
        
          - Total number of attempts for each skill by each user.
          - Average time between attempts for each skill by each user.
        
          #### Summary of Features Created:
  
          - Attempt_Number: The number of times a user has attempted questions for each skill.
          - Time_Diff: The time difference between consecutive attempts for each skill by the same user.
          - Time_Diff_Minutes: The time difference in minutes.
          - Aggregated Features:
  
            - Total Attempts: The total number of attempts for each skill by each user.
            - Average Time Between Attempts: The average time between attempts for each skill by each user.

4. Implement Knowledge Tracing Using Deep Knowledge Tracing (DKT):

    This process includes preparing the data, splitting it into training and testing sets, and training the model using an LSTM-based architecture.
    
    1. Step-by-Step Guide
       - Step 1: Preprocess the data to create input sequences for each student.
       - Step 2: Split the data into training and testing sets.
       - Step 3: Train the DKT model using an LSTM to predict the probability of mastery for each skill.
       - Step 4: Evaluate the model on the test set.
  
    2. Code for the Full Process

        ```python
    
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from keras.models import Sequential
        from keras.layers import LSTM, Dense
        from keras.utils import to_categorical
        from sklearn.metrics import accuracy_score
        
        # Load dataset (replace 'your_dataset.csv' with the actual dataset path)
        df = pd.read_csv('ednet_kt3.csv')
        
        # Ensure Timestamp column is in datetime format
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Sort by User_ID and Timestamp to create interaction sequences
        df = df.sort_values(by=['User_ID', 'Timestamp'])
        
        # Step 1: Prepare Input Data for DKT Model
        # Create the necessary features: User_ID, Skill_ID, Correct (1 or 0 for correct/incorrect)
        df = df[['User_ID', 'Skill_ID', 'Correct']]
        
        # One-hot encode Skill_ID to prepare for LSTM input
        num_skills = df['Skill_ID'].nunique()
        df['Skill_ID'] = df['Skill_ID'].astype('category').cat.codes
        
        # Function to generate sequences for each user
        def generate_sequences(df):
            sequences = []
            labels = []
            users = df['User_ID'].unique()
        
            for user in users:
                user_data = df[df['User_ID'] == user]
                skill_sequence = user_data['Skill_ID'].values
                correct_sequence = user_data['Correct'].values
        
                # Create input-output pairs for the model
                for i in range(1, len(skill_sequence)):
                    # Input: skill and response history up to current question
                    sequences.append((skill_sequence[:i], correct_sequence[:i]))
                    # Output: correct/incorrect on the next question
                    labels.append(correct_sequence[i])
        
            return sequences, labels
    
          # Generate sequences and labels from the dataset
          sequences, labels = generate_sequences(df)
          
          # Pad the sequences to ensure equal length and convert to numpy arrays
          max_seq_length = max([len(seq[0]) for seq in sequences])
          
          X = np.zeros((len(sequences), max_seq_length, num_skills))
          y = np.array(labels)
          
          for i, (skills, responses) in enumerate(sequences):
              for j, skill in enumerate(skills):
                  X[i, j, skill] = 1  # One-hot encoding for skill
                  X[i, j, -1] = responses[j]  # Response (correct/incorrect)
          
          # Step 2: Split into training and testing sets
          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
          
          # Step 3: Build and Train the DKT Model using LSTM
          model = Sequential()
          model.add(LSTM(128, input_shape=(max_seq_length, num_skills), return_sequences=False))
          model.add(Dense(1, activation='sigmoid'))
          
          model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
          
          # Train the model
          history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
          
          # Step 4: Evaluate the model
          y_pred = model.predict(X_test)
          y_pred_binary = (y_pred > 0.5).astype(int)
          
          # Calculate accuracy
          accuracy = accuracy_score(y_test, y_pred_binary)
          print(f'Accuracy on the test set: {accuracy:.4f}')
        ```

    3. Explanation of the Code:
   
        Step 1: Preprocessing Data
       
        - We sort the data by User_ID and Timestamp to ensure the sequences are in the correct order.
        - We only use the necessary columns (User_ID, Skill_ID, and Correct), where Correct is 1 if the student answered correctly, and 0 if they didn’t.
        - The Skill_ID column is one-hot encoded for use in the LSTM input.
        - We generate sequences of skills and responses for each user, which are needed for training the model.
      
        Step 2: Splitting the Data
  
        - The input data (X) is padded to the length of the longest sequence and split into training (80%) and testing (20%) sets using train_test_split.
        
        Step 3: Building the DKT Model
        
        - We use an LSTM (Long Short-Term Memory) network to process sequences of skills and responses. The LSTM layer has 128 units and takes as input the skill-response sequences for each student.
        - The output is a single value (a probability between 0 and 1) indicating whether the student is likely to answer the next question correctly.
        
        Step 4: Model Evaluation
        
        - We evaluate the model on the test set by predicting the likelihood of correctness for each question. We then threshold the predictions at 0.5 to create binary predictions (1 for correct, 0 for incorrect).
        - The final accuracy score is calculated based on the test set predictions.
      
    4. Potential Improvements:
    
    - Hyperparameter tuning: You can experiment with the number of LSTM units, learning rate, batch size, and other parameters.
    - Data augmentation: Add more features such as the time between attempts or attempts per skill to enrich the model.
    - Early stopping: You can implement early stopping to prevent overfitting based on validation loss.

6. Training the Knowledge Tracing Model:

    - Train the model using student interaction sequences. The input to the model is the sequence of questions and responses, and the output is the probability of mastery for each skill.
    - Split the data into training and testing sets to evaluate model performance.

    Training example:

    ```python

    # Assuming X_train and y_train are the training sequences for users
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    ```

7. Predict Mastery and Suggest Reviews or Challenges:

    - Once the model is trained, predict each student's knowledge state for each skill. If the predicted probability of mastery P(Mastery) for a skill is low (below 0.5), suggest review material. If the probability is high (above 0.8), introduce new challenges for the student.

    Example code to suggest reviews or challenges:

    ```python

    P_mastery = model.predict(X_test)  # Predict mastery for each user in the test set
    
    # Suggest review or new challenges based on P_mastery threshold
    for skill, mastery in enumerate(P_mastery):
        if mastery < 0.5:
            print(f"Suggest review material for Skill {skill}")
        elif mastery > 0.8:
            print(f"Introduce new challenges for Skill {skill}")
    ```

8. Visualize Mastery Progress:

    - Create visualizations to track the knowledge state of each student over time. Plot P(Mastery) for different skills and how mastery changes as students complete more questions.

    Example visualization using matplotlib:

    ```python

    import matplotlib.pyplot as plt
    
    # Plot mastery progress for a single user
    plt.plot(P_mastery)
    plt.title('Student Mastery Progress Over Time')
    plt.xlabel('Question Number')
    plt.ylabel('P(Mastery)')
    plt.show()
    ```

9. Deliverable:

    - A working knowledge tracing system that dynamically suggests review materials or new challenges for students based on their predicted mastery state. Include a report showing how this system adapts based on user interactions and performance.

## Conclusion:

Using the EdNet KT3 Dataset (2023) provides a more recent and rich data source for experimenting with Deep Knowledge Tracing and adaptive learning techniques. This project will give you hands-on experience in implementing DKT models and building a dynamic, personalized learning system that can suggest reviews or new challenges based on user readiness.
