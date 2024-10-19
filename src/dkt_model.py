# src/dkt_model.py
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

def build_dkt_model(input_shape):
    """
    Build the LSTM-based Deep Knowledge Tracing (DKT) model.
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_dkt_model(model, X_train, y_train, X_test, y_test, epochs=10):
    """
    Train the DKT model using the training data.
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))
    return history

def generate_sequences(df):
    """
    Create sequences of student interactions to be used as input for the DKT model.
    """
    sequences = []
    labels = []

    for user_id in df['User_ID'].unique():
        user_data = df[df['User_ID'] == user_id]

        skill_sequence = user_data['Skill_ID'].values
        correct_sequence = user_data['Correct'].values

        # Create sequences and labels for training
        for i in range(1, len(skill_sequence)):
            sequences.append((skill_sequence[:i], correct_sequence[:i]))
            labels.append(correct_sequence[i])

    return sequences, labels

def prepare_input_data(sequences, num_skills):
    """
    Convert the generated sequences into a format suitable for LSTM input.
    """
    max_seq_length = max([len(seq[0]) for seq in sequences])
    num_features = num_skills  # Number of unique skills

    # Initialize input arrays with zeros for padding
    X = np.zeros((len(sequences), max_seq_length, num_features))
    y = np.array([label for _, label in sequences], dtype=object)

    # Fill the input array with sequences
    for i, (skills, correctness) in enumerate(sequences):
        for j, skill in enumerate(skills):
            X[i, j, skill] = 1  # One-hot encode the skill index

    return X, y
