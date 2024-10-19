# 🎓 Dynamic Knowledge Tracing for Personalized Learning Pathways Using Deep Learning 🚀

Hey there, education enthusiasts and data adventurers! 👋 Welcome to my project on making learning smarter, more adaptive, and personalized using **Deep Knowledge Tracing (DKT)**. In this journey, I’m using the **[EdNet KT3 (2023)](http://bit.ly/ednet-kt3)** dataset to predict a learner's mastery over time and dynamically adjust their learning pathway based on their progress. Through this approach, the model aims to provide personalized learning experiences by dynamically adjusting the difficulty of future lessons and recommending review sessions when necessary. The DKT model continuously adapts to the learner's progress, making it a valuable tool for improving learning efficiency and retention in online education platforms.

## 🧩 About the Dataset: EdNet-KT3 🔍
The **EdNet-KT3 dataset** isn’t just about solving questions; it also logs various learning activities:
- **Reading Explanations:** After answering a question, students can view detailed explanations. These are tagged based on their source, such as `after_sprint`, `after_review`, or `my_note` for re-reads.
- **Watching Lectures:** The dataset tracks when students start or stop watching lecture videos, which can come from sources like `archive`, `adaptive_offer`, or `todays_recommendation`.
- **Actions Tracked:** Includes actions like `enter`, `quit`, `respond`, and `submit` for different activities, making it possible to analyze learning behaviors in depth.

### 📊 Data Snapshot
With over **297,915 files**, this dataset provides detailed insights into students' learning activities:
- **Compressed Size:** 0.8GB (762.8MB)
- **Uncompressed Size:** 4.3GB
- **Learning Activities:** From answering questions to watching lectures, it logs everything.

Here’s a quick look at the kind of data we’re working with:

| timestamp       | action_type | item_id | source            | user_answer | platform |
|-----------------|-------------|---------|-------------------|-------------|----------|
| 1573364188664   | enter       | b790    | sprint            |             | mobile   |
| 1573364206572   | respond     | q790    | sprint            | b           | mobile   |
| 1573364209710   | submit      | b790    | sprint            |             | mobile   |
| 1573364209745   | enter       | e790    | sprint            |             | mobile   |
| 1573364218306   | quit        | e790    | sprint            |             | mobile   |
| 1573364391205   | enter       | l540    | adaptive_offer    |             | mobile   |
| 1573364686796   | quit        | l540    | adaptive_offer    |             | mobile   |

## 🎯 Project Structure
The project is organized into various folders and scripts to keep things clean and modular:

### 📁 `src/` - Source Code
- **[data_preprocessing.py](https://github.com/safrin96/DKT/blob/main/src/data_processing.py)**: Functions to clean the dataset, convert timestamps, and calculate time spent on explanations/lectures.
- **[dkt_model.py](https://github.com/safrin96/DKT/blob/main/src/dkt_model.py)**: Code to build and train the LSTM-based DKT model.
- **[ab_testing.py](https://github.com/safrin96/DKT/blob/main/src/ab_testing.py)**: Methods for conducting A/B testing to compare different learning strategies.
- **[dynamic_adjustments.py](https://github.com/safrin96/DKT/blob/main/src/dynamic_adjustments.py)**: Functions to dynamically adjust learning paths based on the model's predictions.

### 📁 `data/` - Sample Data
Contains sample data for testing and development. The full dataset is not included for privacy reasons, but you can download it [here](http://bit.ly/ednet-kt3).

### 📁 `scripts/` - Running the Project
Scripts to execute various parts of the project:
- **[run_data_preprocessing.py](https://github.com/safrin96/DKT/blob/main/scripts/run_data_preprocessing.py)**: Preprocesses the KT3 data.
- **[run_dkt_model.py](https://github.com/safrin96/DKT/blob/main/scripts/run_dkt_model.py)**: Trains the DKT model.
- **[run_ab_testing.py](https://github.com/safrin96/DKT/blob/main/scripts/run_ab_testing.py)**: Performs A/B testing.
- **[run_dynamic_adjustments.py](https://github.com/safrin96/DKT/blob/main/scripts/run_dynamic_adjustments.py)**: Applies the model's predictions to adjust learning paths dynamically.

## 🏃‍♂️ Getting Started
1. **Set Up the Environment**: Create a Conda environment using [environment.yml](https://github.com/safrin96/DKT/blob/main/environment.yml).
   ```bash
       conda env create -f environment.yml
       conda activate kt3-dkt

2. Download the Data: Get the EdNet KT3 dataset from here and place the unzipped files in the data/ folder.
3. Run Data Preprocessing: Clean and prepare the data.
   
    ```bash
    python scripts/run_data_preprocessing.py

5. Train the Model: Run the script to train the DKT model.
   
    ```bash
    python scripts/run_dkt_model.py
7. Evaluate with A/B Testing: Compare different recommendation strategies.
   
    ```bash
    python scripts/run_ab_testing.py

8. Apply Dynamic Adjustments: Use predictions to adjust the learning paths.
   
    ```bash
    python scripts/run_dynamic_adjustments.py
    
## 🔬 A/B Testing Approach

I compared two groups:

- Group A: Static learning recommendations.
- Group B: Dynamic recommendations based on DKT predictions.
<!--Results are logged in ab_test_results.csv and visualized in ab_testing_plot.png.-->
## 📊 Visualizing Progress

Tracking students' mastery over time helps understand their learning journey better:

 - Accuracy Plot: Tracks model accuracy during training.
 - Mastery Prediction Plot: Shows how predicted mastery changes based on interactions.
    
    ```python
    
    import matplotlib.pyplot as plt
    
    # Plot mastery progress for a student
    plt.plot(predicted_mastery)
    plt.title('Student Mastery Progress Over Time 🎢')
    plt.xlabel('Interaction Number')
    plt.ylabel('P(Mastery)')
    plt.show()

## 📦 Deliverable:

A working knowledge tracing system that dynamically suggests review materials or new challenges for students based on their predicted mastery state. Include a report showing how this system adapts based on user interactions and performance.

## 🚀 Future Work

There’s still plenty of room to grow:

- Tuning Hyperparameters: Experiment with different model configurations.
- Feature Engineering: Incorporate more behavioral data, like frequency of explanation views.
- Real-time Adaptations: Integrate the model into an actual platform to provide real-time feedback.

## 🔍 Conclusion:

Using the EdNet KT3 Dataset (2023) provides a more recent and rich data source for experimenting with Deep Knowledge Tracing and adaptive learning techniques. This project gives a hands-on experience in implementing DKT models and building a dynamic, personalized learning system that can suggest reviews or new challenges based on user readiness.
