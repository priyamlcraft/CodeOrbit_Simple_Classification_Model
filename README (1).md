# CodeOrbit Task 2: Simple Classification Model

**Organization:** CodeOrbit Tech  
**Internship Domain:** Machine Learning  
**Task:** Task 2 — Simple Classification Model

## Project Overview
This project builds a machine learning classification model using Python and the Iris dataset. It uses Logistic Regression to predict an Iris flower species from its measurements. The workflow includes splitting the data, scaling features, training the model, and evaluating its performance.

## Objectives
- Load and explore the Iris dataset.
- Split data into training and testing sets.
- Scale features using `StandardScaler`.
- Train a Logistic Regression classifier.
- Evaluate performance using accuracy, precision, recall, and a classification report.
- Visualize results with a confusion matrix.

## Dataset
The Iris dataset is included in `scikit-learn`. It contains four numerical measurements for three flower species: Setosa, Versicolor, and Virginica.

## Tools and Libraries
- Python
- Pandas
- Matplotlib
- Scikit-learn

## Project Workflow
1. Load the Iris dataset.
2. Separate input features and target labels.
3. Split the data using a stratified train-test split.
4. Scale features and train a Logistic Regression model in a pipeline.
5. Predict labels for the test set.
6. Calculate accuracy, weighted precision, and weighted recall.
7. Print a classification report and sample predictions.
8. Create and save a confusion matrix plot.

## Installation
Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

## How to Run
Keep the Python script and `requirements.txt` in the same folder, then run:

```bash
python CodeOrbit_Task_2_Simple_Classification_Model.py
```

The script prints evaluation results in the terminal, displays a confusion matrix, and saves it as `confusion_matrix.png` in the current folder.

## Evaluation Metrics
- **Accuracy:** Proportion of correct predictions.
- **Precision:** Correctness of predicted classes, reported as a weighted average.
- **Recall:** Identification of true classes, reported as a weighted average.
- **Confusion Matrix:** Compares actual labels with predicted labels.

Exact metric values are generated when the script runs.

## Learning Outcomes
- Practiced building a classification model with Scikit-learn.
- Learned train-test splitting and feature scaling.
- Evaluated a model using standard classification metrics.
- Visualized model predictions using a confusion matrix.

## Conclusion
This task provides practical experience with a supervised machine learning workflow, from loading data to training and evaluating a classification model.

---
*Completed as part of the CodeOrbit Tech Machine Learning Internship.*
