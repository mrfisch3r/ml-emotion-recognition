"""
===============================================================================
Title       : svm_baseline.py
Project     : Speech Emotion Recognition
Authors     : Alexander Dimayuga
Created     : November 3, 2025
Description : 
    SVM is one of our baseline machine learning models for the Speech
    Emotion Recognition project. It uses extracted audio features (like MFCCs) 
    to predict the emotion label (happy, angry, neutral). Currently, this 
    pipeline includes loading features from "features.csv", splitting dataset
    into train/validation/test (80/10/10), standard scaling of features, 
    training a SVM classifier (linear or RBF kernel), and evaluating using
    metrics like accuracy.

Dependencies:
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - sklearn
    - os

Usage:
    python .\svm_baseline.py

Notes:
    TODO: MORE IMPROVEMENTS
    - Needs hyperparameter tuning
    - Include more data visualizations
    - Feature importance?
    - Save model 
    - Make CLI friendly
    - Validation set not used
    - And more...
===============================================================================
"""
# Imports
# =============================================================================
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import os
# =============================================================================

# Global Variables
# =============================================================================
param_grid_linear = {
    'C': [0.1, 1, 10],                          # OG: 0.001, 0.01, 0.1, 1, 10, 100
    'class_weight': [None, 'balanced'],
}

param_grid_RBF = {
    'C': [1, 10],                               # OG: 0.1, 1, 10, 100
    'gamma': ['scale', 0.01],                   # OG: 'scale', 1e-3, 1e-2, 1e-1, 1
    'class_weight': ['balanced', None]
}
# =============================================================================

# Helper Functions
# =============================================================================
def printVersions():
    print()
    print("Python environment versions:")
    print("================================================")
    print("Pandas version:       ", pd.__version__)
    print("NumPy version:        ", np.__version__)
    print("scikit-learn version: ", sklearn.__version__)
    print("Seaborn version:      ", sns.__version__)
    print("Matplotlib version:   ", matplotlib.__version__)
    print("================================================")
    print()

def loadAndPreprocessDataset():
    df = pd.read_csv("../data/features.csv")    # Change if need be...

    X = df.drop(columns=["label"])              # Includes all features
    y = df["label"]                             # Predicting for label

    # Split into Train (80%) and Temp (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size = 0.2, random_state=42, stratify=y
    )

    # Split Temp (20%) into Validation (10%) and Test (10%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Confirm these shapes
    print()
    print("Train | Validation | Test Shapes")
    print("================================================")
    print("Train: ", X_train.shape)
    print("Validation: ", X_val.shape)
    print("Test: ", X_test.shape)
    print("================================================")
    print()

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

def gridSearch(model, param_grid, X_train, y_train):
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,               # 5-fold cross-validation
        scoring='accuracy', # or F1, etc...
        n_jobs=-1,
        verbose=3
    )

    grid.fit(X_train, y_train)

    print("================================================")
    print("Best parameters: ", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    print("================================================")
    print()

    best_svm = grid.best_estimator_
    
    return best_svm

def svmModel(X_train, y_train):
    # Create a SVM Classifier [NOTE: Comment one or the other out]
    clf = svm.SVC(kernel='linear', verbose=True) # Linear Kernel
    # clf = svm.SVC(kernel='rbf', verbose=True)  # RBF kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    return clf

def predictAndEvaluate(clf, X_test, y_test, class_names=None):
    # Make predictions using our SVM Model
    y_pred = clf.predict(X_test)

    # Compute our metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Display our metrics
    print()
    print("Model Evaluation Metrics")
    print("================================================")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-Score : {f1:.3f}")
    print("================================================")
    print()

    # Classification Report
    print("Classification Report:")
    print("================================================")
    print(classification_report(y_test, y_pred, digits=3))
    print("================================================")
    print()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Display CM 
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()
# =============================================================================

# Main Function
# =============================================================================
def main():
    printVersions()

    # 1.) Retrieve scaled training/validation/test features and targets
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler = loadAndPreprocessDataset()
    
    # 2a.) Train tuned model [NOTE: Use whichever model produces more accurate results (Linear vs RBF kernel)]
    print("\n--- Grid Search: Linear SVM ---")
    best_linear = gridSearch(svm.SVC(kernel='linear'), param_grid_linear, X_train_scaled, y_train)
    print("\n--- Grid Search: RBF SVM ---")
    best_rbf = gridSearch(svm.SVC(kernel='rbf'), param_grid_RBF, X_train_scaled, y_train)

    # 2b.) Train untuned model [NOTE: Comment out if using tuned and vice versa]
    # svm_model = svmModel(X_train_scaled, y_train)

    # 3.) Predict and Evaluate
    # class_names = sorted(y_train.unique())
    # predictAndEvaluate(svm_model, X_test_scaled, y_test, class_names=class_names) # NOTE: Change model name as needed

if __name__ == "__main__":
    main()
# =============================================================================
