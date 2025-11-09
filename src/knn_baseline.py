"""
===============================================================================
Title       : knn_baseline.py
Project     : Speech Emotion Recognition
Authors     : Alexander Dimayuga
Created     : November 3, 2025
Description : 
    KNN is one of our baseline baseline machine learning models for the Speech
    Emotion Recognition project. It uses extracted audio features (like MFCCs) 
    to predict the emotion label (happy, angry, neutral). Currently, this 
    pipeline includes loading features from "features.csv", splitting dataset
    into train/validation/test (80/10/10), standard scaling of features, 
    training a KNeighborsClassifier (with default k=5), and evaluating using
    metricsl ike accuracy.

Dependencies:
    pandas
    numpy
    seaborn
    matplotlib
    sklearn
    os

Usage:
    python .\knn_baseline.py

Notes:
    TODO: MORE IMPROVEMENTS
    - Could do more hyperparameter tuning (find best k)
    - Try different distance metrics
    - Include more data visualizations
    - Cross-validation?
    - Feature importance?
    - Save model 
    - Make CLI friendly
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    print("================================================")
    print("Train | Validation | Test Shapes")
    print("Train: ", X_train.shape)
    print("Validation: ", X_val.shape)
    print("Test: ", X_test.shape)
    print("================================================")
    print()

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

def knnModel(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train, y_train)

    return knn

def predictAndEvaluate(knn, X_test, y_test, class_names=None):
    # Make predictions using our KNN Model
    y_pred = knn.predict(X_test)

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

    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler = loadAndPreprocessDataset()

    knn = knnModel(X_train_scaled, y_train)

    class_names = sorted(y_train.unique())
    predictAndEvaluate(knn, X_test_scaled, y_test, class_names=class_names)


if __name__ == "__main__":
    main()
# =============================================================================
