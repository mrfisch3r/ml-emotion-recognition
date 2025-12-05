"""
===============================================================================
Title       : svm_baseline.py
Project     : Speech Emotion Recognition
Authors     : Alexander Dimayuga, Harrison Jacob
Created     : November 3, 2025
Last Modified: December 4, 2025
Description : 
    SVM is one of our baseline machine learning models for the Speech
    Emotion Recognition project. It uses extracted audio features (like MFCCs) 
    to predict the emotion label (happy, angry, neutral). Currently, this 
    pipeline includes loading features from "features.csv", splitting dataset
    into train/test (80/20), standard scaling of features, training a SVM 
    classifier (linear or RBF kernel), and evaluating using metrics like accuracy.

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
    - Continue tuning
    - Include more data visualizations
    - Feature importance?
    - Find a way to automatically save best parameters, confusion matrix, model evaluation metrics, classifcation report, etc
    - Refactor code to reduce redundancy and improve modularity/efficiency
    - Standardize between SVM and KNN baseline scripts (pipeline structure, scalar, function names, etc)
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
from sklearn.model_selection import learning_curve
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
import joblib
from datetime import datetime
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
    df = pd.read_csv("CS460G-Speech-Emotion-Recognition/data/features.csv")    # Change if need be...

    X = df.drop(columns=["label"])              # Includes all features
    y = df["label"]                             # Predicting for label

    # Split into Train (80%) and Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Confirm these shapes
    print()
    print("Train | Test Shapes")
    print("================================================")
    print("Train: ", X_train.shape)
    print("Test: ", X_test.shape)
    print("================================================")
    print()

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

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
    
    return best_svm, grid.best_score_

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

def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    """
    Plots training vs validation score as dataset size grows.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=1, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring="accuracy"
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")

    # Plot the variance bands (standard deviation)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")

    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.grid()
    plt.show()
# =============================================================================

# Main Function
# =============================================================================
def main():
    printVersions()

    # 1.) Retrieve scaled training/test features and targets
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = loadAndPreprocessDataset()
    
    # 2a.) Train tuned model [NOTE: Use whichever model produces more accurate results (Linear vs RBF kernel)] and visualize
    print("\n--- Grid Search: Linear SVM ---")
    best_linear, linear_score = gridSearch(svm.SVC(kernel='linear'), param_grid_linear, X_train_scaled, y_train)
    plot_learning_curve(best_linear, X_train_scaled, y_train, title=f"Linear SVM Learning Curve (Acc: {linear_score:.3f})")
    print("\n--- Grid Search: RBF SVM ---")
    best_rbf, rbf_score = gridSearch(svm.SVC(kernel='rbf'), param_grid_RBF, X_train_scaled, y_train)
    plot_learning_curve(best_rbf, X_train_scaled, y_train, title=f"RBF SVM Learning Curve (Acc: {rbf_score:.3f})")

    # Select best model based on CV score
    if rbf_score >= linear_score:
        print(f"Selected RBF SVM with CV Score: {rbf_score}")
        svm_model = best_rbf
    else:
        print(f"Selected Linear SVM with CV Score: {linear_score}")
        svm_model = best_linear

    # 2b.) Train untuned model [NOTE: Comment out if using tuned and vice versa]
    # svm_model = svmModel(X_train_scaled, y_train)

    # 3.) Predict and Evaluate
    class_names = sorted(y_train.unique())
    predictAndEvaluate(svm_model, X_test_scaled, y_test, class_names=class_names) # NOTE: Change model name as needed

    # 4.) Save model
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"../models/svm/svm_model_{date_str}.joblib"
    joblib.dump(svm_model, filename)
    print(f"SVM model saved to: {filename}")

if __name__ == "__main__":
    main()
# =============================================================================
