"""
===============================================================================
Title       : knn_baseline.py
Project     : Speech Emotion Recognition
Authors     : Alexander Dimayuga, Harrison Jacob
Created     : November 3, 2025
Last Modified: December 4, 2025
Description : 
    KNN is one of our baseline machine learning models for the Speech
    Emotion Recognition project. It uses extracted audio features (like MFCCs) 
    to predict the emotion label (happy, angry, neutral). Currently, this 
    pipeline includes loading features from "features.csv", splitting dataset
    into train/test (80/20), standard scaling of features, training a KNeighborsClassifier 
    (with default k=5), and evaluating using metrics like accuracy.

Dependencies:
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - sklearn
    - os

Usage:
    python .\knn_baseline.py

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
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder # <--- Add LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA
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
param_grid = [
    # Case 1: Metrics that do NOT use the 'p' parameter
    {
            'pca__n_components': [0.95, 0.99, None], # None means "keep all features"
            'knn__n_neighbors': [3, 5, 7],
            'knn__weights': ['distance'],
            'knn__metric': ['manhattan'] 
    },
    # Case 2: euclidian & cosine
    {
        'knn__n_neighbors': [1, 3, 5, 7, 9, 15, 25],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'cosine'] 
    },
    # Case 3: minkowski
    {
        'knn__n_neighbors': [1, 3, 5, 7, 9, 15, 25],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['minkowski'],
        'knn__p': [1, 2]
    }
]
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

    # --- NEW: Encode Labels (String -> Integer) ---
    le = LabelEncoder()
    y = le.fit_transform(y)  # Converts 'anger','happy' -> 0, 1
    
    # Save the original class names for plotting later
    class_names = le.classes_ 
    # ----------------------------------------------

    # Split into Train (80%) and Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state=42, stratify=y
    )

    # Scale features --> Alternatively could do Min-Max, Robust, etc...
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

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test, class_names

def gridSearch(X_train, y_train): 
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('knn', KNeighborsClassifier())
    ])

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=5,               # 5-fold cross-validation
        scoring='accuracy', # or F1, roc_auc, etc...
        n_jobs=1,   
        verbose=3
    )

    grid.fit(X_train, y_train)

    print("================================================")
    print("Best parameters: ", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    print("================================================")
    print()

    best_knn = grid.best_estimator_
    
    return best_knn

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

def plot_decision_boundary(X, y, class_names, title="KNN Decision Boundary (2D PCA Projection)"):
    from sklearn.decomposition import PCA
    from matplotlib.colors import ListedColormap

    # Squash the 76 features down to 2 just for visualization
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # We use the same k=5 and 'manhattan' metric as your main model
    clf = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')
    clf.fit(X_reduced, y)

    # Create a meshgrid (a background of empty points)
    h = .02  # step size in the mesh
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict the emotion for every pixel in the background
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 5. Plot
    plt.figure(figsize=(10, 8))
    
    # Create color maps (Red=Anger, Blue=Happy, Grey/Green=Neutral usually)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3) # Background colors

    # Plot the actual training points
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=cmap_bold,
                          edgecolor='k', s=20)
    
    plt.legend(handles=scatter.legend_elements()[0], labels=list(class_names))
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

# =============================================================================

# Main Function
# =============================================================================
def main():
    printVersions()

    # 1.) Retrieve scaled + unscaled training/validation/test features and targets
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test, class_names = loadAndPreprocessDataset()

    # 2a.) Train tuned model 
    knn_model = gridSearch(X_train, y_train)

    # 2b.) Train untuned model [NOTE: Comment out if using tuned and vice versa]
    # knn = knnModel(X_train_scaled, y_train)

    # 3.) Predict and Evaluate
    predictAndEvaluate(knn_model, X_test, y_test, class_names=class_names) # NOTE: X_test or X_test_scaled based on if model is tuned or not
    
    # 4.) Visualizations
    print("Calculating decision boundaries")
    plot_decision_boundary(X_train_scaled, y_train, class_names)

    print("Calculating learning curve")
    plot_learning_curve(knn_model, X_train, y_train, title="KNN Learning Curve")

    # 5.) Save model
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"CS460G-Speech-Emotion-Recognition/models/knn/knn_model_{date_str}.joblib"
    joblib.dump(knn_model, filename)
    print(f"KNN model saved to: {filename}")


if __name__ == "__main__":
    main()
# =============================================================================
