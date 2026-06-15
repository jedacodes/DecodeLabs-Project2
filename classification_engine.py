"""
DecodeLabs AI Engineering Track
Project 2: Data Classification Using AI (Supervised Learning Phase)
Author: Paul Femi-Adejobi
Batch: 2026
Description: A complete production pipeline implementing the Iris benchmark 
             using the K-Nearest Neighbors (KNN) algorithm.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def run_classification_pipeline():
    print("============================================================")
    # Target Blueprint: Data Classification Track [cite: 256, 257]
    print("       DECODELABS DATA CLASSIFICATION ENGINE v2.0.0         ")
    print("      [Pipeline Framework | Module 02: KNN Classifier]      ")
    print("============================================================\n")

    # ----------------------------------------------------------------
    # PHASE 1: INPUT LAYER (Load & Understand Dataset) [cite: 313, 314]
    # ----------------------------------------------------------------
    print("[PHASE 1] Loading the raw materials: The Iris Benchmark...")
    iris = load_iris()
    X = iris.data  # Dimensions: Sepal/Petal Length and Width [cite: 332, 333, 336]
    y = iris.target  # Classes: Setosa, Versicolor, Virginica [cite: 325, 326, 327]
    
    print(f" -> Samples Loaded: {len(X)} (Balanced dataset) [cite: 329]")
    print(f" -> Target Classes identified: {iris.target_names} ({len(iris.target_names)} classes) [cite: 331]")
    print(f" -> Dimensions tracking: {iris.feature_names}\n")

    # ----------------------------------------------------------------
    # PHASE 2: PROCESS LAYER (Data split & Normalization) [cite: 313, 315]
    # ----------------------------------------------------------------
    # Requirement: Shuffle and Split into 80% Training and 20% Testing [cite: 377, 434, 435]
    print("[PHASE 2] Executing Train-Test split (80% Train / 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f" -> Training Partition size: {X_train.shape[0]} samples")
    print(f" -> Testing Partition size: {X_test.shape[0]} samples\n")

    # Gatekeeper Rule: Feature Scaling via StandardScaler [cite: 342, 344]
    # Balances variance and brings dimensions into scale (-2 to +2 space) [cite: 351, 371]
    print("[PHASE 2.1] Applying Gatekeeper Scaling (StandardScaler: Mean=0, Var=1)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(" -> Scaling normalization optimized successfully.\n")

    # ----------------------------------------------------------------
    # PHASE 3: TRAINING LAYER (KNN Implementation) [cite: 321]
    # ----------------------------------------------------------------
    # Optimal Hyperparameter Tuning selection based on the Elbow point (K=5) [cite: 381, 388]
    print("[PHASE 3] Instantiating and fitting the logic model (K=5 Neighbors)...")
    model = KNeighborsClassifier(n_neighbors=5)  # Instantiate (Build the frame) [cite: 396, 399]
    model.fit(X_train_scaled, y_train)           # Fit (Memorize the map) [cite: 397, 400]
    print(" -> Model mapping completed successfully.\n")

    # ----------------------------------------------------------------
    # PHASE 4: OUTPUT VALIDATION LAYER (Predictions & Diagnostics) [cite: 313, 316, 403]
    # ----------------------------------------------------------------
    print("[PHASE 4] Running output diagnostic tests...")
    predictions = model.predict(X_test_scaled)  # Predict (Apply logic) [cite: 398, 401]

    # Diagnostic Tool 1: Confusion Matrix layout tracking [cite: 319, 407]
    print("\n--- Diagnostic Tool: Confusion Matrix ---")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    print(" -> Tracks True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN) [cite: 409, 411, 414, 416]")

    # Diagnostic Tool 2: Classification Report checking strategic trade-offs [cite: 420]
    print("\n--- Strategy Diagnostics: Classification Metrics ---")
    print(classification_report(y_test, predictions, target_names=iris.target_names))
    
    # Extract total Harmonic Mean for verification [cite: 322, 423]
    macro_f1 = f1_score(y_test, predictions, average='macro')
    print(f"============================================================")
    print(f" PIPELINE SUCCESS | FINAL MACRO F1 SCORE: {macro_f1:.4f}")
    print(f"============================================================")

if __name__ == "__main__":
    run_classification_pipeline()