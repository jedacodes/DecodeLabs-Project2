# DecodeLabs AI Engineering Track - Project 2
## Module 02: Data Classification Using AI (Supervised Learning Phase)

### 📌 Project Overview
This repository contains the official milestone submission for Project 2 of the DecodeLabs Industrial Training Track. Shifting away from hard-coded heuristic rules, this module implements a complete machine learning pipeline to execute supervised learning patterns. The engine processes historical training metrics to derive data classification boundaries autonomously.

### 📂 File Structure & Formats
* **`classification_engine.py`**: Production-grade machine learning script executing data splitting, scaling normalization, and neighborhood modeling.
* **`README.md`**: Comprehensive pipeline documentation and evaluation metrics.

### ⚙️ Key Requirements Met
* **The Iris Benchmark Ingestion**: Loads, structures, and logs a balanced dataset across 4 geometric dimensions (Sepal/Petal Lengths and Widths).
* **Structural Train-Test Split**: Implements a randomized shuffle split isolating exactly **80% for training** pattern recognition and **20% for testing** validation.
* **Gatekeeper Feature Scaling**: Standardizes data arrays to a unified mean space (Mean = 0, Variance = 1) via `StandardScaler` to cancel out variable magnitude bias.
* **Supervised KNN Selection**: Leverages proximity principles using an optimized hyperparameter selection based on the elbow point (K=5 majority votes) to eliminate underfitting and noise overfitting.
* **Diagnostic Tool Verification**: Rejects superficial metrics by logging a multi-class **Confusion Matrix** mapping (TP, FP, FN, TN) and computing the precise macro **F1 Score harmonic mean** to balance strategic trade-offs between precision and recall.

### 🚀 Local Execution Setup
Activate your workspace environment, ensure packages are installed, and trigger the pipeline script:
```bash
pip install numpy scikit-learn
python classification_engine.py
