# DecodeLabs AI Engineering Track - Project 2
## Module 02: Data Classification Using AI (Supervised Learning Phase)

### 📌 Project Overview
[cite_start]This repository contains the official milestone submission for Project 2 of the DecodeLabs Industrial Training Track[cite: 7]. [cite_start]Shifting away from hard-coded heuristic rules, this module implements a complete machine learning pipeline to execute supervised learning patterns[cite: 7, 44]. [cite_start]The engine processes historical training metrics to derive data classification boundaries autonomously[cite: 44, 74].

### 📂 File Structure & Formats
* **`classification_engine.py`**: Production-grade machine learning script executing data splitting, scaling normalization, and neighborhood modeling.
* **`README.md`**: Comprehensive pipeline documentation and evaluation metrics.

### ⚙️ Key Requirements Met
* [cite_start]**The Iris Benchmark Ingestion**: Loads, structures, and logs a balanced dataset across 4 geometric dimensions (Sepal/Petal Lengths and Widths)[cite: 80, 87, 92, 95].
* [cite_start]**Structural Train-Test Split**: Implements a randomized shuffle split isolating exactly **80% for training** pattern recognition and **20% for testing** validation[cite: 81, 138, 139, 140, 141].
* **Gatekeeper Feature Scaling**: Standardizes data arrays to a unified mean space ($Mean = 0, Variance = 1$) via `StandardScaler` to cancel out variable magnitude bias[cite: 83, 105, 134].
* [cite_start]**Supervised KNN Selection**: Leverages proximity principles using an optimized hyperparameter selection based on the elbow point ($K=5$ majority votes) to eliminate underfitting and noise overfitting[cite: 84, 143, 144, 149, 151].
* [cite_start]**Diagnostic Tool Verification**: Rejects superficial metrics by logging a multi-class **Confusion Matrix** mapping (TP, FP, FN, TN) and computing the precise macro **F1 Score harmonic mean** to balance strategic trade-offs between precision and recall[cite: 82, 85, 170, 183, 186].

### 🚀 Local Execution Setup
Activate your workspace environment, ensure packages are installed, and trigger the pipeline script:
```bash
pip install numpy scikit-learn
python classification_engine.py
