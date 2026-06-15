---

### 📝 2. For `DecodeLabs-Project2` Repository

```markdown
# DecodeLabs AI Engineering Track - Project 2
## Module 02: Data Classification Using AI (Supervised Learning Phase)

### 📌 Project Overview
This repository contains the official milestone submission for Project 2[cite: 7]. Shifting away from hard-coded heuristic rules, this module implements a complete machine learning pipeline to execute supervised learning patterns[cite: 7, 31]. The engine processes historical training metrics to derive data classification boundaries autonomously[cite: 7, 74].

### 📂 File Structure & Formats
* **`classification_engine.py`**: Production-grade machine learning script executing data splitting, scaling normalization, and neighborhood modeling.
* **`README.md`**: Comprehensive pipeline documentation and evaluation metrics.

### ⚙️ Key Requirements Met
* **The Iris Benchmark Ingestion**: Loads, structures, and logs balanced dataset arrays across 4 geometric dimensions[cite: 25, 80, 87, 95].
* **Structural Train-Test Split**: Implements a randomized shuffle split isolating exactly **80% for training** pattern recognition and **20% for testing** validation[cite: 25, 26, 81, 138, 139, 140, 141].
* **Gatekeeper Feature Scaling**: Standardizes data arrays to a unified mean space ($Mean = 0, Variance = 1$) via `StandardScaler` to cancel out variable magnitude bias[cite: 83, 134].
* **Supervised KNN Selection**: Leverages proximity principles using an optimized hyperparameter value of **$K=5$ majority votes** to eliminate underfitting and noise overfitting[cite: 27, 28, 84, 144, 149, 151, 152].
* **Diagnostic Tool Verification**: Rejects superficial metrics by logging a multi-class **Confusion Matrix** mapping (TP, FP, FN, TN) and computing the precise macro **F1 Score harmonic mean**[cite: 82, 85, 166, 167, 170, 186].

### 🚀 Local Execution Setup
Activate your environment, ensure packages are installed, and trigger the pipeline script:
```bash
pip install numpy scikit-learn
python classification_engine.py
