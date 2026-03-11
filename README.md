# 🔍 Fraud Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SMOTE](https://img.shields.io/badge/SMOTE-Augmented-2ecc71?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**A machine learning system for real-time financial fraud detection — with SMOTE augmentation, ROC/PR-AUC evaluation, and a live Streamlit scoring dashboard.**

</div>

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Getting Started](#-getting-started)
- [Training the Model](#-training-the-model)
- [Running the App](#-running-the-streamlit-app)
- [Notes](#-notes)

---

## 🔍 Overview

Financial fraud detection is a classic **class imbalance** problem — fraudulent transactions are rare but costly. This project tackles it with:

- A **baseline** gradient-boosted pipeline for a fair comparison
- A **SMOTE-augmented** pipeline that oversamples the minority class
- Side-by-side **ROC-AUC and PR-AUC** comparison to quantify improvement
- A **Streamlit dashboard** for scoring individual transactions in real time

---

## 📊 Results

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Baseline (no SMOTE) | — | — |
| SMOTE Augmented | — | — |
| **Improvement** | **↑** | **↑** |

> Run `python train_model.py` to populate these metrics for your dataset.

---

## 📁 Project Structure

```
fraud-detection/
│
├── 📄 train_model.py              # Training entrypoint — baseline vs SMOTE comparison
├── 📄 app.py                      # Streamlit app — real-time transaction scoring
│
├── 📓 ml_pipeline.ipynb           # Advanced experimentation notebook
├── 📓 01_eda.ipynb                # Exploratory data analysis
├── 📓 main.ipynb                  # Experimentation notebook
│
├── src/                           # Reusable Python package
│   ├── features.py                # Feature engineering (create_features)
│   ├── modeling.py                # Pipeline building (preprocessing, SMOTE)
│   └── evaluation.py             # Evaluation utilities (ROC-AUC, PR-AUC, reports)
│
├── models/
│   └── fraud_detection_pipeline.pkl   # Saved trained model artifact
│
├── Fraud.csv                      # ⚠️ Local only — not committed (size/licensing)
├── requirements.txt               # Pinned dependencies
└── README.md
```

---

## ⚙️ How It Works

```
Fraud.csv
    │
    ▼
┌─────────────────────────────┐
│  Feature Engineering        │  create_features() from src/features.py
│  (shared across train/app)  │
└────────────┬────────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
┌─────────┐     ┌──────────────┐
│Baseline │     │ SMOTE (0.1)  │   sampling_strategy = 0.1
│Pipeline │     │ Pipeline     │
└────┬────┘     └──────┬───────┘
     │                 │
     ▼                 ▼
┌─────────────────────────────┐
│  Evaluation & Comparison    │  ROC-AUC · PR-AUC · Classification Report
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Save Best Model            │  models/fraud_detection_pipeline.pkl
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Streamlit App              │  Real-time single-transaction scoring
└─────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- `Fraud.csv` placed in the project root (Kaggle transaction dataset)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

> ⚠️ **Important:** `requirements.txt` pins `scikit-learn==1.8.0` and `joblib==1.5.3` to match the trained model artifact. Do not upgrade these without retraining.

---

## 🏋️ Training the Model

```bash
python train_model.py
```

The script will:

1. Load `Fraud.csv` — sampled to **1,000,000 rows** for speed if larger
2. Build a **Baseline** pipeline (no resampling)
3. Build a **SMOTE** pipeline (`sampling_strategy=0.1`)
4. Evaluate both on a stratified train/test split
5. Print **ROC-AUC**, **PR-AUC**, and the improvement delta
6. Save the final model to `models/fraud_detection_pipeline.pkl`

**Sample output:**
```
Baseline  →  ROC-AUC: 0.XXXX  |  PR-AUC: 0.XXXX
SMOTE     →  ROC-AUC: 0.XXXX  |  PR-AUC: 0.XXXX
Improvement: ROC +X.XX%  |  PR +X.XX%
Model saved → models/fraud_detection_pipeline.pkl ✅
```

---

## 🖥️ Running the Streamlit App

Make sure the model is trained first (`models/fraud_detection_pipeline.pkl` exists), then:

```bash
streamlit run app.py
```

Open the URL shown in your terminal. The app lets you:

- 📝 **Enter transaction details** in the sidebar
- 🎯 **Get a Fraud / Non-Fraud prediction** with a risk score
- 📊 **View visualizations** for transaction and account balance patterns

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | 1.8.0 | Model training, pipelines, evaluation |
| `imbalanced-learn` | latest | SMOTE oversampling |
| `pandas` | latest | Data manipulation |
| `numpy` | latest | Numerical computing |
| `streamlit` | latest | Web dashboard |
| `joblib` | 1.5.3 | Model serialization |
| `matplotlib` / `seaborn` | latest | Visualizations |

---

## ⚠️ Notes

**Data file:** `Fraud.csv` is intentionally excluded from version control due to size and licensing. Place your local copy in the project root before training.

**Version compatibility:** The pickled model is tied to the scikit-learn/joblib versions used during training. The `requirements.txt` pins `scikit-learn==1.8.0` and `joblib==1.5.3` to prevent the dreaded `AttributeError: Can't get attribute '_RemainderColsList'` error. If you upgrade scikit-learn, retrain the model first and update the pinned versions accordingly.

**Retraining:** After any code or data changes, re-run `python train_model.py` and restart the Streamlit app to load the updated artifact.

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
Built to catch fraud before it catches you 🛡️
</div>
