# Fraud Detection System

This repository contains a fraud detection model and a Streamlit dashboard for real-time transaction scoring.

The core components are:

- `train_model.py` – training script that builds a baseline model and a SMOTE-augmented model, compares their performance, and saves the final model artifact.
- `app.py` – Streamlit app that loads the trained model and exposes a simple UI for scoring individual transactions.
- `ml_pipeline.ipynb` – advanced notebook for experimentation with feature engineering, SMOTE, and model tuning.
- `01_eda.ipynb` / `main.ipynb` – exploratory data analysis and experimentation notebooks.

## Project structure

- `Fraud.csv` – **local data file (not committed due to size)**. The training scripts expect this file at the project root by default.
- `src/` – reusable Python package:
  - `src/features.py` – shared feature engineering (`create_features`) used by training and compatible with saved models.
  - `src/modeling.py` – pipeline building (preprocessing, baseline, SMOTE pipeline)
  - `src/evaluation.py` – evaluation utilities (ROC-AUC, PR-AUC, reports)
- `train_model.py` – training entrypoint that uses SMOTE, reports ROC-AUC / PR-AUC improvements versus a baseline, and saves the model.
- `app.py` – Streamlit app for single-transaction fraud prediction (loads the latest trained model).
- `notebooks/` – experimentation and analysis notebooks.
- `README.md`, `requirements.txt` – documentation and dependencies for GitHub.

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place `Fraud.csv` (Kaggle / transaction dataset) in the project root alongside `train_model.py`.

## Training with SMOTE and evaluating improvements

Run the training script:

```bash
python train_model.py
```

The script will:

- Load `Fraud.csv` (sampling to 1,000,000 rows for speed if necessary).
- Build a **baseline** pipeline (no SMOTE) and a **SMOTE** pipeline (with `SMOTE(sampling_strategy=0.1)`).
- Evaluate both using a stratified train/test split (ROC-AUC and PR-AUC).
- Print the metrics and the **improvement of SMOTE vs baseline**.
- Train the final SMOTE pipeline on the full sampled dataset and save it as `models/fraud_detection_pipeline.pkl` (fallback: project root).

The saved model is fully compatible with the Streamlit app.

## Running the Streamlit app

After training (and ensuring `models/fraud_detection_pipeline.pkl` exists), run:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal. You can:

- Enter transaction details in the sidebar.
- Get a fraud / non-fraud prediction and risk score.
- View simple visualizations for the transaction and account balances.

## Notes

- `Fraud.csv` is intentionally kept out of version control (due to size and licensing); make sure you have a local copy.
- If you retrain the model, re-run `python train_model.py` and restart the Streamlit app to use the updated artifact.

