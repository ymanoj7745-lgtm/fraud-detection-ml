import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🕵️",
    layout="wide"
)

st.title("💳 Fraud Detection ML System")

MODEL_NAME = "fraud_detection_pipeline.pkl"

POSSIBLE_PATHS = [
    Path("models") / MODEL_NAME,
    Path(MODEL_NAME),
]

@st.cache_resource
def load_model():
    for path in POSSIBLE_PATHS:
        if path.exists():
            model = joblib.load(path)
            return model, path
    return None, None


model, MODEL_PATH = load_model()

if model:
    st.success(f"✅ Model loaded successfully from: {MODEL_PATH}")
else:
    st.error("❌ Model file not found.")
    st.stop()


st.header("Transaction Input")

amount = st.number_input("Transaction Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0)
newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0)
oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0)
newbalanceDest = st.number_input("New Balance Destination", min_value=0.0)

type_txn = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]
)

if st.button("Predict Fraud"):
    input_data = pd.DataFrame({
        "type": [type_txn],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest],
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraud Detected! Probability: {probability:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction. Fraud Probability: {probability:.2f}")