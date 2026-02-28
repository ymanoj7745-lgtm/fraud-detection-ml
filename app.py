import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection ML",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Fraud Detection System")
st.write("Machine Learning model to detect fraudulent transactions.")

# -----------------------------
# Model Paths
# -----------------------------
MODEL_PATHS = [
    Path("models/fraud_detection_pipeline.pkl"),
    Path("fraud_detection_pipeline.pkl")
]

# -----------------------------
# Load Model (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    for path in MODEL_PATHS:
        if path.exists():
            model = joblib.load(path)
            return model, path
    return None, None


model, MODEL_PATH = load_model()

# -----------------------------
# Model Status
# -----------------------------
if model is not None:
    st.success(f"✅ Model loaded successfully from: {MODEL_PATH}")
else:
    st.error("❌ Model not found. Please upload the trained model.")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Transaction Input")

step = st.sidebar.number_input("Step", min_value=0)
amount = st.sidebar.number_input("Amount", min_value=0.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance (Sender)", min_value=0.0)
newbalanceOrig = st.sidebar.number_input("New Balance (Sender)", min_value=0.0)
oldbalanceDest = st.sidebar.number_input("Old Balance (Receiver)", min_value=0.0)
newbalanceDest = st.sidebar.number_input("New Balance (Receiver)", min_value=0.0)

type_transaction = st.sidebar.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]
)

# -----------------------------
# Input Data
# -----------------------------
input_data = pd.DataFrame({
    "step": [step],
    "type": [type_transaction],
    "amount": [amount],
    "oldbalanceOrg": [oldbalanceOrg],
    "newbalanceOrig": [newbalanceOrig],
    "oldbalanceDest": [oldbalanceDest],
    "newbalanceDest": [newbalanceDest]
})

st.subheader("Input Data")
st.dataframe(input_data)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Fraud"):

    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")

        st.write(f"Fraud Probability: **{probability:.4f}**")

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Built by Manoj | Data Science Project")