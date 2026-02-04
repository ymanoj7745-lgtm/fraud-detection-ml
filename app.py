import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Detection", layout="centered")

@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("fraud_detection_pipeline.pkl")

import joblib
from pathlib import Path
import streamlit as st

MODEL_PATH = Path("fraud_detection_pipeline.pkl")

try:
    model = joblib.load("fraud_detection_pipeline.pkl")
    st.write("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

st.title("💳 Fraud Detection System")
st.write("Predict whether a financial transaction is fraudulent")

st.markdown("### Transaction Details")

step = st.number_input("Step (time step)", min_value=0)
type_ = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
amount = st.number_input("Transaction Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)

if st.button("Check Fraud"):
    input_df = pd.DataFrame([{
        "step": step,
        "type": type_,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Transaction is Legitimate")