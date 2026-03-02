import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection ML",
    page_icon="🛡️",
    layout="wide"
)

# ------------------------------------------------
# Title
# ------------------------------------------------
st.title("🛡️ Fraud Detection System")
st.markdown("### Real-time Machine Learning Fraud Detection")

st.markdown(
"""
This system predicts whether a transaction is **fraudulent or legitimate**
using a trained Machine Learning model.
"""
)

# ------------------------------------------------
# Model Loading
# ------------------------------------------------
MODEL_PATH = Path("models/fraud_detection_pipeline.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Model not found. Please upload the trained model.")
    st.stop()

model = load_model()
st.success("Model loaded successfully")

# ------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------
st.sidebar.header("Transaction Details")

step = st.sidebar.number_input("Step", min_value=0)
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0)

oldbalanceOrg = st.sidebar.number_input("Sender Old Balance", min_value=0.0)
newbalanceOrig = st.sidebar.number_input("Sender New Balance", min_value=0.0)

oldbalanceDest = st.sidebar.number_input("Receiver Old Balance", min_value=0.0)
newbalanceDest = st.sidebar.number_input("Receiver New Balance", min_value=0.0)

type_transaction = st.sidebar.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]
)

# ------------------------------------------------
# Input DataFrame
# ------------------------------------------------
input_df = pd.DataFrame({
    "step": [step],
    "type": [type_transaction],
    "amount": [amount],
    "oldbalanceOrg": [oldbalanceOrg],
    "newbalanceOrig": [newbalanceOrig],
    "oldbalanceDest": [oldbalanceDest],
    "newbalanceDest": [newbalanceDest]
})

st.subheader("Transaction Data")
st.dataframe(input_df)

# ------------------------------------------------
# Prediction
# ------------------------------------------------
if st.button("Predict Transaction"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("🚨 Fraud Detected")
        else:
            st.success("✅ Legitimate Transaction")

    with col2:
        st.metric("Fraud Probability", f"{probability:.2%}")

    fig = px.pie(
        values=[probability, 1 - probability],
        names=["Fraud", "Safe"],
        title="Risk Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# Model Info
# ------------------------------------------------
st.markdown("---")
st.header("Model Information")

st.write("""
Model: Random Forest + SMOTE  
Feature Engineering + ML Pipeline  
Built by **Manoj**
""")