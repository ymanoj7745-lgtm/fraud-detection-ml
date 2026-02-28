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
using a trained Machine Learning model with SMOTE balancing.
"""
)

# ------------------------------------------------
# Model Loading
# ------------------------------------------------
MODEL_PATHS = [
    Path("models/fraud_detection_pipeline.pkl"),
    Path("fraud_detection_pipeline.pkl")
]

@st.cache_resource
def load_model():
    for path in MODEL_PATHS:
        if path.exists():
            return joblib.load(path), path
    return None, None

model, MODEL_PATH = load_model()

if model is None:
    st.error("Model not found. Please upload the trained model.")
    st.stop()

st.success(f"Model loaded from: {MODEL_PATH}")

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
# Prediction Section
# ------------------------------------------------
if st.button("Predict Transaction"):

    try:
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

        # Gauge chart
        fig = px.pie(
            values=[probability, 1 - probability],
            names=["Fraud Probability", "Safe Probability"],
            title="Risk Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

# ------------------------------------------------
# EDA Dashboard
# ------------------------------------------------
st.markdown("---")
st.header("EDA Dashboard")

@st.cache_data
def load_sample_data():
    path = Path("Fraud.csv")
    if path.exists():
        return pd.read_csv(path, nrows=50000)
    return None

data = load_sample_data()

if data is not None:

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(data, x="amount", title="Transaction Amount Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(data["type"].value_counts(),
                     title="Transaction Type Distribution")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Dataset not available for EDA.")

# ------------------------------------------------
# Model Information
# ------------------------------------------------
st.markdown("---")
st.header("Model Information")

st.write("""
Model Type: Machine Learning Pipeline  
Tech Stack:
- Scikit-learn
- SMOTE
- Feature Engineering
- Streamlit Deployment
""")

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown("---")
st.markdown(
"""
Built by **Manoj**  
AI/ML Engineer | Data Science Project
"""
)