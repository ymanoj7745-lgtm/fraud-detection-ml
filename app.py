import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import altair as alt

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(
    page_title="💳 Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------
# Load Model Safely
# ----------------------
MODEL_PATH = Path("fraud_detection_pipeline.pkl")

try:
    model = joblib.load(MODEL_PATH)
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")

# ----------------------
# Custom CSS for UI
# ----------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #0099ff;
        color: white;
        font-size:16px;
        border-radius:8px;
        padding:10px 20px;
    }
    .stMetric {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True
)

# ----------------------
# Header Section
# ----------------------
st.title("💳 Fraud Detection System")
st.markdown("Predict whether a financial transaction is **fraudulent** or **legitimate**")
st.image("https://cdn-icons-png.flaticon.com/512/2910/2910762.png", width=120)

# ----------------------
# Sidebar Inputs
# ----------------------
st.sidebar.header("Transaction Details")

step = st.sidebar.number_input("Step (time step)", min_value=0)
type_ = st.sidebar.selectbox(
    "Transaction Type",
    ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
)
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance (Sender)", min_value=0.0)
newbalanceOrig = st.sidebar.number_input("New Balance (Sender)", min_value=0.0)
oldbalanceDest = st.sidebar.number_input("Old Balance (Receiver)", min_value=0.0)
newbalanceDest = st.sidebar.number_input("New Balance (Receiver)", min_value=0.0)

# ----------------------
# Predict Button
# ----------------------
if st.sidebar.button("Check Fraud"):
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

    # ----------------------
    # Transaction Breakdown Chart
    # ----------------------
    chart_data = pd.DataFrame({
        "Feature": ["Amount", "OldBalance(Sender)", "NewBalance(Sender)", "OldBalance(Receiver)", "NewBalance(Receiver)"],
        "Value": [amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]
    })

    st.markdown("### 📊 Transaction Breakdown")
    chart = alt.Chart(chart_data).mark_bar(color="#0099ff").encode(
        x='Feature',
        y='Value',
        tooltip=["Feature", "Value"]
    ).properties(width=700)
    st.altair_chart(chart, use_container_width=True)

# ----------------------
# Metrics Section
# ----------------------
st.markdown("---")
st.subheader("📈 Model Insights")
metric_col1, metric_col2, metric_col3 = st.columns(3)

# Replace the values below with your real metrics if available
metric_col1.metric("Total Transactions", "6,000,000")
metric_col2.metric("Frauds Detected", "15,234")
metric_col3.metric("Model Accuracy", "97%")

# ----------------------
# Interactive History Chart (optional)
# ----------------------
st.markdown("### 🔍 Transaction History Simulation")
history_df = pd.DataFrame({
    "Time Step": [1,2,3,4,5,6,7],
    "Transactions": [5000, 6000, 5500, 5800, 6200, 6400, 7000],
    "Frauds": [5, 6, 4, 7, 8, 10, 12]
})
line_chart = alt.Chart(history_df).transform_fold(
    ["Transactions", "Frauds"],
    as_=['Type', 'Count']
).mark_line(point=True).encode(
    x='Time Step',
    y='Count',
    color='Type',
    tooltip=["Type", "Count", "Time Step"]
).properties(width=800, height=350)
st.altair_chart(line_chart, use_container_width=True)

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown(
    "<center>Made with ❤️ by [Your Name] | Data Science Project</center>",
    unsafe_allow_html=True
)
