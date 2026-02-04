import streamlit as st
import pandas as pd
import joblib
import altair as alt
from pathlib import Path

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="💳 Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💳 Fraud Detection Dashboard")
st.markdown("""
Predict fraudulent transactions in real-time and get visual insights.  
Enter transaction details on the sidebar and check the risk instantly.
""")

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = Path("fraud_detection_pipeline.pkl")
try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Transaction Details")
step = st.sidebar.number_input("Step (time step)", min_value=0)
type_ = st.sidebar.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance (Sender)", min_value=0.0)
newbalanceOrig = st.sidebar.number_input("New Balance (Sender)", min_value=0.0)
oldbalanceDest = st.sidebar.number_input("Old Balance (Receiver)", min_value=0.0)
newbalanceDest = st.sidebar.number_input("New Balance (Receiver)", min_value=0.0)

# -----------------------------
# Predict button
# -----------------------------
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

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

    # -----------------------------
    # Display as colored cards
    # -----------------------------
    col1, col2 = st.columns(2)
    if prediction == 1:
        col1.metric("Status", "Fraudulent 🚨", delta=f"Risk: {probability:.2f}" if probability else "")
    else:
        col1.metric("Status", "Legitimate ✅", delta=f"Risk: {probability:.2f}" if probability else "")
    
    col2.metric("Transaction Amount", f"${amount:,.2f}")

    # -----------------------------
    # Visualizations
    # -----------------------------
    st.markdown("### 💹 Transaction Overview")

    # Add Fraud column for visualization
    input_df["Fraudulent"] = ["Yes" if prediction == 1 else "No"]

    # Line chart for transaction amount over step
    line_chart = alt.Chart(input_df).mark_line(point=True, color='red' if prediction==1 else 'green').encode(
        x=alt.X('step', title="Step"),
        y=alt.Y('amount', title="Transaction Amount"),
        tooltip=['step', 'amount', 'type', 'Fraudulent']
    ).properties(
        width=700,
        height=400,
        title="Transaction Amount over Time"
    )

    # Bar chart for sender vs receiver balances
    balance_df = pd.DataFrame({
        "Account": ["Sender (Old)", "Sender (New)", "Receiver (Old)", "Receiver (New)"],
        "Balance": [oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]
    })

    bar_chart = alt.Chart(balance_df).mark_bar(color='blue').encode(
        x='Account',
        y='Balance',
        tooltip=['Account', 'Balance']
    ).properties(
        width=700,
        height=300,
        title="Sender & Receiver Balances"
    )

    st.altair_chart(line_chart, use_container_width=True)
    st.altair_chart(bar_chart, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & scikit-learn | Designed for Real-time Fraud Detection")
