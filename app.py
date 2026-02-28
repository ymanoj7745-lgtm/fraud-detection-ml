import streamlit as st
import pandas as pd
import joblib
import altair as alt
from pathlib import Path
import sys
import importlib

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
# BASE_DIR for project
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

# Ensure imports work for pickled pipeline
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    main_mod = importlib.import_module("main")
    sys.modules["main"] = main_mod
    sys.modules["__main__"] = main_mod
except ModuleNotFoundError:
    pass

# -----------------------------
# Load pipeline (FAST + RELIABLE)
# -----------------------------
MODEL_FILENAME = "fraud_detection_pipeline.pkl"

POSSIBLE_PATHS = [
    BASE_DIR / MODEL_FILENAME,
    BASE_DIR / "models" / MODEL_FILENAME,
    Path("models") / MODEL_FILENAME,
    Path(MODEL_FILENAME),
]

@st.cache_resource
def load_model():
    for path in POSSIBLE_PATHS:
        if path.exists():
            return joblib.load(path), path
    return None, None

model, MODEL_PATH = load_model()

if model:
    st.success(f"✅ Model loaded from: {MODEL_PATH}")
else:
    st.error("❌ Model file not found. Please check deployment.")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Transaction Details")

threshold = st.sidebar.slider("Fraud threshold", 0.01, 0.99, 0.90, 0.01)

step = st.sidebar.number_input("Step (time step)", min_value=0, value=0, step=1)
type_ = st.sidebar.selectbox(
    "Transaction Type",
    ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
)

amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=0.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance (Sender)", min_value=0.0, value=0.0)
newbalanceOrig = st.sidebar.number_input("New Balance (Sender)", min_value=0.0, value=0.0)
oldbalanceDest = st.sidebar.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.sidebar.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

# -----------------------------
# Predict button
# -----------------------------
if st.button("Check Fraud", disabled=model is None):

    input_df = pd.DataFrame([{
        "step": int(step),
        "type": type_,
        "amount": float(amount),
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    try:
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_df)[0][1])

        prediction = (
            int(probability >= threshold)
            if probability is not None
            else int(model.predict(input_df)[0])
        )

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # -----------------------------
    # Display cards
    # -----------------------------
    col1, col2 = st.columns(2)

    risk_text = f"{probability:.4f}" if probability is not None else "N/A"

    if prediction == 1:
        col1.metric("Status", "Fraudulent 🚨", delta=risk_text)
    else:
        col1.metric("Status", "Legitimate ✅", delta=risk_text)

    col2.metric("Transaction Amount", f"${amount:,.2f}")

    # -----------------------------
    # Visualizations
    # -----------------------------
    st.markdown("### 💹 Transaction Overview")

    input_df["Fraudulent"] = ["Yes" if prediction == 1 else "No"]

    line_chart = alt.Chart(input_df).mark_line(
        point=True,
        color="red" if prediction == 1 else "green"
    ).encode(
        x=alt.X('step', title="Step"),
        y=alt.Y('amount', title="Transaction Amount"),
        tooltip=['step', 'amount', 'type', 'Fraudulent']
    )

    balance_df = pd.DataFrame({
        "Account": ["Sender (Old)", "Sender (New)", "Receiver (Old)", "Receiver (New)"],
        "Balance": [oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]
    })

    bar_chart = alt.Chart(balance_df).mark_bar(color="blue").encode(
        x='Account',
        y='Balance',
        tooltip=['Account', 'Balance']
    )

    st.altair_chart(line_chart, use_container_width=True)
    st.altair_chart(bar_chart, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "Made with ❤️ using Streamlit & scikit-learn | Designed for Real-time Fraud Detection"
)