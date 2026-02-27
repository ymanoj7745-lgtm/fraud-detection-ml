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
# Load model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CANDIDATES = [
    BASE_DIR / "models" / "fraud_detection_pipeline.pkl",
    BASE_DIR / "fraud_detection_pipeline.pkl",
]


def _install_pickle_compat_shims() -> None:
    # Ensure the repo folder is on sys.path even if Streamlit
    # is launched from a different working directory.
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    # Compatibility shim: older pickles may reference `main.create_features`
    # (or sometimes `__main__.create_features`). We map those module names
    # to our local `main.py`, which provides `create_features`.
    main_mod = importlib.import_module("main")
    sys.modules["main"] = main_mod
    sys.modules["__main__"] = main_mod


def _cache_singleton(func):
    """
    Streamlit caching API changed across versions.
    Prefer cache_resource; fall back to experimental_singleton; then cache.
    """
    cache_resource = getattr(st, "cache_resource", None)
    if callable(cache_resource):
        return cache_resource(show_spinner=False)(func)

    experimental_singleton = getattr(st, "experimental_singleton", None)
    if callable(experimental_singleton):
        try:
            return experimental_singleton(show_spinner=False)(func)
        except TypeError:
            return experimental_singleton()(func)

    cache = getattr(st, "cache", None)
    if callable(cache):
        try:
            return cache(show_spinner=False, allow_output_mutation=True)(func)
        except TypeError:
            return cache(allow_output_mutation=True)(func)

    return func


@_cache_singleton
def load_model(model_path: str):
    _install_pickle_compat_shims()
    return joblib.load(model_path)


def resolve_model_path() -> Path:
    for p in DEFAULT_MODEL_CANDIDATES:
        if p.exists():
            return p
    return DEFAULT_MODEL_CANDIDATES[-1]


MODEL_PATH = resolve_model_path()

try:
    model = load_model(str(MODEL_PATH))
    st.success(f"✅ Model loaded: `{MODEL_PATH.name}`")
except Exception as e:
    model = None
    st.error(f"❌ Error loading model: {e}")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Transaction Details")
threshold = st.sidebar.slider("Fraud threshold", min_value=0.01, max_value=0.99, value=0.90, step=0.01)
step = st.sidebar.number_input("Step (time step)", min_value=0, value=0, step=1)
type_ = st.sidebar.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=0.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance (Sender)", min_value=0.0, value=0.0)
newbalanceOrig = st.sidebar.number_input("New Balance (Sender)", min_value=0.0, value=0.0)
oldbalanceDest = st.sidebar.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.sidebar.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

# -----------------------------
# Predict button
# -----------------------------
def _st_button(label: str, *, disabled: bool) -> bool:
    """
    Streamlit button() signature varies across versions.
    - `type=` is not available in older versions
    - `disabled=` may not be available in very old versions
    """
    try:
        return bool(st.button(label, disabled=disabled))
    except TypeError:
        return bool(st.button(label))


if _st_button("Check Fraud", disabled=model is None):
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

        if probability is not None:
            prediction = int(probability >= threshold)
        else:
            prediction = int(model.predict(input_df)[0])
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # -----------------------------
    # Display as colored cards
    # -----------------------------
    col1, col2 = st.columns(2)
    risk_text = f"Risk: {probability:.4f}" if probability is not None else "Risk: N/A"
    if prediction == 1:
        col1.metric("Status", "Fraudulent 🚨", delta=risk_text)
    else:
        col1.metric("Status", "Legitimate ✅", delta=risk_text)
    
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
