import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import pickle
from pathlib import Path
from typing import Optional, Tuple, Any
import os
import sys
import importlib

# Page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💳 Fraud Detection Dashboard")
st.markdown("""
Predict fraudulent transactions in real-time and get visual insights.  
Enter transaction details on the sidebar and check the risk instantly.
""")

# BASE_DIR for project
BASE_DIR = Path(__file__).resolve().parent

# Ensure main.py is importable for pickled pipeline
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    main_mod = importlib.import_module("main")
    sys.modules["main"] = main_mod
    sys.modules["__main__"] = main_mod
except ModuleNotFoundError:
    pass

# Load pipeline with error handling
MODEL_FILENAME = "fraud_detection_pipeline.pkl"

def find_pipeline_file(filename: str, start_dir: Path) -> Optional[Path]:
    """Recursively search for the pipeline file starting from start_dir."""
    for root, dirs, files in os.walk(start_dir):
        if filename in files:
            return Path(root) / filename
    return None

@st.cache_resource
def load_model() -> Tuple[Optional[Any], Optional[Path]]:
    """Load the fraud detection model with proper error handling."""
    model_path = find_pipeline_file(MODEL_FILENAME, BASE_DIR)
    
    if not model_path or not model_path.exists():
        return None, None
    
    # ==================================================================
    # CRITICAL: Compatibility shim for scikit-learn version mismatches
    # Models pickled with older sklearn (1.6.x) break with newer sklearn
    # (1.8.x) due to removed private classes. This shim injects stubs.
    # ==================================================================
    _inject_sklearn_compatibility_stubs()
    
    try:
        model = joblib.load(model_path)
        return model, model_path
    except (EOFError, pickle.UnpicklingError):
        st.error("❌ Model file is corrupted or incompatible with current scikit-learn version.")
        return None, None
    except ValueError as ve:
        # often triggered by sklearn Trees when there is a version mismatch
        st.error(
            "❌ Model loading error: incompatible scipy/sklearn versions or corrupted file. "
            "Try retraining the model or regenerating with: `python create_dummy_model.py`"
        )
        return None, None
    except AttributeError as ae:
        # missing classes are usually due to version drift
        st.error(
            "❌ Model loading failed: missing sklearn class during unpickle. "
            "This usually means a scikit-learn version mismatch. "
            "Regenerate the model with: `python create_dummy_model.py`"
        )
        return None, None
    except Exception as e:
        st.error(f"❌ Unexpected error loading model: {type(e).__name__}: {str(e)[:100]}")
        return None, None


def _inject_sklearn_compatibility_stubs():
    """
    Inject missing sklearn private classes for backward compatibility.
    Handles models pickled with sklearn 1.6.x loaded with sklearn 1.8.x+
    """
    try:
        from sklearn.compose import _column_transformer as _ct
        from sklearn import tree
        
        # Handle _RemainderColsList (removed in sklearn 1.8.x)
        if not hasattr(_ct, "_RemainderColsList"):
            class _RemainderColsList:  # type: ignore
                """Stub for backward compat with pkl from sklearn <1.8"""
                def __init__(self, *args, **kwargs):
                    pass
                def __reduce__(self):
                    return (_RemainderColsList, ())
            _ct._RemainderColsList = _RemainderColsList
            
        # Handle potential tree node class changes
        if not hasattr(tree, "TREE_UNDEFINED"):
            tree.TREE_UNDEFINED = -2
            
    except Exception as e:
        # Silently fail - shim is best-effort
        pass

model, MODEL_PATH = load_model()

if MODEL_PATH:
    st.success(f"✓ Model loaded: {MODEL_PATH.name}")
else:
    st.warning(f"Model file `{MODEL_FILENAME}` not found. Run create_dummy_model.py or train_model.py to create one.")

# Sidebar inputs
st.sidebar.header("Transaction Details")
threshold = st.sidebar.slider("Fraud threshold", 0.01, 0.99, 0.90, 0.01)
step = st.sidebar.number_input("Step (time step)", min_value=0, value=0, step=1)
type_ = st.sidebar.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=0.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance (Sender)", min_value=0.0, value=0.0)
newbalanceOrig = st.sidebar.number_input("New Balance (Sender)", min_value=0.0, value=0.0)
oldbalanceDest = st.sidebar.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.sidebar.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

# Predict button
def _st_button(label: str, *, disabled: bool) -> bool:
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

        prediction = int(probability >= threshold) if probability is not None else int(model.predict(input_df)[0])
    except Exception as e:
        st.error(f"Prediction failed: {type(e).__name__}")
        st.stop()

    # Display cards
    col1, col2 = st.columns(2)
    risk_text = f"Risk: {probability:.4f}" if probability is not None else "Risk: N/A"
    if prediction == 1:
        col1.metric("Status", "Fraudulent [ALERT]", delta=risk_text)
    else:
        col1.metric("Status", "Legitimate [SAFE]", delta=risk_text)
    col2.metric("Transaction Amount", f"${amount:,.2f}")

    # Visualizations
    st.markdown("### Transaction Overview")
    input_df["Fraudulent"] = ["Yes" if prediction == 1 else "No"]

    line_chart = alt.Chart(input_df).mark_line(point=True, color='red' if prediction==1 else 'green').encode(
        x=alt.X('step', title="Step"),
        y=alt.Y('amount', title="Transaction Amount"),
        tooltip=['step', 'amount', 'type', 'Fraudulent']
    ).properties(width=700, height=400, title="Transaction Amount over Time")

    balance_df = pd.DataFrame({
        "Account": ["Sender (Old)", "Sender (New)", "Receiver (Old)", "Receiver (New)"],
        "Balance": [oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]
    })

    bar_chart = alt.Chart(balance_df).mark_bar(color='blue').encode(
        x='Account',
        y='Balance',
        tooltip=['Account', 'Balance']
    ).properties(width=700, height=300, title="Sender & Receiver Balances")

    st.altair_chart(line_chart, use_container_width=True)
    st.altair_chart(bar_chart, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Made with care using Streamlit & scikit-learn | Designed for Real-time Fraud Detection")
