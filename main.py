import pandas as pd
import numpy as np


def create_features(X: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering used in the fraud detection pipeline.

    This must stay in this module (`main.create_features`) so that
    pickled pipelines depending on it can be loaded by other scripts
    (e.g. the Streamlit app).
    """
    X = X.copy()
    X["orig_balance_diff"] = X["oldbalanceOrg"] - X["newbalanceOrig"]
    X["dest_balance_diff"] = X["newbalanceDest"] - X["oldbalanceDest"]
    X["amount_log"] = np.log1p(X["amount"])
    X["hour"] = X["step"] % 24
    X["is_night"] = (X["hour"] < 6).astype(int)
    X["orig_empty"] = (X["oldbalanceOrg"] == 0).astype(int)
    X["dest_empty"] = (X["newbalanceDest"] == 0).astype(int)
    return X


__all__ = ["create_features"]

