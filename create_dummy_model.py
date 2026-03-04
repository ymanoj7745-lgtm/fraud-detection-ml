#!/usr/bin/env python3
"""
Create a dummy fraud detection model for testing.
This ensures the Streamlit app can load without a full dataset.

This script generates a pickled model compatible with the current
scikit-learn version in the environment.

Usage:
    python create_dummy_model.py

Output:
    models/fraud_detection_pipeline.pkl
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

print(f"📦 Creating dummy model with scikit-learn {sklearn.__version__}...")

# Create a minimal dataset
X_dummy = pd.DataFrame([
    {"step": 1, "type": "CASH_IN", "amount": 100, "oldbalanceOrg": 1000, 
     "newbalanceOrig": 900, "oldbalanceDest": 2000, "newbalanceDest": 2100},
    {"step": 2, "type": "CASH_OUT", "amount": 500, "oldbalanceOrg": 2000, 
     "newbalanceOrig": 1500, "oldbalanceDest": 3000, "newbalanceDest": 3500},
    {"step": 3, "type": "DEBIT", "amount": 50, "oldbalanceOrg": 500, 
     "newbalanceOrig": 450, "oldbalanceDest": 1000, "newbalanceDest": 1050},
    {"step": 4, "type": "PAYMENT", "amount": 1000, "oldbalanceOrg": 5000, 
     "newbalanceOrig": 4000, "oldbalanceDest": 10000, "newbalanceDest": 11000},
    {"step": 5, "type": "TRANSFER", "amount": 200, "oldbalanceOrg": 1500, 
     "newbalanceOrig": 1300, "oldbalanceDest": 2500, "newbalanceDest": 2700},
])
y_dummy = np.array([0, 1, 0, 1, 0])

# Create proper preprocessing pipeline
# Use sparse_output parameter that's compatible with both old and new sklearn
try:
    # sklearn >= 1.2
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['type']),
            ('num', StandardScaler(), ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                                        'oldbalanceDest', 'newbalanceDest'])
        ])
except TypeError:
    # sklearn < 1.2 uses 'sparse' instead of 'sparse_output'
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), ['type']),
            ('num', StandardScaler(), ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                                        'oldbalanceDest', 'newbalanceDest'])
        ])

# Create full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5))
])

# Train on dummy data
pipeline.fit(X_dummy, y_dummy)

# Save the model
model_path = Path(__file__).parent / "models" / "fraud_detection_pipeline.pkl"
model_path.parent.mkdir(exist_ok=True)

joblib.dump(pipeline, model_path)
print(f"✅ Dummy model created successfully!")
print(f"📍 Location: {model_path.resolve()}")
print(f"📊 Model type: {type(pipeline).__name__}")
print(f"🧠 Features: step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest")
print(f"🎯 Classes: Legitimate (0) and Fraudulent (1)")
print(f"\n✨ Your Streamlit app should now load without errors!")
print(f"   Run: streamlit run app.py")
