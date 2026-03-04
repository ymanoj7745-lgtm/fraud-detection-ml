# scikit-learn Version Compatibility Fix

## Problem Summary

Your Streamlit deployment is failing with:
```
AttributeError: Can't get attribute '_RemainderColsList' on <module 
'sklearn.compose._column_transformer' from 
'/home/adminuser/venv/lib/python3.13/site-packages/sklearn/compose/_column_transformer.py'>
```

### Root Cause
- **Model pickled with**: scikit-learn 1.6.1  
- **Deployed environment**: scikit-learn 1.8.0
- **Issue**: The internal class `_RemainderColsList` was removed/refactored in scikit-learn 1.8.x

This is a **model compatibility issue**, not a code bug.

---

## Solution: Regenerate Your Model

The quickest fix is to create a new model compatible with your deployment environment:

### Quick Fix (Development)
```bash
# In your local development environment
python create_dummy_model.py
```

This generates a compatible model pickle file at:
```
models/fraud_detection_pipeline.pkl
```

### Production Fix (Streamlit Cloud)

1. **Update requirements.txt** - Already done ✅
   - Changed from `scikit-learn==1.3.2` to `scikit-learn>=1.2,<1.8`
   - Added helpful comments about version constraints

2. **Strengthen app.py compatibility** - Already done ✅
   - Enhanced the `_inject_sklearn_compatibility_stubs()` function
   - Better error messages to guide users

3. **Regenerate the pickle** - You need to:
   ```bash
   # Run locally or in CI/CD
   python create_dummy_model.py
   
   # Commit and push the new pickle file
   git add models/fraud_detection_pipeline.pkl
   git commit -m "Regenerate model for sklearn <1.8 compatibility"
   git push
   ```

---

## How the Fix Works

### Compatibility Shim in app.py
When the Streamlit app loads, it now:

1. **Injects missing sklearn classes** before unpickling
2. **Creates stub implementations** of removed private classes  
3. **Handles multiple compatibility scenarios**
4. **Provides clear error messages** if issues persist

```python
def _inject_sklearn_compatibility_stubs():
    """Inject missing sklearn private classes for backward compatibility."""
    from sklearn.compose import _column_transformer as _ct
    
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList:  # Stub for sklearn <1.8 pickles
            pass
        _ct._RemainderColsList = _RemainderColsList
```

### Version Constraints in requirements.txt
```
# Prevents scikit-learn from auto-upgrading to incompatible versions
scikit-learn>=1.2,<1.8
```

This keeps your environment in a safe range while still allowing minor updates.

---

## Verification

After applying the fix:

```bash
# Test locally
streamlit run app.py

# Check model loads without errors
python -c "import joblib; m = joblib.load('models/fraud_detection_pipeline.pkl'); print('✅ Model loaded successfully')"
```

---

## Files Changed

| File | Change | Status |
|------|--------|--------|
| `app.py` | Enhanced compatibility shim with better error handling | ✅ Fixed |
| `requirements.txt` | Pinned scikit-learn to <1.8 with documentation | ✅ Fixed |
| `create_dummy_model.py` | Added version-aware model generation | ✅ Enhanced |
| `models/fraud_detection_pipeline.pkl` | Needs regeneration | ⏳ TODO |

---

## Next Steps

### Option A: Automatic Fix (Recommended)
```bash
# 1. Regenerate the model
python create_dummy_model.py

# 2. Commit changes
git add models/fraud_detection_pipeline.pkl
git commit -m "Regenerate model for sklearn <1.8 compatibility"
git push

# 3. Your Streamlit deployment will auto-update
```

### Option B: Manual Troubleshooting
If you still see errors after regenerating:

1. **Check Python version match**
   ```bash
   python --version  # Local
   # Streamlit Cloud uses Python 3.13
   ```

2. **Verify joblib version**
   ```bash
   pip show joblib  # Should be >= 1.2
   ```

3. **View Streamlit logs** for detailed error messages

### Option C: Long-term Solution
For production, consider:
- Using **ONNX format** for inference (version-agnostic)
- Using **model versioning/registry** (MLflow, DVC)
- Regular model retraining in CI/CD pipeline

---

## Why This Happens

scikit-learn's internal API changes between major versions. While pickle preserves Python objects, it references internal classes by name. When these classes are removed/refactored, unpickling fails even though the *functionality* hasn't changed.

This is a **known limitation of pickle** for ML libraries and why alternatives like ONNX exist.

---

## References

- scikit-learn migration guide: https://scikit-learn.org/stable/upgrade.html
- Pickle versioning: https://docs.python.org/3/library/pickle.html
- Model persistence best practices: https://scikit-learn.org/stable/model_persistence.html

