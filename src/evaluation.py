from dataclasses import dataclass

import numpy as np
from sklearn.metrics import auc, classification_report, precision_recall_curve, roc_auc_score


@dataclass(frozen=True)
class EvalResult:
    roc_auc: float
    pr_auc: float
    threshold: float
    report: str


def evaluate_binary_classifier(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    threshold: float = 0.9,
) -> EvalResult:
    model.fit(X_train, y_train)

    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must implement predict_proba for ROC-AUC/PR-AUC evaluation.")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)

    roc = float(roc_auc_score(y_test, y_pred_proba))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr = float(auc(recall, precision))

    y_pred = (y_pred_proba >= threshold).astype(int)
    report = classification_report(y_test, y_pred, zero_division=0)

    return EvalResult(roc_auc=roc, pr_auc=pr, threshold=threshold, report=report)


__all__ = ["EvalResult", "evaluate_binary_classifier"]

