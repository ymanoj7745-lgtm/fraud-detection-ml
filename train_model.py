import pandas as pd
import logging
from dataclasses import dataclass
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
import joblib
from src.evaluation import evaluate_binary_classifier
from src.modeling import build_baseline_pipeline, build_preprocessor, build_smote_pipeline


USE_COLS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
]

LOGGER = logging.getLogger("fraud_detection")


@dataclass(frozen=True)
class TrainConfig:
    data_path: Path = Path("Fraud.csv")
    model_path: Path = Path("models/fraud_detection_pipeline.pkl")
    sample_rows: int = 1_000_000
    test_size: float = 0.2
    random_state: int = 42
    smote_sampling_strategy: float = 0.1
    threshold: float = 0.9
    skip_smote: bool = False


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a fraud detection model and save the pipeline.")
    parser.add_argument("--data-path", default="Fraud.csv", help="Path to the training CSV file.")
    parser.add_argument(
        "--model-path",
        default="models/fraud_detection_pipeline.pkl",
        help="Where to save the trained model pipeline.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=1_000_000,
        help="Max rows to sample for training (<=0 means no sampling).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--smote-sampling-strategy",
        type=float,
        default=0.1,
        help="SMOTE sampling_strategy (e.g., 0.1).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Threshold used for the classification report (probability cutoff).",
    )
    parser.add_argument(
        "--skip-smote",
        action="store_true",
        help="Train baseline model only (no SMOTE). Useful if imbalanced-learn is not available.",
    )

    args = parser.parse_args()
    return TrainConfig(
        data_path=Path(args.data_path),
        model_path=Path(args.model_path),
        sample_rows=args.sample_rows,
        test_size=args.test_size,
        random_state=args.random_state,
        smote_sampling_strategy=args.smote_sampling_strategy,
        threshold=args.threshold,
        skip_smote=bool(args.skip_smote),
    )


def main() -> None:
    _setup_logging()
    cfg = _parse_args()

    LOGGER.info("Loading data from %s", cfg.data_path)
    df = pd.read_csv(cfg.data_path, usecols=USE_COLS, low_memory=True)

    if cfg.sample_rows and cfg.sample_rows > 0 and len(df) > cfg.sample_rows:
        df = df.sample(n=cfg.sample_rows, random_state=cfg.random_state)
        LOGGER.info("Sampled %d rows for training.", cfg.sample_rows)

    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.random_state,
    )

    preprocessor = build_preprocessor()

    LOGGER.info("Evaluating baseline pipeline (no SMOTE)...")
    baseline_pipeline = build_baseline_pipeline(preprocessor, random_state=cfg.random_state)
    baseline_eval = evaluate_binary_classifier(
        baseline_pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
        threshold=cfg.threshold,
    )
    LOGGER.info("Baseline ROC-AUC=%.4f PR-AUC=%.4f", baseline_eval.roc_auc, baseline_eval.pr_auc)

    if cfg.skip_smote:
        LOGGER.warning("Skipping SMOTE as requested. Training baseline model only.")
        LOGGER.info("Training final baseline pipeline on full dataset...")
        final_pipeline = build_baseline_pipeline(preprocessor, random_state=cfg.random_state)
        final_pipeline.fit(X, y)
    else:
        LOGGER.info("Evaluating SMOTE pipeline...")
        try:
            smote_pipeline = build_smote_pipeline(
                preprocessor,
                sampling_strategy=cfg.smote_sampling_strategy,
                random_state=cfg.random_state,
            )
        except ModuleNotFoundError as e:
            if getattr(e, "name", "") == "imblearn":
                raise ModuleNotFoundError(
                    "Missing dependency 'imbalanced-learn' required for SMOTE. "
                    "Install dependencies with: pip install -r requirements.txt "
                    "or run: python train_model.py --skip-smote"
                ) from e
            raise

        smote_eval = evaluate_binary_classifier(
            smote_pipeline,
            X_train,
            y_train,
            X_test,
            y_test,
            threshold=cfg.threshold,
        )
        LOGGER.info("SMOTE ROC-AUC=%.4f PR-AUC=%.4f", smote_eval.roc_auc, smote_eval.pr_auc)

        LOGGER.info(
            "Metric improvements (SMOTE - Baseline): ROC-AUC=%.4f PR-AUC=%.4f",
            smote_eval.roc_auc - baseline_eval.roc_auc,
            smote_eval.pr_auc - baseline_eval.pr_auc,
        )

        LOGGER.info("Training final SMOTE pipeline on full dataset...")
        final_pipeline = build_smote_pipeline(
            preprocessor,
            sampling_strategy=cfg.smote_sampling_strategy,
            random_state=cfg.random_state,
        )
        final_pipeline.fit(X, y)

    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, cfg.model_path)
    LOGGER.info("Saved pipeline to %s", cfg.model_path)


if __name__ == "__main__":
    main()

