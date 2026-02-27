from dataclasses import dataclass
from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import create_features


@dataclass(frozen=True)
class FeatureSpec:
    # Use typing.List for compatibility with Python < 3.9
    numeric_features: List[str]
    categorical_features: List[str]


DEFAULT_FEATURE_SPEC = FeatureSpec(
    numeric_features=[
        "amount_log",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "orig_balance_diff",
        "dest_balance_diff",
        "hour",
    ],
    categorical_features=["type", "orig_empty", "dest_empty", "is_night"],
)


def build_preprocessor(spec: FeatureSpec = DEFAULT_FEATURE_SPEC) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, spec.numeric_features),
            ("cat", categorical_transformer, spec.categorical_features),
        ]
    )


def build_classifier(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=50,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )


def build_baseline_pipeline(
    preprocessor: ColumnTransformer,
    *,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(create_features)),
            ("preprocessor", preprocessor),
            ("classifier", build_classifier(random_state=random_state)),
        ]
    )


def build_smote_pipeline(
    preprocessor: ColumnTransformer,
    *,
    sampling_strategy: float = 0.1,
    random_state: int = 42,
) :
    # Imported lazily so the baseline pipeline can run without imbalanced-learn.
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    return ImbPipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(create_features)),
            ("preprocessor", preprocessor),
            (
                "smote",
                SMOTE(sampling_strategy=sampling_strategy, random_state=random_state),
            ),
            ("classifier", build_classifier(random_state=random_state)),
        ]
    )


__all__ = [
    "DEFAULT_FEATURE_SPEC",
    "FeatureSpec",
    "build_baseline_pipeline",
    "build_preprocessor",
    "build_smote_pipeline",
]

