"""
Trains expected-goals (xG) models on the engineered shot features and reports
standard classification metrics. Two model types are supported so they can be
compared head-to-head on the exact same train/test split:

  - XGBoost (gradient-boosted trees) -- the primary model.
  - Logistic regression -- a simple, interpretable baseline, to quantify how
    much lift the gradient-boosted model actually provides over a linear model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

NUMERIC_FEATURES = ["distance_ft", "angle_deg", "game_elapsed_s", "score_diff", "period", "is_rebound", "is_rush"]
CATEGORICAL_FEATURES = ["shot_type", "strength_state"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_data(df: pd.DataFrame, random_state: int = 42):
    """One shared split so every model is judged on identical train/test data."""
    X = df[ALL_FEATURES]
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)


def build_xgb_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",  # trees don't need numeric features scaled
    )
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
    )
    return Pipeline([("prep", preprocessor), ("clf", clf)])


def build_logistic_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
    )
    clf = LogisticRegression(max_iter=1000)
    return Pipeline([("prep", preprocessor), ("clf", clf)])


def _evaluate(pipeline, X_test, y_test) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "log_loss": log_loss(y_test, proba),
        "brier_score": brier_score_loss(y_test, proba),
        "n_test": len(y_test),
        "goal_rate_test": y_test.mean(),
    }, proba


def train_xgb(X_train, X_test, y_train, y_test):
    pipeline = build_xgb_pipeline()
    pipeline.fit(X_train, y_train)
    metrics, proba = _evaluate(pipeline, X_test, y_test)
    return pipeline, metrics, proba


def train_logistic(X_train, X_test, y_train, y_test):
    pipeline = build_logistic_pipeline()
    pipeline.fit(X_train, y_train)
    metrics, proba = _evaluate(pipeline, X_test, y_test)
    return pipeline, metrics, proba


def train_and_evaluate(df: pd.DataFrame, random_state: int = 42):
    """Back-compat wrapper: trains just the XGBoost model, as the original
    single-model version of this module did."""
    X_train, X_test, y_train, y_test = split_data(df, random_state=random_state)
    pipeline, metrics, proba = train_xgb(X_train, X_test, y_train, y_test)
    return pipeline, metrics, (X_test, y_test, proba)


if __name__ == "__main__":
    print("Import this module from the notebook; it's not meant to run standalone without data.")
