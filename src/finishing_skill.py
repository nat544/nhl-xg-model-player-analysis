"""
Aggregates actual goals vs. expected goals (xG) by player and by team to find
over/under-performers relative to shot quality.

Uses OUT-OF-FOLD predictions (via cross_val_predict).
"""

import pandas as pd
from sklearn.model_selection import cross_val_predict

from model import build_xgb_pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def compute_oof_xg(df: pd.DataFrame, n_splits: int = 5) -> pd.Series:
    """Returns an out-of-fold predicted goal-probability for every row in df."""
    X = df[ALL_FEATURES]
    y = df["target"]
    pipeline = build_xgb_pipeline()
    oof_proba = cross_val_predict(pipeline, X, y, cv=n_splits, method="predict_proba")[:, 1]
    return pd.Series(oof_proba, index=df.index)


def _finishing_table(df: pd.DataFrame, group_col: str, name_lookup: dict, min_shots: int) -> pd.DataFrame:
    grouped = df.groupby(group_col).agg(
        shots=("target", "size"),
        goals=("target", "sum"),
        xg=("xg_oof", "sum"),
    )
    grouped = grouped[grouped["shots"] >= min_shots].copy()
    grouped["goals_above_expected"] = grouped["goals"] - grouped["xg"]
    grouped["goals_above_expected_per_100_shots"] = 100 * grouped["goals_above_expected"] / grouped["shots"]
    grouped["name"] = [name_lookup.get(int(idx), str(idx)) for idx in grouped.index]

    cols = ["name", "shots", "goals", "xg", "goals_above_expected", "goals_above_expected_per_100_shots"]
    return grouped[cols].sort_values("goals_above_expected_per_100_shots", ascending=False)


def player_finishing_skill(df: pd.DataFrame, player_lookup: dict, min_shots: int = 30) -> pd.DataFrame:
    """Ranks shooters by goals scored above/below what shot quality (xG) predicted."""
    return _finishing_table(df, "shooting_player_id", player_lookup, min_shots)


def team_finishing_skill(df: pd.DataFrame, team_lookup: dict, min_shots: int = 100) -> pd.DataFrame:
    """Same idea, aggregated by shooting team."""
    return _finishing_table(df, "owner_team_id", team_lookup, min_shots)
