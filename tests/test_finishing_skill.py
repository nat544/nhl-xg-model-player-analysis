"""
Unit tests for finishing_skill.py's aggregation logic (_finishing_table),
plus a small end-to-end check that compute_oof_xg runs and returns
sensible output shape on a tiny synthetic dataset.
"""

import pandas as pd
import pytest

from finishing_skill import _finishing_table, player_finishing_skill, compute_oof_xg


def test_finishing_table_computes_goals_above_expected():
    df = pd.DataFrame({
        "shooting_player_id": [1, 1, 1, 2, 2, 2],
        "target": [1, 1, 0, 0, 0, 0],
        "xg_oof": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    })
    table = _finishing_table(df, "shooting_player_id", {1: "Overperformer", 2: "Underperformer"}, min_shots=1)

    row1 = table.loc[1]
    assert row1["shots"] == 3
    assert row1["goals"] == 2
    assert row1["xg"] == pytest.approx(0.6)
    assert row1["goals_above_expected"] == pytest.approx(1.4)

    row2 = table.loc[2]
    assert row2["goals"] == 0
    assert row2["goals_above_expected"] == pytest.approx(-0.6)


def test_finishing_table_filters_out_small_samples():
    df = pd.DataFrame({
        "shooting_player_id": [1, 1, 2, 2, 2, 2],
        "target": [1, 0, 1, 0, 1, 0],
        "xg_oof": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    })
    table = _finishing_table(df, "shooting_player_id", {1: "Low volume", 2: "High volume"}, min_shots=3)
    assert 1 not in table.index
    assert 2 in table.index


def test_finishing_table_sorted_best_to_worst():
    df = pd.DataFrame({
        "shooting_player_id": [1, 1, 2, 2],
        "target": [1, 1, 0, 0],
        "xg_oof": [0.1, 0.1, 0.1, 0.1],
    })
    table = _finishing_table(df, "shooting_player_id", {1: "Best", 2: "Worst"}, min_shots=1)
    assert table.index.tolist() == [1, 2]


def test_finishing_table_falls_back_to_id_string_when_name_missing():
    df = pd.DataFrame({
        "shooting_player_id": [999],
        "target": [1],
        "xg_oof": [0.5],
    })
    table = _finishing_table(df, "shooting_player_id", {}, min_shots=1)
    assert table.loc[999, "name"] == "999"


def test_player_finishing_skill_uses_default_min_shots_of_30():
    df = pd.DataFrame({
        "shooting_player_id": [1] * 29,
        "target": [0] * 29,
        "xg_oof": [0.1] * 29,
    })
    table = player_finishing_skill(df, {1: "Just under threshold"})
    assert table.empty


def _synthetic_shot_df(n=40):
    import numpy as np
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "distance_ft": rng.uniform(5, 60, n),
        "angle_deg": rng.uniform(0, 90, n),
        "game_elapsed_s": rng.uniform(0, 3600, n),
        "score_diff": rng.integers(-2, 2, n),
        "period": rng.integers(1, 4, n),
        "is_rebound": rng.choice([True, False], n),
        "is_rush": rng.choice([True, False], n),
        "shot_type": rng.choice(["wrist", "slap", "unknown"], n),
        "strength_state": rng.choice(["5v5", "5v4"], n),
        "target": rng.integers(0, 2, n),
    })


def test_compute_oof_xg_returns_a_probability_per_row():
    df = _synthetic_shot_df(40)
    oof = compute_oof_xg(df, n_splits=2)
    assert len(oof) == len(df)
    assert oof.between(0, 1).all()
