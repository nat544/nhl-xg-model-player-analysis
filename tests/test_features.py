"""
Unit tests for features.py -- geometry, strength-state parsing, CSV-string
type coercion, and the rebound/rush classification logic.
"""

import pytest

from features import (
    _strength_state,
    _to_bool,
    build_features,
    REBOUND_MAX_TIME_S,
    REBOUND_MAX_DISTANCE_FT,
    RUSH_MAX_TIME_S,
    RUSH_MIN_DISTANCE_FT,
)


# ---- _strength_state ----------------------------------------------------

def test_strength_state_even_strength():
    assert _strength_state("1551", is_home_shot=True) == "5v5"


def test_strength_state_power_play():
    assert _strength_state("1451", is_home_shot=True) == "5v4"
    assert _strength_state("1451", is_home_shot=False) == "4v5"


def test_strength_state_handles_malformed_code():
    assert _strength_state("", is_home_shot=True) == "unknown"
    assert _strength_state(None, is_home_shot=True) == "unknown"


def test_strength_state_accepts_int_like_code_from_csv():
    # Cached rows loaded from CSV may hand this in as a numeric-looking string
    assert _strength_state("1551", is_home_shot=True) == "5v5"


# ---- _to_bool (needed because CSV-cached rows are all strings) -------------

def test_to_bool_passes_through_real_booleans():
    assert _to_bool(True) is True
    assert _to_bool(False) is False


def test_to_bool_parses_string_true_case_insensitively():
    assert _to_bool("True") is True
    assert _to_bool("true") is True
    assert _to_bool("TRUE") is True


def test_to_bool_parses_string_false():
    assert _to_bool("False") is False
    assert _to_bool("false") is False


def test_to_bool_treats_missing_value_as_false():
    assert _to_bool(None) is False
    assert _to_bool(float("nan")) is False


# ---- build_features ---------------------------------------------------

def _base_row(**overrides):
    row = {
        "game_id": 1,
        "event_id": 1,
        "event_type": "shot-on-goal",
        "period": 1,
        "time_in_period": "00:00",
        "time_remaining": "20:00",
        "situation_code": "1551",
        "home_defending_side": "left",
        "x": 89,
        "y": 0,
        "zone": "O",
        "shot_type": "wrist",
        "shooting_player_id": 123,
        "owner_team_id": 10,
        "is_home_team_shot": True,
        "is_goal": False,
        "home_score_before": 0,
        "away_score_before": 0,
        "game_elapsed_s": 0,
        "time_since_prev_event_s": None,
        "distance_from_prev_event_ft": None,
        "prev_event_same_team": None,
        "prev_event_was_shot": None,
    }
    row.update(overrides)
    return row


def test_build_features_on_empty_input_returns_empty_dataframe():
    assert build_features([]).empty


def test_distance_and_angle_geometry_matches_basic_project():
    df = build_features([_base_row(x=89, y=0)])
    assert df.iloc[0]["distance_ft"] == pytest.approx(0, abs=0.01)
    df2 = build_features([_base_row(x=0, y=0)])
    assert df2.iloc[0]["distance_ft"] == pytest.approx(89, abs=0.01)


def test_is_rebound_true_when_quick_close_follow_up_on_own_shot():
    row = _base_row(
        prev_event_was_shot=True,
        prev_event_same_team=True,
        time_since_prev_event_s=1,
        distance_from_prev_event_ft=5,
    )
    df = build_features([row])
    assert bool(df.iloc[0]["is_rebound"]) is True
    assert bool(df.iloc[0]["is_rush"]) is False


def test_is_rebound_false_when_too_slow():
    row = _base_row(
        prev_event_was_shot=True,
        prev_event_same_team=True,
        time_since_prev_event_s=REBOUND_MAX_TIME_S + 1,
        distance_from_prev_event_ft=5,
    )
    df = build_features([row])
    assert bool(df.iloc[0]["is_rebound"]) is False


def test_is_rebound_false_when_too_far_away():
    row = _base_row(
        prev_event_was_shot=True,
        prev_event_same_team=True,
        time_since_prev_event_s=1,
        distance_from_prev_event_ft=REBOUND_MAX_DISTANCE_FT + 1,
    )
    df = build_features([row])
    assert bool(df.iloc[0]["is_rebound"]) is False


def test_is_rebound_false_when_previous_event_was_not_a_shot():
    row = _base_row(
        prev_event_was_shot=False,
        prev_event_same_team=True,
        time_since_prev_event_s=1,
        distance_from_prev_event_ft=5,
    )
    df = build_features([row])
    assert bool(df.iloc[0]["is_rebound"]) is False


def test_is_rush_true_for_quick_long_transition_from_other_team():
    row = _base_row(
        prev_event_was_shot=False,
        prev_event_same_team=False,
        time_since_prev_event_s=2,
        distance_from_prev_event_ft=RUSH_MIN_DISTANCE_FT + 10,
    )
    df = build_features([row])
    assert bool(df.iloc[0]["is_rush"]) is True
    assert bool(df.iloc[0]["is_rebound"]) is False


def test_is_rush_false_when_distance_too_short():
    row = _base_row(
        prev_event_same_team=False,
        time_since_prev_event_s=2,
        distance_from_prev_event_ft=RUSH_MIN_DISTANCE_FT - 1,
    )
    df = build_features([row])
    assert bool(df.iloc[0]["is_rush"]) is False


def test_is_rush_false_when_too_slow():
    row = _base_row(
        prev_event_same_team=False,
        time_since_prev_event_s=RUSH_MAX_TIME_S + 1,
        distance_from_prev_event_ft=RUSH_MIN_DISTANCE_FT + 10,
    )
    df = build_features([row])
    assert bool(df.iloc[0]["is_rush"]) is False


def test_build_features_handles_csv_string_types():
    # Simulates a row loaded back from the CSV cache, where every value is a string.
    row = _base_row(
        x="89", y="0", period="1", home_score_before="0", away_score_before="0",
        game_elapsed_s="0", is_home_team_shot="True", is_goal="False",
        prev_event_was_shot="True", prev_event_same_team="True",
        time_since_prev_event_s="1", distance_from_prev_event_ft="5",
    )
    df = build_features([row])
    assert df.iloc[0]["distance_ft"] == pytest.approx(0, abs=0.01)
    assert bool(df.iloc[0]["is_rebound"]) is True
    assert df.iloc[0]["target"] == 0
