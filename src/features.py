"""
Features: 
- shot distance/angle to the net, 
- time elapsed in the game, 
- score differential, and 
- rebound/rush-shot context.
"""

import math
import pandas as pd

NET_X = 89.0  # NHL rink: net sits ~89ft from center ice along the long axis

# Heuristics used by public NHL xG models (e.g. MoneyPuck, Evolving-Hockey)
# for what counts as a "rebound" or "rush" shot attempt.
REBOUND_MAX_TIME_S = 3
REBOUND_MAX_DISTANCE_FT = 15
RUSH_MAX_TIME_S = 4
RUSH_MIN_DISTANCE_FT = 60


def _strength_state(situation_code: str, is_home_shot: bool) -> str:
    """
    situationCode is 4 digits: [away_goalie][away_skaters][home_skaters][home_goalie]
    e.g. "1551" = away has goalie(1) + 5 skaters, home has 5 skaters + goalie(1) -> even strength.
    Returns something like "5v5", "5v4", "4v5", etc. from the shooting team's perspective.
    """
    if not situation_code or len(str(situation_code)) != 4:
        return "unknown"
    situation_code = str(situation_code)
    away_skaters, home_skaters = int(situation_code[1]), int(situation_code[2])
    shooter_skaters, opp_skaters = (home_skaters, away_skaters) if is_home_shot else (away_skaters, home_skaters)
    return f"{shooter_skaters}v{opp_skaters}"


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    return str(v).strip().lower() == "true"


def build_features(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Rows may come straight from the API (native types) or from a CSV cache
    # (everything is a string) -- coerce the fields we do arithmetic on.
    numeric_cols = ["x", "y", "period", "home_score_before", "away_score_before",
                     "game_elapsed_s", "time_since_prev_event_s", "distance_from_prev_event_ft"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_home_team_shot"] = df["is_home_team_shot"].map(_to_bool)
    df["is_goal"] = df["is_goal"].map(_to_bool)
    df["prev_event_same_team"] = df["prev_event_same_team"].map(_to_bool)
    df["prev_event_was_shot"] = df["prev_event_was_shot"].map(_to_bool)

    # Normalize x so "distance to net" always measures toward the attacking net,
    # regardless of which side of the rink the team is shooting from.
    def attacking_net_x(row):
        # Shots recorded with home_defending_side telling us which side HOME defends;
        # the shooting team is attacking the opposite net from whichever net they're near.
        # Simplify: assume shot is always toward the net on the far side of center ice
        # from where it was taken (standard approach used in public xG models).
        return NET_X if row["x"] >= 0 else -NET_X

    df["net_x"] = df.apply(attacking_net_x, axis=1)
    df["distance_ft"] = ((df["net_x"] - df["x"]) ** 2 + df["y"] ** 2) ** 0.5
    df["angle_deg"] = df.apply(
        lambda r: math.degrees(math.atan2(abs(r["y"]), abs(r["net_x"] - r["x"]) + 1e-6)),
        axis=1,
    )

    df["score_diff"] = df.apply(
        lambda r: (r["home_score_before"] - r["away_score_before"]) if r["is_home_team_shot"]
        else (r["away_score_before"] - r["home_score_before"]),
        axis=1,
    )

    df["strength_state"] = df.apply(
        lambda r: _strength_state(r["situation_code"], r["is_home_team_shot"]), axis=1
    )

    # Rebound: quick follow-up shot on the team's own prior shot attempt.
    df["is_rebound"] = (
        df["prev_event_was_shot"]
        & df["prev_event_same_team"]
        & (df["time_since_prev_event_s"] <= REBOUND_MAX_TIME_S)
        & (df["distance_from_prev_event_ft"] <= REBOUND_MAX_DISTANCE_FT)
    )

    # Rush: shot quickly following an event far away on the ice (the other
    # team's or a neutral event), suggesting a fast transition up the ice.
    df["is_rush"] = (
        (~df["prev_event_same_team"])
        & (df["time_since_prev_event_s"] <= RUSH_MAX_TIME_S)
        & (df["distance_from_prev_event_ft"] >= RUSH_MIN_DISTANCE_FT)
    )

    df["shot_type"] = df["shot_type"].fillna("unknown")
    df["target"] = df["is_goal"].astype(int)

    feature_cols = [
        "distance_ft", "angle_deg", "game_elapsed_s", "score_diff",
        "period", "shot_type", "strength_state", "is_rebound", "is_rush",
    ]
    return df[feature_cols + [
        "target", "game_id", "event_id", "x", "y", "is_home_team_shot",
        "owner_team_id", "shooting_player_id", "is_goal",
    ]]
