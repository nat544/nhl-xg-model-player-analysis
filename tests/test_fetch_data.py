"""
Unit tests for fetch_data.py: game-clock math, shot extraction with
rebound/rush context, player/team lookups, and the cache round-trip.
None of these hit the live NHL API -- they use small hand-built fixtures.
"""

from fetch_data import (
    _time_to_seconds,
    _game_elapsed_seconds,
    extract_shot_events,
    extract_lookups,
    save_cache,
    load_cache,
)

HOME_TEAM_ID = 10
AWAY_TEAM_ID = 20


# ---- time helpers ---------------------------------------------------------

def test_time_to_seconds():
    assert _time_to_seconds("05:30") == 330
    assert _time_to_seconds(None) is None


def test_game_elapsed_seconds_within_regulation():
    assert _game_elapsed_seconds(1, "05:00") == 300
    assert _game_elapsed_seconds(2, "00:00") == 1200
    assert _game_elapsed_seconds(3, "00:00") == 2400


def test_game_elapsed_seconds_overtime():
    # 3 full 20-minute periods (3600s) plus time into OT
    assert _game_elapsed_seconds(4, "02:00") == 3600 + 120


def test_game_elapsed_seconds_missing_input():
    assert _game_elapsed_seconds(None, "05:00") is None
    assert _game_elapsed_seconds(1, None) is None


# ---- extract_shot_events: fixtures -----------------------------------------

def _fake_pbp(plays, roster=None):
    return {
        "id": 999,
        "homeTeam": {"id": HOME_TEAM_ID, "abbrev": "HOM"},
        "awayTeam": {"id": AWAY_TEAM_ID, "abbrev": "AWY"},
        "plays": plays,
        "rosterSpots": roster or [],
    }


def _play(event_id, event_type, team_id, x=50, y=10, period=1, time_in_period="01:00"):
    return {
        "eventId": event_id,
        "typeDescKey": event_type,
        "periodDescriptor": {"number": period},
        "timeInPeriod": time_in_period,
        "timeRemaining": "19:00",
        "situationCode": "1551",
        "homeTeamDefendingSide": "left",
        "details": {
            "eventOwnerTeamId": team_id,
            "xCoord": x,
            "yCoord": y,
            "zoneCode": "O",
            "shotType": "wrist",
            "shootingPlayerId": 111,
        },
    }


def test_non_shot_events_produce_no_rows_but_still_update_prev_event():
    plays = [
        _play(1, "faceoff", HOME_TEAM_ID, x=0, y=0, time_in_period="00:00"),
        _play(2, "shot-on-goal", HOME_TEAM_ID, x=10, y=0, time_in_period="00:02"),
    ]
    rows = extract_shot_events(_fake_pbp(plays))
    assert len(rows) == 1
    # previous event (the faceoff) was not a shot
    assert rows[0]["prev_event_was_shot"] is False
    assert rows[0]["time_since_prev_event_s"] == 2
    assert rows[0]["distance_from_prev_event_ft"] == 10.0


def test_rebound_context_true_for_quick_close_second_shot():
    plays = [
        _play(1, "shot-on-goal", HOME_TEAM_ID, x=80, y=0, time_in_period="00:00"),
        _play(2, "shot-on-goal", HOME_TEAM_ID, x=82, y=1, time_in_period="00:02"),
    ]
    rows = extract_shot_events(_fake_pbp(plays))
    second = rows[1]
    assert second["prev_event_was_shot"] is True
    assert second["prev_event_same_team"] is True
    assert second["time_since_prev_event_s"] == 2
    assert second["distance_from_prev_event_ft"] == pytest_approx_sqrt5()


def pytest_approx_sqrt5():
    # (82-80)^2 + (1-0)^2 = 5 -> sqrt(5)
    return 5 ** 0.5


def test_prev_event_same_team_false_for_opponent_shot():
    plays = [
        _play(1, "shot-on-goal", HOME_TEAM_ID, x=80, y=0, time_in_period="00:00"),
        _play(2, "shot-on-goal", AWAY_TEAM_ID, x=-80, y=0, time_in_period="00:05"),
    ]
    rows = extract_shot_events(_fake_pbp(plays))
    assert rows[1]["prev_event_same_team"] is False


def test_first_event_in_game_has_no_prev_event_context():
    plays = [_play(1, "shot-on-goal", HOME_TEAM_ID)]
    rows = extract_shot_events(_fake_pbp(plays))
    assert rows[0]["time_since_prev_event_s"] is None
    assert rows[0]["distance_from_prev_event_ft"] is None
    assert rows[0]["prev_event_same_team"] is None
    assert rows[0]["prev_event_was_shot"] is None


def test_running_score_increments_on_goals():
    plays = [
        _play(1, "goal", HOME_TEAM_ID),
        _play(2, "shot-on-goal", AWAY_TEAM_ID),
    ]
    rows = extract_shot_events(_fake_pbp(plays))
    assert rows[0]["home_score_before"] == 0
    assert rows[1]["home_score_before"] == 1


# ---- extract_lookups ----------------------------------------------------

def test_extract_lookups_builds_player_and_team_maps():
    roster = [{"playerId": 1, "firstName": {"default": "Connor"}, "lastName": {"default": "McDavid"}}]
    players, teams = extract_lookups(_fake_pbp([], roster=roster))
    assert players[1] == "Connor McDavid"
    assert teams == {HOME_TEAM_ID: "HOM", AWAY_TEAM_ID: "AWY"}


# ---- save_cache / load_cache round-trip ------------------------------------

def test_cache_round_trip_preserves_row_count_and_lookups(tmp_path):
    plays = [_play(1, "goal", HOME_TEAM_ID)]
    rows = extract_shot_events(_fake_pbp(plays))
    for r in rows:
        r["_source_date"] = "2024-01-01"

    players = {111: "Test Player"}
    teams = {HOME_TEAM_ID: "HOM", AWAY_TEAM_ID: "AWY"}

    cache_dir = str(tmp_path / "cache")
    save_cache(cache_dir, rows, players, teams)

    loaded_rows, loaded_players, loaded_teams = load_cache(cache_dir)
    assert len(loaded_rows) == len(rows)
    assert loaded_players == players
    assert loaded_teams == teams


def test_load_cache_returns_none_when_missing(tmp_path):
    assert load_cache(str(tmp_path / "does_not_exist")) is None
