"""
Pulls play-by-play data from the NHL's public API and extracts shot events
(shot-on-goal, goal, missed-shot, blocked-shot) into a flat list of dicts,
along with player-name and team-abbreviation lookup tables.

No API key needed. Endpoints used:
  - schedule:      https://api-web.nhle.com/v1/schedule/{YYYY-MM-DD}
  - play-by-play:  https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play

Schema verified directly against the live API (Sept 2026). If the NHL changes
their API shape, tweak the field names in `extract_shot_events`.
"""

import csv
import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://api-web.nhle.com/v1"
SHOT_EVENT_TYPES = {"shot-on-goal", "goal", "missed-shot", "blocked-shot"}


def _make_session() -> requests.Session:
    """A session with retries on connection resets and transient server
    errors -- a single blip shouldn't kill a fetch spanning hundreds of games."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_SESSION = _make_session()

SHOT_FIELDNAMES = [
    "game_id", "event_id", "event_type", "period", "time_in_period", "time_remaining",
    "situation_code", "home_defending_side", "x", "y", "zone", "shot_type",
    "shooting_player_id", "owner_team_id", "is_home_team_shot", "is_goal",
    "home_score_before", "away_score_before", "game_elapsed_s",
    "time_since_prev_event_s", "distance_from_prev_event_ft",
    "prev_event_same_team", "prev_event_was_shot", "_source_date",
]


def get_game_ids_for_date(date_str: str) -> list[int]:
    """date_str format: 'YYYY-MM-DD'. Returns NHL game IDs played that day."""
    resp = _SESSION.get(f"{BASE_URL}/schedule/{date_str}", timeout=15)
    resp.raise_for_status()
    data = resp.json()
    game_ids = []
    for week in data.get("gameWeek", []):
        if week.get("date") == date_str:
            game_ids.extend(g["id"] for g in week.get("games", []))
    return game_ids


def get_play_by_play(game_id: int) -> dict:
    resp = _SESSION.get(f"{BASE_URL}/gamecenter/{game_id}/play-by-play", timeout=15)
    resp.raise_for_status()
    return resp.json()


def _time_to_seconds(mmss) -> int | None:
    if not mmss:
        return None
    m, s = mmss.split(":")
    return int(m) * 60 + int(s)


def _game_elapsed_seconds(period: int, time_in_period: str) -> int | None:
    """timeInPeriod counts UP from 0:00, so elapsed time is just the sum of
    full periods before this one, plus time so far in the current period."""
    t = _time_to_seconds(time_in_period)
    if t is None or period is None:
        return None
    periods_before = min(period - 1, 3) * 20 * 60 + max(0, period - 4) * 5 * 60
    return periods_before + t


def extract_shot_events(pbp: dict) -> list[dict]:
    """
    Flattens the play-by-play JSON into one row per shot attempt. Tracks:
      - a running score (needed for the "score state" feature)
      - the immediately preceding event (any type, not just shots) so each
        shot can carry rebound/rush context: how long ago and how far away
        the previous event on the ice was, and whether it was the same
        team's own prior shot attempt.
    """
    game_id = pbp.get("id")
    home_team_id = pbp.get("homeTeam", {}).get("id")
    away_team_id = pbp.get("awayTeam", {}).get("id")

    rows = []
    home_score = 0
    away_score = 0
    prev_event = None  # {"x", "y", "elapsed_s", "owner_team_id", "is_shot"}

    for play in pbp.get("plays", []):
        event_type = play.get("typeDescKey")
        details = play.get("details", {})
        x, y = details.get("xCoord"), details.get("yCoord")
        period = play.get("periodDescriptor", {}).get("number")
        elapsed_s = _game_elapsed_seconds(period, play.get("timeInPeriod"))
        owner_team_id = details.get("eventOwnerTeamId")
        is_shot_type = event_type in SHOT_EVENT_TYPES

        if is_shot_type:
            is_home_team_shot = owner_team_id == home_team_id

            if prev_event is not None and prev_event["x"] is not None and x is not None and elapsed_s is not None and prev_event["elapsed_s"] is not None:
                time_since_prev = elapsed_s - prev_event["elapsed_s"]
                distance_from_prev = ((x - prev_event["x"]) ** 2 + (y - prev_event["y"]) ** 2) ** 0.5
                prev_same_team = prev_event["owner_team_id"] == owner_team_id
                prev_was_shot = prev_event["is_shot"]
            else:
                time_since_prev = None
                distance_from_prev = None
                prev_same_team = None
                prev_was_shot = None

            rows.append({
                "game_id": game_id,
                "event_id": play.get("eventId"),
                "event_type": event_type,
                "period": period,
                "time_in_period": play.get("timeInPeriod"),
                "time_remaining": play.get("timeRemaining"),
                "situation_code": play.get("situationCode"),
                "home_defending_side": play.get("homeTeamDefendingSide"),
                "x": x,
                "y": y,
                "zone": details.get("zoneCode"),
                "shot_type": details.get("shotType"),
                "shooting_player_id": details.get("shootingPlayerId") or details.get("scoringPlayerId"),
                "owner_team_id": owner_team_id,
                "is_home_team_shot": is_home_team_shot,
                "is_goal": event_type == "goal",
                "home_score_before": home_score,
                "away_score_before": away_score,
                "game_elapsed_s": elapsed_s,
                "time_since_prev_event_s": time_since_prev,
                "distance_from_prev_event_ft": distance_from_prev,
                "prev_event_same_team": prev_same_team,
                "prev_event_was_shot": prev_was_shot,
            })

            if event_type == "goal":
                if owner_team_id == home_team_id:
                    home_score += 1
                elif owner_team_id == away_team_id:
                    away_score += 1

        if x is not None and y is not None:
            prev_event = {"x": x, "y": y, "elapsed_s": elapsed_s, "owner_team_id": owner_team_id, "is_shot": is_shot_type}

    return rows


def extract_lookups(pbp: dict) -> tuple[dict, dict]:
    """Returns (player_id -> full name, team_id -> abbrev) for one game."""
    players = {}
    for spot in pbp.get("rosterSpots", []):
        first = spot.get("firstName", {}).get("default", "")
        last = spot.get("lastName", {}).get("default", "")
        players[spot["playerId"]] = f"{first} {last}".strip()

    teams = {}
    for side in ("homeTeam", "awayTeam"):
        team = pbp.get(side, {})
        if team.get("id") is not None:
            teams[team["id"]] = team.get("abbrev", str(team["id"]))

    return players, teams


def fetch_shots_for_date_range(dates: list[str], sleep_sec: float = 0.3, progress: bool = True,
                                 checkpoint_path: str = None):
    """
    Given a list of 'YYYY-MM-DD' strings, pull every shot event from every
    game played on those dates, plus player-name and team-abbreviation
    lookup tables built up across all fetched games.

    A single game that fails to fetch (even after the session's built-in
    retries are exhausted) is skipped rather than aborting the whole run --
    losing one game out of hundreds isn't worth losing everything already
    pulled. If `checkpoint_path` is given, progress is saved to disk after
    each date, so a fatal error partway through doesn't lose everything;
    re-running with the same path will pick up where it left off.
    """
    all_rows = []
    player_lookup = {}
    team_lookup = {}
    done_dates = set()

    if checkpoint_path:
        cached = load_cache(checkpoint_path)
        if cached:
            all_rows, player_lookup, team_lookup = cached
            done_dates = {str(r.get("_source_date")) for r in all_rows if r.get("_source_date")}
            if progress:
                print(f"Resuming from checkpoint: {len(all_rows)} shots already fetched, "
                      f"{len(done_dates)} date(s) already done.")

    for date_str in dates:
        if date_str in done_dates:
            continue
        try:
            game_ids = get_game_ids_for_date(date_str)
        except requests.exceptions.RequestException as e:
            print(f"skipping date {date_str} (couldn't fetch schedule): {e}")
            continue

        if progress:
            print(f"{date_str}: {len(game_ids)} game(s)")

        for gid in game_ids:
            try:
                pbp = get_play_by_play(gid)
            except requests.exceptions.RequestException as e:
                print(f"  skipping game {gid}: {e}")
                continue
            shots = extract_shot_events(pbp)
            for s in shots:
                s["_source_date"] = date_str
            all_rows.extend(shots)
            players, teams = extract_lookups(pbp)
            player_lookup.update(players)
            team_lookup.update(teams)
            time.sleep(sleep_sec)

        if checkpoint_path:
            save_cache(checkpoint_path, all_rows, player_lookup, team_lookup)

    return all_rows, player_lookup, team_lookup


def save_cache(path: str, rows: list[dict], player_lookup: dict, team_lookup: dict):
    """Saves fetched shots + lookups to disk so a large pull doesn't need to
    be re-fetched every time the notebook runs."""
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "shots.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SHOT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(path, "players.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["player_id", "name"])
        writer.writerows(player_lookup.items())

    with open(os.path.join(path, "teams.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["team_id", "abbrev"])
        writer.writerows(team_lookup.items())


def load_cache(path: str):
    """Loads a cache written by save_cache(). Returns (rows, player_lookup, team_lookup)
    or None if no cache exists at that path."""
    shots_path = os.path.join(path, "shots.csv")
    if not os.path.exists(shots_path):
        return None

    with open(shots_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with open(os.path.join(path, "players.csv"), newline="", encoding="utf-8") as f:
        player_lookup = {int(r["player_id"]): r["name"] for r in csv.DictReader(f)}

    with open(os.path.join(path, "teams.csv"), newline="", encoding="utf-8") as f:
        team_lookup = {int(r["team_id"]): r["abbrev"] for r in csv.DictReader(f)}

    return rows, player_lookup, team_lookup


if __name__ == "__main__":
    # Quick smoke test on a single known game.
    pbp = get_play_by_play(2023020672)
    shots = extract_shot_events(pbp)
    players, teams = extract_lookups(pbp)
    print(f"Extracted {len(shots)} shot events from game {pbp.get('id')}")
    print(shots[5])
    print("sample player:", list(players.items())[0])
    print("teams:", teams)
