"""
Two visualizations 
  1. A rink shot-location scatter, colored by predicted goal probability (xG).
  2. A cumulative xG-over-game-time line chart for a single game.
"""

import matplotlib.pyplot as plt


def plot_shot_map(df, xg_col="xg", ax=None):
    """df needs columns: x, y, and an xg probability column."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    sc = ax.scatter(df["x"], df["y"], c=df[xg_col], cmap="Reds", s=25, alpha=0.8, edgecolors="k", linewidths=0.3)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-45, 45)
    ax.set_title("Shot locations colored by predicted xG")
    ax.set_xlabel("x (ft from center ice)")
    ax.set_ylabel("y (ft from center line)")
    plt.colorbar(sc, ax=ax, label="xG")
    return ax


def plot_cumulative_xg(df_game, xg_col="xg", ax=None):
    """df_game: rows for ONE game, sorted by game_elapsed_s, with xg + is_home_team_shot."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    df_sorted = df_game.sort_values("game_elapsed_s")
    for team_flag, label in [(True, "Home"), (False, "Away")]:
        team_df = df_sorted[df_sorted["is_home_team_shot"] == team_flag]
        ax.step(team_df["game_elapsed_s"] / 60, team_df[xg_col].cumsum(), where="post", label=label)

    ax.set_xlabel("Game time (minutes)")
    ax.set_ylabel("Cumulative xG")
    ax.set_title("Cumulative expected goals over game time")
    ax.legend()
    return ax
