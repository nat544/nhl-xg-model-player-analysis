# NHL Expected Goals (xG) Model — Spatio-Temporal Shot Quality

A shot-quality ("expected goals") model for NHL shots, built to demonstrate
**spatio-temporal modeling**: predicting an outcome (did this shot become a goal?)
from *where* on the ice a shot was taken combined with *when* in the game it
happened and the context at that moment (score, strength state, rebound/rush).

Data comes directly from the NHL's public play-by-play API — no API key or
paid data source required.

## Pipeline

1. **`src/fetch_data.py`** — pulls play-by-play JSON from the NHL API and
   extracts every shot attempt (shot-on-goal, goal, missed-shot, blocked-shot)
   into flat rows, tracking the running score and the immediately preceding
   event (any type) as it goes, so each shot carries rebound/rush context.
   Includes automatic retries on connection resets, and checkpoints progress
   to disk after each date fetched — a large multi-hundred-game pull can
   survive a network blip partway through instead of losing everything.
2. **`src/features.py`** — engineers the spatio-temporal features:
   - *Spatial:* shot distance and angle to the net (from rink x/y coordinates)
   - *Temporal:* elapsed game time, period, score differential, strength
     state (5v5, power play, etc.), and rebound/rush flags — all **at the
     moment of the shot**
3. **`src/model.py`** — trains a gradient-boosted classifier (XGBoost) **and**
   a logistic regression baseline on the identical train/test split, with
   standard evaluation metrics (ROC-AUC, log loss, Brier score) for both, so
   the two can be compared head-to-head rather than judged in isolation.
4. **`src/visualize.py`** — plots shot locations colored by predicted xG, and
   a cumulative-xG-over-game-time chart showing when each team generated
   their scoring chances.
5. **`src/finishing_skill.py`** — ranks players and teams by actual goals
   scored vs. *out-of-fold* expected goals, to find shooters who consistently
   outperform (or underperform) their shot quality.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook nhl_xg_model.ipynb
```

The notebook defaults to a 60-day window (~450 games, ~47,000 shots), which
takes about 4 minutes to fetch on a fresh run. Results are cached to
`data/cache/` afterward, so re-running the notebook loads instantly instead
of re-fetching. Set `FORCE_REFETCH = True` to pull fresh data, or widen
`NUM_DAYS` for an even bigger sample — a full season is roughly 180 days
and takes 30+ minutes, so that's best kicked off as a background script
(see `src/fetch_data.py`'s `checkpoint_path` option) rather than run inline.

## Results (60-day sample: 46,898 shots, ~450 games, Jan 1 - Feb 29, 2024)

| Model | ROC-AUC | Log loss | Brier score |
|---|---|---|---|
| XGBoost | 0.8198 | 0.1662 | 0.0445 |
| Logistic Regression | 0.8201 | 0.1668 | 0.0447 |

Goal rate in sample: ~5.1% (consistent with real NHL shooting percentages).
Rebound shots: 3.6% of attempts. Rush shots: 1.9% of attempts.

**The logistic regression baseline is not worse than XGBoost here** — the two
are statistically indistinguishable, and this held up after quadrupling the
sample size from the original 3-day test. That's a legitimate result, not a
bug: with a handful of well-chosen features (distance, angle, game time,
score state, strength state, rebound/rush), the relationship to goal
probability is fairly smooth and close to linear in log-odds space, which is
exactly what logistic regression is built to capture. Gradient boosting's
advantage shows up more with larger feature sets or messier, more
interaction-heavy relationships. Reporting this honestly (rather than
defaulting to "the fancier model wins") is itself the point of including
the baseline.

### Finishing skill: goals above expected

Aggregating actual goals vs. out-of-fold xG by player surfaces exactly the
players known for elite finishing — e.g., in this sample, Auston Matthews
scored 26 goals on 14.5 expected across 202 shots (+11.5 goals above
expected), consistent with his real-world reputation as one of the league's
best shooters. That the analysis independently recovers a well-known result
is a good sanity check that the underlying model and methodology are sound,
not just fitting noise.

This uses **out-of-fold** predictions (`cross_val_predict`), not the model's
in-sample predictions — a model evaluated on data it was trained on partially
"knows" which shots went in, which quietly shrinks the apparent skill gap
between good and bad finishers.

## Why this project

Built to close a specific skills gap: hands-on **spatio-temporal modeling**
combining spatial (location) and temporal (game-time/context) features into
a single predictive model — the same class of problem as player-tracking-based
performance modeling in professional sports analytics, just applied to public
shot-event data instead of proprietary tracking data.

## Possible extensions

- Pull a full season (`NUM_DAYS=180`+) for an even more robust model — the
  pipeline already supports this via checkpointed fetching.
- Add more granular shot-context features (screened shots, one-timers) with
  a richer text-parsing pass.
- Extend the finishing-skill analysis to goaltenders (goals-saved-above-
  expected — the defensive mirror of this same idea).
