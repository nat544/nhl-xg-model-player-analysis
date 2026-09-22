# NHL Expected Goals (xG) Model — Shot Quality

A shot-quality ("expected goals") model for NHL shots, predicting an outcome (did this shot become a goal?)
from *where* on the ice a shot was taken combined with *when* in the game it
happened and the context at that moment (score, strength state, rebound/rush).

Data comes directly from the NHL's public play-by-play API

## Pipeline

1. **`src/fetch_data.py`** — pulls play-by-play JSON from the NHL API and
   extracts every shot attempt (shot-on-goal, goal, missed-shot, blocked-shot)
   into flat rows, tracking the running score and the immediately preceding
   event (any type) as it goes.
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
takes about 4 minutes to fetch on a fresh run. 

## Results (60-day sample: 46,898 shots, ~450 games, Jan 1 - Feb 29, 2024)

| Model | ROC-AUC | Log loss | Brier score |
|---|---|---|---|
| XGBoost | 0.8198 | 0.1662 | 0.0445 |
| Logistic Regression | 0.8201 | 0.1668 | 0.0447 |

Goal rate in sample: ~5.1% (consistent with real NHL shooting percentages).
Rebound shots: 3.6% of attempts. Rush shots: 1.9% of attempts.

**The logistic regression baseline is comparable to XGBoost here** — the two
are statistically very similar. That's a legitimate result, using well-selected features (distance, angle, game time,
score state, strength state, rebound/rush), the relationship to goal
probability is fairly smooth and close to linear in log-odds space, which is what logistic regression captures.

### Finishing skill: goals above expected

Aggregating actual goals vs. out-of-fold xG by player surfaces exactly the
players known for elite finishing — e.g., in this sample, Auston Matthews
scored 26 goals on 14.5 expected across 202 shots (+11.5 goals above
expected), consistent with his real-world reputation as one of the league's
best shooters. That the analysis independently recovers a well-known result
is a good sanity check that the underlying model and methodology are sound.

This uses **out-of-fold** predictions (`cross_val_predict`).


