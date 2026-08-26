# 🏈 NFL Projection Model — 2026

Player yardage and touchdown projections built on public play-by-play, with an
explicit model of what happens when a coaching staff changes.

Ten head-coaching jobs turned over before the 2026 season, so "what this team did
last year" is the wrong starting point for a third of the league. This model
carries a play caller's measured scheme signature from their previous stop, then
bends it around the players they actually inherited.

```bash
uv sync
uv run streamlit run streamlit_app.py
```

First run downloads roughly 90 MB of nflverse data into `data/raw/` and caches it.

## What it produces

- **Projection board** — per-game receiving, rushing and passing yardage with full
  distributions, plus anytime-touchdown probability and the chance of clearing any
  yardage line. Reported two ways: *if active*, and *expected* with availability
  priced in.
- **Team scouting** — projected 2026 identity against league average, in plain
  language, with the reasoning shown: whose scheme it came from and how much weight
  it carries.
- **Signature concepts** — the calls each team runs more than the league does, with
  what they produced. The 2025 Rams ran under-center play-action deep shots off
  motion at 2.8× league rate for 15.3 yards a play; Miami's shotgun jet-motion
  outside zone ran at 4.5×.
- **Matchup** — where one team's offensive structure meets the other's defensive
  habits.
- **Game predictions** — independent margin, total and win probability per game
  from opponent-adjusted EPA ratings, shown next to the market line. It beats the
  naive baselines and loses to the closing line; see below.
- **Scheme explorer** — all 32 projected fingerprints in one sortable table.

## Coaching changes are configuration, not code

`data/coaching_2026.yaml` holds every staff assignment and where each coach came
from. It is meant to be edited — coaching news moves faster than any data feed,
and the aggregators contradicted each other while this was assembled. Each team
carries a `confidence` flag that controls how far the model may move off last
season's baseline. Fix a line, rerun, and the projections change.

## Validation

`scripts/backtest.py` holds out 2025 completely and projects it from 2022–2024.

| | bias | correlation |
| --- | --- | --- |
| Targets | +0.04 | 0.54 |
| Carries | +0.22 | 0.72 |
| Receiving yards | +0.26 | 0.50 |
| Rushing yards | +0.98 | 0.67 |
| Scrimmage yards | +1.24 | 0.54 |
| Anytime TD | +1.6 pts | — |

The backtest is what forced the availability model: projecting every listed player
as if he dresses every week overstated volume by 25–40%, because 46% of projected
player-games ended with no touches at all.

`scripts/backtest_games.py` validates the game model over 544 held-out games:

| | model | market |
| --- | --- | --- |
| Margin MAE | 10.26 | **9.67** |
| Straight-up winners | 64.7% | **68.4%** |

The market wins on every measure, and betting the model's disagreements lost
money in backtest (48.7% ATS against a 52.4% break-even). It clears the naive
baselines comfortably, so it knows something — it just knows less than the
closing line. Read a large edge as "the model is missing news," not as a signal.

## Layout

```
nflproj/
  data.py          nflverse fetch + cache
  schemes.py       play-by-play -> coordinator tendency vectors
  coaches.py       staff registry + scheme-transfer model
  personnel.py     depth chart, QB profiles, personnel constraints
  usage.py         target/carry/goal-line shares
  availability.py  probability a player is on the field
  projections.py   volume, bootstrapped yardage, TD simulation
  board.py         assembles full projection boards
  playbook.py      situational tendencies and signature concepts
  report.py        scouting language
scripts/
  backtest.py           2025 holdout validation
  build_projections.py  season-long CSV exports
```

`METHOD.md` documents the modelling in detail, including what the model does not do.

## Data

All public, via [nflverse](https://github.com/nflverse/nflverse-data): play-by-play
and FTN charting (2022–2025), snap counts, depth charts, rosters, injury reports,
and schedules with closing market lines.
