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

Opens at `http://localhost:8501`. The first run downloads roughly 140 MB of
public nflverse data into `data/raw/` and caches it; later starts take about ten
seconds.

## Deploying it

The app needs no secrets, no database and no API keys — every source is public —
so hosting is mostly a matter of picking somewhere to run it.

**Streamlit Community Cloud** (free, easiest). Push this repo to GitHub, then at
[share.streamlit.io](https://share.streamlit.io) point a new app at
`streamlit_app.py` on this branch. It reads `requirements.txt`. The first boot
downloads the nflverse cache and shows progress while it does.

**Any container host** (Render, Railway, Fly.io, Cloud Run, your own box):

```bash
docker build -t nfl-projections .
docker run -p 8501:8501 nfl-projections
```

The image warms the data cache at build time so the first visitor is not left
waiting. Comment out that line in the `Dockerfile` to fetch lazily instead.

### What to watch when hosting

- **Memory.** A cold start peaks around 620 MB, which fits the 1 GB free tier
  but not with much room. Loading play-by-play season by season and storing
  repeated strings as categories is what keeps it there; if you widen
  `HISTORY_SEASONS`, re-measure before deploying.
- **Ephemeral disk.** Most free hosts wipe the filesystem on restart, so the
  140 MB download repeats on every cold boot. A small persistent volume mounted
  at `data/` removes that.
- **Refreshing during the season.** The cache never expires on its own. Re-run
  `python -c "from nflproj import data; data.sync_all(projection_season=2026)"`
  (or redeploy) to pull new results, depth charts and injury reports.
- **It is a single-process app.** Streamlit re-runs the script per interaction
  and `@st.cache_resource` keeps the model in memory, so one container serves
  many readers fine — but every viewer shares that one cache.

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
- **Players** — every projected player by position group, with distributions,
  percentiles and a full line sheet.
- **Best picks** — the model's highest-confidence sides across a week, with an
  expected-value calculator once you enter a price.
- **Parlay builder** — legs priced against one shared simulation, so correlation
  is carried rather than assumed away. Same-team stacks come out 15–25% likelier
  than a naive calculator says; cross-game legs correctly show no lift.
- **War room** — the weekly coordinator view: where two schemes collide, which
  players the matchup swings, and what changed since last week (depth-chart moves,
  injury report, snap-share trend, your own notes).
- **Coverage** — man/zone rates and coverage shells per defence, how each offence
  fares against each structure, route menus and personnel groupings, from nflverse
  participation charting.
- **Situational context** — wind, venue and divisional status feed both the game
  and player models; travel and body clock are shown but not applied, because
  they no longer predict anything (see below).
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

## What drives a projection

Volume dominates, so the usage signals matter most. Year-over-year stability,
measured across 2022–2025, is what decides how much weight each carries:

| signal | stability |
| --- | --- |
| Share of intended air yards | **0.727** |
| Avg separation | 0.547 |
| Yards before contact/carry (blocking) | 0.433 |
| Pressure allowed | 0.395 |
| Yards after contact/carry (the back) | **0.106** |

That last row is why rushing is split into blocking and back rather than
projected as one yards-per-carry number: the line persists, broken tackles do
not.

## In-season updating

Before Week 1 the model runs on prior seasons and the depth chart. Once games are
played it rebuilds from them, and the app switches automatically.

How fast this season displaces last was measured, not guessed: the best weight on
season-to-date is `n / (n + K)` with K = 8 for targets and 4 for carries — so a
target share is 79% current by 30 targets, and a backfield settles faster still.

The bigger effect is at team level. A defence barely carries across seasons
(EPA allowed r = 0.11) but predicts its own second half from its first at
r = 0.32. Preseason, opponent multipliers span just 0.988–1.011; rebuilt through
week 12 they span **0.905–1.102**. Matchup adjustment goes from nearly inert to a
real term, which is also what makes X-factors and the war room bite once the
season starts.

## Narratives: tested, not assumed

Familiar football narratives were tested against closing lines back to 2006 —
revenge games, divisional rematches, byes, short weeks, primetime, letdown spots.
None move outcomes. **National Tight Ends Day**, a real league-promoted event,
shows a tight end target share of 0.2179 against 0.2173 on every other day
(t = 0.11); other October Sundays run slightly higher than the holiday itself.

That is not a claim that narratives don't matter to football. It is that
narratives *everyone already knows* are in the price, so a model adding them
double-counts.

What is not priced is information that has not propagated yet — first-team reps,
a coordinator's stated intent, a snap-count plan. `data/news_2026.yaml` is where
that goes, and the model also detects role changes on its own from depth charts,
injury reports and snap-share trends. Snap share is the most useful signal there,
because it moves before target share does.

## Parlays are priced on correlation, not independence

Multiplying leg probabilities assumes they are independent. Same-game legs never
are — a quarterback over his passing yards and his receiver over his receiving
yards are close to the same bet. Because the game is simulated once with shared
randomness, a parlay's probability is simply the share of simulations in which
every leg lands, and the naive figure is shown alongside so the correction is
visible.

Measured against real 2022–2025 same-game correlations, the dominant
relationship lands close: QB passing yards with his WR1 simulates at +0.47
against a measured +0.51. Quarterback-versus-running-back and cross-sideline
pairs are directionally right but understated, so those slips price
conservatively.

**There is no odds feed here.** "Best picks" ranks confidence, not value — a 90%
leg at −1200 is still a bad bet. Enter a price and the app computes real expected
value and a Kelly stake.

## Travel, weather and divisional games

All four were tested over 7,276 games with a closing line (1999–2025) and
checked across three eras. `scripts/measure_context.py` reproduces it.

**Applied**: wind (−0.19 pts of total per mph above 8), indoor venue (+1.0),
divisional game (−0.91). Wind also reshapes play-calling — offenses throw less,
shorter, and complete fewer — so it moves player projections, not just totals.

**Not applied**: travel distance, time zones, body clock. The east-to-west
effect was worth −2.8 points against the spread in 1999–2009 and has decayed to
zero (−0.11, then +0.24). The famous West-Coast-team-at-1pm-Eastern body-clock
effect never shows up at all. These are computed and displayed for context, and
carry no weight in any number.

Cold weather does almost nothing on its own: about −0.009 points of total per
degree, so a 40-degree swing is worth a third of a point. Wind is the weather
variable that matters.

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
  coverage.py      man/zone, coverage shells, routes, personnel groupings
  blocking.py      offensive line vs running back decomposition
  kicking.py       field goal and extra point projections
  gamemodel.py     team ratings, margin/total/win probability
  joint.py         correlated whole-game simulation
  gameplan.py      scheme collisions and X-factor players
  news.py          hand-entered notes and detected role changes
  picks.py         player line sheets, best picks, odds arithmetic
  parlay.py        correlated parlay pricing
  venues.py        stadium geography, climate, travel and environment
  report.py        scouting language
scripts/
  backtest.py           2025 holdout validation
  backtest_games.py     game model vs the closing line
  measure_context.py    travel, weather and divisional effect tests
  build_projections.py  season-long CSV exports
```

`METHOD.md` documents the modelling in detail, including what the model does not do.

## Data

All public, via [nflverse](https://github.com/nflverse/nflverse-data): play-by-play
and FTN charting (2022–2025), snap counts, depth charts, rosters, injury reports,
and schedules with closing market lines.

**Preseason is never used.** Starters play a series and sit, most snaps go to
players who will not make the roster, and the play-calling is vanilla by design.
nflverse does not ship preseason play-by-play today, but the filter is enforced
regardless so a feed change cannot contaminate the model later.

**Postseason is off by default** — not because it is unserious, but because only
fourteen teams have any, which would hand extra sample to teams that were already
good. Flip `INCLUDE_POSTSEASON` in `config.py` to include it.

The depth chart is the one preseason-shaped input, and deliberately so: it states
intended role rather than measuring performance. It is also the most fragile
input in the model, since roles move through cutdowns.
