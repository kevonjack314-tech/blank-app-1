"""Route charting: a better-behaved restatement of depth of target.

nflverse charts the route run on each charted target - twelve concepts once the
vocabulary is made consistent - and the league table it produces is exactly what
you would hope for. Over 68,063 charted targets from 2022-2025:

| route | aDOT | catch rate | yards/target | EPA |
| --- | --- | --- | --- | --- |
| post | 20.3 | 0.53 | 12.19 | +0.527 |
| corner | 19.7 | 0.45 | 10.36 | +0.438 |
| wheel | 15.0 | 0.48 | 10.27 | +0.379 |
| in/dig | 12.2 | 0.60 | 9.34 | +0.317 |
| go | 24.5 | 0.36 | 10.13 | +0.309 |
| slant | 6.3 | 0.68 | 7.32 | +0.236 |
| cross/drag | 5.0 | 0.71 | 7.19 | +0.167 |
| hitch/curl | 6.3 | 0.74 | 6.83 | +0.162 |
| out | 5.8 | 0.71 | 6.23 | +0.146 |
| angle | 3.1 | 0.76 | 6.37 | +0.064 |
| flat/swing | -1.6 | 0.82 | 5.10 | -0.032 |
| screen | -3.0 | 0.88 | 5.56 | -0.050 |

A receiver's route mix is one of the most persistent things about him:
year-over-year r = 0.860 for his mix priced at league rates, against 0.482 for
his own yards per target. That is genuinely useful description.

It is not, however, new information for the projection, because the model
already carries depth of target - and route mix and aDOT turn out to be near
substitutes. Predicting a receiver's next-season yards per target over 341
consecutive receiver-seasons:

| predictors | multiple r |
| --- | --- |
| his own yards per target | 0.482 |
| + depth of target | 0.534 |
| + route mix | **0.541** |

Seven thousandths. And once route mix enters, the aDOT coefficient collapses
from 0.107 to 0.018 - they are measuring the same thing, and aDOT is measured on
every target rather than only the charted ones. The obvious escape route, that
screens and flat routes are short in a way aDOT flattens, was tested against
yards after catch and adds +0.003 on the same design.

So routes are carried as description and as matchup material - the concept menu
an offense actually calls, which is the coordinator-level content - and nothing
here multiplies a projection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import normalize_team

# The charting vocabulary changed after 2022: HITCH became HITCH/CURL, OUT split
# into QUICK OUT and DEEP OUT, CROSS became SHALLOW CROSS/DRAG, IN became
# IN/DIG, ANGLE became TEXAS/ANGLE. Comparing raw labels across seasons would
# read a taxonomy change as a change in play-calling, so everything is mapped to
# one vocabulary first.
CANONICAL_ROUTES = {
    "HITCH": "hitch/curl", "HITCH/CURL": "hitch/curl",
    "OUT": "out", "QUICK OUT": "out", "DEEP OUT": "out",
    "IN": "in/dig", "IN/DIG": "in/dig",
    "CROSS": "cross/drag", "SHALLOW CROSS/DRAG": "cross/drag",
    "ANGLE": "angle", "TEXAS/ANGLE": "angle",
    "FLAT": "flat/swing", "SWING": "flat/swing",
    "GO": "go", "POST": "post", "CORNER": "corner", "SLANT": "slant",
    "SCREEN": "screen", "WHEEL": "wheel",
}

# League production per concept, from 68,063 charted targets 2022-2025.
ROUTE_RATES = {
    "post":       {"adot": 20.3, "catch": 0.529, "ypt": 12.19, "epa": +0.527},
    "corner":     {"adot": 19.7, "catch": 0.452, "ypt": 10.36, "epa": +0.438},
    "wheel":      {"adot": 15.0, "catch": 0.483, "ypt": 10.27, "epa": +0.379},
    "in/dig":     {"adot": 12.2, "catch": 0.597, "ypt": 9.34,  "epa": +0.317},
    "go":         {"adot": 24.5, "catch": 0.359, "ypt": 10.13, "epa": +0.309},
    "slant":      {"adot": 6.3,  "catch": 0.677, "ypt": 7.32,  "epa": +0.236},
    "cross/drag": {"adot": 5.0,  "catch": 0.707, "ypt": 7.19,  "epa": +0.167},
    "hitch/curl": {"adot": 6.3,  "catch": 0.744, "ypt": 6.83,  "epa": +0.162},
    "out":        {"adot": 5.8,  "catch": 0.707, "ypt": 6.23,  "epa": +0.146},
    "angle":      {"adot": 3.1,  "catch": 0.762, "ypt": 6.37,  "epa": +0.064},
    "flat/swing": {"adot": -1.6, "catch": 0.820, "ypt": 5.10,  "epa": -0.032},
    "screen":     {"adot": -3.0, "catch": 0.881, "ypt": 5.56,  "epa": -0.050},
}

# Measured, and the reason none of it is wired in.
ROUTE_MIX_PERSISTENCE = 0.860
ROUTE_MIX_GAIN_OVER_ADOT = 0.007
APPLIED_TO_PROJECTION = False

MIN_CHARTED_TARGETS = 20


def canonicalize(routes: pd.Series) -> pd.Series:
    """Map a season's charting labels onto the one vocabulary."""
    s = routes.astype(str).str.strip().str.upper()
    return s.map(CANONICAL_ROUTES)


def attach_routes(plays: pd.DataFrame, participation: pd.DataFrame) -> pd.DataFrame:
    """Put the charted route on each target, canonicalised.

    Only the targeted receiver's route is charted, so this describes the routes
    a player is *thrown* on rather than every route he runs.
    """
    if participation is None or participation.empty or "route" not in participation.columns:
        return pd.DataFrame()
    part = participation.copy()
    key = "nflverse_game_id" if "nflverse_game_id" in part.columns else "game_id"
    part = part[[key, "play_id", "route"] + (["season"] if "season" in part.columns else [])]
    part = part.rename(columns={key: "game_id"})
    part["route"] = canonicalize(part["route"])
    part = part[part["route"].notna()]
    if part.empty:
        return pd.DataFrame()

    on = ["game_id", "play_id"] + (["season"] if "season" in part.columns
                                   and "season" in plays.columns else [])
    tgt = plays[plays["receiver_player_id"].notna()]
    return tgt.merge(part, on=on, how="inner")


def player_route_profile(routed: pd.DataFrame, player_id: str | None = None,
                         seasons: tuple | None = None,
                         min_targets: int = MIN_CHARTED_TARGETS) -> pd.DataFrame:
    """A receiver's concept menu: what he is thrown, and what it produced."""
    if routed is None or routed.empty:
        return pd.DataFrame()
    d = routed
    if seasons and "season" in d.columns:
        d = d[d["season"].isin(seasons)]
    if player_id:
        d = d[d["receiver_player_id"] == player_id]
    if d.empty or len(d) < min_targets:
        return pd.DataFrame()

    caught = d["complete_pass"].fillna(0) > 0
    d = d.assign(_yards=np.where(caught, d["yards_gained"], 0.0))
    g = (
        d.groupby("route")
        .agg(targets=("_yards", "size"), yards=("_yards", "sum"),
             catch_rate=("complete_pass", "mean"), adot=("air_yards", "mean"),
             epa=("epa", "mean"))
        .reset_index()
    )
    g["share"] = g["targets"] / g["targets"].sum()
    g["yards_per_target"] = g["yards"] / g["targets"].clip(lower=1)
    g["league_ypt"] = g["route"].map(lambda r: ROUTE_RATES.get(r, {}).get("ypt", np.nan))
    g["vs_league"] = g["yards_per_target"] - g["league_ypt"]
    return g.sort_values("targets", ascending=False).reset_index(drop=True)


def team_route_menu(routed: pd.DataFrame, team: str, seasons: tuple | None = None,
                    top_n: int = 8) -> pd.DataFrame:
    """The concepts an offense calls more than the league does."""
    if routed is None or routed.empty or "posteam" not in routed.columns:
        return pd.DataFrame()
    d = routed
    if seasons and "season" in d.columns:
        d = d[d["season"].isin(seasons)]
    if d.empty:
        return pd.DataFrame()
    league = d.groupby("route").size()
    league = league / league.sum()

    t = d[d["posteam"] == normalize_team(team)]
    if t.empty:
        return pd.DataFrame()
    caught = t["complete_pass"].fillna(0) > 0
    t = t.assign(_yards=np.where(caught, t["yards_gained"], 0.0))
    g = (
        t.groupby("route")
        .agg(targets=("_yards", "size"), yards_per_target=("_yards", "mean"),
             epa=("epa", "mean"), adot=("air_yards", "mean"))
        .reset_index()
    )
    g["share"] = g["targets"] / g["targets"].sum()
    g["league_share"] = g["route"].map(league)
    g["vs_league"] = g["share"] / g["league_share"].replace(0, np.nan)
    return g.sort_values("vs_league", ascending=False).head(top_n).reset_index(drop=True)


def route_implied_depth(profile: pd.DataFrame) -> float:
    """The depth a player's concept menu implies, at league rates.

    Reported next to his measured aDOT. A gap between the two says his offense
    is using him at a different depth than the concepts alone would suggest -
    interesting to read, and not a projection input: the two carry the same
    information and aDOT is measured on every target rather than the charted
    subset.
    """
    if profile is None or profile.empty:
        return float("nan")
    w = profile["share"].to_numpy(float)
    a = profile["route"].map(lambda r: ROUTE_RATES.get(r, {}).get("adot", np.nan)).to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(w)
    return float((a[ok] * w[ok]).sum() / w[ok].sum()) if w[ok].sum() > 0 else float("nan")
