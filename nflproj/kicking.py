"""Field goal and extra point projections.

Distance is overwhelmingly the story. Fitted on 4,325 regular-season attempts
(2022-2025), the make probability is a clean logistic in distance:

    logit(make) = 5.915 - 0.0985 x distance

which puts a 25-yarder at 97%, a 45-yarder at 82%, and a 58-yarder at 55%.

Wind was tested and is **not** applied. Controlling for distance, the wind
coefficient is -0.017 per mph with a bootstrap t of -1.0 and a 95% interval
spanning zero: a 45-yard attempt at 20mph comes out at 76% against 81% calm,
which is directionally sensible but not distinguishable from noise on the 200
attempts available in high wind. Coaches also simply attempt shorter kicks in
wind, which absorbs part of the effect at the decision level rather than the
kick level.

Wind still reduces a kicker's output, just not through accuracy: it suppresses
scoring generally, which is already modelled at the drive level and flows
through to fewer attempts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PRIOR_STRENGTH, normalize_team

FG_INTERCEPT = 5.9147
FG_SLOPE = -0.09854
XP_MAKE_RATE = 0.956
XP_DISTANCE = 33

# Attempts per team-game and the distance distribution they come from.
LEAGUE_FG_ATTEMPTS = 2.25
FG_DISTANCE_MEAN = 38.7
FG_DISTANCE_SD = 10.7

KICKER_PRIOR_ATTEMPTS = 60.0   # regression strength for a kicker's own accuracy


def make_probability(distance: float | np.ndarray) -> float | np.ndarray:
    """Probability a field goal from this distance is good."""
    z = FG_INTERCEPT + FG_SLOPE * np.asarray(distance, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def kicker_history(plays: pd.DataFrame) -> pd.DataFrame:
    """Per-kicker accuracy above or below what distance alone predicts."""
    need = {"field_goal_attempt", "field_goal_result", "kicker_player_id", "yardline_100"}
    if not need.issubset(plays.columns):
        return pd.DataFrame()
    fg = plays[(plays["field_goal_attempt"].fillna(0) > 0)
               & plays["field_goal_result"].notna()
               & plays["kicker_player_id"].notna()].copy()
    if fg.empty:
        return pd.DataFrame()
    fg["distance"] = fg["yardline_100"] + 17
    fg["made"] = (fg["field_goal_result"] == "made").astype(float)
    fg["expected"] = make_probability(fg["distance"])
    g = (
        fg.groupby(["season", "kicker_player_id"])
        .agg(attempts=("made", "size"), made=("made", "sum"),
             expected=("expected", "sum"),
             avg_distance=("distance", "mean"))
        .reset_index().rename(columns={"kicker_player_id": "player_id"})
    )
    # Makes above expectation, per attempt: the kicker's own signal.
    g["fg_over_expected"] = (g["made"] - g["expected"]) / g["attempts"]
    return g


def project_kicker_accuracy(hist: pd.DataFrame, player_id: str | None,
                            anchor: int | None = None, halflife: float = 1.6) -> float:
    """Kicker skill as makes above expectation per attempt, heavily regressed.

    Kicking is noisy and a season is only about thirty attempts, so a kicker's
    own record moves the projection modestly.
    """
    if not player_id or hist is None or hist.empty:
        return 0.0
    h = hist[hist["player_id"] == player_id]
    if h.empty:
        return 0.0
    anchor = anchor if anchor is not None else int(h["season"].max())
    w = 0.5 ** ((anchor - h["season"]) / halflife)
    n = float((h["attempts"] * w).sum())
    if n <= 0:
        return 0.0
    observed = float((h["fg_over_expected"] * h["attempts"] * w).sum() / n)
    return float(observed * n / (n + KICKER_PRIOR_ATTEMPTS))


def project_kicker(
    *, player_id: str | None, name: str, team: str, hist: pd.DataFrame,
    volume: dict, env: dict | None = None, n_sims: int = 20000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Simulate a kicker's game: attempts, makes, and points.

    Attempts scale with how often the offence reaches scoring range without
    finding the end zone, which is why a good offence in a low-scoring
    environment is the ideal kicker spot.
    """
    rng = rng or np.random.default_rng()
    skill = project_kicker_accuracy(hist, player_id)

    # Drives that stall in range become attempts. Teams that score more
    # touchdowns convert some of those into scores instead, so attempts scale
    # with points but flatten as touchdowns rise.
    points = float(volume.get("implied_points", 22.5))
    tds = float(volume.get("expected_off_td", 2.3))
    exp_attempts = float(np.clip(LEAGUE_FG_ATTEMPTS * (points / 22.5) ** 0.55
                                 * (2.3 / max(tds, 0.6)) ** 0.30, 0.5, 5.0))
    if env:
        # Wind does not measurably change accuracy, but it does suppress
        # scoring, which slightly reduces trips into range.
        exp_attempts *= float(np.clip(1.0 + env.get("total_delta", 0.0) * 0.010, 0.85, 1.10))

    attempts = rng.poisson(exp_attempts, n_sims)
    xp_attempts = rng.poisson(max(tds * 0.94, 0.0), n_sims)

    made = np.zeros(n_sims, dtype=float)
    total = int(attempts.sum())
    if total:
        distances = np.clip(rng.normal(FG_DISTANCE_MEAN, FG_DISTANCE_SD, total), 18, 66)
        probs = np.clip(make_probability(distances) + skill, 0.01, 0.995)
        hits = (rng.random(total) < probs).astype(float)
        idx = np.concatenate([[0], np.cumsum(attempts)])
        csum = np.concatenate([[0.0], np.cumsum(hits)])
        made = csum[idx[1:]] - csum[idx[:-1]]

    xp_made = rng.binomial(xp_attempts, XP_MAKE_RATE)
    points_scored = made * 3 + xp_made

    return {
        "player": name, "team": team, "position": "K",
        "exp_attempts": exp_attempts, "skill_over_expected": skill,
        "samples": {
            "fg_attempts": attempts.astype(float), "fg_made": made,
            "xp_made": xp_made.astype(float), "kicking_points": points_scored,
        },
    }
