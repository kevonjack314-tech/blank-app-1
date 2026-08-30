"""Longshot props: the far right tail, priced honestly.

A lotto play is a line a player clears only when the game goes his way - a
receiver projected for fifty yards, bet to reach a hundred. The market prices
these at long odds and they lose most weeks by design. What decides whether
they are worth taking is not whether the player is good; it is the *shape* of
his distribution, which is a different question from its mean and is where a
model can actually say something a price cannot.

Three findings shaped this module, and the first two overturned what the work
started out assuming.

**Doubling a projection gets much harder as the projection grows, and the
effect is enormous.** On the 2025 holdout a receiver projected around 27 yards
reached twice that 18.3% of the time; one projected around 78 yards did it
0.9% of the time. In fair-odds terms that is +446 against +9900. Any statement
about longshots that does not condition on the projection level is measuring
that gradient and nothing else.

**Explosiveness looked like it made the tail *thinner*, and that was an
artifact.** Quartiling receivers by their share of catches of twenty-plus yards
appeared to show the most explosive quartile doubling its own average less
often than the least (6.8% against 10.6%). That comparison is confounded:
explosive receivers have higher averages, so "twice his average" is a much
larger number of yards for them. Holding the projection fixed reverses it - at
20-40 projected yards, explosive receivers double 22.5% of the time against
13.1% for the rest (z = 3.9), and depth of target orders the same way (12.4% at
the shallowest third, 22.4% at the deepest). So the tail is a property of how a
player gets his yards, but only visibly so among the players a lotto ticket is
actually written on. Above about sixty projected yards the effect washes out.

**The simulator's own tail is too fat above forty projected yards.** Comparing
the joint simulation's P(twice the projection) against the holdout: at around
66 projected receiving yards the simulator said 10.5% where the season says
roughly 5%, and at 71 projected rushing yards 7.0% against roughly 3.5%. Below
forty it is close to right. A bootstrap of per-touch outcomes has no mechanism
that stops a good day compounding, and reality does. So the simulation is not
trusted on its own out here: it is blended with a tail curve fitted directly to
what happened, and the curve takes over as the multiple grows.

None of this is a claim to beat a price. There are no book odds in this data,
so what is reported is the model's probability, the fair price it implies, and
how a player's own tail compares with a generic player at the same projection.
Whether that beats what a book is offering is the user's half of the work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .picks import (american_to_probability, expected_value, kelly_fraction,
                    probability_to_american)

# Logistic tail regressions, fitted on the 2025 holdout over multiples of
# 1.25x to 3x. The response is P(actual >= multiple * projection); the features
# are log(multiple), its square, log(projection / 40) and their interaction.
# The interaction carries the level gradient described above and is the largest
# term in every fit.
#
# ``burst_x_log_mult`` is present only where it was supported: receiving and
# scrimmage yards. The same term fitted on rushing was indistinguishable from
# zero (z = -0.02), so it is not carried there - a running back's long run does
# not seem to be predictable from how often he has broken one before, at least
# not at the sample sizes available.
TAIL_COEFFICIENTS = {
    "rec_yards": {
        "intercept": -0.3732, "log_mult": -1.6080, "log_mult_sq": -1.3645,
        "log_level": 0.1345, "log_mult_x_level": -2.4294,
        "burst_x_log_mult": 1.0242,
    },
    "rush_yards": {
        "intercept": -0.8813, "log_mult": -1.0505, "log_mult_sq": -1.6217,
        "log_level": 0.4861, "log_mult_x_level": -2.0315,
    },
    "scrimmage_yards": {
        "intercept": -0.4853, "log_mult": -1.0452, "log_mult_sq": -1.5646,
        "log_level": 0.1203, "log_mult_x_level": -2.3019,
        "burst_x_log_mult": 0.4968,
    },
}

# The projection the level term is measured against.
TAIL_REFERENCE_LEVEL = 40.0

# League explosive rates, used to centre the burst term. A player at the league
# rate contributes nothing; the term is the log of his ratio to it.
LEAGUE_EXPLOSIVE_RATE = {"rec_yards": 0.138, "scrimmage_yards": 0.138}
BURST_RATIO_CLIP = (0.25, 4.0)

# How much of the answer the simulation keeps as the line moves out. At the
# projection itself the simulation is the better estimate - it knows the
# opponent, the game total, the player's usage and who else is on the field,
# none of which the tail curve sees. By twice the projection it has been
# measured too fat, so it holds a quarter of the weight and the fitted curve
# carries the rest.
SIM_WEIGHT_SPAN = 1.0
SIM_WEIGHT_FLOOR = 0.25

# Outside this range the fit is extrapolating past its data and says so.
FITTED_MULTIPLE_RANGE = (1.25, 3.0)

# What makes a play a lotto play rather than an ordinary prop.
MIN_LOTTO_ODDS = 250.0        # fair price of +250 or longer
MIN_LOTTO_PROBABILITY = 0.03  # below this it is a raffle ticket, not a bet
MIN_PROJECTION = {"rec_yards": 20.0, "rush_yards": 20.0, "scrimmage_yards": 25.0}

# Quarterback passing yards are deliberately absent. There is no fitted tail
# curve for them, so a longshot would be priced off the simulation alone -
# which is the thing measured too fat out here - and the passing projection
# underneath it only just matches the trivial baseline of a passer's own prior
# yards per game (correlation +0.07 against +0.18, MAE 61.5 against 61.1 on the
# 2025 holdout). Pricing longshots on top of that would be stacking an
# extrapolation on a weak estimate. The market is left out until the passing
# line earns it.

# The multiples a book tends to hang a longshot at.
LOTTO_MULTIPLES = (1.5, 1.75, 2.0, 2.5)

STAT_LABELS = {"rec_yards": "Receiving yards", "rush_yards": "Rushing yards",
               "scrimmage_yards": "Scrimmage yards", "pass_yards": "Passing yards",
               "total_td": "Anytime TD"}


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def _expit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0))))


def tail_probability(stat: str, projection: float, line: float,
                     burst: float | None = None) -> float | None:
    """P(actual >= line) from the fitted tail curve, ignoring the simulation.

    ``burst`` is the player's own explosive rate - his share of catches gaining
    twenty or more yards - not a ratio. It is converted to a ratio against the
    league rate here so callers can pass what ``gameplan.explosive_profile``
    measures. Passing ``None`` gives the generic player at that projection,
    which is what ``baseline`` means everywhere below.
    """
    coef = TAIL_COEFFICIENTS.get(stat)
    if coef is None or not np.isfinite(projection) or projection <= 0:
        return None
    mult = float(line) / float(projection)
    if mult <= 0:
        return None
    lm = np.log(mult)
    lu = np.log(float(projection) / TAIL_REFERENCE_LEVEL)
    x = (coef["intercept"] + coef["log_mult"] * lm + coef["log_mult_sq"] * lm ** 2
         + coef["log_level"] * lu + coef["log_mult_x_level"] * lm * lu)
    if "burst_x_log_mult" in coef and burst is not None and np.isfinite(burst):
        lg = LEAGUE_EXPLOSIVE_RATE.get(stat, 0.138)
        ratio = float(np.clip(burst / lg, *BURST_RATIO_CLIP))
        x += coef["burst_x_log_mult"] * np.log(ratio) * lm
    return _expit(x)


def sim_weight(multiple: float) -> float:
    """How much of the blend the simulation keeps at this multiple."""
    w = 1.0 - (float(multiple) - 1.0) / SIM_WEIGHT_SPAN
    return float(np.clip(w, SIM_WEIGHT_FLOOR, 1.0))


def blended_probability(sim_p: float | None, stat: str, projection: float,
                        line: float, burst: float | None = None) -> float | None:
    """The simulation and the fitted tail curve, combined in log-odds.

    Where no curve exists for a statistic the simulation is returned unchanged,
    and the caller is told so through ``calibrated`` in the board below.
    """
    fitted = tail_probability(stat, projection, line, burst)
    if fitted is None:
        return sim_p
    if sim_p is None or not np.isfinite(sim_p):
        return fitted
    w = sim_weight(float(line) / float(projection)) if projection > 0 else 1.0
    return _expit(w * _logit(sim_p) + (1.0 - w) * _logit(fitted))


def _burst_lookup(explosive: pd.DataFrame | None) -> dict:
    """Player id -> explosive rate, from ``gameplan.explosive_profile``."""
    if explosive is None or explosive.empty:
        return {}
    rec = explosive[explosive["kind"] == "rec"] if "kind" in explosive else explosive
    return dict(zip(rec["player_id"], rec["explosive"]))


def player_lotto_lines(game, player: str, burst: float | None = None,
                       multiples=LOTTO_MULTIPLES,
                       conditional: bool = True) -> pd.DataFrame:
    """Every longshot line for one player, at each multiple of his projection.

    ``conditional`` prices the bet as "given he plays", which is how a book
    treats it - a scratch is a void, not a loss. The unconditional probability
    is reported alongside so the difference is visible.
    """
    meta = game.meta.get(player)
    if meta is None:
        return pd.DataFrame()
    mask = game.active_mask(player) if conditional else None

    rows = []
    for stat in ("rec_yards", "rush_yards", "scrimmage_yards"):
        s = game.stat(player, stat)
        if s is None or len(s) == 0:
            continue
        vals = s[mask] if mask is not None else s
        if len(vals) == 0:
            continue
        projection = float(np.mean(vals))
        if projection < MIN_PROJECTION.get(stat, 20.0):
            continue
        b = burst if stat in TAIL_COEFFICIENTS and "burst_x_log_mult" in TAIL_COEFFICIENTS[stat] else None
        for mult in multiples:
            line = _round_line(projection * mult, stat)
            actual_mult = line / projection
            sim_p = float((vals > line).mean())
            p = blended_probability(sim_p, stat, projection, line, b)
            if p is None or p < MIN_LOTTO_PROBABILITY:
                continue
            fair = probability_to_american(p)
            if fair < MIN_LOTTO_ODDS:
                continue
            base = tail_probability(stat, projection, line, None)
            rows.append({
                "player": player, "team": meta.get("team"),
                "position": meta.get("position"), "stat": stat,
                "market": STAT_LABELS.get(stat, stat), "line": line,
                "projection": round(projection, 1),
                "multiple": round(actual_mult, 2),
                "p_model": p, "p_sim": sim_p,
                "p_fitted": tail_probability(stat, projection, line, b),
                "p_baseline": base,
                "edge_ratio": (p / base) if base and base > 0 else np.nan,
                "fair_odds": fair,
                "p_active": float(meta.get("p_active", 1.0)),
                "calibrated": stat in TAIL_COEFFICIENTS,
                "extrapolated": not (FITTED_MULTIPLE_RANGE[0] <= actual_mult
                                     <= FITTED_MULTIPLE_RANGE[1]),
            })
    return pd.DataFrame(rows)


def _round_line(value: float, stat: str) -> float:
    """Books hang round numbers on yardage: 75.5, 100.5, 124.5."""
    step = 25.0 if stat == "pass_yards" else (25.0 if value >= 90 else 5.0)
    return float(np.round(value / step) * step) + 0.5


def multi_touchdown_lines(game, player: str, conditional: bool = True) -> pd.DataFrame:
    """Two or more touchdowns - the other longshot every book hangs.

    Priced from the simulation alone. There is no ratio to take a multiple of
    here, so the tail curve does not apply; what stands behind this number is
    the anytime-touchdown calibration, which the backtest put within about a
    point across most of its range.
    """
    meta = game.meta.get(player)
    if meta is None:
        return pd.DataFrame()
    s = game.stat(player, "total_td")
    if s is None or len(s) == 0:
        return pd.DataFrame()
    mask = game.active_mask(player) if conditional else None
    vals = s[mask] if mask is not None else s
    if len(vals) == 0:
        return pd.DataFrame()
    rows = []
    for k in (2, 3):
        p = float((vals >= k).mean())
        if p < MIN_LOTTO_PROBABILITY:
            continue
        fair = probability_to_american(p)
        if fair < MIN_LOTTO_ODDS:
            continue
        rows.append({
            "player": player, "team": meta.get("team"),
            "position": meta.get("position"), "stat": "total_td",
            "market": f"{k}+ touchdowns", "line": float(k) - 0.5,
            "projection": round(float(np.mean(vals)), 2), "multiple": np.nan,
            "p_model": p, "p_sim": p, "p_fitted": np.nan, "p_baseline": np.nan,
            "edge_ratio": np.nan, "fair_odds": fair,
            "p_active": float(meta.get("p_active", 1.0)),
            "calibrated": False, "extrapolated": False,
        })
    return pd.DataFrame(rows)


def lotto_board(games: list, explosive: pd.DataFrame | None = None,
                top_n: int = 40, min_active: float = 0.70,
                max_per_player: int = 2, include_touchdowns: bool = True,
                conditional: bool = True) -> pd.DataFrame:
    """Longshot plays across a slate, ranked by tail shape rather than by size.

    The ranking is ``edge_ratio``: how much likelier this player is to reach the
    line than a generic player projected for the same total. That is the only
    edge statement this data supports. Ranking by probability instead would
    just re-sort the board by who has the lowest projection, since a small
    number is the easiest one to double - which is the trap the whole module
    exists to avoid.

    Touchdown longshots have no baseline to compare against, so they sort last
    within the board on probability alone.
    """
    burst = _burst_lookup(explosive)
    frames = []
    for g in games:
        for player, meta in g.meta.items():
            if float(meta.get("p_active", 1.0)) < min_active:
                continue
            pid = meta.get("player_id")
            f = player_lotto_lines(g, player, burst.get(pid), conditional=conditional)
            if not f.empty:
                frames.append(f)
            if include_touchdowns:
                t = multi_touchdown_lines(g, player, conditional=conditional)
                if not t.empty:
                    frames.append(t)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["_rank"] = out["edge_ratio"].fillna(0.0)
    out = out.sort_values(["_rank", "p_model"], ascending=False)
    if max_per_player:
        out = out.groupby("player", sort=False).head(max_per_player)
    return out.drop(columns=["_rank"]).head(top_n).reset_index(drop=True)


def price_lotto(board: pd.DataFrame, odds: dict | None = None) -> pd.DataFrame:
    """Attach expected value where a real price is supplied.

    ``odds`` maps ``(player, market, line)`` to an American price. Rows with no
    price keep the fair number and nothing else - a probability without a price
    is not an edge, and the module will not pretend otherwise.
    """
    if board.empty:
        return board
    out = board.copy()
    prices, evs, kellys, edges = [], [], [], []
    for r in out.itertuples():
        key = (r.player, r.market, float(r.line))
        price = (odds or {}).get(key)
        if price is None:
            prices.append(np.nan); evs.append(np.nan)
            kellys.append(np.nan); edges.append(np.nan)
            continue
        prices.append(float(price))
        evs.append(expected_value(float(r.p_model), float(price)))
        kellys.append(kelly_fraction(float(r.p_model), float(price)))
        edges.append(float(r.p_model) - american_to_probability(float(price)))
    out["price"] = prices
    out["ev_per_100"] = evs
    out["kelly"] = kellys
    out["prob_edge"] = edges
    return out
