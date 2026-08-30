"""Player browsing, best picks, and odds arithmetic.

Two things this module is careful about.

There is no sportsbook feed here. Without prices, "best pick" cannot mean "the
biggest edge over the market" - it can only mean "what the model is most
confident about", which is a different and much weaker claim. Confidence is not
value: a 90% leg priced at -1200 is a bad bet. Where a price is supplied, real
expected value is computed and shown instead, and that is the number worth
acting on.

Second, the model's own calibration is the ceiling on all of this. Backtesting
put anytime-touchdown probability within about a point across most of its range
and yardage correlations between 0.53 and 0.82 depending on the statistic. That
is useful, not omniscient.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Positions grouped the way a person looks for a player.
POSITION_GROUPS = {
    "Quarterbacks": ["QB"],
    "Running backs": ["RB", "FB"],
    "Wide receivers": ["WR"],
    "Tight ends": ["TE"],
}

# Stats that make sense to price, per position.
STAT_MENU = {
    "QB": [("pass_yards", "Passing yards"), ("attempts", "Pass attempts"),
           ("completions", "Completions"), ("pass_td", "Passing TDs"),
           ("interceptions", "Interceptions"), ("rush_yards", "Rushing yards"),
           ("carries", "Carries"), ("total_td", "Anytime TD (rushing)")],
    "RB": [("rush_yards", "Rushing yards"), ("carries", "Carries"),
           ("rec_yards", "Receiving yards"), ("receptions", "Receptions"),
           ("targets", "Targets"), ("scrimmage_yards", "Scrimmage yards"),
           ("total_td", "Anytime TD")],
    "WR": [("rec_yards", "Receiving yards"), ("receptions", "Receptions"),
           ("targets", "Targets"), ("scrimmage_yards", "Scrimmage yards"),
           ("total_td", "Anytime TD")],
}
STAT_MENU["FB"] = STAT_MENU["RB"]
STAT_MENU["TE"] = STAT_MENU["WR"]

# Where a plausible line sits, as quantiles of the player's own simulated
# distribution. A book hangs a number near the middle, so lines are generated
# per player and per statistic rather than from a fixed grid.
#
# Quantiles rather than percentage offsets, because the two behave very
# differently across statistics. A line 25% above a receiver's yardage is a
# close call; 25% above a quarterback's pass attempts is nearly a lock, since
# attempts vary far less. Offsetting by percent therefore manufactured
# 85%-confidence quarterback unders and pushed every other position off the
# board. Quantiles put every market in the same probability range by
# construction.
LINE_QUANTILES = (0.35, 0.45, 0.55, 0.65)

# A line is only offered if its implied probability sits near a coin flip, which
# is where a book hangs one and where the model's own error matters least.
TARGET_BAND = (0.28, 0.72)

# Counting stats get whole- or half-number lines close to the projection.
COUNT_STATS = {"receptions", "targets", "carries", "attempts", "completions",
               "pass_td", "interceptions", "total_td"}

# Below these, a market is not offered at all.
MIN_PROJECTION = {
    "pass_yards": 90.0, "rush_yards": 15.0, "rec_yards": 15.0,
    "scrimmage_yards": 25.0, "receptions": 1.5, "targets": 2.0,
    "carries": 4.0, "attempts": 12.0, "completions": 8.0,
    "pass_td": 0.5, "interceptions": 0.3, "total_td": 0.12,
}


def candidate_lines(values: np.ndarray | float, stat: str) -> list[float]:
    """Lines a book might plausibly hang, from the player's own distribution.

    Accepts the simulated samples. A bare number is still accepted for
    convenience and treated as a point estimate with no spread.
    """
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    if arr.size == 0:
        return []
    projection = float(arr.mean())
    if not np.isfinite(projection) or projection < MIN_PROJECTION.get(stat, 0.0):
        return []

    if stat in ("total_td", "pass_td", "interceptions"):
        # Scoring markets are quoted at fixed half-points. They still have to
        # clear the probability band: an anytime-touchdown under on a tight end
        # who rarely scores is a genuine market, but at 79% it is a near-lock
        # that would otherwise fill the whole board.
        return [x for x in (0.5, 1.5)
                if TARGET_BAND[0] <= float((arr > x).mean()) <= TARGET_BAND[1]]

    if arr.size <= 1:
        return []

    # Enumerate the half-point lines a book could hang around this distribution,
    # then keep the ones whose implied probability actually lands near a coin
    # flip. Rounding a quantile to the nearest half-point looks equivalent but
    # is not: for a small count, a 3.1 quantile becomes a 3.5 line, and the
    # under on a mean-2.4 distribution is then a 79% shot. Selecting on
    # probability keeps overs and unders both representable at every market.
    lo, hi = np.quantile(arr, [0.08, 0.92])
    if stat in COUNT_STATS:
        grid = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5, 1.0)
    else:
        step = 5.0 if projection >= 40 else 2.5
        grid = np.arange(np.floor(lo / step) * step - 0.5,
                         np.ceil(hi / step) * step + step, step)

    out = []
    for line in grid:
        if line <= 0:
            continue
        p_over = float((arr > line).mean())
        if TARGET_BAND[0] <= p_over <= TARGET_BAND[1]:
            out.append(float(line))
    return sorted(set(out))


# ---------------------------------------------------------------- odds maths
def american_to_probability(odds: float) -> float:
    """Implied probability of an American price, including the vig."""
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / -odds)


def probability_to_american(p: float) -> float:
    """Fair price for a probability, with no margin added.

    At exactly even money the two forms are the same price; +100 is the
    conventional way to write it, so the strict comparison matters.
    """
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return -100.0 * p / (1.0 - p) if p > 0.5 else 100.0 * (1.0 - p) / p


def expected_value(p: float, odds: float, stake: float = 100.0) -> float:
    """Expected profit on a stake at this price, given the model's probability."""
    dec = american_to_decimal(odds)
    return float(p * (dec - 1.0) * stake - (1.0 - p) * stake)


def kelly_fraction(p: float, odds: float) -> float:
    """Full-Kelly stake as a fraction of bankroll. Negative means no bet."""
    b = american_to_decimal(odds) - 1.0
    if b <= 0:
        return 0.0
    return float((p * b - (1.0 - p)) / b)


# ------------------------------------------------------------------- picks
def player_lines(game, player: str, conditional: bool = True) -> pd.DataFrame:
    """Every plausible line for one player, with the model's probability."""
    meta = game.meta.get(player)
    if meta is None:
        return pd.DataFrame()
    pos = meta["position"]
    mask = game.active_mask(player) if conditional else None

    rows = []
    for stat, label in STAT_MENU.get(pos, []):
        s = game.stat(player, stat)
        if s is None or len(s) == 0:
            continue
        vals = s[mask] if mask is not None else s
        if len(vals) == 0:
            continue
        projection = float(vals.mean())
        for line in candidate_lines(vals, stat):
            p_over = float((vals > line).mean())
            # Skip lines nobody would hang: near-certainties either way.
            if p_over < 0.15 or p_over > 0.85:
                continue
            rows.append({
                "player": player, "team": meta["team"], "pos": pos,
                "stat": stat, "market": label, "line": line,
                "projection": float(vals.mean()),
                "p_over": p_over, "p_under": 1.0 - p_over,
                "fair_over": probability_to_american(p_over),
                "fair_under": probability_to_american(1.0 - p_over),
                "p_active": meta["p_active"],
            })
    return pd.DataFrame(rows)


def best_picks(games: list, min_prob: float = 0.58, max_prob: float = 0.80,
               min_active: float = 0.70, top_n: int = 40,
               conditional: bool = True, max_per_player: int = 2) -> pd.DataFrame:
    """Rank the model's highest-confidence sides across a slate.

    This is a confidence ranking, not a value ranking - there are no prices in
    the data. A leg near the top is one the model likes, not one the market has
    mispriced. Supply a price in the app to turn it into expected value.

    Near-certainties are excluded because they are where a book's margin is
    heaviest and where the model's own error is proportionally largest.
    """
    frames = []
    for g in games:
        for player in g.players:
            df = player_lines(g, player, conditional=conditional)
            if not df.empty:
                df["matchup"] = f"{g.away} @ {g.home}"
                frames.append(df)
    if not frames:
        return pd.DataFrame()

    all_lines = pd.concat(frames, ignore_index=True)
    rows = []
    for _, r in all_lines.iterrows():
        for side, p in (("Over", r["p_over"]), ("Under", r["p_under"])):
            if min_prob <= p <= max_prob and r["p_active"] >= min_active:
                rows.append({**r.to_dict(), "side": side, "probability": p,
                             "fair_odds": probability_to_american(p)})
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["edge_score"] = out["probability"] * out["p_active"]
    out = out.sort_values("edge_score", ascending=False)

    # One market yields two sides, and for a mid-range line both can sit in the
    # band at once - "over 14.5" and "under 49.5" are both true most weeks and
    # both say the same thing, that he lands in the middle. Keep the stronger.
    out = out.groupby(["player", "stat"], sort=False).head(1)

    # One player offers many correlated markets, and quarterbacks offer the most
    # predictable ones. Without a cap the board fills with a handful of names.
    out = out.sort_values("edge_score", ascending=False)
    out = out.groupby("player", sort=False).head(max_per_player)

    cols = ["matchup", "player", "team", "pos", "market", "line", "side",
            "projection", "probability", "fair_odds", "p_active", "edge_score", "stat"]
    return out.head(top_n)[cols].reset_index(drop=True)
