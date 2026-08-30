"""Parlay evaluation on correlated simulations.

The usual way to price a parlay is to multiply the legs' probabilities. That
assumes the legs are independent, and same-game legs never are. A quarterback
over his passing yards and his receiver over his receiving yards are close to
the same bet twice; a quarterback over and his running back over pull against
each other. Multiplying either together gives the wrong number, and the error
runs in opposite directions.

Because the game is simulated once with shared randomness, a parlay's
probability is just the fraction of simulations in which every leg happens to
land. No independence assumption is needed, and the naive figure is reported
alongside so the size and direction of the correction is visible.

Legs from different games are genuinely close to independent, so the two numbers
converge there - which is the correct behaviour, not a missing feature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .picks import (american_to_decimal, expected_value, kelly_fraction,
                    probability_to_american)


@dataclass(frozen=True)
class Leg:
    """One selection: a player, a statistic, a line and a side."""
    player: str
    stat: str
    line: float
    side: str = "over"          # "over" or "under"
    odds: float | None = None   # American price, if you have one
    label: str = ""

    def describe(self) -> str:
        name = self.label or self.stat.replace("_", " ")
        return f"{self.player} {self.side} {self.line:g} {name}"


def evaluate(legs: list[Leg], games: list, conditional: bool = True) -> dict:
    """Probability, fair price and expected value for a parlay.

    ``conditional`` prices the parlay as a book effectively does - assuming the
    named players are active, since a scratch usually voids the leg. Turning it
    off prices scratch risk into the parlay itself, which is the right view if
    your book grades a non-appearance as a loss.
    """
    if not legs:
        return {"error": "no legs"}

    # Resolve every leg first, then condition once on all named players being
    # active. Conditioning leg by leg would trim each to a different length and
    # silently break the alignment that makes the joint reading correct.
    seen = set()
    for leg in legs:
        key = (leg.player, leg.stat, leg.line, leg.side)
        if key in seen:
            return {"error": f"the same selection is in the slip twice: "
                             f"{leg.describe()}"}
        seen.add(key)

    resolved, missing = [], []
    for leg in legs:
        game = _find_game(games, leg.player)
        if game is None:
            missing.append(leg.player)
            continue
        m = game.leg_mask(leg.player, leg.stat, leg.line, leg.side)
        if m is None:
            missing.append(f"{leg.player} ({leg.stat})")
            continue
        resolved.append((leg, game, m))

    if missing:
        return {"error": f"not simulated: {', '.join(missing)}"}
    if not resolved:
        return {"error": "no valid legs"}

    lengths = {len(m) for _, _, m in resolved}
    if len(lengths) > 1:
        return {"error": "simulations of differing length; re-run with one n_sims"}

    keep = np.ones(lengths.pop(), dtype=bool)
    if conditional:
        for leg, game, _ in resolved:
            keep &= game.active_mask(leg.player)
        if keep.sum() < 200:
            return {"error": "too few simulations with every player active; "
                             "price this one unconditionally instead"}

    masks, per_leg = [], []
    for leg, game, m in resolved:
        m = m[keep]
        masks.append(m)
        per_leg.append({
            "leg": leg.describe(), "player": leg.player, "stat": leg.stat,
            "line": leg.line, "side": leg.side,
            "probability": float(m.mean()),
            "fair_odds": probability_to_american(float(m.mean())),
            "odds": leg.odds,
            "p_active": game.meta.get(leg.player, {}).get("p_active"),
        })

    joint_hit = np.logical_and.reduce(masks)
    p_joint = float(joint_hit.mean())
    p_naive = float(np.prod([m.mean() for m in masks]))

    out = {
        "legs": per_leg,
        "n_legs": len(masks),
        "probability": p_joint,
        "naive_probability": p_naive,
        "correlation_lift": (p_joint / p_naive) if p_naive > 0 else np.nan,
        "fair_odds": probability_to_american(p_joint),
        "naive_fair_odds": probability_to_american(p_naive),
        "conditional": conditional,
        "all_active_probability": float(keep.mean()) if conditional else 1.0,
    }

    priced = [l.odds for l in legs if l.odds is not None]
    if len(priced) == len(legs) and legs:
        dec = float(np.prod([american_to_decimal(o) for o in priced]))
        out["offered_decimal"] = dec
        out["offered_american"] = probability_to_american(1.0 / dec)
        out["expected_value_per_100"] = float(p_joint * (dec - 1.0) * 100 - (1 - p_joint) * 100)
        out["kelly_fraction"] = float((p_joint * (dec - 1.0) - (1 - p_joint)) / (dec - 1.0)) \
            if dec > 1 else 0.0
        out["breakeven_probability"] = 1.0 / dec
    return out


def _find_game(games: list, player: str):
    for g in games:
        if player in g.players:
            return g
    return None


def correlation_matrix(legs: list[Leg], games: list) -> pd.DataFrame:
    """Pairwise correlation between the legs' outcomes.

    Positive means the legs tend to land together and the parlay is friendlier
    than independence implies; negative means they fight each other.
    """
    cols, names = [], []
    for leg in legs:
        g = _find_game(games, leg.player)
        if g is None:
            continue
        m = g.leg_mask(leg.player, leg.stat, leg.line, leg.side)
        if m is None:
            continue
        cols.append(m.astype(float))
        # Two identical legs describe identically, and a frame with repeated
        # labels cannot be rendered. Number the repeats rather than dropping
        # them: the caller asked about these legs, duplicates included.
        name = leg.describe()
        if name in names:
            name = f"{name} ({names.count(name) + 1})"
        names.append(name)
    if len(cols) < 2:
        return pd.DataFrame()
    corr = np.corrcoef(np.vstack(cols))
    return pd.DataFrame(np.atleast_2d(corr), index=names, columns=names)


def suggest(games: list, n_legs: int = 2, min_leg_prob: float = 0.55,
            max_leg_prob: float = 0.85, min_active: float = 0.80,
            target: str = "correlated", top_n: int = 8,
            candidates_per_player: int = 2, max_candidates: int = 60) -> pd.DataFrame:
    """Build candidate parlays and rank them by their correlated probability.

    ``target`` chooses the flavour:

    * ``correlated`` builds same-*team* stacks. Correlation lives inside an
      offence - a quarterback and the receivers he throws to - so pairing two
      players from opposite sidelines finds nothing, which is why this requires
      a shared team rather than merely a shared game.
    * ``independent`` spreads across games, which lowers variance rather than
      raising probability.

    Neither is a claim about value. Without prices these are just parlays the
    model rates highly, and a parlay is a worse bet than its legs regardless of
    how the correlation falls.
    """
    from itertools import combinations

    from .picks import player_lines

    pool = []
    for g in games:
        for player in g.players:
            meta = g.meta.get(player, {})
            if meta.get("p_active", 0) < min_active:
                continue
            df = player_lines(g, player, conditional=True)
            if df.empty:
                continue
            picked = 0
            for _, r in df.iterrows():
                if picked >= candidates_per_player:
                    break
                for side, p in (("over", r["p_over"]), ("under", r["p_under"])):
                    if min_leg_prob <= p <= max_leg_prob and picked < candidates_per_player:
                        pool.append({
                            "leg": Leg(player=player, stat=r["stat"], line=r["line"],
                                       side=side, label=r["market"]),
                            "p": p, "team": r["team"], "pos": meta.get("position"),
                            "game": g, "matchup": f"{g.away} @ {g.home}",
                        })
                        picked += 1
    if len(pool) < n_legs:
        return pd.DataFrame()

    pool = sorted(pool, key=lambda d: -d["p"])[:max_candidates]

    rows = []
    for combo in combinations(pool, n_legs):
        if len({c["leg"].player for c in combo}) < n_legs:
            continue                      # never stack one player against himself
        teams = {c["team"] for c in combo}
        games_in = {id(c["game"]) for c in combo}
        same_team = len(teams) == 1
        same_game = len(games_in) == 1

        if target == "correlated" and not same_team:
            continue
        if target == "independent" and same_game:
            continue

        res = evaluate([c["leg"] for c in combo], games, conditional=same_game)
        if "error" in res:
            continue
        rows.append({
            "legs": "  +  ".join(c["leg"].describe() for c in combo),
            "leg_objects": [c["leg"] for c in combo],
            "leg_detail": res["legs"],
            "matchups": ", ".join(sorted({c["matchup"] for c in combo})),
            "team": ", ".join(sorted(teams)),
            "probability": res["probability"],
            "naive": res["naive_probability"],
            "correlation_lift": res["correlation_lift"],
            "fair_odds": res["fair_odds"],
            "naive_fair_odds": res["naive_fair_odds"],
        })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if target == "correlated":
        # Surface the slips whose correlation actually helps, not merely the
        # likeliest - a naive calculator already finds those.
        df = df.sort_values(["correlation_lift", "probability"], ascending=False)
    else:
        df = df.sort_values("probability", ascending=False)
    return df.head(top_n).reset_index(drop=True)


# Presets for the one-press generator, as leg-probability bands. A parlay is a
# worse bet than its legs at every setting; what changes is how the payout and
# the hit rate trade off against each other.
STYLES = {
    "Safer": {"min_leg_prob": 0.62, "max_leg_prob": 0.88},
    "Balanced": {"min_leg_prob": 0.52, "max_leg_prob": 0.80},
    "Longshot": {"min_leg_prob": 0.25, "max_leg_prob": 0.58},
}


def generate(games: list, n_legs: int = 3, style: str = "Balanced",
             correlated: bool = True, teams: list[str] | None = None,
             seed: int | None = None) -> dict:
    """One parlay, ready to show. The button behind the button.

    Returns the best slip ``suggest`` can find at this size and style, already
    evaluated, or a message saying why it could not find one. ``seed`` picks a
    different slip from the same ranked shortlist so pressing generate again
    gives something new rather than the same answer.
    """
    band = STYLES.get(style, STYLES["Balanced"])
    df = suggest(games, n_legs=n_legs,
                 target="correlated" if correlated else "independent",
                 top_n=12, **band)
    if df.empty:
        return {"error": f"No {n_legs}-leg {style.lower()} parlay available this week. "
                         "Try fewer legs or a different style."}
    if teams:
        keep = df[df["team"].apply(lambda t: any(x in t.split(", ") for x in teams))]
        if keep.empty:
            return {"error": "No parlay found for that team. Clear the filter or "
                             "try a different style."}
        df = keep.reset_index(drop=True)
    row = df.iloc[int(seed) % len(df)] if seed is not None else df.iloc[0]
    return {
        "legs": pd.DataFrame(row["leg_detail"]),
        "leg_objects": row["leg_objects"],
        "summary": row.drop(labels=["leg_detail", "leg_objects"]).to_dict(),
        "alternatives": df.drop(columns=["leg_detail", "leg_objects"]),
    }
