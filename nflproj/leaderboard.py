"""One table with everyone on it.

The rest of the model answers questions about a single player at a time. This
answers the question people actually open the app with: who are the top twenty
receivers this week, what does each one project for, and how wide is the range
around it - all on one screen, without clicking through a roster.

Everything here reads the same joint simulation the picks and parlay code
reads, so the numbers agree with those tabs by construction rather than by
coincidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .picks import probability_to_american

# What each position group is worth ranking by, in the order a person would
# reach for them.
LEADERBOARD_STATS = {
    "QB": [("pass_yards", "Passing yards"), ("pass_td", "Passing TDs"),
           ("attempts", "Attempts"), ("completions", "Completions"),
           ("rush_yards", "Rushing yards"), ("interceptions", "Interceptions")],
    "RB": [("rush_yards", "Rushing yards"), ("scrimmage_yards", "Scrimmage yards"),
           ("carries", "Carries"), ("rec_yards", "Receiving yards"),
           ("receptions", "Receptions"), ("targets", "Targets"),
           ("total_td", "Touchdowns")],
    "WR": [("rec_yards", "Receiving yards"), ("receptions", "Receptions"),
           ("targets", "Targets"), ("scrimmage_yards", "Scrimmage yards"),
           ("total_td", "Touchdowns")],
}
LEADERBOARD_STATS["FB"] = LEADERBOARD_STATS["RB"]
LEADERBOARD_STATS["TE"] = LEADERBOARD_STATS["WR"]

# A round number worth showing a hit rate for, per statistic. These are the
# thresholds people actually talk about, not quantiles.
MILESTONES = {
    "rec_yards": 100.0, "rush_yards": 100.0, "scrimmage_yards": 100.0,
    "pass_yards": 300.0, "receptions": 5.5, "targets": 7.5, "carries": 15.5,
    "attempts": 34.5, "completions": 22.5, "pass_td": 1.5, "total_td": 0.5,
    "interceptions": 0.5,
}


def stats_for(positions: list[str]) -> list[tuple[str, str]]:
    """The stat menu for a group of positions, de-duplicated in order."""
    seen, out = set(), []
    for pos in positions:
        for key, label in LEADERBOARD_STATS.get(pos, []):
            if key not in seen:
                seen.add(key)
                out.append((key, label))
    return out


def leaderboard(games: list, positions: list[str], stat: str,
                conditional: bool = True, min_active: float = 0.0,
                teams: list[str] | None = None) -> pd.DataFrame:
    """Every player at these positions, ranked by one statistic.

    ``conditional`` reports the line assuming the player dresses, which is how
    a book prices a prop; the chance he does not is carried separately in
    ``Active %`` rather than folded silently into the projection.
    """
    rows = []
    for g in games:
        for player, meta in g.meta.items():
            if meta.get("position") not in positions:
                continue
            if teams and meta.get("team") not in teams:
                continue
            p_active = float(meta.get("p_active", 1.0))
            if p_active < min_active:
                continue
            v = g.stat(player, stat)
            if v is None or len(v) == 0:
                continue
            vals = v[g.active_mask(player)] if conditional else v
            if len(vals) == 0:
                continue
            row = {
                "player": player,
                "team": meta.get("team"),
                "pos": meta.get("position"),
                "matchup": f"{g.away} @ {g.home}",
                "projection": float(np.mean(vals)),
                "floor": float(np.percentile(vals, 10)),
                "median": float(np.median(vals)),
                "ceiling": float(np.percentile(vals, 90)),
                "p_active": p_active,
            }
            milestone = MILESTONES.get(stat)
            if milestone is not None:
                row["milestone"] = milestone
                row["p_milestone"] = float((vals > milestone).mean())
            td = g.stat(player, "total_td")
            if td is not None and len(td):
                td_vals = td[g.active_mask(player)] if conditional else td
                row["p_anytime_td"] = float((td_vals >= 1).mean()) if len(td_vals) else np.nan
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("projection", ascending=False)
    return out.reset_index(drop=True)


def player_card(games: list, player: str, conditional: bool = True) -> dict:
    """Everything worth knowing about one player, without leaving the table.

    Returns the full stat line plus the samples for whichever statistics exist,
    so a caller can draw a distribution without reaching back into the game.
    """
    game = next((g for g in games if player in g.players), None)
    if game is None:
        return {}
    meta = game.meta.get(player, {})
    mask = game.active_mask(player) if conditional else None
    lines = []
    for stat, label in LEADERBOARD_STATS.get(meta.get("position", ""), []):
        v = game.stat(player, stat)
        if v is None or len(v) == 0:
            continue
        vals = v[mask] if mask is not None else v
        if len(vals) == 0:
            continue
        milestone = MILESTONES.get(stat)
        lines.append({
            "stat": stat, "market": label,
            "projection": float(np.mean(vals)),
            "floor": float(np.percentile(vals, 10)),
            "median": float(np.median(vals)),
            "ceiling": float(np.percentile(vals, 90)),
            "milestone": milestone,
            "p_milestone": float((vals > milestone).mean()) if milestone is not None else np.nan,
            "fair_milestone": (probability_to_american(float((vals > milestone).mean()))
                               if milestone is not None else np.nan),
        })
    return {
        "player": player, "game": game, "meta": meta,
        "matchup": f"{game.away} @ {game.home}",
        "lines": pd.DataFrame(lines),
        "samples": {s: (game.stat(player, s)[mask] if mask is not None else game.stat(player, s))
                    for s, _ in LEADERBOARD_STATS.get(meta.get("position", ""), [])
                    if game.stat(player, s) is not None},
    }
