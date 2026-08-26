"""Whether a player is on the field at all.

A projection that assumes every listed player suits up every week overstates
everything. Backtesting the 2025 season showed roughly 46% of projected
player-games ended with no touches - injuries, inactives, healthy scratches and
rotational players who simply never got the ball. Conditional on playing the
projections were close to unbiased; unconditionally they ran 25-40% hot.

This module estimates the probability a player is a live participant in a given
game, so projections can be reported two ways: what a player does when he plays,
and what he is worth once availability is priced in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import normalize_team

# Share of team games a player at each depth-chart slot actually takes an
# offensive snap in. Measured from 2023-2025 snap counts.
AVAILABILITY_PRIOR = {
    ("QB", 1): 0.85, ("QB", 2): 0.33,
    ("RB", 1): 0.82, ("RB", 2): 0.81, ("RB", 3): 0.54, ("RB", 4): 0.37,
    ("WR", 1): 0.86, ("WR", 2): 0.77, ("WR", 3): 0.76, ("WR", 4): 0.70, ("WR", 5): 0.52,
    ("TE", 1): 0.89, ("TE", 2): 0.81, ("TE", 3): 0.67,
    ("FB", 1): 0.85,
}
# Typical share of offensive snaps by slot, used as a secondary signal.
SNAP_SHARE_PRIOR = {
    ("QB", 1): 0.95, ("QB", 2): 0.53,
    ("RB", 1): 0.60, ("RB", 2): 0.36, ("RB", 3): 0.20, ("RB", 4): 0.15,
    ("WR", 1): 0.83, ("WR", 2): 0.74, ("WR", 3): 0.58, ("WR", 4): 0.46, ("WR", 5): 0.34,
    ("TE", 1): 0.74, ("TE", 2): 0.47, ("TE", 3): 0.31,
    ("FB", 1): 0.28,
}
PRIOR_GAMES = 10.0  # strength of the role prior, in games

# Injury report designations that materially change availability.
STATUS_MULTIPLIER = {
    "Out": 0.0, "Doubtful": 0.12, "Questionable": 0.72,
    "Injured Reserve": 0.0, "IR": 0.0, "PUP": 0.0, "Suspended": 0.0,
}


def id_crosswalk(players: pd.DataFrame) -> dict[str, str]:
    """Map pfr player ids to gsis ids so snap counts join to the depth chart."""
    if players is None or players.empty or "pfr_id" not in players.columns:
        return {}
    d = players[["gsis_id", "pfr_id"]].dropna()
    return dict(zip(d["pfr_id"], d["gsis_id"]))


def player_availability(snaps: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Per player-season availability and mean snap share, keyed by gsis id."""
    if snaps is None or snaps.empty:
        return pd.DataFrame(columns=["player_id", "season", "availability", "snap_pct", "games"])

    s = snaps[snaps.get("game_type", "REG") == "REG"].copy()
    s["team"] = s["team"].map(normalize_team)
    team_games = s.groupby(["season", "team"])["game_id"].nunique().rename("team_games").reset_index()

    active = s[s["offense_snaps"].fillna(0) > 0]
    g = (
        active.groupby(["season", "team", "pfr_player_id"])
        .agg(games=("game_id", "nunique"), snap_pct=("offense_pct", "mean"))
        .reset_index()
        .merge(team_games, on=["season", "team"], how="left")
    )
    g["availability"] = (g["games"] / g["team_games"].clip(lower=1)).clip(0, 1)

    xw = id_crosswalk(players)
    g["player_id"] = g["pfr_player_id"].map(xw)
    return g.dropna(subset=["player_id"])[
        ["player_id", "season", "team", "games", "team_games", "availability", "snap_pct"]
    ]


def project_availability(
    avail: pd.DataFrame, player_id: str | None, pos: str, rank: int,
    halflife: float = 1.5, injury_status: str | None = None,
) -> tuple[float, float]:
    """Probability a player is an active participant, plus his snap share.

    A player's own record is regressed toward the prior for the role he holds.
    Availability is genuinely persistent - some players are chronically banged
    up - but a single lost season should not condemn a starter, so the prior
    carries real weight.
    """
    prior_a = AVAILABILITY_PRIOR.get((pos, int(rank)), 0.45)
    prior_s = SNAP_SHARE_PRIOR.get((pos, int(rank)), 0.30)

    p_active, snap = prior_a, prior_s
    if player_id is not None and avail is not None and not avail.empty:
        h = avail[avail["player_id"] == player_id]
        if not h.empty:
            w = 0.5 ** ((h["season"].max() - h["season"]) / halflife)
            n = float((h["team_games"] * w).sum())
            obs_a = float((h["availability"] * h["team_games"] * w).sum() / max(n, 1e-9))
            obs_s = float((h["snap_pct"].fillna(prior_s) * h["team_games"] * w).sum() / max(n, 1e-9))
            p_active = (obs_a * n + prior_a * PRIOR_GAMES) / (n + PRIOR_GAMES)
            snap = (obs_s * n + prior_s * PRIOR_GAMES) / (n + PRIOR_GAMES)

    if injury_status:
        p_active *= STATUS_MULTIPLIER.get(str(injury_status).strip(), 1.0)

    return float(np.clip(p_active, 0.0, 0.99)), float(np.clip(snap, 0.0, 1.0))


def current_injuries(inj: pd.DataFrame, players: pd.DataFrame) -> dict[str, str]:
    """Latest reported designation per player, keyed by gsis id."""
    if inj is None or inj.empty:
        return {}
    d = inj.copy()
    idcol = next((c for c in ("gsis_id", "player_id") if c in d.columns), None)
    statuscol = next((c for c in ("report_status", "game_status", "status") if c in d.columns), None)
    if idcol is None or statuscol is None:
        return {}
    if "week" in d.columns:
        d = d.sort_values("week").groupby(idcol).tail(1)
    return dict(zip(d[idcol], d[statuscol]))
