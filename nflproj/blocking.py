"""Separate what the offensive line does from what the running back does.

A back's yards per carry blends two things that behave completely differently.
Measured over 2022-2025:

    yards BEFORE contact per carry (the blocking)   r = 0.433 year over year
    yards AFTER  contact per carry (the back)       r = 0.106

Blocking persists; broken tackles mostly do not. Regressing a back's raw yards
per carry toward a league mean treats those as one quantity and throws away the
part that is actually predictable. This module keeps them apart: the line's
contribution is projected from the team, and the back's from the player with
heavy regression, because it barely carries.

The 2025 spread is not small - Chicago generated 3.31 yards before contact per
carry, Las Vegas 1.63. That is most of a yard and a half of blocking that a
league-average prior would erase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LAST_COMPLETED_SEASON, normalize_team

LEAGUE_YBC = 2.45   # league mean yards before contact per carry
LEAGUE_YAC = 1.85   # league mean yards after contact per carry

# Blocking persists, so a team's own record dominates. Elusiveness does not, so
# it is pulled hard toward the league mean.
TEAM_BLOCKING_PERSISTENCE = 0.65
PLAYER_ELUSIVENESS_PERSISTENCE = 0.25
TEAM_PRIOR_CARRIES = 260.0
PLAYER_PRIOR_CARRIES = 190.0
RECENCY_HALFLIFE = 1.5


def team_blocking(pfr_rush: pd.DataFrame) -> pd.DataFrame:
    """Yards before and after contact per carry, by team-season."""
    if pfr_rush is None or pfr_rush.empty:
        return pd.DataFrame()
    need = {"rushing_yards_before_contact", "rushing_yards_after_contact", "carries"}
    if not need.issubset(pfr_rush.columns):
        return pd.DataFrame()
    g = (
        pfr_rush.groupby(["season", "team"])
        .agg(carries=("carries", "sum"),
             ybc=("rushing_yards_before_contact", "sum"),
             yac=("rushing_yards_after_contact", "sum"))
        .reset_index()
    )
    g = g[g["carries"] > 0]
    g["ybc_per_carry"] = g["ybc"] / g["carries"]
    g["yac_per_carry"] = g["yac"] / g["carries"]
    return g


def player_elusiveness(pfr_rush: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Yards after contact per carry, by player-season, keyed on gsis id."""
    if pfr_rush is None or pfr_rush.empty:
        return pd.DataFrame()
    g = (
        pfr_rush.groupby(["season", "pfr_player_id"])
        .agg(carries=("carries", "sum"),
             yac=("rushing_yards_after_contact", "sum"),
             broken=("rushing_broken_tackles", "sum"))
        .reset_index()
    )
    g = g[g["carries"] >= 20]
    g["yac_per_carry"] = g["yac"] / g["carries"]
    g["broken_per_carry"] = g["broken"] / g["carries"]

    if players is not None and not players.empty and "pfr_id" in players.columns:
        xw = players[["gsis_id", "pfr_id"]].dropna()
        g = g.merge(xw, left_on="pfr_player_id", right_on="pfr_id", how="left")
        g = g.rename(columns={"gsis_id": "player_id"})
    return g


def _shrink(hist: pd.DataFrame, col: str, weight_col: str, prior: float,
            prior_weight: float, persistence: float,
            anchor: int = LAST_COMPLETED_SEASON) -> float:
    """Recency-weighted mean, regressed to the prior, then decayed by persistence."""
    if hist is None or hist.empty or col not in hist.columns:
        return float(prior)
    w = 0.5 ** ((anchor - hist["season"]) / RECENCY_HALFLIFE)
    n = (hist[weight_col].astype(float) * w)
    v = hist[col].astype(float)
    m = v.notna() & (n > 0)
    if not m.any():
        return float(prior)
    observed = float((v[m] * n[m]).sum() / n[m].sum())
    n_eff = float(n[m].sum())
    shrunk = (observed * n_eff + prior * prior_weight) / (n_eff + prior_weight)
    # Only the persistent fraction of the edge carries into next season.
    return float(prior + (shrunk - prior) * persistence)


def project_rushing_efficiency(
    team: str, player_id: str | None,
    team_block: pd.DataFrame, player_elus: pd.DataFrame,
    anchor: int = LAST_COMPLETED_SEASON,
) -> dict:
    """Projected yards per carry, split into blocking and back.

    Returns both components so a projection can explain itself: a back behind a
    good line and a back creating on his own produce the same number for very
    different reasons, and only one of them travels if he changes teams.
    """
    tb = team_block[team_block["team"] == normalize_team(team)] if not team_block.empty else pd.DataFrame()
    ybc = _shrink(tb, "ybc_per_carry", "carries", LEAGUE_YBC,
                  TEAM_PRIOR_CARRIES, TEAM_BLOCKING_PERSISTENCE, anchor)

    pe = pd.DataFrame()
    if player_id and not player_elus.empty and "player_id" in player_elus.columns:
        pe = player_elus[player_elus["player_id"] == player_id]
    yac = _shrink(pe, "yac_per_carry", "carries", LEAGUE_YAC,
                  PLAYER_PRIOR_CARRIES, PLAYER_ELUSIVENESS_PERSISTENCE, anchor)

    return {
        "ypc": float(ybc + yac),
        "yards_before_contact": float(ybc),
        "yards_after_contact": float(yac),
        "blocking_vs_league": float(ybc - LEAGUE_YBC),
        "elusiveness_vs_league": float(yac - LEAGUE_YAC),
        "has_player_sample": bool(len(pe)),
    }


# Offensive line positions as snap counts label them.
OL_POSITIONS = {"T", "G", "C", "OL", "OT", "OG"}
# A lineman needs this many snaps in a season to count as having been a starter.
STARTER_SNAPS = 200

# Line continuity is measured and reported and deliberately not applied.
#
# Over 223 team-seasons from 2019-2025, counting how many of a team's five
# highest-snap linemen also started for it the previous year, the sign is right
# and the size is not there:
#
#   sack rate    -0.00165 per returning starter (t = -1.54)
#   rush EPA     +0.00098                        (t = +0.23)
#   yards/carry  -0.00636                        (t = -0.24)
#
# Added to a model that already has the team's own prior sack rate, continuity
# lifts the multiple correlation from 0.376 to 0.388. The whole range - one
# returning starter against five - is worth about two thirds of a percentage
# point of sack rate, and that is inside the noise. Rushing shows nothing at
# all, which is the more surprising half: continuity is usually argued for on
# run blocking.
#
# This is "not established" rather than "established zero", so the measure is
# kept and surfaced. If a later season pushes it past significance, applying it
# is a one-line change.
CONTINUITY_SACK_RATE_COEF = -0.00165
CONTINUITY_SACK_RATE_T = -1.54
CONTINUITY_APPLIED = False


def line_continuity(snaps: pd.DataFrame) -> pd.DataFrame:
    """How much of each offensive line is back from last season.

    Two measures, because they disagree about what continuity means: the share
    of line snaps taken by players who were starters here last year, and the
    plain count of returning starters among this year's top five. The count is
    the one the football conversation uses.
    """
    if snaps is None or snaps.empty or "position" not in snaps.columns:
        return pd.DataFrame()
    ol = snaps[snaps["position"].isin(OL_POSITIONS)]
    if ol.empty:
        return pd.DataFrame()
    key = "pfr_player_id" if "pfr_player_id" in ol.columns else "player"
    ts = (ol.groupby(["season", "team", key])["offense_snaps"].sum()
            .rename("snaps").reset_index())

    prior = ts.copy()
    prior["season"] += 1
    prior = prior.rename(columns={"snaps": "prior_snaps"})
    j = ts.merge(prior, on=["season", "team", key], how="left")
    j["prior_snaps"] = j["prior_snaps"].fillna(0.0)

    j["returning_snaps"] = np.where(j["prior_snaps"] >= STARTER_SNAPS, j["snaps"], 0.0)
    share = (j.groupby(["season", "team"])
               .agg(line_snaps=("snaps", "sum"),
                    returning_snaps=("returning_snaps", "sum")).reset_index())
    share["snap_continuity"] = share["returning_snaps"] / share["line_snaps"].clip(lower=1)

    j["rank"] = j.groupby(["season", "team"])["snaps"].rank(ascending=False, method="first")
    five = j[j["rank"] <= 5]
    count = (five.assign(back=(five["prior_snaps"] >= STARTER_SNAPS).astype(int))
                 .groupby(["season", "team"])["back"].sum()
                 .rename("returning_starters").reset_index())

    out = share.merge(count, on=["season", "team"], how="left")
    out["returning_starters"] = out["returning_starters"].fillna(0).astype(int)
    return out[["season", "team", "line_snaps", "snap_continuity",
                "returning_starters"]].sort_values(["season", "team"]).reset_index(drop=True)


def team_pass_protection(pfr_pass: pd.DataFrame) -> pd.DataFrame:
    """Pressure allowed per dropback, by team-season - a protection signal."""
    if pfr_pass is None or pfr_pass.empty:
        return pd.DataFrame()
    cols = {"times_pressured", "times_sacked", "times_blitzed"}
    if not cols.issubset(pfr_pass.columns):
        return pd.DataFrame()
    g = (
        pfr_pass.groupby(["season", "team"])
        .agg(pressured=("times_pressured", "sum"), sacked=("times_sacked", "sum"),
             blitzed=("times_blitzed", "sum"), hurried=("times_hurried", "sum"))
        .reset_index()
    )
    return g
