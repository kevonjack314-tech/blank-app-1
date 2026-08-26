"""Who is actually on the field, and how they bend a coach's scheme.

A fingerprint carried over from a coach's previous stop describes what they did
with different players. Some of that signature is genuinely portable - tempo,
motion usage, how often they go on fourth down. Some of it is not: Todd Monken's
Baltimore offense ran designed quarterback runs at a high rate because Lamar
Jackson was taking the snaps, and none of that transfers to a different roster.

This module projects the depth chart into usable roles, measures each player's
own tendencies from history, and then constrains the scheme projection so that
personnel-bound traits follow the players rather than the coach.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LAST_COMPLETED_SEASON, PRIOR_STRENGTH, normalize_team

# Skill roles the projection cares about, and how deep to read the chart.
ROLE_DEPTH = {"QB": 2, "RB": 4, "WR": 6, "TE": 3, "FB": 1}
SKILL_POSITIONS = ("QB", "RB", "WR", "TE", "FB")

# For each personnel-bound trait: how much of the projection is dictated by the
# player rather than the play caller. 1.0 means the coach has essentially no say.
PERSONNEL_BOUND = {
    "qb_designed_run_rate": 0.85,   # a coach cannot scheme mobility into a statue
    "scramble_rate": 0.80,          # almost entirely a QB trait
    "adot": 0.40,                   # scheme sets the menu, the QB picks from it
    "deep_rate": 0.40,
    "yac_share": 0.30,              # depends on who is catching it
    "sack_rate_allowed": 0.45,      # QB pocket habits plus the line in front
}


def latest_depth_chart(depth: pd.DataFrame) -> pd.DataFrame:
    """Skill-position depth chart, one row per player per role."""
    if depth is None or depth.empty:
        return pd.DataFrame()
    df = depth.copy()
    df["team"] = df["team"].map(normalize_team)
    df = df[df["pos_abb"].isin(SKILL_POSITIONS)]
    df = df[df["player_name"].notna()]
    df["pos_rank"] = pd.to_numeric(df["pos_rank"], errors="coerce")
    df = df[df["pos_rank"].notna()]

    keep = []
    for pos, depth_n in ROLE_DEPTH.items():
        keep.append(df[(df["pos_abb"] == pos) & (df["pos_rank"] <= depth_n)])
    out = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()
    if out.empty:
        return out
    out = out.sort_values(["team", "pos_abb", "pos_rank"])
    return out[["team", "player_name", "gsis_id", "pos_abb", "pos_rank", "pos_grp"]].reset_index(drop=True)


def base_defensive_front(depth: pd.DataFrame) -> pd.DataFrame:
    """Each team's listed base front (3-4 / 4-3), straight off the depth chart."""
    if depth is None or depth.empty or "pos_grp" not in depth.columns:
        return pd.DataFrame(columns=["team", "base_front"])
    d = depth.copy()
    d["team"] = d["team"].map(normalize_team)
    d = d[d["pos_grp"].astype(str).str.contains("Base", na=False)]
    if d.empty:
        return pd.DataFrame(columns=["team", "base_front"])
    front = (
        d.groupby("team")["pos_grp"]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("base_front")
        .reset_index()
    )
    front["base_front"] = (
        front["base_front"].astype(str).str.replace("Base ", "", regex=False).str.replace(" D", "", regex=False)
    )
    return front


def qb_profiles(plays: pd.DataFrame) -> pd.DataFrame:
    """Per-quarterback tendencies that a new coordinator has to work around."""
    df = plays.copy()
    dropbacks = df[df["is_dropback"]]
    if dropbacks.empty:
        return pd.DataFrame()

    # Attribute team snaps to whoever was the passer, so rates share a denominator.
    snaps = (
        dropbacks.groupby(["season", "passer_player_id"])
        .agg(
            dropbacks=("play_id", "size"),
            team=("posteam", lambda s: s.value_counts().idxmax()),
            adot=("air_yards", "mean"),
            deep_rate=("air_yards", lambda s: float((s >= 20).mean()) if s.notna().any() else np.nan),
            sacks=("sack", "sum"),
            cpoe=("cpoe", "mean"),
            epa=("epa", "mean"),
            comp_yards=("yards_gained", "sum"),
        )
        .reset_index()
        .rename(columns={"passer_player_id": "player_id"})
    )

    runs = df[df["is_designed_run"]]
    qb_runs = (
        runs.groupby(["season", "rusher_player_id"])
        .agg(designed_runs=("play_id", "size"), rush_yards=("yards_gained", "sum"),
             rush_tds=("rush_touchdown", "sum"))
        .reset_index()
        .rename(columns={"rusher_player_id": "player_id"})
    )

    # A scramble is charted as a run, so it carries a rusher and no passer.
    # Attribute it through the ball carrier or it disappears from the profile.
    scrambles = (
        df[df["qb_scramble"].fillna(0) > 0]
        .groupby(["season", "rusher_player_id"])
        .size()
        .rename("scrambles")
        .reset_index()
        .rename(columns={"rusher_player_id": "player_id"})
    )

    completions = dropbacks[dropbacks["complete_pass"].fillna(0) > 0]
    yac = (
        completions.groupby(["season", "passer_player_id"])
        .agg(_yac=("yards_after_catch", "sum"), _rec_yards=("yards_gained", "sum"))
        .reset_index()
        .rename(columns={"passer_player_id": "player_id"})
    )
    yac["yac_share"] = yac["_yac"] / yac["_rec_yards"].replace(0, np.nan)

    team_plays = df.groupby(["season", "posteam"]).size().rename("team_plays").reset_index()
    team_plays = team_plays.rename(columns={"posteam": "team"})

    prof = snaps.merge(qb_runs, on=["season", "player_id"], how="left")
    prof = prof.merge(scrambles, on=["season", "player_id"], how="left")
    prof = prof.merge(yac[["season", "player_id", "yac_share"]], on=["season", "player_id"], how="left")
    prof["scrambles"] = prof["scrambles"].fillna(0.0)
    prof = prof.merge(team_plays, on=["season", "team"], how="left")
    prof["designed_runs"] = prof["designed_runs"].fillna(0.0)

    # Only count seasons where the player actually ran the offense.
    prof = prof[prof["dropbacks"] >= 100].copy()
    prof["qb_designed_run_rate"] = prof["designed_runs"] / prof["team_plays"].clip(lower=1)
    # Scrambles are dropbacks that became runs; the denominator is all dropbacks.
    prof["scramble_rate"] = prof["scrambles"] / (prof["dropbacks"] + prof["scrambles"]).clip(lower=1)
    prof["sack_rate_allowed"] = prof["sacks"] / prof["dropbacks"].clip(lower=1)
    return prof


def blend_player_history(prof: pd.DataFrame, player_id: str, cols: list[str],
                         halflife: float = 1.8) -> pd.Series | None:
    """Recency-weighted average of one player's seasons."""
    d = prof[prof["player_id"] == player_id]
    if d.empty:
        return None
    w = 0.5 ** ((LAST_COMPLETED_SEASON - d["season"]) / halflife)
    w = w * np.sqrt(d["dropbacks"].clip(lower=1) / 400.0).clip(upper=1.5)
    out = {}
    for c in cols:
        if c not in d.columns:
            out[c] = np.nan          # trait not measured for this position
            continue
        v = d[c].astype(float)
        m = v.notna()
        out[c] = float((v[m] * w[m]).sum() / w[m].sum()) if m.any() and w[m].sum() > 0 else np.nan
    s = pd.Series(out)
    s["_evidence"] = float(min(w.sum(), 1.0))
    return s


def projected_qb(chart: pd.DataFrame, team: str) -> dict | None:
    """The QB1 listed on the most recent depth chart."""
    d = chart[(chart["team"] == team) & (chart["pos_abb"] == "QB")].sort_values("pos_rank")
    if d.empty:
        return None
    row = d.iloc[0]
    return {"name": row["player_name"], "gsis_id": row.get("gsis_id")}


def apply_personnel_constraints(
    projected: pd.Series,
    team: str,
    chart: pd.DataFrame,
    qb_prof: pd.DataFrame,
    league: pd.Series,
) -> tuple[pd.Series, dict]:
    """Pull personnel-bound traits toward the players actually rostered.

    Returns the adjusted fingerprint and a note describing what moved and why.
    """
    out = projected.copy()
    info: dict = {"qb": None, "adjusted": {}, "qb_evidence": 0.0}

    qb = projected_qb(chart, team)
    if qb is None:
        return out, info
    info["qb"] = qb["name"]

    traits = [t for t in PERSONNEL_BOUND if t in out.index]
    hist = None
    if qb.get("gsis_id"):
        hist = blend_player_history(qb_prof, qb["gsis_id"], traits)

    if hist is None:
        # An unproven starter - rookie or a career backup with no starting
        # sample. Regress the personnel-bound traits toward the league mean
        # rather than inheriting a previous quarterback's athletic profile.
        info["qb_evidence"] = 0.0
        for t in traits:
            if t in league.index and np.isfinite(league.get(t, np.nan)):
                before = float(out[t])
                pull = 0.5 * PERSONNEL_BOUND[t]
                out[t] = (1 - pull) * before + pull * float(league[t])
                info["adjusted"][t] = {"from": before, "to": float(out[t]), "driver": "unproven starter"}
        return out, info

    evidence = float(hist.get("_evidence", 0.0))
    info["qb_evidence"] = evidence
    for t in traits:
        pv = hist.get(t, np.nan)
        if not np.isfinite(pv):
            continue
        w = PERSONNEL_BOUND[t] * evidence
        before = float(out[t])
        out[t] = (1 - w) * before + w * float(pv)
        info["adjusted"][t] = {"from": before, "to": float(out[t]), "driver": qb["name"]}
    return out, info
