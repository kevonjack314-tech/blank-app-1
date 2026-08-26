"""Situational tendencies: what a staff actually calls, and when.

A fingerprint says a team motions on 55% of snaps. A playbook says they motion
on 71% of first-and-ten from under center and almost never on third-and-long -
which is the difference between a number and something a defensive coordinator
can use. These functions pull the concrete call tendencies out of play-by-play
and charting: down-and-distance splits, run direction and gap, pass depth and
location, and the specific concept combinations a team leans on most.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PLAYS = 12  # below this a split is noise, not a tendency


def _sit(df: pd.DataFrame, label: str) -> dict | None:
    if len(df) < MIN_PLAYS:
        return None
    d = {
        "situation": label,
        "plays": len(df),
        "pass_rate": float(df["is_dropback"].mean()),
        "shotgun_rate": float(df["shotgun"].fillna(0).mean()),
        "epa": float(df["epa"].mean(skipna=True)),
        "success": float(df["success"].mean(skipna=True)) if "success" in df else np.nan,
    }
    for col, name in (("is_motion", "motion_rate"), ("is_play_action", "play_action_rate"),
                      ("is_rpo", "rpo_rate"), ("is_screen_pass", "screen_rate"),
                      ("is_no_huddle", "no_huddle_rate")):
        if col in df.columns and df[col].notna().any():
            d[name] = float(df[col].astype(float).mean())
    return d


def situational_tendencies(plays: pd.DataFrame, team: str, side: str = "offense",
                           seasons: tuple | None = None) -> pd.DataFrame:
    """Down-and-distance and field-zone breakdown for one team."""
    key = "posteam" if side == "offense" else "defteam"
    df = plays[plays[key] == team]
    if seasons:
        df = df[df["season"].isin(seasons)]
    if df.empty:
        return pd.DataFrame()

    rows = []
    situations = [
        ("1st & 10", (df["down"] == 1) & (df["ydstogo"].between(9, 11))),
        ("2nd & short (1-3)", (df["down"] == 2) & (df["ydstogo"] <= 3)),
        ("2nd & medium (4-7)", (df["down"] == 2) & df["ydstogo"].between(4, 7)),
        ("2nd & long (8+)", (df["down"] == 2) & (df["ydstogo"] >= 8)),
        ("3rd & short (1-2)", (df["down"] == 3) & (df["ydstogo"] <= 2)),
        ("3rd & medium (3-6)", (df["down"] == 3) & df["ydstogo"].between(3, 6)),
        ("3rd & long (7+)", (df["down"] == 3) & (df["ydstogo"] >= 7)),
        ("Red zone", df["yardline_100"] <= 20),
        ("Goal to go", df["goal_to_go"].fillna(0) > 0),
        ("Inside the 5", df["yardline_100"] <= 5),
        ("Backed up (own 10)", df["yardline_100"] >= 90),
        ("Two-minute drill", (df["half_seconds_remaining"] <= 120)),
        ("Leading by 8+", df["score_differential"] >= 8),
        ("Trailing by 8+", df["score_differential"] <= -8),
    ]
    for label, mask in situations:
        rec = _sit(df[mask.fillna(False)], label)
        if rec:
            rows.append(rec)
    return pd.DataFrame(rows)


def run_direction_profile(plays: pd.DataFrame, team: str, seasons: tuple | None = None) -> pd.DataFrame:
    """Where a team's runs actually go, by direction and gap."""
    df = plays[(plays["posteam"] == team) & plays["is_designed_run"]]
    if seasons:
        df = df[df["season"].isin(seasons)]
    df = df[df["run_location"].notna()]
    if df.empty:
        return pd.DataFrame()
    g = (
        df.groupby(["run_location", "run_gap"], dropna=False)
        .agg(plays=("play_id", "size"), ypc=("yards_gained", "mean"), epa=("epa", "mean"))
        .reset_index()
    )
    g["share"] = g["plays"] / g["plays"].sum()
    return g.sort_values("plays", ascending=False)


def pass_location_profile(plays: pd.DataFrame, team: str, seasons: tuple | None = None) -> pd.DataFrame:
    """Distribution of throws by field location and depth."""
    df = plays[(plays["posteam"] == team) & plays["is_dropback"]]
    if seasons:
        df = df[df["season"].isin(seasons)]
    df = df[df["pass_location"].notna() & df["pass_length"].notna()]
    if df.empty:
        return pd.DataFrame()
    g = (
        df.groupby(["pass_length", "pass_location"])
        .agg(plays=("play_id", "size"), ypa=("yards_gained", "mean"),
             epa=("epa", "mean"), comp_rate=("complete_pass", "mean"))
        .reset_index()
    )
    g["share"] = g["plays"] / g["plays"].sum()
    return g.sort_values("plays", ascending=False)


def signature_concepts(plays: pd.DataFrame, team: str, seasons: tuple | None = None,
                       min_plays: int = 25, top_n: int = 12) -> pd.DataFrame:
    """The concept combinations a team runs most, versus league baseline rates.

    A concept here is the observable shell of a call - shotgun or under centre,
    motion or not, play-action, RPO, screen - crossed with where the ball went.
    Ranking by how far a team's usage sits above the league average surfaces
    what is distinctive about them rather than what everyone does.
    """
    df = plays[plays["posteam"].notna()]
    if seasons:
        df = df[df["season"].isin(seasons)]
    if df.empty:
        return pd.DataFrame()

    # Sacks and scrambles carry no charted target, so they would collapse into
    # a phantom concept with badly negative EPA. They are plays that broke down,
    # not calls, and are excluded from the concept mix.
    df = df[~((df["sack"].fillna(0) > 0) | (df["qb_scramble"].fillna(0) > 0))]
    if df.empty:
        return pd.DataFrame()

    def tag(d: pd.DataFrame) -> pd.Series:
        parts = [pd.Series(np.where(d["shotgun"].fillna(0) > 0, "Shotgun", "Under center"), index=d.index)]
        for col, name in (("is_play_action", "play-action"), ("is_rpo", "RPO"),
                          ("is_screen_pass", "screen"), ("is_motion", "motion")):
            if col in d.columns:
                parts.append(pd.Series(np.where(d[col].fillna(False), f" + {name}", ""), index=d.index))
        # Pass depth carries the scheme signal; left/right mostly splits the
        # sample without adding meaning, so runs keep their gap and passes
        # keep their depth.
        action = np.where(
            d["is_dropback"],
            " pass (" + d["pass_length"].fillna("?").astype(str) + ")",
            " run " + d["run_location"].fillna("?").astype(str)
            + np.where(d["run_gap"].notna(), "/" + d["run_gap"].fillna("").astype(str), ""),
        )
        parts.append(pd.Series(action, index=d.index))
        return pd.concat(parts, axis=1).sum(axis=1).str.replace(r"\s+", " ", regex=True).str.strip()

    df = df.assign(concept=tag(df))
    league = df.groupby("concept").size()
    league_share = league / league.sum()

    t = df[df["posteam"] == team]
    if t.empty:
        return pd.DataFrame()
    g = (
        t.groupby("concept")
        .agg(plays=("play_id", "size"), epa=("epa", "mean"),
             yards=("yards_gained", "mean"), success=("success", "mean"))
        .reset_index()
    )
    g = g[g["plays"] >= min_plays]
    if g.empty:
        return pd.DataFrame()
    g["share"] = g["plays"] / len(t)
    g["league_share"] = g["concept"].map(league_share).fillna(0.0)
    g["lift"] = g["share"] / g["league_share"].replace(0, np.nan)
    g["over_index"] = (g["share"] - g["league_share"]) * 100
    # A signature call is one used both often and more than the league does.
    g["signature"] = g["over_index"] * np.log1p(g["lift"].clip(lower=0.01))
    return g.sort_values("signature", ascending=False).head(top_n)


def defensive_profile(plays: pd.DataFrame, team: str, seasons: tuple | None = None) -> pd.DataFrame:
    """How a defence changes its front and pressure by down and distance."""
    df = plays[plays["defteam"] == team]
    if seasons:
        df = df[df["season"].isin(seasons)]
    if df.empty:
        return pd.DataFrame()

    rows = []
    buckets = [
        ("1st & 10", (df["down"] == 1) & df["ydstogo"].between(9, 11)),
        ("2nd & long (8+)", (df["down"] == 2) & (df["ydstogo"] >= 8)),
        ("3rd & short (1-2)", (df["down"] == 3) & (df["ydstogo"] <= 2)),
        ("3rd & medium (3-6)", (df["down"] == 3) & df["ydstogo"].between(3, 6)),
        ("3rd & long (7+)", (df["down"] == 3) & (df["ydstogo"] >= 7)),
        ("Red zone", df["yardline_100"] <= 20),
        ("Goal to go", df["goal_to_go"].fillna(0) > 0),
    ]
    for label, mask in buckets:
        d = df[mask.fillna(False)]
        if len(d) < MIN_PLAYS:
            continue
        rushers = d["n_pass_rushers"] if "n_pass_rushers" in d else pd.Series(dtype=float)
        rushers = rushers[rushers > 0]
        box = d["n_defense_box"] if "n_defense_box" in d else pd.Series(dtype=float)
        box = box[box > 0]
        rows.append({
            "situation": label,
            "plays": len(d),
            "blitz_rate": float((rushers >= 5).mean()) if len(rushers) else np.nan,
            "avg_rushers": float(rushers.mean()) if len(rushers) else np.nan,
            "avg_box": float(box.mean()) if len(box) else np.nan,
            "heavy_box_rate": float((box >= 8).mean()) if len(box) else np.nan,
            "epa_allowed": float(d["epa"].mean(skipna=True)),
            "success_allowed": float(d["success"].mean(skipna=True)) if "success" in d else np.nan,
        })
    return pd.DataFrame(rows)
