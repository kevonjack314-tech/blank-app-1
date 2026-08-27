"""Coverage charting: what shells a defence plays, and who beats them.

The scheme fingerprints describe fronts and pressure - how many rushers, how
many in the box. This module covers the other half of a defensive call: what
happens behind it. Man or zone, and which shell.

The source is nflverse participation charting, which labels ``COVER_0`` through
``COVER_6``, ``2_MAN`` and a man/zone flag on roughly every charted pass play.
It also carries the route each receiver ran, which makes it possible to ask not
just what a defence plays but what an offence does against it.

Coverage is charted on pass plays only, so about half of all snaps carry a
label. Rates here are therefore shares of *charted dropbacks*, not of all plays.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PRIOR_STRENGTH, normalize_team

# Grouped so that thin shells do not fragment the sample. Single-high and
# two-high is the distinction that actually changes how an offence attacks.
SHELL_GROUPS = {
    "COVER_0": "man-free (0)",
    "COVER_1": "single-high man (1)",
    "2_MAN": "two-high man (2-man)",
    "COVER_2": "two-high zone (2)",
    "COVER_3": "single-high zone (3)",
    "COVER_4": "quarters (4)",
    "COVER_6": "quarter-quarter-half (6)",
    "COVER_9": "match (9)",
    "COMBO": "combo",
    "BLOWN": "blown",
}
SINGLE_HIGH = {"COVER_1", "COVER_3", "COVER_0"}
TWO_HIGH = {"COVER_2", "COVER_4", "COVER_6", "2_MAN"}

MIN_CHARTED = 80  # below this a team's coverage profile is not stable


def attach(plays: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    """Join coverage charting onto prepared plays."""
    if part is None or part.empty:
        return plays
    cols = [c for c in ("game_id", "play_id", "defense_man_zone_type",
                        "defense_coverage_type", "route", "was_pressure",
                        "offense_formation", "offense_personnel",
                        "defense_personnel", "time_to_throw") if c in part.columns]
    return plays.merge(part[cols], on=["game_id", "play_id"], how="left")


def _personnel_group(s: pd.Series) -> pd.Series:
    """Reduce a personnel string to the standard RB-TE shorthand (11, 12, 21)."""
    rb = s.str.extract(r"(\d+)\s*RB", expand=False).astype(float).fillna(0)
    te = s.str.extract(r"(\d+)\s*TE", expand=False).astype(float).fillna(0)
    out = (rb * 10 + te).astype(int).astype(str).str.zfill(2)
    return out.where(s.notna(), other=pd.NA)


def defense_coverage_profile(plays: pd.DataFrame, team: str | None = None,
                             seasons: tuple | None = None) -> pd.DataFrame:
    """How often each defence plays each coverage, and what it allows."""
    if "defense_coverage_type" not in plays.columns:
        return pd.DataFrame()
    df = plays[plays["defense_coverage_type"].notna()]
    if seasons:
        df = df[df["season"].isin(seasons)]
    if team:
        df = df[df["defteam"] == normalize_team(team)]
    if df.empty:
        return pd.DataFrame()

    g = (
        df.groupby("defense_coverage_type")
        .agg(plays=("play_id", "size"), epa=("epa", "mean"),
             ypa=("yards_gained", "mean"),
             comp_rate=("complete_pass", "mean"),
             adot=("air_yards", "mean"))
        .reset_index()
    )
    g["shell"] = g["defense_coverage_type"].map(SHELL_GROUPS).fillna(g["defense_coverage_type"])
    g["rate"] = g["plays"] / g["plays"].sum()
    return g.sort_values("plays", ascending=False)


def coverage_fingerprint(plays: pd.DataFrame, seasons: tuple | None = None) -> pd.DataFrame:
    """One row per defence: man rate, shell mix, and what each surrenders."""
    if "defense_man_zone_type" not in plays.columns:
        return pd.DataFrame()
    df = plays[plays["defense_coverage_type"].notna() | plays["defense_man_zone_type"].notna()]
    if seasons:
        df = df[df["season"].isin(seasons)]
    if df.empty:
        return pd.DataFrame()

    rows = []
    for (season, team), grp in df.groupby(["season", "defteam"]):
        if not team or len(grp) < MIN_CHARTED:
            continue
        mz = grp["defense_man_zone_type"].dropna()
        cov = grp["defense_coverage_type"].dropna()
        rec = {
            "season": season, "team": team, "charted_plays": len(grp),
            "man_rate": float((mz == "MAN_COVERAGE").mean()) if len(mz) else np.nan,
            "zone_rate": float((mz == "ZONE_COVERAGE").mean()) if len(mz) else np.nan,
            "single_high_rate": float(cov.isin(SINGLE_HIGH).mean()) if len(cov) else np.nan,
            "two_high_rate": float(cov.isin(TWO_HIGH).mean()) if len(cov) else np.nan,
            "cover0_rate": float((cov == "COVER_0").mean()) if len(cov) else np.nan,
        }
        for shell in ("COVER_1", "COVER_2", "COVER_3", "COVER_4", "COVER_6", "2_MAN"):
            key = shell.lower().replace("_", "") + "_rate"     # COVER_1 -> cover1_rate
            rec[key] = float((cov == shell).mean()) if len(cov) else np.nan
        # What it gives up, by structure.
        for label, mask in (("vs_man", grp["defense_man_zone_type"] == "MAN_COVERAGE"),
                            ("vs_zone", grp["defense_man_zone_type"] == "ZONE_COVERAGE")):
            sub = grp[mask.fillna(False)]
            rec[f"epa_{label}"] = float(sub["epa"].mean(skipna=True)) if len(sub) > 20 else np.nan
        if "was_pressure" in grp.columns:
            pr = grp["was_pressure"].dropna()
            rec["pressure_rate"] = float(pr.astype(float).mean()) if len(pr) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def offense_vs_coverage(plays: pd.DataFrame, team: str, seasons: tuple | None = None) -> pd.DataFrame:
    """How one offence performs against man versus zone, and by shell."""
    if "defense_man_zone_type" not in plays.columns:
        return pd.DataFrame()
    df = plays[(plays["posteam"] == normalize_team(team)) & plays["is_dropback"]]
    if seasons:
        df = df[df["season"].isin(seasons)]
    df = df[df["defense_man_zone_type"].notna()]
    if df.empty:
        return pd.DataFrame()

    rows = []
    for label, mask in [("vs man", df["defense_man_zone_type"] == "MAN_COVERAGE"),
                        ("vs zone", df["defense_man_zone_type"] == "ZONE_COVERAGE")]:
        sub = df[mask]
        if len(sub) < 40:
            continue
        rows.append({
            "split": label, "plays": len(sub),
            "epa": float(sub["epa"].mean(skipna=True)),
            "ypa": float(sub["yards_gained"].mean(skipna=True)),
            "comp_rate": float(sub["complete_pass"].mean(skipna=True)),
            "adot": float(sub["air_yards"].mean(skipna=True)),
            "sack_rate": float(sub["sack"].fillna(0).mean()),
            "motion_rate": float(sub["is_motion"].fillna(False).mean()) if "is_motion" in sub else np.nan,
        })
    return pd.DataFrame(rows)


def route_profile(plays: pd.DataFrame, team: str | None = None,
                  seasons: tuple | None = None, top_n: int = 12) -> pd.DataFrame:
    """Route distribution and productivity - the concept menu, by name."""
    if "route" not in plays.columns:
        return pd.DataFrame()
    df = plays[plays["route"].notna()]
    if seasons:
        df = df[df["season"].isin(seasons)]
    league = df.groupby("route").size()
    league_share = league / league.sum()

    if team:
        df = df[df["posteam"] == normalize_team(team)]
    if df.empty:
        return pd.DataFrame()
    g = (
        df.groupby("route")
        .agg(plays=("play_id", "size"), epa=("epa", "mean"),
             yards=("yards_gained", "mean"), comp_rate=("complete_pass", "mean"))
        .reset_index()
    )
    g["rate"] = g["plays"] / g["plays"].sum()
    g["league_rate"] = g["route"].map(league_share).fillna(0.0)
    g["lift"] = g["rate"] / g["league_rate"].replace(0, np.nan)
    return g.sort_values("plays", ascending=False).head(top_n)


def personnel_profile(plays: pd.DataFrame, team: str, seasons: tuple | None = None) -> pd.DataFrame:
    """Offensive personnel groupings (11, 12, 21) and what they produce."""
    if "offense_personnel" not in plays.columns:
        return pd.DataFrame()
    df = plays[(plays["posteam"] == normalize_team(team)) & plays["offense_personnel"].notna()]
    if seasons:
        df = df[df["season"].isin(seasons)]
    if df.empty:
        return pd.DataFrame()
    df = df.assign(grouping=_personnel_group(df["offense_personnel"].astype(str)))
    g = (
        df.groupby("grouping")
        .agg(plays=("play_id", "size"), pass_rate=("is_dropback", "mean"),
             epa=("epa", "mean"), yards=("yards_gained", "mean"))
        .reset_index()
    )
    g["rate"] = g["plays"] / g["plays"].sum()
    return g[g["plays"] >= 25].sort_values("plays", ascending=False)


def coverage_matchup(off_team: str, def_team: str, plays: pd.DataFrame,
                     seasons: tuple | None = None) -> list[str]:
    """Plain-language read of one offence against one defence's coverage habits."""
    notes: list[str] = []
    d = coverage_fingerprint(plays, seasons)
    if d.empty:
        return notes
    row = d[d["team"] == normalize_team(def_team)]
    if row.empty:
        return notes
    r = row.iloc[0]
    lg_man = d["man_rate"].mean()
    lg_single = d["single_high_rate"].mean()

    o = offense_vs_coverage(plays, off_team, seasons)
    if not o.empty and len(o) == 2:
        man = o[o["split"] == "vs man"]
        zone = o[o["split"] == "vs zone"]
        if len(man) and len(zone):
            gap = float(man["epa"].iloc[0]) - float(zone["epa"].iloc[0])
            better = "man" if gap > 0 else "zone"
            if abs(gap) > 0.05:
                notes.append(
                    f"{off_team} are markedly better against {better} "
                    f"({float(man['epa'].iloc[0]):+.3f} EPA vs man, "
                    f"{float(zone['epa'].iloc[0]):+.3f} vs zone), and "
                    f"{def_team} play man on {r['man_rate']*100:.0f}% of charted snaps "
                    f"against a {lg_man*100:.0f}% league rate."
                )
    if np.isfinite(r.get("single_high_rate", np.nan)):
        if r["single_high_rate"] > lg_single + 0.06:
            notes.append(
                f"{def_team} sit in single-high on {r['single_high_rate']*100:.0f}% of "
                f"charted snaps ({lg_single*100:.0f}% league) - one fewer deep defender, "
                "so the deep shot is there if the protection holds."
            )
        elif r["single_high_rate"] < lg_single - 0.06:
            notes.append(
                f"{def_team} play two-high on {r['two_high_rate']*100:.0f}% of charted "
                "snaps, taking away the deep ball and conceding underneath throws "
                "and the run."
            )
    if np.isfinite(r.get("cover0_rate", np.nan)) and r["cover0_rate"] > 0.03:
        notes.append(
            f"{def_team} bring Cover 0 on {r['cover0_rate']*100:.1f}% of charted snaps - "
            "no safety help, so a beaten protection is a touchdown and a hot read is a big play."
        )
    return notes
