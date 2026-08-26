"""Scheme fingerprints: turn play-by-play into coordinator tendency vectors.

An offensive or defensive staff leaves a measurable signature - how often they
drop back relative to situation, how much pre-snap motion and play-action they
use, which gaps they run at, how many defenders they put in the box, how often
they send a fifth rusher. These functions reduce a team-season to that vector so
staffs can be compared, and so a coach's signature can be carried to a new team.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PRIOR_STRENGTH, allowed_season_types, normalize_team

# Plays that reflect deliberate play-calling rather than clock management.
NEUTRAL_WP = (0.20, 0.80)

PBP_COLUMNS = [
    "game_id", "play_id", "season", "week", "season_type", "posteam", "defteam",
    "play_type", "pass", "rush", "qb_dropback", "qb_scramble", "sack", "shotgun",
    "no_huddle", "down", "ydstogo", "yardline_100", "goal_to_go", "yards_gained",
    "air_yards", "yards_after_catch", "epa", "success", "wp", "xpass", "pass_oe",
    "half_seconds_remaining", "game_seconds_remaining", "fixed_drive",
    "fixed_drive_result", "touchdown", "pass_touchdown", "rush_touchdown",
    "complete_pass", "interception", "fumble_lost", "penalty",
    "receiver_player_id", "rusher_player_id", "passer_player_id",
    "qb_kneel", "qb_spike", "play", "special", "run_location", "run_gap",
    "pass_location", "pass_length", "first_down", "third_down_converted",
    "fourth_down_converted", "fourth_down_failed", "score_differential",
    "drive", "qb_hit", "cpoe",
]


def _rate(numer: float, denom: float, prior: float, strength: float) -> float:
    """Empirical-Bayes rate: shrink a small sample toward a league prior."""
    if denom is None or denom <= 0 or not np.isfinite(denom):
        return float(prior)
    return float((numer + prior * strength) / (denom + strength))


def prepare_plays(pbp: pd.DataFrame, charting: pd.DataFrame | None = None,
                  include_postseason: bool | None = None) -> pd.DataFrame:
    """Filter to real scrimmage plays and attach charting flags.

    Preseason is dropped unconditionally: starters play a series and sit, the
    play-calling is deliberately vanilla, and most snaps go to players who will
    not be on the roster. Postseason is dropped by default because only some
    teams have any, which would hand extra sample to teams that were already
    good.
    """
    df = pbp.copy()

    if "season_type" in df.columns:
        allowed = allowed_season_types(include_postseason)
        df = df[df["season_type"].astype(str).str.upper().isin(allowed)]
    elif "game_type" in df.columns:
        allowed = allowed_season_types(include_postseason)
        gt = df["game_type"].astype(str).str.upper()
        # Playoff rounds are labelled individually in some feeds.
        post_rounds = {"WC", "DIV", "CON", "SB"}
        keep = gt.eq("REG") | (gt.isin(post_rounds) if "POST" in allowed else False)
        df = df[keep]
    for col in ("posteam", "defteam"):
        if col in df.columns:
            df[col] = df[col].map(normalize_team)

    keep = df["posteam"].notna()
    if "special" in df.columns:
        keep &= df["special"].fillna(0) == 0
    for col in ("qb_kneel", "qb_spike"):
        if col in df.columns:
            keep &= df[col].fillna(0) == 0
    if "play_type" in df.columns:
        keep &= df["play_type"].isin(["pass", "run"])
    df = df[keep].copy()

    if charting is not None and not charting.empty:
        cols = [
            "nflverse_game_id", "nflverse_play_id", "is_motion", "is_play_action",
            "is_rpo", "is_screen_pass", "is_no_huddle", "is_qb_out_of_pocket",
            "n_blitzers", "n_pass_rushers", "n_defense_box", "n_offense_backfield",
            "is_qb_sneak", "is_drop", "is_catchable_ball", "is_contested_ball",
        ]
        ch = charting[[c for c in cols if c in charting.columns]].copy()
        ch = ch.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
        df = df.merge(ch, on=["game_id", "play_id"], how="left")

    df["neutral"] = df["wp"].between(*NEUTRAL_WP, inclusive="both").fillna(False)
    df["early_down"] = df["down"].isin([1, 2])
    df["is_dropback"] = df["qb_dropback"].fillna(0).astype(float) > 0
    df["is_designed_run"] = (df["rush"].fillna(0) > 0) & (df["qb_scramble"].fillna(0) == 0)
    df["red_zone"] = df["yardline_100"] <= 20
    df["explosive"] = np.where(
        df["pass"].fillna(0) > 0, df["yards_gained"] >= 20, df["yards_gained"] >= 10
    )
    return df


def _pace(df: pd.DataFrame) -> float:
    """Median seconds burned per neutral-script snap (lower = faster tempo)."""
    d = df[df["neutral"] & df["early_down"]].sort_values(["game_id", "fixed_drive", "play_id"])
    delta = d.groupby(["game_id", "fixed_drive"])["game_seconds_remaining"].diff(-1)
    delta = delta[(delta > 0) & (delta < 60)]
    if len(delta) < 20:
        return np.nan
    lo, hi = delta.quantile([0.05, 0.95])
    return float(delta[delta.between(lo, hi)].mean())


def offense_fingerprint(df: pd.DataFrame, league: dict | None = None) -> dict:
    """Reduce one team's offensive snaps to a tendency vector."""
    league = league or {}
    n = len(df)
    if n == 0:
        return {}
    S = PRIOR_STRENGTH["team_rate"]
    neutral = df[df["neutral"]]
    early_neutral = df[df["neutral"] & df["early_down"]]
    dropbacks = df[df["is_dropback"]]
    runs = df[df["is_designed_run"]]
    rz = df[df["red_zone"]]
    g2g = df[df["goal_to_go"].fillna(0) > 0]

    def rate(mask_sum, total, key, default=0.0):
        return _rate(mask_sum, total, league.get(key, default), S)

    out = {
        "plays": n,
        "games": df["game_id"].nunique(),
        "plays_per_game": n / max(df["game_id"].nunique(), 1),
        "sec_per_play": _pace(df),
        # --- pass/run identity -------------------------------------------------
        "dropback_rate": rate(df["is_dropback"].sum(), n, "dropback_rate", 0.57),
        "early_down_pass_rate": rate(
            early_neutral["is_dropback"].sum(), len(early_neutral), "early_down_pass_rate", 0.53
        ),
        "proe": float(neutral["pass_oe"].mean()) if len(neutral) else np.nan,
        # --- formation & tempo -------------------------------------------------
        "shotgun_rate": rate(df["shotgun"].fillna(0).sum(), n, "shotgun_rate", 0.65),
        "no_huddle_rate": rate(df["no_huddle"].fillna(0).sum(), n, "no_huddle_rate", 0.07),
        # --- charted scheme markers -------------------------------------------
        "motion_rate": _charted_rate(df, "is_motion", league, S),
        "play_action_rate": _charted_rate(dropbacks, "is_play_action", league, S),
        "rpo_rate": _charted_rate(df, "is_rpo", league, S),
        "screen_rate": _charted_rate(dropbacks, "is_screen_pass", league, S),
        # --- passing shape -----------------------------------------------------
        "adot": float(df["air_yards"].mean(skipna=True)),
        "deep_rate": rate((df["air_yards"] >= 20).sum(), df["air_yards"].notna().sum(), "deep_rate", 0.11),
        "yac_share": _yac_share(df),
        "cpoe": float(df["cpoe"].mean(skipna=True)) if "cpoe" in df else np.nan,
        # --- rushing shape -----------------------------------------------------
        "outside_run_rate": rate(
            runs["run_location"].isin(["left", "right"]).sum(), len(runs), "outside_run_rate", 0.62
        ),
        "run_gap_end_rate": rate(
            (runs["run_gap"] == "end").sum(), runs["run_gap"].notna().sum(), "run_gap_end_rate", 0.28
        ),
        "qb_designed_run_rate": _qb_designed_run_rate(df, runs, league, S),
        "scramble_rate": rate(
            df["qb_scramble"].fillna(0).sum(), max(df["is_dropback"].sum(), 1), "scramble_rate", 0.07
        ),
        # --- situational aggression -------------------------------------------
        "rz_pass_rate": rate(rz["is_dropback"].sum(), len(rz), "rz_pass_rate", 0.52),
        "g2g_run_rate": rate(len(g2g) - g2g["is_dropback"].sum(), len(g2g), "g2g_run_rate", 0.55),
        # --- results (used as efficiency priors, not identity) -----------------
        "epa_per_play": float(df["epa"].mean(skipna=True)),
        "pass_epa": float(dropbacks["epa"].mean(skipna=True)) if len(dropbacks) else np.nan,
        "rush_epa": float(runs["epa"].mean(skipna=True)) if len(runs) else np.nan,
        "success_rate": float(df["success"].mean(skipna=True)) if "success" in df else np.nan,
        "explosive_rate": float(df["explosive"].mean()),
        "sack_rate_allowed": rate(
            df["sack"].fillna(0).sum(), max(df["is_dropback"].sum(), 1), "sack_rate_allowed", 0.065
        ),
        "turnover_rate": rate(
            df["interception"].fillna(0).sum() + df["fumble_lost"].fillna(0).sum(),
            n, "turnover_rate", 0.025,
        ),
        "points_per_drive": _points_per_drive(df),
        "rz_td_rate": _rz_td_rate(df),
    }
    return out


def _charted_rate(df: pd.DataFrame, col: str, league: dict, strength: float) -> float:
    if col not in df.columns:
        return np.nan
    s = df[col]
    valid = s.notna()
    if valid.sum() == 0:
        return np.nan
    return _rate(s[valid].astype(float).sum(), valid.sum(), league.get(col.replace("is_", "") + "_rate", 0.2), strength)


def _yac_share(df: pd.DataFrame) -> float:
    comp = df[df["complete_pass"].fillna(0) > 0]
    if comp.empty:
        return np.nan
    total = comp["yards_gained"].sum()
    yac = comp["yards_after_catch"].sum(skipna=True)
    return float(yac / total) if total else np.nan


def _qb_designed_run_rate(df: pd.DataFrame, runs: pd.DataFrame, league: dict, strength: float) -> float:
    """Designed quarterback runs per offensive snap.

    Scrambles are excluded upstream, so this isolates called QB run game -
    zone read, power read, sneaks - which is a scheme choice rather than a
    reaction to pressure.
    """
    if df.empty or runs.empty:
        return np.nan
    passers = df.loc[df["is_dropback"], "passer_player_id"].dropna()
    if passers.empty:
        return np.nan
    # Anyone taking a meaningful share of dropbacks counts as a quarterback.
    share = passers.value_counts(normalize=True)
    qbs = set(share[share >= 0.10].index)
    qb_runs = runs["rusher_player_id"].isin(qbs).sum()
    return _rate(qb_runs, len(df), league.get("qb_designed_run_rate", 0.045), strength)


def fourth_down_rates(pbp: pd.DataFrame) -> pd.DataFrame:
    """Go-for-it rate on 4th down, measured before non-scrimmage plays are cut.

    Restricted to the decision zone - 5 or fewer to gain, outside field-goal
    garbage - where punting, kicking and going are all live options.
    """
    d = pbp[
        (pbp["down"] == 4)
        & (pbp["ydstogo"] <= 5)
        & (pbp["yardline_100"].between(20, 70))
        & (pbp["wp"].between(*NEUTRAL_WP))
        & (pbp["play_type"].isin(["pass", "run", "punt", "field_goal"]))
    ].copy()
    if d.empty:
        return pd.DataFrame(columns=["season", "team", "fourth_down_go_rate", "fourth_down_opps"])
    d["went"] = d["play_type"].isin(["pass", "run"]).astype(float)
    g = d.groupby(["season", "posteam"]).agg(
        fourth_down_go_rate=("went", "mean"), fourth_down_opps=("went", "size")
    ).reset_index().rename(columns={"posteam": "team"})
    return g


def _points_per_drive(df: pd.DataFrame) -> float:
    if "fixed_drive_result" not in df.columns:
        return np.nan
    drives = df.groupby(["game_id", "fixed_drive"])["fixed_drive_result"].first()
    pts = drives.map({"Touchdown": 7.0, "Field goal": 3.0, "Opp touchdown": -7.0, "Safety": -2.0}).fillna(0.0)
    return float(pts.mean()) if len(drives) else np.nan


def _rz_td_rate(df: pd.DataFrame) -> float:
    """Share of red-zone trips that end in an offensive touchdown."""
    rz = df[df["red_zone"]]
    if rz.empty:
        return np.nan
    trips = rz.groupby(["game_id", "fixed_drive"])["fixed_drive_result"].first()
    return float((trips == "Touchdown").mean()) if len(trips) else np.nan


def defense_fingerprint(df: pd.DataFrame, league: dict | None = None) -> dict:
    """Reduce one team's defensive snaps to a tendency vector."""
    league = league or {}
    n = len(df)
    if n == 0:
        return {}
    S = PRIOR_STRENGTH["team_rate"]
    dropbacks = df[df["is_dropback"]]
    runs = df[df["is_designed_run"]]

    box = df["n_defense_box"] if "n_defense_box" in df.columns else pd.Series(dtype=float)
    box = box[box > 0]
    rushers = df["n_pass_rushers"] if "n_pass_rushers" in df.columns else pd.Series(dtype=float)
    rushers = rushers[rushers > 0]
    blitz = df["n_blitzers"] if "n_blitzers" in df.columns else pd.Series(dtype=float)

    out = {
        "plays": n,
        "games": df["game_id"].nunique(),
        # --- front & pressure structure ---------------------------------------
        "blitz_rate": float((rushers >= 5).mean()) if len(rushers) else np.nan,
        "heavy_blitz_rate": float((rushers >= 6).mean()) if len(rushers) else np.nan,
        "avg_pass_rushers": float(rushers.mean()) if len(rushers) else np.nan,
        "avg_box": float(box.mean()) if len(box) else np.nan,
        "light_box_rate": float((box <= 6).mean()) if len(box) else np.nan,
        "heavy_box_rate": float((box >= 8).mean()) if len(box) else np.nan,
        "sack_rate": _rate(df["sack"].fillna(0).sum(), max(len(dropbacks), 1), league.get("sack_rate", 0.065), S),
        "qb_hit_rate": _rate(
            df["qb_hit"].fillna(0).sum() if "qb_hit" in df else 0,
            max(len(dropbacks), 1), league.get("qb_hit_rate", 0.13), S,
        ),
        # --- what it surrenders ------------------------------------------------
        "epa_allowed": float(df["epa"].mean(skipna=True)),
        "pass_epa_allowed": float(dropbacks["epa"].mean(skipna=True)) if len(dropbacks) else np.nan,
        "rush_epa_allowed": float(runs["epa"].mean(skipna=True)) if len(runs) else np.nan,
        "success_allowed": float(df["success"].mean(skipna=True)) if "success" in df else np.nan,
        "explosive_allowed": float(df["explosive"].mean()),
        "adot_allowed": float(df["air_yards"].mean(skipna=True)),
        "yac_allowed_share": _yac_share(df),
        "rz_td_allowed": _rz_td_rate(df),
        "points_per_drive_allowed": _points_per_drive(df),
        "dropback_rate_faced": _rate(len(dropbacks), n, 0.57, S),
        "ypc_allowed": float(runs["yards_gained"].mean(skipna=True)) if len(runs) else np.nan,
        "ypa_allowed": float(dropbacks["yards_gained"].mean(skipna=True)) if len(dropbacks) else np.nan,
    }
    return out


def build_fingerprints(plays: pd.DataFrame, raw_pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Offense and defense fingerprints for every team-season in ``plays``.

    ``raw_pbp`` is the unfiltered play-by-play; when supplied it is used for
    decisions like fourth-down aggression that need the plays this module
    otherwise filters away.
    """
    rows = []
    for (season, team), grp in plays.groupby(["season", "posteam"]):
        if not team:
            continue
        rec = {"season": season, "team": team, "side": "offense"}
        rec.update(offense_fingerprint(grp))
        rows.append(rec)
    for (season, team), grp in plays.groupby(["season", "defteam"]):
        if not team:
            continue
        rec = {"season": season, "team": team, "side": "defense"}
        rec.update(defense_fingerprint(grp))
        rows.append(rec)
    fp = pd.DataFrame(rows)

    if raw_pbp is not None and not raw_pbp.empty:
        fd = fourth_down_rates(raw_pbp)
        if not fd.empty:
            fp = fp.merge(fd, on=["season", "team"], how="left")
            # Fourth-down aggression is an offensive call; leave defense rows blank.
            fp.loc[fp["side"] != "offense", ["fourth_down_go_rate", "fourth_down_opps"]] = np.nan
    return fp


def league_means(fp: pd.DataFrame, side: str, season: int | None = None) -> pd.Series:
    d = fp[fp["side"] == side]
    if season is not None:
        d = d[d["season"] == season]
    return d.select_dtypes(include=[np.number]).mean(numeric_only=True)


# Identity traits that genuinely travel with a coaching staff, as opposed to
# result columns (EPA, points) that mostly measure the roster they inherited.
OFFENSE_IDENTITY = [
    "early_down_pass_rate", "proe", "shotgun_rate", "no_huddle_rate", "sec_per_play",
    "motion_rate", "play_action_rate", "rpo_rate", "screen_rate", "adot", "deep_rate",
    "yac_share", "outside_run_rate", "run_gap_end_rate", "qb_designed_run_rate",
    "scramble_rate", "sack_rate_allowed", "rz_pass_rate", "g2g_run_rate",
    "plays_per_game", "fourth_down_go_rate",
]
DEFENSE_IDENTITY = [
    "blitz_rate", "heavy_blitz_rate", "avg_pass_rushers", "avg_box",
    "light_box_rate", "heavy_box_rate",
]
