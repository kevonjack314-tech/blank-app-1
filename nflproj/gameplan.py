"""Weekly game planning: scheme collisions and the players a matchup unlocks.

A coordinator does not look at a season average. He looks at what this opponent
does badly, on which downs, out of which structures, and asks which of his own
players is positioned to attack it.

Two things here.

**Scheme collisions** put one team's offensive identity against the other's
defensive habits on the specific axes where they interact - motion against man
coverage, play-action against a four-man rush, outside runs against light boxes,
three-receiver sets against heavy personnel. Each read is generated from measured
rates on both sides, not written in advance, so it changes when the data does.

**X-factor players** are the ones who can change the trajectory of a game on any
given Sunday - not the steadiest producers, and not merely the ones with a
favourable matchup. What matters is the top of the range: how high the ceiling
goes, how often the projection produces a genuinely game-breaking line, and
whether the player is the sort who takes one snap eighty yards. A back projected
for a reliable 70 yards is valuable and is not an X-factor; a receiver projected
for 55 with a real chance of 140 and two scores is.

All three ingredients are measured. Ceiling and boom probability come from the
simulated distribution; explosiveness comes from how often the player's own
touches have actually gone twenty and forty yards. Matchup sensitivity is
reported alongside, because it is useful, but it is not what makes an X-factor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import coverage as cvg
from . import projections as pj
from . import schemes
from .config import TEAM_NAMES, normalize_team


def scheme_collisions(off_team: str, def_team: str, projections_map: dict,
                      cov_plays: pd.DataFrame, league_off: pd.Series,
                      league_def: pd.Series, seasons: tuple | None = None) -> list[dict]:
    """Where one offence's identity meets the other defence's habits."""
    off = projections_map[off_team]["offense"]["projected"]
    dfn = projections_map[def_team]["defense"]["projected"]
    notes: list[dict] = []

    def _read(label, edge, detail, favours):
        notes.append({"axis": label, "edge": edge, "detail": detail, "favours": favours})

    # --- motion against man coverage -------------------------------------
    cov = cvg.coverage_fingerprint(cov_plays, seasons) if cov_plays is not None else pd.DataFrame()
    man_rate = lg_man = np.nan
    if not cov.empty:
        row = cov[cov["team"] == normalize_team(def_team)]
        lg_man = float(cov["man_rate"].mean())
        if len(row):
            man_rate = float(row["man_rate"].iloc[0])
    motion = float(off.get("motion_rate", np.nan))
    lg_motion = float(league_off.get("motion_rate", np.nan))
    if np.isfinite(motion) and np.isfinite(man_rate):
        if motion > lg_motion * 1.10 and man_rate > lg_man * 1.10:
            _read("Motion vs man",
                  f"{off_team} motion {motion*100:.0f}% · {def_team} man {man_rate*100:.0f}%",
                  "Motion against man forces defenders to travel and declares the "
                  "coverage pre-snap. This is the matchup that produces manufactured "
                  "touches and rub concepts.", off_team)
        elif motion > lg_motion * 1.10 and man_rate < lg_man * 0.90:
            _read("Motion vs zone",
                  f"{off_team} motion {motion*100:.0f}% · {def_team} man only {man_rate*100:.0f}%",
                  "Heavy motion against a zone team buys less: nobody follows, so it "
                  "reveals little and the leverage has to be won after the snap.", "neutral")

    # --- play-action against the pass rush --------------------------------
    pa = float(off.get("play_action_rate", np.nan))
    lg_pa = float(league_off.get("play_action_rate", np.nan))
    blitz = float(dfn.get("blitz_rate", np.nan))
    lg_blitz = float(league_def.get("blitz_rate", np.nan))
    if np.isfinite(pa) and np.isfinite(blitz):
        if pa > lg_pa * 1.10 and blitz < lg_blitz * 0.90:
            _read("Play-action vs four-man rush",
                  f"{off_team} play-action {pa*100:.0f}% · {def_team} blitz {blitz*100:.0f}%",
                  "Play-action holds linebackers but the coverage behind stays whole. "
                  "Expect the throws to come off schedule rather than wide open.", "neutral")
        elif pa > lg_pa * 1.10 and blitz > lg_blitz * 1.10:
            _read("Play-action vs pressure",
                  f"{off_team} play-action {pa*100:.0f}% · {def_team} blitz {blitz*100:.0f}%",
                  "Deep play-action against a blitzing team is the highest-variance "
                  "exchange on the field: explosive when the protection holds, a sack "
                  "when it does not.", "variance")

    # --- run game against the box -----------------------------------------
    box = float(dfn.get("avg_box", np.nan))
    lg_box = float(league_def.get("avg_box", np.nan))
    outside = float(off.get("outside_run_rate", np.nan))
    lg_outside = float(league_off.get("outside_run_rate", np.nan))
    if np.isfinite(box) and np.isfinite(outside):
        if box < lg_box - 0.05 and outside > lg_outside * 1.05:
            _read("Perimeter runs vs light boxes",
                  f"{def_team} box {box:.2f} vs {lg_box:.2f} league · "
                  f"{off_team} outside {outside*100:.0f}%",
                  "Light boxes and a perimeter run game is the cleanest structural "
                  "advantage available. Expect the run rate to climb if it works early.",
                  off_team)
        elif box > lg_box + 0.05:
            _read("Loaded box",
                  f"{def_team} box {box:.2f} vs {lg_box:.2f} league",
                  f"{def_team} commit bodies to the run, so {off_team} will have to "
                  "throw to move it - which lifts pass volume and lowers rushing "
                  "efficiency.", def_team)

    # --- deep shots against the shell -------------------------------------
    if not cov.empty:
        row = cov[cov["team"] == normalize_team(def_team)]
        if len(row):
            single = float(row["single_high_rate"].iloc[0])
            lg_single = float(cov["single_high_rate"].mean())
            adot = float(off.get("adot", np.nan))
            lg_adot = float(league_off.get("adot", np.nan))
            if np.isfinite(single) and np.isfinite(adot):
                if single > lg_single + 0.05 and adot > lg_adot:
                    _read("Deep shots vs single-high",
                          f"{def_team} single-high {single*100:.0f}% · "
                          f"{off_team} aDOT {adot:.1f}",
                          "One deep defender against an offence that pushes the ball "
                          "downfield. The shot is there if the protection holds.", off_team)
                elif single < lg_single - 0.05 and adot > lg_adot:
                    _read("Deep shots vs two-high",
                          f"{def_team} two-high {(1-single)*100:.0f}% · "
                          f"{off_team} aDOT {adot:.1f}",
                          "Two-high takes the deep ball away from a team that wants it. "
                          "Expect checkdowns and a heavier run share than usual.", def_team)

    # --- tempo -------------------------------------------------------------
    sec = float(off.get("sec_per_play", np.nan))
    lg_sec = float(league_off.get("sec_per_play", np.nan))
    if np.isfinite(sec) and sec < lg_sec - 1.0:
        _read("Tempo", f"{off_team} {sec:.1f}s per play vs {lg_sec:.1f}s league",
              "A fast offence limits substitution, which pins the defence in whatever "
              "personnel it was in and inflates the play count for both sides.", off_team)
    return notes


# What counts as a game-breaking line, by position. These are the performances
# that swing a result rather than fill a box score.
BOOM_THRESHOLDS = {
    "QB": {"pass_yards": 300.0, "total_td": 1.0, "pass_td": 3.0},
    "RB": {"scrimmage_yards": 120.0, "total_td": 2.0},
    "FB": {"scrimmage_yards": 60.0, "total_td": 1.0},
    "WR": {"scrimmage_yards": 110.0, "total_td": 2.0},
    "TE": {"scrimmage_yards": 90.0, "total_td": 2.0},
}

# League-average explosive rates, for scaling a player's own.
LEAGUE_EXPLOSIVE = {"rec": 0.138, "rush": 0.023}


def explosive_profile(plays: pd.DataFrame, min_touches: int = 25) -> pd.DataFrame:
    """How often each player's touches actually break for twenty or forty yards.

    This is the part of an X-factor that no distribution of totals captures: a
    player who gains his yards in chunks can change a game on one snap, and one
    who grinds for the same total cannot.
    """
    rec = plays[plays["receiver_player_id"].notna() & (plays["complete_pass"].fillna(0) > 0)]
    run = plays[plays["is_designed_run"] & plays["rusher_player_id"].notna()]

    frames = []
    for df, key, kind in ((rec, "receiver_player_id", "rec"), (run, "rusher_player_id", "rush")):
        if df.empty:
            continue
        g = (df.groupby(key)["yards_gained"]
             .agg(touches="size",
                  explosive=lambda s: float((s >= 20).mean()),
                  breakaway=lambda s: float((s >= 40).mean()),
                  longest="max")
             .reset_index().rename(columns={key: "player_id"}))
        g = g[g["touches"] >= min_touches]
        g["kind"] = kind
        g["explosive_index"] = g["explosive"] / LEAGUE_EXPLOSIVE[kind]
        frames.append(g)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _boom_probability(proj, position: str) -> float:
    """Chance of a line that actually swings a game."""
    thresholds = BOOM_THRESHOLDS.get(position)
    if not thresholds:
        return float("nan")
    hit = None
    for stat, cut in thresholds.items():
        s = proj.samples.get(stat)
        if s is None or len(s) == 0:
            continue
        m = s >= cut
        hit = m if hit is None else (hit | m)
    return float(hit.mean()) if hit is not None else float("nan")


def x_factors(team: str, opponent: str, ctx, projections_map: dict,
              usage_hist: pd.DataFrame, sampler: pj.TouchSampler,
              game_ctx, league_def: pd.Series, env: dict | None = None,
              n_sims: int = 8000, top_n: int = 6) -> pd.DataFrame:
    """Players who can change the trajectory of a game.

    Ranked on the top of their range rather than the middle: the ceiling the
    simulation produces, how often it produces a game-breaking line, and how
    explosive the player's own touches have historically been. Matchup
    sensitivity is carried as a column because it is informative, but a player
    is not an X-factor merely for facing a soft defence.
    """
    from . import board

    real = board.project_team(team, ctx, projections_map, usage_hist, sampler,
                              game_ctx, league_def, n_sims=n_sims, seed=11, env=env)

    # Same team against a league-average defence, to isolate matchup effect.
    flat = {t: dict(v) for t, v in projections_map.items()}
    if opponent in flat:
        flat[opponent] = dict(flat[opponent])
        flat[opponent]["defense"] = dict(flat[opponent]["defense"])
        flat[opponent]["defense"]["projected"] = league_def.copy()
    base = {r.name: r for r in board.project_team(
        team, ctx, flat, usage_hist, sampler, game_ctx, league_def,
        n_sims=n_sims, seed=11, env=env)}

    expl = explosive_profile(ctx.plays)
    expl_idx = {}
    if not expl.empty:
        for pid, grp in expl.groupby("player_id"):
            # A player's headline explosiveness is whichever way he gets the
            # ball most, weighted by touches.
            w = grp["touches"] / grp["touches"].sum()
            expl_idx[pid] = float((grp["explosive_index"] * w).sum())

    rows = []
    for r in real:
        stat = "pass_yards" if r.position == "QB" else "scrimmage_yards"
        s = r.samples.get(stat)
        if s is None or len(s) == 0:
            continue
        mean = float(np.mean(s))
        if mean < (60.0 if r.position == "QB" else 8.0):
            continue                      # not enough involvement to swing a game

        ceiling = float(np.percentile(s, 90))
        boom = _boom_probability(r, r.position)
        ei = expl_idx.get(r.player_id, 1.0)

        other = base.get(r.name)
        swing = np.nan
        if other is not None and stat in other.samples:
            flat_mean = float(np.mean(other.samples[stat]))
            swing = mean - flat_mean

        # Upside relative to a typical day, times how often it actually lands,
        # times how explosively the player gets there. Availability gates it:
        # a player who may not dress cannot swing anything.
        upside = ceiling / max(mean, 1e-6)
        score = (boom if np.isfinite(boom) else 0.0) * upside * (ei ** 0.5) * r.p_active

        rows.append({
            "player": r.name, "pos": r.position, "rank": r.depth_rank,
            "projection": mean, "ceiling_90th": ceiling,
            "boom_pct": boom * 100 if np.isfinite(boom) else np.nan,
            "explosive_index": ei,
            "matchup_swing": swing,
            "p_active": r.p_active,
            "x_factor_score": score,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("x_factor_score", ascending=False)
    return df.head(top_n).reset_index(drop=True)


def weekly_plan(home: str, away: str, ctx, projections_map: dict,
                cov_plays: pd.DataFrame, league_off: pd.Series,
                league_def: pd.Series, seasons: tuple | None = None) -> dict:
    """A two-sided scouting package for one game."""
    out = {"home": home, "away": away,
           "home_name": TEAM_NAMES.get(home, home),
           "away_name": TEAM_NAMES.get(away, away)}
    for side, off, dfn in (("home_offense", home, away), ("away_offense", away, home)):
        out[side] = {
            "offense": off, "defense": dfn,
            "collisions": scheme_collisions(off, dfn, projections_map, cov_plays,
                                            league_off, league_def, seasons),
            "coverage": cvg.coverage_matchup(off, dfn, cov_plays, seasons)
            if cov_plays is not None else [],
        }
    return out


def situational_edges(off_team: str, def_team: str, plays: pd.DataFrame,
                      seasons: tuple | None = None) -> pd.DataFrame:
    """Down-and-distance comparison: what one does against what the other allows."""
    from . import playbook

    o = playbook.situational_tendencies(plays, off_team, "offense", seasons)
    d = playbook.defensive_profile(plays, def_team, seasons)
    if o.empty or d.empty:
        return pd.DataFrame()
    merged = o.merge(d, on="situation", how="inner", suffixes=("_off", "_def"))
    keep = [c for c in ["situation", "plays_off", "pass_rate", "epa",
                        "blitz_rate", "avg_box", "epa_allowed"] if c in merged.columns]
    merged = merged[keep].rename(columns={
        "plays_off": "off plays", "pass_rate": "off pass rate", "epa": "off EPA",
        "blitz_rate": "def blitz", "avg_box": "def box", "epa_allowed": "def EPA allowed"})
    if "off EPA" in merged and "def EPA allowed" in merged:
        merged["net"] = merged["off EPA"] + merged["def EPA allowed"]
        merged = merged.sort_values("net", ascending=False)
    return merged.reset_index(drop=True)
