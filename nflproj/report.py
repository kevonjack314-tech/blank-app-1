"""Turn the model's numbers into a scouting read.

Everything here is derived from the projections and the charted tendencies -
the phrasing is generated from thresholds, not written in advance - so a report
changes when the underlying data does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import playbook
from .config import TEAM_NAMES
from .schemes import DEFENSE_IDENTITY, OFFENSE_IDENTITY

# Trait -> (label, how to phrase a high value, how to phrase a low value, unit)
OFFENSE_LANGUAGE = {
    "early_down_pass_rate": ("early-down pass rate", "throws early and often", "runs on early downs", "pct"),
    "proe": ("pass rate over expected", "passes well above situation-neutral expectation", "runs well below expectation", "raw"),
    "motion_rate": ("pre-snap motion", "lives in motion", "plays largely static", "pct"),
    "play_action_rate": ("play-action rate", "leans hard on play-action", "rarely fakes the run", "pct"),
    "rpo_rate": ("RPO rate", "builds in run-pass options", "does not run RPO", "pct"),
    "screen_rate": ("screen rate", "uses screens as an extension of the run game", "seldom screens", "pct"),
    "shotgun_rate": ("shotgun rate", "operates from the gun", "works under center", "pct"),
    "no_huddle_rate": ("no-huddle rate", "pushes tempo", "huddles up", "pct"),
    "adot": ("average depth of target", "pushes the ball downfield", "works underneath", "yards"),
    "deep_rate": ("deep-shot rate", "takes shots", "avoids low-percentage throws", "pct"),
    "qb_designed_run_rate": ("designed QB run rate", "makes the quarterback a runner", "keeps the quarterback in the pocket", "pct"),
    "outside_run_rate": ("outside run rate", "attacks the perimeter", "runs between the tackles", "pct"),
    "g2g_run_rate": ("goal-to-go run rate", "runs it in close", "throws at the goal line", "pct"),
    "fourth_down_go_rate": ("fourth-down aggression", "goes for it", "takes the points", "pct"),
    "sec_per_play": ("seconds per play", "plays slowly", "plays fast", "sec"),
}
DEFENSE_LANGUAGE = {
    "blitz_rate": ("blitz rate", "sends extra rushers", "rushes four and drops seven", "pct"),
    "heavy_blitz_rate": ("six-plus rusher rate", "brings the house", "rarely overloads", "pct"),
    "avg_box": ("average box count", "loads the box", "plays light boxes", "count"),
    "heavy_box_rate": ("heavy box rate", "commits bodies to the run", "stays in nickel", "pct"),
    "light_box_rate": ("light box rate", "spends most snaps in light boxes", "plays big personnel", "pct"),
}


def _fmt(v: float, unit: str) -> str:
    if not np.isfinite(v):
        return "n/a"
    if unit == "pct":
        return f"{v * 100:.1f}%"
    if unit == "yards":
        return f"{v:.1f} yds"
    if unit == "sec":
        return f"{v:.1f}s"
    if unit == "count":
        return f"{v:.2f}"
    return f"{v:+.2f}"


def identity_deltas(projected: pd.Series, league: pd.Series, side: str = "offense",
                    top_n: int = 6) -> pd.DataFrame:
    """Traits where a team most separates from league average, in z-score terms."""
    lang = OFFENSE_LANGUAGE if side == "offense" else DEFENSE_LANGUAGE
    traits = [t for t in (OFFENSE_IDENTITY if side == "offense" else DEFENSE_IDENTITY)
              if t in lang and t in projected.index]
    rows = []
    for t in traits:
        v, m = projected.get(t, np.nan), league.get(t, np.nan)
        if not (np.isfinite(v) and np.isfinite(m)):
            continue
        label, hi, lo, unit = lang[t]
        rel = (v - m) / abs(m) if m else np.nan
        # Within a few percent of league average there is no real tendency to
        # describe, and asserting one either way would be misleading.
        if np.isfinite(rel) and abs(rel) < 0.04:
            phrase = "at league average"
        else:
            phrase = hi if v >= m else lo
        rows.append({
            "trait": label, "key": t, "value": v, "league": m,
            "diff": v - m, "rel": rel, "phrase": phrase,
            "display": _fmt(v, unit), "league_display": _fmt(m, unit),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Rank by standardized distance so percentages and counts compare fairly.
    df["z"] = (df["diff"] - df["diff"].mean()) / (df["diff"].std(ddof=0) or 1)
    df["magnitude"] = df["rel"].abs()
    return df.sort_values("magnitude", ascending=False).head(top_n)


def coaching_note(entry: dict) -> str:
    """Explain, in plain terms, where a team's projection is coming from."""
    w = entry["weights"]
    name = entry.get("coach_name") or "the incumbent staff"
    if not entry.get("is_new"):
        return (f"Continuity: {name} returns, so the projection sits mostly on this team's own "
                f"{int(w['team'] * 100)}% weighted baseline.")
    srcs = ", ".join(entry.get("coach_sources") or []) or "no measurable NFL sample"
    return (
        f"New voice: {name}. Scheme signature drawn from {srcs}, carrying "
        f"{int(w['coach'] * 100)}% of the projection against {int(w['team'] * 100)}% team "
        f"continuity and {int(w['league'] * 100)}% league regression. "
        f"Staff confidence: {entry.get('confidence', 'medium')}."
    )


def personnel_note(entry: dict) -> str:
    p = entry.get("personnel") or {}
    qb = p.get("qb")
    if not qb:
        return ""
    adj = p.get("adjusted") or {}
    if not adj:
        return f"Quarterback: {qb}."
    # Compare on a relative scale: rates and yardages are not commensurable in
    # absolute terms, and a 0.2-yard shift in depth of target should not outrank
    # a 50% swing in designed quarterback runs.
    def _relative(kv):
        _, d = kv
        base = abs(d["from"]) or 1e-9
        return abs(d["to"] - d["from"]) / base
    biggest = max(adj.items(), key=_relative)
    key, d = biggest
    label = OFFENSE_LANGUAGE.get(key, (key,))[0]
    direction = "up" if d["to"] > d["from"] else "down"
    return (
        f"Quarterback: {qb}. Personnel pulls {label} {direction} from "
        f"{_fmt(d['from'], 'pct' if key.endswith('rate') else 'yards')} to "
        f"{_fmt(d['to'], 'pct' if key.endswith('rate') else 'yards')} - "
        f"the scheme bends to the player, not the other way round."
    )


def team_report(team: str, projections_map: dict, ctx, league_off: pd.Series,
                league_def: pd.Series, seasons: tuple | None = None) -> dict:
    """Assemble a full scouting package for one team."""
    entry = projections_map[team]
    off, dfn = entry["offense"], entry["defense"]
    staff = entry["staff"]

    return {
        "team": team,
        "name": TEAM_NAMES.get(team, team),
        "staff": staff,
        "coaching_note": coaching_note(off),
        "defense_note": coaching_note(dfn),
        "personnel_note": personnel_note(off),
        "offense_identity": identity_deltas(off["projected"], league_off, "offense"),
        "defense_identity": identity_deltas(dfn["projected"], league_def, "defense"),
        "base_front": dfn.get("base_front"),
        "situational": playbook.situational_tendencies(ctx.plays, team, "offense", seasons),
        "def_situational": playbook.defensive_profile(ctx.plays, team, seasons),
        "concepts": playbook.signature_concepts(ctx.plays, team, seasons),
        "run_directions": playbook.run_direction_profile(ctx.plays, team, seasons),
        "pass_map": playbook.pass_location_profile(ctx.plays, team, seasons),
        "notes": staff.notes,
    }


def matchup_edges(off_report: dict, def_report: dict) -> list[str]:
    """Where one team's offensive identity meets the other's defensive habits."""
    edges: list[str] = []
    o = off_report["offense_identity"]
    d = def_report["defense_identity"]
    if o.empty or d.empty:
        return edges

    def val(df, key):
        m = df[df["key"] == key]
        return (float(m["value"].iloc[0]), float(m["league"].iloc[0])) if len(m) else (np.nan, np.nan)

    pa, pa_lg = val(o, "play_action_rate")
    blitz, blitz_lg = val(d, "blitz_rate")
    if np.isfinite(pa) and np.isfinite(blitz):
        if pa > pa_lg and blitz > blitz_lg:
            edges.append(
                f"{off_report['team']} run play-action at {pa*100:.0f}% against a defence that "
                f"blitzes {blitz*100:.0f}% - play-action against pressure is a big-play lever, "
                "but it is also where sacks come from.")
        elif pa > pa_lg and blitz < blitz_lg:
            edges.append(
                f"{off_report['team']}'s play-action game meets a four-man-rush defence "
                f"({blitz*100:.0f}% blitz); coverage will be intact behind it, so expect the "
                "throws to come off schedule rather than wide open.")

    box, box_lg = val(d, "avg_box")
    run, run_lg = val(o, "outside_run_rate")
    if np.isfinite(box) and np.isfinite(run):
        if box < box_lg and run > run_lg:
            edges.append(
                f"{def_report['team']} average {box:.2f} in the box, below the {box_lg:.2f} "
                f"league mark, while {off_report['team']} run outside on {run*100:.0f}% of "
                "carries - light boxes and perimeter runs favour the offence.")
        elif box > box_lg:
            edges.append(
                f"{def_report['team']} load the box ({box:.2f} average) - "
                f"{off_report['team']} will need to throw to move it.")

    motion, motion_lg = val(o, "motion_rate")
    if np.isfinite(motion) and motion > motion_lg * 1.15:
        edges.append(
            f"{off_report['team']} motion on {motion*100:.0f}% of snaps; motion is the primary "
            "tool for diagnosing man versus zone before the snap.")
    return edges
