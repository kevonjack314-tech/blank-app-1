"""Player and team news, and the role changes that show up in the data.

Narratives divide into two kinds, and they behave completely differently.

**Known narratives are priced.** Tested against closing lines back to 2006, none
of the familiar ones move outcomes: revenge games, divisional rematches, teams
off a bye, short weeks, primetime, letdown spots after a blowout. National Tight
Ends Day - a real, league-promoted event that plausibly changes play-calling -
shows a tight end target share of 0.2179 against 0.2173 on every other day
(t = 0.11). If everyone knows it, it is in the number.

**News is not priced until it is known.** A beat report that a back is taking
first-team reps, a coordinator saying they intend to feature someone, a
practice-squad elevation - these move projections precisely because they have not
propagated yet. That is where attention pays.

So this module does not model narratives. It tracks *information*: notes a person
enters by hand, plus the role changes the model can detect on its own by watching
depth charts, injury reports and snap trends move week to week.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import DATA, normalize_team

log = logging.getLogger(__name__)

NOTES_PATH = DATA / "news_2026.yaml"

# What a note is allowed to do. Bounded deliberately: a note is a nudge from
# information the data has not caught up with, not a licence to hand-write a
# projection.
EFFECTS = {
    "usage_up": "raises a player's projected share",
    "usage_down": "lowers it",
    "out": "rules the player out entirely",
    "questionable": "lowers the chance he plays",
    "role_change": "moves him up or down the depth chart",
    "workload_cap": "limits snaps on return from injury",
}
MAX_USAGE_MULTIPLIER = 1.60
MIN_USAGE_MULTIPLIER = 0.40


def load_notes(path: Path | None = None, as_of: date | None = None) -> pd.DataFrame:
    """Read hand-entered notes, dropping any that have expired."""
    path = path or NOTES_PATH
    if not path.exists():
        return pd.DataFrame(columns=["player", "team", "effect", "magnitude",
                                     "note", "source", "expires"])
    raw = yaml.safe_load(path.read_text()) or {}
    rows = raw.get("notes") or []
    if not rows:
        return pd.DataFrame(columns=["player", "team", "effect", "magnitude",
                                     "note", "source", "expires"])
    df = pd.DataFrame(rows)
    if "team" in df.columns:
        df["team"] = df["team"].map(normalize_team)

    today = as_of or date.today()
    if "expires" in df.columns:
        def _live(v):
            if v in (None, "", "never"):
                return True
            try:
                return pd.to_datetime(v).date() >= today
            except Exception:
                return True
        df = df[df["expires"].map(_live)]

    bad = set(df.get("effect", pd.Series(dtype=str))) - set(EFFECTS)
    if bad:
        log.warning("unknown note effects ignored: %s", bad)
        df = df[df["effect"].isin(EFFECTS)]
    return df.reset_index(drop=True)


def usage_multiplier(notes: pd.DataFrame, player_id: str | None,
                     player_name: str | None = None) -> float:
    """Combined usage effect of every live note about one player."""
    if notes is None or notes.empty:
        return 1.0
    m = pd.Series(False, index=notes.index)
    if player_id is not None and "gsis_id" in notes.columns:
        m |= notes["gsis_id"].eq(player_id)
    if player_name:
        m |= notes["player"].astype(str).str.casefold().eq(str(player_name).casefold())
    hits = notes[m]
    if hits.empty:
        return 1.0

    mult = 1.0
    for _, r in hits.iterrows():
        mag = float(r.get("magnitude", 0.15) or 0.15)
        if r["effect"] == "usage_up":
            mult *= 1.0 + abs(mag)
        elif r["effect"] in ("usage_down", "workload_cap"):
            mult *= 1.0 - abs(mag)
    return float(np.clip(mult, MIN_USAGE_MULTIPLIER, MAX_USAGE_MULTIPLIER))


def availability_override(notes: pd.DataFrame, player_id: str | None,
                          player_name: str | None = None) -> float | None:
    """Hard availability set by a note, or ``None`` to leave the model alone."""
    if notes is None or notes.empty:
        return None
    m = pd.Series(False, index=notes.index)
    if player_id is not None and "gsis_id" in notes.columns:
        m |= notes["gsis_id"].eq(player_id)
    if player_name:
        m |= notes["player"].astype(str).str.casefold().eq(str(player_name).casefold())
    hits = notes[m]
    if hits.empty:
        return None
    if (hits["effect"] == "out").any():
        return 0.0
    if (hits["effect"] == "questionable").any():
        return 0.55
    return None


# ---------------------------------------------------------------------------
# Changes the model can see for itself
# ---------------------------------------------------------------------------

def depth_chart_moves(season: int, from_week: int, to_week: int,
                      games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Players whose depth-chart position changed between two weeks.

    A promotion is the earliest hard signal of a role change that exists in
    structured data - usually days before it shows up in production.
    """
    from . import data as ndata
    from . import personnel

    a = personnel.latest_depth_chart(ndata.depth_chart_for_week(season, from_week, games))
    b = personnel.latest_depth_chart(ndata.depth_chart_for_week(season, to_week, games))
    if a.empty or b.empty:
        return pd.DataFrame()

    key = ["team", "player_name", "pos_abb"]
    merged = a[key + ["pos_rank", "gsis_id"]].merge(
        b[key + ["pos_rank"]], on=key, how="outer", suffixes=("_before", "_after"))
    merged["moved"] = merged["pos_rank_before"] - merged["pos_rank_after"]
    changed = merged[merged["moved"].fillna(0) != 0].copy()
    if changed.empty:
        return changed
    changed["direction"] = np.where(changed["moved"] > 0, "promoted", "demoted")
    changed["from_week"], changed["to_week"] = from_week, to_week
    return changed.sort_values("moved", ascending=False).reset_index(drop=True)


def snap_trend(snaps: pd.DataFrame, season: int, window: int = 3) -> pd.DataFrame:
    """Players whose snap share is trending hard, recent weeks against earlier.

    Snap share moves before target share does: a receiver on the field more is
    about to be thrown to more, and the box score has not registered it yet.
    """
    if snaps is None or snaps.empty:
        return pd.DataFrame()
    s = snaps[(snaps["season"] == season) & (snaps.get("game_type", "REG") == "REG")].copy()
    if s.empty or "offense_pct" not in s.columns:
        return pd.DataFrame()
    s = s.sort_values("week")
    latest = s["week"].max()
    recent = s[s["week"] > latest - window]
    earlier = s[(s["week"] <= latest - window) & (s["week"] > latest - 2 * window)]
    if recent.empty or earlier.empty:
        return pd.DataFrame()

    r = recent.groupby(["team", "player", "position"])["offense_pct"].mean().rename("recent")
    e = earlier.groupby(["team", "player", "position"])["offense_pct"].mean().rename("earlier")
    out = pd.concat([r, e], axis=1).dropna().reset_index()
    out["change"] = out["recent"] - out["earlier"]
    out = out[out["earlier"].gt(0.02) | out["recent"].gt(0.15)]
    return out.reindex(out["change"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def injury_watch(injuries: pd.DataFrame, week: int | None = None) -> pd.DataFrame:
    """Current designations and practice participation, ranked by seriousness."""
    if injuries is None or injuries.empty:
        return pd.DataFrame()
    d = injuries.copy()
    if week is not None and "week" in d.columns:
        d = d[d["week"] == week]
    keep = [c for c in ("team", "full_name", "position", "report_status",
                        "practice_status", "report_primary_injury", "week")
            if c in d.columns]
    d = d[keep]
    order = {"Out": 0, "Doubtful": 1, "Questionable": 2}
    if "report_status" in d.columns:
        d = d[d["report_status"].notna()]
        d["_rank"] = d["report_status"].map(order).fillna(3)
        d = d.sort_values("_rank").drop(columns=["_rank"])
    if "team" in d.columns:
        d["team"] = d["team"].map(normalize_team)
    return d.reset_index(drop=True)


def briefing(season: int, week: int, ctx) -> dict:
    """Everything that changed going into a week, in one place."""
    out = {}
    try:
        out["depth_moves"] = depth_chart_moves(season, max(week - 1, 1), week, ctx.games)
    except Exception as exc:
        log.warning("depth-chart comparison failed (%s)", exc)
        out["depth_moves"] = pd.DataFrame()
    out["snap_trend"] = snap_trend(ctx.snaps, season)
    from . import data as ndata
    out["injuries"] = injury_watch(ndata.fetch("injuries", season), week)
    out["notes"] = load_notes()
    return out
