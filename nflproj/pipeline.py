"""Assemble the pieces into a projected 2026 scheme profile for every team."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import availability, coaches, data, personnel, schemes
from .config import CACHE, HISTORY_SEASONS, LAST_COMPLETED_SEASON, PROJECTION_SEASON, ROOT, TEAMS

log = logging.getLogger(__name__)


@dataclass
class Context:
    """Everything downstream models need, loaded once."""
    plays: pd.DataFrame
    raw_pbp: pd.DataFrame
    fingerprints: pd.DataFrame
    depth: pd.DataFrame
    chart: pd.DataFrame
    fronts: pd.DataFrame
    qb_profiles: pd.DataFrame
    staffs: dict
    games: pd.DataFrame
    weekly: pd.DataFrame
    snaps: pd.DataFrame
    availability: pd.DataFrame = None
    injuries: dict = None


def build_context(seasons=HISTORY_SEASONS, projection_season=PROJECTION_SEASON,
                  use_cache: bool = True) -> Context:
    cache = CACHE / f"plays_{min(seasons)}_{max(seasons)}.parquet"
    fp_cache = CACHE / f"fingerprints_{min(seasons)}_{max(seasons)}.parquet"

    raw = data.play_by_play(seasons, columns=schemes.PBP_COLUMNS)
    if use_cache and cache.exists() and fp_cache.exists():
        plays = pd.read_parquet(cache)
        fp = pd.read_parquet(fp_cache)
    else:
        plays = schemes.prepare_plays(raw, data.charting(seasons))
        fp = schemes.build_fingerprints(plays, raw_pbp=raw)
        plays.to_parquet(cache)
        fp.to_parquet(fp_cache)

    depth = data.depth_chart(projection_season)
    snaps = data.snap_counts(seasons)
    players_df = data.players()
    return Context(
        plays=plays,
        raw_pbp=raw,
        fingerprints=fp,
        depth=depth,
        chart=personnel.latest_depth_chart(depth),
        fronts=personnel.base_defensive_front(depth),
        qb_profiles=personnel.qb_profiles(plays),
        staffs=coaches.load_registry(),
        games=data.games(),
        weekly=data.weekly_stats(seasons),
        snaps=snaps,
        availability=availability.player_availability(snaps, players_df),
        # Injury designations only count for the season being projected. Last
        # season's final report is not current status - a player ruled out in
        # Week 18 is not thereby out for the following September, and chronic
        # availability is already carried by the snap-count history.
        injuries=availability.current_injuries(
            data.fetch("injuries", projection_season), players_df),
    )


def project_team_schemes(ctx: Context, anchor_season: int | None = None) -> dict[str, dict]:
    """Projected offensive and defensive fingerprint for each team.

    ``anchor_season`` is the most recent season of evidence; it defaults to the
    latest one present in the fingerprints so that backtests anchored on an
    earlier year cannot accidentally read the season they are predicting.
    """
    if anchor_season is None:
        anchor_season = int(ctx.fingerprints["season"].max())
    lg_off = schemes.league_means(ctx.fingerprints, "offense", anchor_season)
    lg_def = schemes.league_means(ctx.fingerprints, "defense", anchor_season)

    out: dict[str, dict] = {}
    for team in TEAMS:
        staff = ctx.staffs[team]

        off = coaches.project_fingerprint(team, staff, ctx.fingerprints, "offense", lg_off,
                                          anchor_season=anchor_season)
        adjusted, note = personnel.apply_personnel_constraints(
            off["projected"], team, ctx.chart, ctx.qb_profiles, lg_off
        )
        off["pre_personnel"] = off["projected"]
        off["projected"] = adjusted
        off["personnel"] = note

        dfn = coaches.project_fingerprint(team, staff, ctx.fingerprints, "defense", lg_def,
                                          anchor_season=anchor_season)
        front = ctx.fronts[ctx.fronts["team"] == team]
        dfn["base_front"] = front["base_front"].iloc[0] if len(front) else None

        out[team] = {"offense": off, "defense": dfn, "staff": staff}
    return out


def scheme_table(projections: dict[str, dict], side: str = "offense") -> pd.DataFrame:
    """Flatten projections into one row per team for display and export."""
    rows = []
    for team, p in projections.items():
        rec = {"team": team}
        rec.update(p[side]["projected"].to_dict())
        rec["coach"] = p[side].get("coach_name")
        rec["new_staff"] = p[side].get("is_new")
        rec["coach_weight"] = p[side]["weights"]["coach"]
        rec["confidence"] = p[side].get("confidence")
        if side == "offense":
            rec["qb"] = p[side].get("personnel", {}).get("qb")
        else:
            rec["base_front"] = p[side].get("base_front")
        rows.append(rec)
    return pd.DataFrame(rows).set_index("team")
