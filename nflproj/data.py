"""Fetch and cache nflverse datasets.

Everything is pulled from the public nflverse-data release assets and cached
to ``data/raw`` so the model runs offline after the first sync.
"""
from __future__ import annotations

import io
import logging
from functools import lru_cache

import pandas as pd
import requests

from .config import HISTORY_SEASONS, NFLDATA_GAMES, NFLVERSE, RAW, normalize_team

log = logging.getLogger(__name__)

# asset key -> (release path template, local filename template)
ASSETS = {
    "pbp": ("pbp/play_by_play_{season}.parquet", "pbp_{season}.parquet"),
    "weekly": ("stats_player/stats_player_week_{season}.parquet", "stats_week_{season}.parquet"),
    "ftn": ("ftn_charting/ftn_charting_{season}.parquet", "ftn_{season}.parquet"),
    "snaps": ("snap_counts/snap_counts_{season}.parquet", "snaps_{season}.parquet"),
    "depth": ("depth_charts/depth_charts_{season}.parquet", "depth_{season}.parquet"),
    "roster": ("weekly_rosters/roster_weekly_{season}.parquet", "roster_{season}.parquet"),
    "injuries": ("injuries/injuries_{season}.parquet", "inj_{season}.parquet"),
    "pfr_pass": ("pfr_advstats/advstats_week_pass_{season}.parquet", "pfr_pass_{season}.parquet"),
    "pfr_rush": ("pfr_advstats/advstats_week_rush_{season}.parquet", "pfr_rush_{season}.parquet"),
    "pfr_rec": ("pfr_advstats/advstats_week_rec_{season}.parquet", "pfr_rec_{season}.parquet"),
}
STATIC_ASSETS = {
    "players": ("players/players.parquet", "players.parquet"),
    "ngs_pass": ("nextgen_stats/ngs_passing.parquet", "ngs_pass.parquet"),
    "ngs_rec": ("nextgen_stats/ngs_receiving.parquet", "ngs_rec.parquet"),
    "ngs_rush": ("nextgen_stats/ngs_rushing.parquet", "ngs_rush.parquet"),
}


def _download(url: str, dest, timeout: int = 120) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
    except requests.HTTPError as exc:
        # A 404 usually means the feed simply has not been published for that
        # season yet - normal when projecting ahead - so it is not a warning.
        if exc.response is not None and exc.response.status_code == 404:
            log.info("asset not published upstream: %s", url)
        else:
            log.warning("could not fetch %s (%s)", url, exc)
        return False
    except Exception as exc:  # network trouble
        log.warning("could not fetch %s (%s)", url, exc)
        return False
    dest.write_bytes(r.content)
    return True


def fetch(asset: str, season: int | None = None, refresh: bool = False):
    """Return a cached parquet asset, downloading it when missing.

    Returns ``None`` when the asset does not exist upstream (for example a
    charting feed that has not been published for the season yet).
    """
    if season is None:
        path_tmpl, file_tmpl = STATIC_ASSETS[asset]
        rel, fname = path_tmpl, file_tmpl
    else:
        path_tmpl, file_tmpl = ASSETS[asset]
        rel, fname = path_tmpl.format(season=season), file_tmpl.format(season=season)

    dest = RAW / fname
    if refresh or not dest.exists():
        if not _download(f"{NFLVERSE}/{rel}", dest):
            return None
    try:
        return pd.read_parquet(dest)
    except Exception as exc:
        log.warning("unreadable cache %s (%s)", dest, exc)
        return None


@lru_cache(maxsize=1)
def games(refresh: bool = False) -> pd.DataFrame:
    """Schedule and market lines for every season, including the upcoming one."""
    dest = RAW / "games.csv"
    if refresh or not dest.exists():
        _download(NFLDATA_GAMES, dest)
    df = pd.read_csv(dest, low_memory=False)
    for col in ("home_team", "away_team"):
        df[col] = df[col].map(normalize_team)
    return df


def play_by_play(seasons=HISTORY_SEASONS, columns=None) -> pd.DataFrame:
    """Play-by-play for several seasons, with team abbreviations normalized."""
    frames = []
    for s in seasons:
        df = fetch("pbp", s)
        if df is None:
            continue
        if columns:
            keep = [c for c in columns if c in df.columns]
            df = df[keep]
        df["season"] = s
        frames.append(df)
    if not frames:
        raise RuntimeError("no play-by-play available; run a sync with network access")
    out = pd.concat(frames, ignore_index=True)
    for col in ("posteam", "defteam", "home_team", "away_team", "td_team"):
        if col in out.columns:
            out[col] = out[col].map(normalize_team)
    return out


def charting(seasons=HISTORY_SEASONS) -> pd.DataFrame:
    """FTN charting: motion, play-action, RPO, screens, blitzers, box counts."""
    frames = [df for s in seasons if (df := fetch("ftn", s)) is not None]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def weekly_stats(seasons=HISTORY_SEASONS) -> pd.DataFrame:
    frames = []
    for s in seasons:
        df = fetch("weekly", s)
        if df is None:
            continue
        df["season"] = s
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in ("team", "opponent_team"):
        if col in out.columns:
            out[col] = out[col].map(normalize_team)
    return out


def snap_counts(seasons=HISTORY_SEASONS) -> pd.DataFrame:
    frames = [df for s in seasons if (df := fetch("snaps", s)) is not None]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in ("team", "opponent"):
        if col in out.columns:
            out[col] = out[col].map(normalize_team)
    return out


def depth_chart(season: int) -> pd.DataFrame:
    """Most recent depth chart snapshot for each team/position slot."""
    df = fetch("depth", season)
    if df is None or df.empty:
        return pd.DataFrame()
    df["team"] = df["team"].map(normalize_team)
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce", utc=True)
        latest = df.groupby("team")["dt"].transform("max")
        df = df[df["dt"] == latest]
    return df.reset_index(drop=True)


def rosters(season: int) -> pd.DataFrame:
    df = fetch("roster", season)
    if df is None or df.empty:
        return pd.DataFrame()
    df["team"] = df["team"].map(normalize_team)
    return df


def players() -> pd.DataFrame:
    df = fetch("players")
    return pd.DataFrame() if df is None else df


def sync_all(seasons=HISTORY_SEASONS, projection_season: int | None = None) -> dict:
    """Warm the whole cache. Returns a per-asset status map."""
    status = {}
    for s in seasons:
        for a in ("pbp", "weekly", "ftn", "snaps", "pfr_pass", "pfr_rush", "pfr_rec", "injuries"):
            status[f"{a}_{s}"] = fetch(a, s) is not None
    if projection_season:
        for a in ("depth", "roster"):
            status[f"{a}_{projection_season}"] = fetch(a, projection_season) is not None
    for a in STATIC_ASSETS:
        status[a] = fetch(a) is not None
    status["games"] = not games().empty
    return status
