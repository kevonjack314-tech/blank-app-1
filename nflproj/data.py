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
    "pfr_def": ("pfr_advstats/advstats_week_def_{season}.parquet", "pfr_def_{season}.parquet"),
    "participation": ("pbp_participation/pbp_participation_{season}.parquet", "part_{season}.parquet"),
}
STATIC_ASSETS = {
    "players": ("players/players.parquet", "players.parquet"),
    "ngs_pass": ("nextgen_stats/ngs_passing.parquet", "ngs_pass.parquet"),
    "ngs_rec": ("nextgen_stats/ngs_receiving.parquet", "ngs_rec.parquet"),
    "ngs_rush": ("nextgen_stats/ngs_rushing.parquet", "ngs_rush.parquet"),
    "draft": ("draft_picks/draft_picks.parquet", "draft_picks.parquet"),
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
    """Play-by-play for several seasons, with team abbreviations normalized.

    Each season is narrowed and compacted before the next is read. Loading five
    full seasons and filtering afterwards briefly holds half a gigabyte that is
    immediately discarded, which is enough to be killed inside a small
    container even though the final frame is modest.
    """
    frames = []
    for season in seasons:
        path = RAW / ASSETS["pbp"][1].format(season=season)
        if not path.exists() and not _download(f"{NFLVERSE}/{ASSETS['pbp'][0].format(season=season)}", path):
            continue
        try:
            # Read only the needed columns off disk rather than after loading.
            df = pd.read_parquet(path, columns=list(columns) if columns else None)
        except Exception as exc:
            log.warning("unreadable play-by-play %s (%s)", path, exc)
            continue
        df["season"] = season
        for col in ("posteam", "defteam", "home_team", "away_team", "td_team"):
            if col in df.columns:
                df[col] = df[col].map(normalize_team)
        frames.append(compact(df))
    if not frames:
        raise RuntimeError("no play-by-play available; run a sync with network access")
    return pd.concat(frames, ignore_index=True)


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


def depth_chart(season: int, as_of=None) -> pd.DataFrame:
    """Depth chart for each team, as of a point in time.

    Charts are published repeatedly through a season, so ``as_of`` selects the
    most recent snapshot at or before that moment. Without it the latest
    available chart is used. Projecting a Week 3 game from a Week 18 chart
    credits late-emerging players with a role they did not have yet, which is a
    quiet source of hindsight in any backtest.
    """
    df = fetch("depth", season)
    if df is None or df.empty:
        return pd.DataFrame()
    df["team"] = df["team"].map(normalize_team)
    if "dt" not in df.columns:
        return df.reset_index(drop=True)

    df["dt"] = pd.to_datetime(df["dt"], errors="coerce", utc=True)
    if as_of is not None:
        cutoff = pd.to_datetime(as_of, utc=True)
        df = df[df["dt"] <= cutoff]
        if df.empty:
            return pd.DataFrame()
    latest = df.groupby("team")["dt"].transform("max")
    return df[df["dt"] == latest].reset_index(drop=True)


def depth_chart_for_week(season: int, week: int, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Depth chart as it stood going into a given week."""
    g = games if games is not None else globals()["games"]()
    wk = g[(g["season"] == season) & (g["week"] == week)]
    if wk.empty or "gameday" not in wk.columns:
        return depth_chart(season)
    kickoff = pd.to_datetime(wk["gameday"], errors="coerce").min()
    if pd.isna(kickoff):
        return depth_chart(season)
    return depth_chart(season, as_of=kickoff)


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
        for a in ("pbp", "weekly", "ftn", "snaps", "pfr_pass", "pfr_rush", "pfr_rec",
                  "injuries", "participation"):
            status[f"{a}_{s}"] = fetch(a, s) is not None
    if projection_season:
        for a in ("depth", "roster"):
            status[f"{a}_{projection_season}"] = fetch(a, projection_season) is not None
    for a in STATIC_ASSETS:
        status[a] = fetch(a) is not None
    status["games"] = not games().empty
    return status


def participation(seasons=HISTORY_SEASONS) -> pd.DataFrame:
    """Personnel, formation and coverage charting.

    This feed carries what FTN does not: the coverage the defence actually
    played (man or zone, and the shell), the route each receiver ran, and
    whether the quarterback was pressured. Coverage is charted on pass plays
    only, so roughly half of all snaps carry it.
    """
    keep_raw = ["nflverse_game_id", "play_id", "offense_formation", "offense_personnel",
                "defense_personnel", "defenders_in_box", "number_of_pass_rushers",
                "was_pressure", "route", "defense_man_zone_type",
                "defense_coverage_type", "time_to_throw", "possession_team"]
    frames = []
    for season in seasons:
        path = RAW / ASSETS["participation"][1].format(season=season)
        if not path.exists() and not _download(
                f"{NFLVERSE}/{ASSETS['participation'][0].format(season=season)}", path):
            continue
        try:
            import pyarrow.parquet as _pq
            avail = set(_pq.read_schema(path).names)
            df = pd.read_parquet(path, columns=[c for c in keep_raw if c in avail])
        except Exception as exc:
            log.warning("unreadable participation %s (%s)", path, exc)
            continue
        df["season"] = season
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"nflverse_game_id": "game_id"})
    # The feed carries full 22-player rosters per snap as delimited strings,
    # which dominate its footprint and are not used. Keep only the charting.
    keep = ["game_id", "play_id", "season", "possession_team", "offense_formation",
            "offense_personnel", "defense_personnel", "defenders_in_box",
            "number_of_pass_rushers", "was_pressure", "route",
            "defense_man_zone_type", "defense_coverage_type", "time_to_throw"]
    out = out[[c for c in keep if c in out.columns]]
    if "possession_team" in out.columns:
        out["possession_team"] = out["possession_team"].map(normalize_team)
    # Empty strings mean "not charted", which is different from a real value.
    for col in ("defense_man_zone_type", "defense_coverage_type", "route", "offense_formation"):
        if col in out.columns:
            out[col] = out[col].replace("", pd.NA)
    return compact(out)


def draft_picks() -> pd.DataFrame:
    """Draft position, the best available prior for a player with no NFL sample."""
    df = fetch("draft")
    if df is None or df.empty:
        return pd.DataFrame()
    if "team" in df.columns:
        df["team"] = df["team"].map(normalize_team)
    return df


DEF_COVERAGE_COLUMNS = {
    "def_targets": "targets", "def_completions_allowed": "completions",
    "def_yards_allowed": "yards", "def_yards_after_catch": "yac",
    "def_receiving_td_allowed": "td", "def_ints": "ints", "def_adot": "adot",
    "def_tackles_combined": "tackles", "def_missed_tackles": "missed",
    "pfr_player_name": "player", "pfr_player_id": "player_id",
}


def defensive_coverage(seasons=HISTORY_SEASONS) -> pd.DataFrame:
    """Per-defender coverage charting: targets, yards and tackles allowed.

    Renamed to short column names on the way through, because the raw feed
    prefixes everything with ``def_`` and the module that consumes it is
    already about defenders.
    """
    d = pfr_advanced("def", seasons)
    if d.empty:
        return d
    keep = {k: v for k, v in DEF_COVERAGE_COLUMNS.items() if k in d.columns}
    out = d[["season", "week", "team", "opponent"] + list(keep)].rename(columns=keep)
    for c in ("targets", "completions", "yards", "yac", "td", "ints", "adot",
              "tackles", "missed"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def pfr_advanced(kind: str, seasons=HISTORY_SEASONS) -> pd.DataFrame:
    """Pro Football Reference advanced splits: pressure, contact, drops."""
    asset = {"pass": "pfr_pass", "rush": "pfr_rush", "rec": "pfr_rec",
             "def": "pfr_def"}[kind]
    frames = []
    for s in seasons:
        df = fetch(asset, s)
        if df is None:
            continue
        df["season"] = s
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "team" in out.columns:
        out["team"] = out["team"].map(normalize_team)
    # Match the model's season-type policy.
    if "game_type" in out.columns:
        out = out[out["game_type"].astype(str).str.upper() == "REG"]
    return out


def next_gen(kind: str) -> pd.DataFrame:
    """Next Gen Stats: separation, cushion, rush yards over expected."""
    df = fetch({"pass": "ngs_pass", "rec": "ngs_rec", "rush": "ngs_rush"}[kind])
    if df is None or df.empty:
        return pd.DataFrame()
    if "season_type" in df.columns:
        df = df[df["season_type"].astype(str).str.upper() == "REG"]
    if "team_abbr" in df.columns:
        df["team_abbr"] = df["team_abbr"].map(normalize_team)
    return df


# Columns whose values repeat heavily; storing them as categories rather than
# free strings is most of the memory saving on these frames.
_CATEGORICAL_HINTS = (
    "team", "posteam", "defteam", "home_team", "away_team", "possession_team",
    "play_type", "season_type", "game_type", "pos_abb", "pos_grp", "position",
    "run_location", "run_gap", "pass_location", "pass_length", "route",
    "defense_man_zone_type", "defense_coverage_type", "offense_formation",
    "offense_personnel", "defense_personnel", "roof", "surface",
    "fixed_drive_result", "field_goal_result",
)


def compact(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink a frame in place-ish: 64-bit floats and repeated strings dominate.

    Play-by-play arrives as float64 and object columns throughout. Downcasting
    numerics and converting low-cardinality strings to categories cuts the
    working set roughly in half, which is the difference between running inside
    a 1 GB container and being killed by it.
    """
    if df is None or df.empty:
        return df
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_float_dtype(s):
            df[col] = pd.to_numeric(s, downcast="float")
        elif pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
        elif col in _CATEGORICAL_HINTS or (
            s.dtype == object and s.nunique(dropna=True) < max(len(s) * 0.02, 64)
        ):
            try:
                df[col] = s.astype("category")
            except (TypeError, ValueError):
                pass
    return df
