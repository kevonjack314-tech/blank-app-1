"""Global configuration: paths, team metadata, and modeling constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
for _d in (RAW, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# Which games count as evidence.
#
# Preseason is excluded unconditionally and is not configurable. Starters play a
# series and sit, roster spots 60-90 take most of the snaps, play-calling is
# vanilla by design, and nobody is trying to win. It measures almost nothing
# about how a team will play in September. nflverse does not currently publish
# preseason play-by-play, but the filter is enforced anyway so that a future
# feed change cannot quietly contaminate the model.
#
# Postseason is excluded by default for a different reason: it is asymmetric.
# Only fourteen teams play it, so including it hands extra sample - against
# stronger opponents, in higher-leverage scripts - to teams that were already
# good, and none to anyone else. Set INCLUDE_POSTSEASON to True to use it.
INCLUDE_POSTSEASON = False
EXCLUDED_SEASON_TYPES = ("PRE", "PRESEASON")


def allowed_season_types(include_postseason: bool | None = None) -> tuple[str, ...]:
    """Season types the model is permitted to learn from."""
    post = INCLUDE_POSTSEASON if include_postseason is None else include_postseason
    return ("REG", "POST") if post else ("REG",)


# Season the model projects, and the seasons of evidence it learns from.
PROJECTION_SEASON = 2026
HISTORY_SEASONS = (2022, 2023, 2024, 2025)
LAST_COMPLETED_SEASON = 2025

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
NFLDATA_GAMES = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

# Teams as they appear in nflverse play-by-play for the projection season.
TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]

# nflverse has used several abbreviations for the same franchise over the years.
TEAM_ALIASES = {"LAR": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV", "SL": "LA"}

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def normalize_team(team: str | None) -> str | None:
    """Map historical/alternate abbreviations onto the current one."""
    if team is None:
        return None
    t = str(team).strip().upper()
    return TEAM_ALIASES.get(t, t)


# ---------------------------------------------------------------------------
# Modeling constants
# ---------------------------------------------------------------------------

# Points on the board per touchdown drive, used to convert Vegas team totals
# into expected touchdowns. A team's non-TD points come from FGs, safeties, XP.
POINTS_PER_TD = 6.95          # TD + expected PAT value
LEAGUE_PLAYS_PER_GAME = 62.5  # offensive plays (excl. kneels/spikes/ST)

# Shrinkage strength: how many plays of evidence it takes before a team's or
# player's own rate outweighs the league prior. Tuned by position group below.
PRIOR_STRENGTH = {
    "team_rate": 180.0,
    "pass_efficiency": 220.0,
    "rush_efficiency": 110.0,
    "rec_efficiency": 45.0,
    "usage_share": 55.0,
    "td_rate": 30.0,
}
