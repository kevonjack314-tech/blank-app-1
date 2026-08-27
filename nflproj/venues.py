"""Stadium geography, climate and travel context.

Three physical facts about a game that the team-strength model cannot see:
how far a team travelled and across how many time zones, what the weather is
doing relative to what that team is used to, and whether the two clubs know
each other well.

Coordinates and time zones are static. Home climate is measured from historical
game conditions rather than assumed, so a team's "normal" is what it actually
plays in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# lat, lon, standard UTC offset (hours), roof type of the home venue.
# Arizona does not observe daylight saving, which is why it sits with Mountain
# in winter and Pacific in September.
VENUES = {
    "ARI": (33.5276, -112.2626, -7, "closed"),
    "ATL": (33.7554, -84.4008, -5, "closed"),
    "BAL": (39.2780, -76.6227, -5, "outdoors"),
    "BUF": (42.7738, -78.7870, -5, "outdoors"),
    "CAR": (35.2258, -80.8528, -5, "outdoors"),
    "CHI": (41.8623, -87.6167, -6, "outdoors"),
    "CIN": (39.0955, -84.5161, -5, "outdoors"),
    "CLE": (41.5061, -81.6995, -5, "outdoors"),
    "DAL": (32.7473, -97.0945, -6, "closed"),
    "DEN": (39.7439, -105.0201, -7, "outdoors"),
    "DET": (42.3400, -83.0456, -5, "dome"),
    "GB":  (44.5013, -88.0622, -6, "outdoors"),
    "HOU": (29.6847, -95.4107, -6, "closed"),
    "IND": (39.7601, -86.1639, -5, "closed"),
    "JAX": (30.3239, -81.6373, -5, "outdoors"),
    "KC":  (39.0489, -94.4839, -6, "outdoors"),
    "LA":  (33.9535, -118.3392, -8, "closed"),
    "LAC": (33.9535, -118.3392, -8, "closed"),
    "LV":  (36.0909, -115.1833, -8, "dome"),
    "MIA": (25.9580, -80.2389, -5, "outdoors"),
    "MIN": (44.9736, -93.2575, -6, "dome"),
    "NE":  (42.0909, -71.2643, -5, "outdoors"),
    "NO":  (29.9511, -90.0812, -6, "dome"),
    "NYG": (40.8135, -74.0745, -5, "outdoors"),
    "NYJ": (40.8135, -74.0745, -5, "outdoors"),
    "PHI": (39.9008, -75.1675, -5, "outdoors"),
    "PIT": (40.4468, -80.0158, -5, "outdoors"),
    "SEA": (47.5952, -122.3316, -8, "outdoors"),
    "SF":  (37.4033, -121.9694, -8, "outdoors"),
    "TB":  (27.9759, -82.5033, -5, "outdoors"),
    "TEN": (36.1665, -86.7713, -6, "outdoors"),
    "WAS": (38.9077, -76.8645, -5, "outdoors"),
}

DIVISIONS = {
    "BUF": "AFC East", "MIA": "AFC East", "NE": "AFC East", "NYJ": "AFC East",
    "BAL": "AFC North", "CIN": "AFC North", "CLE": "AFC North", "PIT": "AFC North",
    "HOU": "AFC South", "IND": "AFC South", "JAX": "AFC South", "TEN": "AFC South",
    "DEN": "AFC West", "KC": "AFC West", "LAC": "AFC West", "LV": "AFC West",
    "DAL": "NFC East", "NYG": "NFC East", "PHI": "NFC East", "WAS": "NFC East",
    "CHI": "NFC North", "DET": "NFC North", "GB": "NFC North", "MIN": "NFC North",
    "ATL": "NFC South", "CAR": "NFC South", "NO": "NFC South", "TB": "NFC South",
    "ARI": "NFC West", "LA": "NFC West", "SEA": "NFC West", "SF": "NFC West",
}

DOME_TEAMS = {t for t, v in VENUES.items() if v[3] in ("dome", "closed")}


def haversine(a: str, b: str) -> float:
    """Great-circle distance between two teams' stadiums, in miles."""
    if a not in VENUES or b not in VENUES:
        return 0.0
    lat1, lon1 = np.radians(VENUES[a][0]), np.radians(VENUES[a][1])
    lat2, lon2 = np.radians(VENUES[b][0]), np.radians(VENUES[b][1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * 3958.8 * np.arcsin(np.sqrt(h)))


def tz_offset(team: str) -> int:
    return VENUES.get(team, (0, 0, -5, ""))[2]


def timezone_shift(travelling: str, host: str) -> int:
    """Time zones crossed, signed from the travelling team's perspective.

    Positive means the team travelled east and loses hours - a 1pm Eastern
    kickoff is 10am to a body clock still on Pacific time. Negative means they
    travelled west.
    """
    return tz_offset(travelling) - tz_offset(host)


def body_clock_hour(kickoff_local: float, travelling: str, host: str) -> float:
    """Kickoff time as the visiting team's body clock experiences it."""
    return kickoff_local + timezone_shift(travelling, host)


def parse_kickoff(gametime) -> float:
    """Kickoff as a float hour in the host stadium's local time."""
    if not isinstance(gametime, str) or ":" not in gametime:
        return np.nan
    try:
        h, m = gametime.split(":")[:2]
        return float(h) + float(m) / 60.0
    except ValueError:
        return np.nan


def home_climate(games: pd.DataFrame, min_games: int = 8) -> pd.DataFrame:
    """Each team's typical home conditions, measured from played games.

    A team's weather baseline is what it actually plays in, so the comparison
    for an away game is against its own norm rather than a league average.
    """
    g = games.dropna(subset=["home_score"]).copy()
    g = g[g["temp"].notna()]
    if g.empty:
        return pd.DataFrame(columns=["team", "home_temp", "home_wind", "n"])
    agg = (
        g.groupby("home_team")
        .agg(home_temp=("temp", "median"), home_wind=("wind", "median"), n=("temp", "size"))
        .reset_index().rename(columns={"home_team": "team"})
    )
    return agg[agg["n"] >= min_games]


def game_context(games: pd.DataFrame, climate: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-game travel, climate and familiarity features for the away team."""
    g = games.copy()
    g["kickoff_local"] = g["gametime"].map(parse_kickoff)

    g["travel_miles"] = [haversine(a, h) for a, h in zip(g["away_team"], g["home_team"])]
    g["tz_shift"] = [timezone_shift(a, h) for a, h in zip(g["away_team"], g["home_team"])]
    g["away_body_clock"] = g["kickoff_local"] + g["tz_shift"]

    # Neutral-site games are not a home game for anybody.
    g["neutral"] = g["location"].astype(str).str.lower().eq("neutral")

    g["is_divisional"] = g.get("div_game", 0).fillna(0).astype(int)
    g["same_conference"] = [
        int(DIVISIONS.get(a, "?").split()[0] == DIVISIONS.get(h, "?").split()[0])
        for a, h in zip(g["away_team"], g["home_team"])
    ]

    g["away_is_dome_team"] = g["away_team"].isin(DOME_TEAMS).astype(int)
    g["outdoor_game"] = g["roof"].astype(str).str.lower().isin(["outdoors", "open"]).astype(int)
    # A dome team playing outdoors is the classic climate-shock case.
    g["dome_team_outdoors"] = g["away_is_dome_team"] * g["outdoor_game"]

    if climate is not None and not climate.empty:
        c = climate.set_index("team")
        g["away_home_temp"] = g["away_team"].map(c["home_temp"])
        # Only meaningful when the game is actually played in the weather.
        g["temp_delta"] = np.where(
            g["outdoor_game"] == 1, g["temp"] - g["away_home_temp"], 0.0
        )
    else:
        g["temp_delta"] = np.nan

    g["rest_diff"] = g.get("home_rest", np.nan) - g.get("away_rest", np.nan)
    return g


# ---------------------------------------------------------------------------
# Environment adjustments
# ---------------------------------------------------------------------------
# Every coefficient below was measured over 7,276 games with a closing line
# (1999-2025) and checked for stability across three eras. Only effects that
# held their sign and rough magnitude in all three are applied.
#
# What is deliberately NOT here is as important as what is. Travel distance,
# time zones crossed, and visiting-team body clock at kickoff were all tested
# and are not applied to the margin. The historical east-to-west effect was
# real and large in 1999-2009 (-2.8 points against the spread, t = -3.6) and
# has since decayed to nothing (-0.11, then +0.24). That is what an efficient
# market does to a publicised edge, and a model that still paid for it would be
# betting on a pattern that stopped existing. Those fields are still computed
# and surfaced, so the context is visible even though it carries no weight.

WIND_POINTS_PER_MPH = -0.191   # scoring lost per mph above the calm threshold
WIND_CALM_MPH = 8.0            # below this, wind is not doing anything
INDOOR_POINTS = 1.00           # domes and closed roofs score more
DIVISIONAL_POINTS = -0.91      # familiarity suppresses scoring

# Effect of wind on how offences play, measured directly from 7,901 charted
# plays in 15+ mph conditions against calm outdoor baseline.
WIND_PASS_RATE_PER_MPH = -0.0024
WIND_YPA_PCT_PER_MPH = -0.0058
WIND_DEEP_RATE_PCT_PER_MPH = -0.0140
WIND_YPC_PCT_PER_MPH = -0.0021
# There is deliberately no field-goal wind coefficient. Controlling for
# distance, the effect is -0.017 per mph with a bootstrap t of -1.0 and a 95%
# interval spanning zero, on only ~200 attempts in 15+ mph. Coaches also
# attempt shorter kicks in wind, absorbing part of it at the decision level.
# See nflproj/kicking.py.


def _wind_excess(wind, roof) -> float:
    """Wind above the calm threshold, and only when it can reach the field."""
    if wind is None or not np.isfinite(wind):
        return 0.0
    if isinstance(roof, str) and roof.strip().lower() not in ("outdoors", "open"):
        return 0.0
    return max(float(wind) - WIND_CALM_MPH, 0.0)


def environment(row) -> dict:
    """Scoring-environment adjustments for one game.

    ``row`` is a schedule record carrying roof, wind, temp and div_game.
    """
    roof = row.get("roof") if hasattr(row, "get") else getattr(row, "roof", None)
    wind = row.get("wind") if hasattr(row, "get") else getattr(row, "wind", None)
    div = row.get("div_game") if hasattr(row, "get") else getattr(row, "div_game", 0)

    indoors = isinstance(roof, str) and roof.strip().lower() in ("dome", "closed", "retractable")
    excess = _wind_excess(wind, roof)

    total_delta = excess * WIND_POINTS_PER_MPH
    if indoors:
        total_delta += INDOOR_POINTS
    if div and float(div) == 1:
        total_delta += DIVISIONAL_POINTS

    return {
        "wind_excess": excess,
        "indoors": indoors,
        "divisional": bool(div and float(div) == 1),
        "total_delta": float(total_delta),
        # Multipliers for player-level projections.
        "pass_rate_delta": excess * WIND_PASS_RATE_PER_MPH,
        "pass_yards_mult": float(max(1.0 + excess * WIND_YPA_PCT_PER_MPH, 0.75)),
        "deep_rate_mult": float(max(1.0 + excess * WIND_DEEP_RATE_PCT_PER_MPH, 0.55)),
        "rush_yards_mult": float(max(1.0 + excess * WIND_YPC_PCT_PER_MPH, 0.90)),
    }


def travel_context(away: str, home: str, kickoff_local: float | None = None) -> dict:
    """Travel facts for a game, reported but not applied to the projection.

    See the note above: these were tested against both the spread and the
    model's own residuals and are not predictive in the modern era. They are
    surfaced so a reader can see the situation and apply their own judgement.
    """
    tz = timezone_shift(away, home)
    out = {
        "travel_miles": haversine(away, home),
        "tz_shift": tz,
        "direction": "east" if tz > 0 else "west" if tz < 0 else "none",
        "applied_to_projection": False,
    }
    if kickoff_local is not None and np.isfinite(kickoff_local):
        out["away_body_clock"] = kickoff_local + tz
    return out
