"""Measure whether travel, climate and familiarity actually move outcomes.

Every effect here is folk wisdom until tested. The test is deliberately strict:
performance is measured against the *closing spread*, not raw results. The
market already prices team quality, home field and rest, so a West Coast team
losing in the East proves nothing - it was probably an underdog. The question is
whether it loses by more than the line expected.

Effects that do not clear their standard error are reported as null and left out
of the model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflproj import data, venues as V

MIN_N = 60


def build() -> pd.DataFrame:
    g = data.games()
    g = g[g["home_score"].notna() & g["spread_line"].notna()].copy()
    climate = V.home_climate(g)
    g = V.game_context(g, climate)

    g["actual_margin"] = g["home_score"] - g["away_score"]
    g["actual_total"] = g["home_score"] + g["away_score"]
    # Positive means the home team beat the number; negative means the road team did.
    g["ats_margin"] = g["actual_margin"] - g["spread_line"]
    # Flip to the travelling team's perspective, which is what these effects are about.
    g["away_ats"] = -g["ats_margin"]
    g["total_error"] = g["actual_total"] - g["total_line"]
    return g


def effect(df: pd.DataFrame, mask, label: str, col: str = "away_ats") -> dict | None:
    sub = df[mask]
    rest = df[~mask]
    if len(sub) < MIN_N:
        return None
    m, s = sub[col].mean(), sub[col].std(ddof=1)
    se = s / np.sqrt(len(sub))
    diff = m - rest[col].mean()
    return {
        "split": label, "n": len(sub), "mean": m, "se": se,
        "vs_rest": diff, "t": diff / se if se else np.nan,
        "significant": abs(diff / se) >= 2.0 if se else False,
    }


def section(title: str, rows: list[dict | None]) -> pd.DataFrame:
    rows = [r for r in rows if r]
    print(f"\n{'=' * 82}\n{title}\n{'=' * 82}")
    if not rows:
        print("  not enough data")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))
    return df


def main() -> None:
    g = build()
    modern = g[g["season"] >= 2010]
    print(f"games with a closing line: {len(g)}  (since 2010: {len(modern)})")
    print("Dependent variable: away team's margin against the spread. Positive means")
    print("the road team beat the number. 't' is versus all other games; |t| >= 2 flagged.")

    # ---- travel and body clock -------------------------------------------
    section("TRAVEL DIRECTION AND TIME ZONES (away team vs the spread)", [
        effect(g, g.tz_shift >= 2, "West team travelling east, 2+ zones"),
        effect(g, g.tz_shift == 1, "Travelling east, 1 zone"),
        effect(g, g.tz_shift == 0, "No time-zone change"),
        effect(g, g.tz_shift == -1, "Travelling west, 1 zone"),
        effect(g, g.tz_shift <= -2, "East team travelling west, 2+ zones"),
        effect(g, g.travel_miles >= 2000, "Trip of 2,000+ miles"),
        effect(g, g.travel_miles < 300, "Trip under 300 miles"),
    ])

    section("BODY CLOCK AT KICKOFF (away team vs the spread)", [
        effect(g, g.away_body_clock <= 10.5, "Kickoff at 10:30am body clock or earlier"),
        effect(g, (g.away_body_clock > 10.5) & (g.away_body_clock <= 13.5), "Midday body clock"),
        effect(g, g.away_body_clock >= 19, "Kickoff at 7pm body clock or later"),
        effect(g, (g.tz_shift >= 2) & (g.away_body_clock <= 10.5),
               "West team, east trip, early body clock"),
    ])

    # ---- climate ----------------------------------------------------------
    section("CLIMATE SHOCK (away team vs the spread)", [
        effect(g, g.dome_team_outdoors == 1, "Dome team playing outdoors"),
        effect(g, (g.dome_team_outdoors == 1) & (g.temp <= 40), "Dome team outdoors, 40F or colder"),
        effect(g, (g.outdoor_game == 1) & (g.temp_delta <= -25), "25F+ colder than the road team's home"),
        effect(g, (g.outdoor_game == 1) & (g.temp_delta >= 20), "20F+ warmer than home"),
        effect(g, (g.outdoor_game == 1) & (g.temp <= 32), "Freezing or below"),
        effect(g, (g.outdoor_game == 1) & (g.wind >= 15), "Wind 15mph or more"),
    ])

    section("CLIMATE AND SCORING (actual total minus the market total)", [
        effect(g, (g.outdoor_game == 1) & (g.temp <= 32), "Freezing or below", "total_error"),
        effect(g, (g.outdoor_game == 1) & (g.wind >= 15), "Wind 15mph or more", "total_error"),
        effect(g, (g.outdoor_game == 1) & (g.wind >= 20), "Wind 20mph or more", "total_error"),
        effect(g, g.outdoor_game == 0, "Indoors", "total_error"),
        effect(g, (g.outdoor_game == 1) & (g.temp >= 75), "75F or warmer", "total_error"),
    ])

    # ---- familiarity ------------------------------------------------------
    section("DIVISIONAL AND CONFERENCE GAMES (away team vs the spread)", [
        effect(g, g.is_divisional == 1, "Divisional matchup"),
        effect(g, (g.is_divisional == 1) & (g.week >= 14), "Divisional, week 14+"),
        effect(g, (g.is_divisional == 0) & (g.same_conference == 1), "Same conference, different division"),
        effect(g, g.same_conference == 0, "Inter-conference"),
    ])

    section("DIVISIONAL GAMES AND SCORING (actual total minus the market total)", [
        effect(g, g.is_divisional == 1, "Divisional matchup", "total_error"),
        effect(g, g.is_divisional == 0, "Non-divisional", "total_error"),
    ])

    section("DIVISIONAL GAMES: ARE THEY CLOSER THAN THE LINE EXPECTS?", [
        {"split": "Divisional |actual margin|", "n": int((g.is_divisional == 1).sum()),
         "mean": float(g[g.is_divisional == 1].actual_margin.abs().mean()), "se": np.nan,
         "vs_rest": float(g[g.is_divisional == 1].actual_margin.abs().mean()
                          - g[g.is_divisional == 0].actual_margin.abs().mean()),
         "t": np.nan, "significant": False},
        {"split": "Non-divisional |actual margin|", "n": int((g.is_divisional == 0).sum()),
         "mean": float(g[g.is_divisional == 0].actual_margin.abs().mean()), "se": np.nan,
         "vs_rest": 0.0, "t": np.nan, "significant": False},
        {"split": "Divisional |spread|", "n": int((g.is_divisional == 1).sum()),
         "mean": float(g[g.is_divisional == 1].spread_line.abs().mean()), "se": np.nan,
         "vs_rest": float(g[g.is_divisional == 1].spread_line.abs().mean()
                          - g[g.is_divisional == 0].spread_line.abs().mean()),
         "t": np.nan, "significant": False},
    ])

    # ---- has anything changed in the modern era? --------------------------
    section("MODERN ERA CHECK, 2010 ONWARDS (away team vs the spread)", [
        effect(modern, modern.tz_shift >= 2, "West team travelling east, 2+ zones"),
        effect(modern, modern.tz_shift <= -2, "East team travelling west, 2+ zones"),
        effect(modern, (modern.tz_shift >= 2) & (modern.away_body_clock <= 10.5),
               "West team, east trip, early body clock"),
        effect(modern, modern.dome_team_outdoors == 1, "Dome team outdoors"),
        effect(modern, modern.is_divisional == 1, "Divisional matchup"),
    ])


if __name__ == "__main__":
    main()
