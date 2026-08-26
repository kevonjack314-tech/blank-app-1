"""Validate the game model on held-out seasons.

Two questions, in order of importance:

  1. Does the model predict games at all - is its margin better than assuming
     every game is a coin flip at league-average scoring?
  2. Does it beat the closing line?

The second bar is the one that matters and the one almost nothing clears. NFL
closing lines aggregate enormous amounts of information, including injury and
personnel news the model never sees. Reporting the comparison honestly is the
point of this script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflproj import data, gamemodel as gm, pipeline, schemes


def run(test_seasons=(2024, 2025)) -> pd.DataFrame:
    raw = data.play_by_play([2021, 2022, 2023, 2024, 2025], columns=schemes.PBP_COLUMNS)
    plays = schemes.prepare_plays(raw, data.charting([2022, 2023, 2024, 2025]))
    ratings = gm.adjusted_ratings(plays)
    games = data.games()

    rows = []
    for test in test_seasons:
        # Ratings and the points mapping may only use seasons before the test.
        hist = ratings[ratings["season"] < test]
        if hist.empty:
            continue
        proj = gm.project_ratings(hist, anchor_season=test - 1)
        past_games = games[games["season"] < test]
        scoring = gm.fit_scoring_map(plays, past_games, hist)

        # Walk the season week by week, refreshing ratings from games already
        # played, so the model sees what the closing line saw.
        weeks = sorted(games[(games["season"] == test) &
                             (games["game_type"] == "REG")]["week"].unique())
        parts = []
        for wk in weeks:
            wk_proj = gm.progressive_ratings(plays, test, int(wk), proj)
            parts.append(gm.predict_slate(games, test, wk_proj, scoring,
                                          week=int(wk), n_sims=4000))
        pred = pd.concat(parts, ignore_index=True)
        actual = games[(games["season"] == test) & (games["game_type"] == "REG")][
            ["game_id", "home_score", "away_score", "spread_line", "total_line"]
        ].dropna(subset=["home_score"])
        m = pred.merge(actual, on="game_id", how="inner", suffixes=("", "_a"))
        m["season"] = test
        m["actual_margin"] = m["home_score"] - m["away_score"]
        m["actual_total"] = m["home_score"] + m["away_score"]
        rows.append(m)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def report(m: pd.DataFrame) -> None:
    print(f"games: {len(m)}  seasons: {sorted(m.season.unique())}\n")

    print("=== margin ===")
    naive = np.abs(m["actual_margin"] - 0.0)
    hfa_only = np.abs(m["actual_margin"] - m["actual_margin"].mean())
    model = np.abs(m["actual_margin"] - m["model_margin"])
    print(f"  pick-em (always 0)        MAE {naive.mean():6.2f}")
    print(f"  home-field only           MAE {hfa_only.mean():6.2f}")
    print(f"  model                     MAE {model.mean():6.2f}   corr {m.model_margin.corr(m.actual_margin):.3f}")
    mk = m.dropna(subset=["spread_line"])
    if len(mk):
        mkt = np.abs(mk["actual_margin"] - mk["spread_line"])
        mdl = np.abs(mk["actual_margin"] - mk["model_margin"])
        print(f"  market closing line       MAE {mkt.mean():6.2f}   corr {mk.spread_line.corr(mk.actual_margin):.3f}   (n={len(mk)})")
        print(f"  model on the same games   MAE {mdl.mean():6.2f}")
        print(f"  -> market is better by {mdl.mean() - mkt.mean():+.2f} points of MAE")

    print("\n=== total ===")
    mt = m.dropna(subset=["total_line"])
    if len(mt):
        print(f"  model                     MAE {np.abs(mt.actual_total - mt.model_total).mean():6.2f}")
        print(f"  market                    MAE {np.abs(mt.actual_total - mt.total_line).mean():6.2f}")

    print("\n=== straight-up winner ===")
    picked = np.where(m["model_margin"] > 0, 1, 0)
    won = np.where(m["actual_margin"] > 0, 1, 0)
    print(f"  model accuracy            {(picked == won).mean() * 100:5.1f}%")
    if len(mk):
        mp = np.where(mk["spread_line"] > 0, 1, 0)
        mw = np.where(mk["actual_margin"] > 0, 1, 0)
        print(f"  market accuracy           {(mp == mw).mean() * 100:5.1f}%")

    print("\n=== win-probability calibration ===")
    m2 = m.copy()
    m2["bucket"] = pd.cut(m2["home_win_pct"], [0, 35, 45, 55, 65, 100])
    cal = m2.groupby("bucket", observed=True).agg(
        n=("home_win_pct", "size"),
        predicted=("home_win_pct", lambda x: x.mean() / 100),
        actual=("actual_margin", lambda x: float((x > 0).mean())),
    )
    print(cal.round(3).to_string())

    print("\n=== against the spread, if you had bet the model's edge ===")
    if len(mk):
        b = mk[np.abs(mk["spread_edge"]) >= 2.0].copy()
        if len(b):
            b["cover"] = np.where(
                b["spread_edge"] > 0,
                b["actual_margin"] > b["spread_line"],
                b["actual_margin"] < b["spread_line"],
            )
            print(f"  edge >= 2 pts: {len(b)} games, {b['cover'].mean() * 100:.1f}% correct "
                  f"(break-even at -110 is 52.4%)")
        b3 = mk[np.abs(mk["spread_edge"]) >= 4.0].copy()
        if len(b3):
            b3["cover"] = np.where(
                b3["spread_edge"] > 0,
                b3["actual_margin"] > b3["spread_line"],
                b3["actual_margin"] < b3["spread_line"],
            )
            print(f"  edge >= 4 pts: {len(b3)} games, {b3['cover'].mean() * 100:.1f}% correct")


if __name__ == "__main__":
    m = run()
    m.to_parquet("data/cache/backtest_games.parquet")
    report(m)
