"""Hold out 2025 and project it from prior seasons only.

The point is not to show the model looks good - it is to find out whether the
volume and scoring chain is calibrated before anyone leans on it. Fingerprints,
usage history and efficiency priors are all built from 2022-2024. The 2025
schedule, its closing market lines and its depth charts are supplied, and the
resulting projections are compared against what actually happened.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflproj import (availability as av, board, coaches, data, personnel, pipeline,
                     projections as pj, schemes, usage as um, venues as V)

TRAIN = (2022, 2023, 2024)
TEST = 2025


def build_holdout_context():
    raw = data.play_by_play(TRAIN, columns=schemes.PBP_COLUMNS)
    plays = schemes.prepare_plays(raw, data.charting(TRAIN))
    fp = schemes.build_fingerprints(plays, raw_pbp=raw)
    depth = data.depth_chart(TEST)
    snaps = data.snap_counts(TRAIN)
    players_df = data.players()
    return pipeline.Context(
        plays=plays, fingerprints=fp, depth=depth,
        chart=personnel.latest_depth_chart(depth),
        fronts=personnel.base_defensive_front(depth),
        qb_profiles=personnel.qb_profiles(plays),
        # No staff registry for a past season: treat every team as continuity so
        # the test measures the projection chain, not the coaching research.
        staffs={t: coaches.Staff(team=t, continuity=True) for t in pipeline.TEAMS},
        games=data.games(), weekly=data.weekly_stats(TRAIN), snaps=snaps,
        # Availability is learned from the training seasons only.
        availability=av.player_availability(snaps, players_df),
        injuries={},
    )


def actuals_2025() -> pd.DataFrame:
    raw = data.play_by_play([TEST], columns=schemes.PBP_COLUMNS)
    p = schemes.prepare_plays(raw, data.charting([TEST]))
    rec = (
        p[p["receiver_player_id"].notna()]
        .groupby(["game_id", "posteam", "receiver_player_id"])
        .agg(targets=("play_id", "size"),
             rec_yards=("yards_gained", lambda s: float(s[p.loc[s.index, "complete_pass"].fillna(0) > 0].sum())),
             rec_td=("pass_touchdown", "sum"))
        .reset_index().rename(columns={"receiver_player_id": "player_id", "posteam": "team"})
    )
    rush = (
        p[p["is_designed_run"] & p["rusher_player_id"].notna()]
        .groupby(["game_id", "posteam", "rusher_player_id"])
        .agg(carries=("play_id", "size"), rush_yards=("yards_gained", "sum"),
             rush_td=("rush_touchdown", "sum"))
        .reset_index().rename(columns={"rusher_player_id": "player_id", "posteam": "team"})
    )
    a = rec.merge(rush, on=["game_id", "team", "player_id"], how="outer")
    for c in ("targets", "rec_yards", "rec_td", "carries", "rush_yards", "rush_td"):
        a[c] = pd.to_numeric(a[c], errors="coerce").fillna(0.0)
    a["scrimmage_yards"] = a["rec_yards"] + a["rush_yards"]
    a["total_td"] = a["rec_td"] + a["rush_td"]
    return a


def run(weeks=range(1, 19), n_sims=4000, inseason: bool = False) -> pd.DataFrame:
    """Project the held-out season week by week.

    With ``inseason`` the model is rebuilt before each week from the games
    already played that year - usage, team form and defensive quality all
    refresh. Weeks are filtered strictly below the target, so a projection never
    sees its own game. Without it the whole season is projected from 2022-2024
    alone, which is what the model does before a season starts.
    """
    ctx = build_holdout_context()
    static_pm = pipeline.project_team_schemes(ctx, anchor_season=2024)
    static_usage = um.player_usage(ctx.plays)
    sampler = pj.TouchSampler(ctx.plays)
    league_def = schemes.league_means(ctx.fingerprints, "defense", 2024)

    # Current-season play, loaded once and sliced per week.
    test_plays = None
    if inseason:
        test_raw = data.play_by_play([TEST], columns=schemes.PBP_COLUMNS)
        test_plays = schemes.prepare_plays(test_raw, data.charting([TEST]))
        print(f"in-season mode: {len(test_plays)} plays available from {TEST}")
    act = actuals_2025()
    sched = board.schedule_for(ctx.games, TEST)

    rows = []
    for wk in weeks:
        pm, usage_hist = static_pm, static_usage
        if inseason and test_plays is not None and wk > 1:
            played = test_plays[test_plays["week"] < wk]
            if not played.empty:
                combined = pd.concat([ctx.plays, played], ignore_index=True)
                wk_ctx = replace(ctx, plays=combined, current_season=TEST,
                                 through_week=int(wk),
                                 fingerprints=schemes.build_fingerprints(combined))
                pm = pipeline.project_team_schemes(wk_ctx, anchor_season=2024)
                usage_hist = um.player_usage(combined)
                ctx_for_week = wk_ctx
            else:
                ctx_for_week = ctx
        else:
            ctx_for_week = ctx
        # Use the depth chart as it stood going into this week. Reusing an
        # end-of-season chart credits late-emerging players with a role they
        # did not yet have, which is hindsight leaking into the test.
        wk_depth = data.depth_chart_for_week(TEST, int(wk), ctx.games)
        wk_chart = personnel.latest_depth_chart(wk_depth)
        if wk_chart.empty:
            wk_chart = ctx.chart
        wk_envs = board.game_environments(ctx.games, TEST, int(wk))
        gcs = board.game_contexts(ctx.games, TEST, int(wk))
        wk_games = sched[sched["week"] == wk]
        gid = {}
        for _, r in wk_games.iterrows():
            gid[r["home_team"]] = r["game_id"]
            gid[r["away_team"]] = r["game_id"]
        for team, gc in gcs.items():
            res = board.project_team(team, ctx_for_week, pm, usage_hist, sampler, gc,
                                     league_def, n_sims=n_sims, seed=int(wk) * 100,
                                     env=wk_envs.get(team), chart=wk_chart)
            for r in res:
                rows.append({
                    "week": wk, "game_id": gid.get(team), "team": team,
                    "player_id": r.player_id, "player": r.name, "pos": r.position,
                    "proj_rec_yards": float(np.mean(r.samples.get("rec_yards", [0]))),
                    "proj_rush_yards": float(np.mean(r.samples.get("rush_yards", [0]))),
                    "proj_scrimmage": float(np.mean(r.samples.get("scrimmage_yards", [0]))) if "scrimmage_yards" in r.samples else np.nan,
                    "proj_pass_yards": float(np.mean(r.samples.get("pass_yards", [0]))) if "pass_yards" in r.samples else np.nan,
                    "proj_targets": float(np.mean(r.samples.get("targets", [0]))) if "targets" in r.samples else np.nan,
                    "proj_carries": float(np.mean(r.samples.get("carries", [0]))),
                    "p_anytime_td": float((r.samples.get("total_td", np.zeros(1)) >= 1).mean()),
                    "p_active": r.p_active,
                    "proj_scrimmage_if_active": float(np.mean(r.conditional.get("scrimmage_yards", [0]))) if "scrimmage_yards" in r.conditional else np.nan,
                })
    proj = pd.DataFrame(rows)
    merged = proj.merge(act, on=["game_id", "team", "player_id"], how="left")
    for c in ("targets", "rec_yards", "rec_td", "carries", "rush_yards", "rush_td",
              "scrimmage_yards", "total_td"):
        merged[c] = merged[c].fillna(0.0)
    return merged


def report(m: pd.DataFrame) -> None:
    skill = m[m["pos"].isin(["WR", "TE", "RB", "FB"])]
    print(f"player-games projected: {len(m)}   skill: {len(skill)}\n")

    print("=== volume and yardage accuracy (skill players) ===")
    for proj_col, act_col, label in (
        ("proj_targets", "targets", "targets"),
        ("proj_carries", "carries", "carries"),
        ("proj_rec_yards", "rec_yards", "receiving yards"),
        ("proj_rush_yards", "rush_yards", "rushing yards"),
        ("proj_scrimmage", "scrimmage_yards", "scrimmage yards"),
    ):
        d = skill[[proj_col, act_col]].dropna()
        if d.empty:
            continue
        err = d[proj_col] - d[act_col]
        print(f"  {label:18s} MAE {err.abs().mean():6.2f}   bias {err.mean():+6.2f}   "
              f"corr {d[proj_col].corr(d[act_col]):.3f}   proj_mean {d[proj_col].mean():6.2f} "
              f"act_mean {d[act_col].mean():6.2f}")

    print()
    print("=== touchdown calibration (skill players) ===")
    s = skill.dropna(subset=["p_anytime_td"]).copy()
    s["bucket"] = pd.cut(s["p_anytime_td"], [0, .05, .10, .15, .20, .30, .45, 1.0])
    cal = s.groupby("bucket", observed=True).agg(
        n=("p_anytime_td", "size"),
        predicted=("p_anytime_td", "mean"),
        actual=("total_td", lambda x: float((x >= 1).mean())),
    )
    cal["gap"] = cal["actual"] - cal["predicted"]
    print(cal.round(3).to_string())
    overall_p = s["p_anytime_td"].mean()
    overall_a = float((s["total_td"] >= 1).mean())
    print(f"\n  overall: predicted {overall_p:.3f}  actual {overall_a:.3f}  "
          f"({'over' if overall_p > overall_a else 'under'}-forecast by {abs(overall_p-overall_a):.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inseason", action="store_true",
                    help="rebuild the model each week from games already played")
    ap.add_argument("--sims", type=int, default=4000)
    args = ap.parse_args()

    m = run(n_sims=args.sims, inseason=args.inseason)
    suffix = "_inseason" if args.inseason else ""
    m.to_parquet(f"data/cache/backtest_2025{suffix}.parquet")
    print(f"\n{'IN-SEASON' if args.inseason else 'PRESEASON'} MODE")
    report(m)
