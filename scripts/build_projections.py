"""Generate season-long projection exports.

Writes a per-game and full-season board for every team to ``data/exports`` so
the numbers can be used outside the app.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nflproj import board, pipeline, projections as pj, usage as um
from nflproj.config import DATA, PROJECTION_SEASON


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=PROJECTION_SEASON)
    ap.add_argument("--sims", type=int, default=4000,
                    help="simulations per player-game (season pass uses many games)")
    ap.add_argument("--weeks", type=int, default=17)
    ap.add_argument("--refresh", action="store_true", help="rebuild cached play frames")
    args = ap.parse_args()

    out = DATA / "exports"
    out.mkdir(parents=True, exist_ok=True)

    print("building context…")
    ctx = pipeline.build_context(use_cache=not args.refresh)
    pm = pipeline.project_team_schemes(ctx)
    usage_hist = um.player_usage(ctx.plays)
    sampler = pj.TouchSampler(ctx.plays)

    print("scheme tables…")
    pipeline.scheme_table(pm, "offense").to_csv(out / f"scheme_offense_{args.season}.csv")
    pipeline.scheme_table(pm, "defense").to_csv(out / f"scheme_defense_{args.season}.csv")

    print(f"season board ({args.weeks} weeks, {args.sims} sims/game)…")
    sb = board.season_board(ctx, pm, usage_hist, sampler, season=args.season,
                            n_sims=args.sims, weeks=args.weeks)
    if sb.empty:
        print("no schedule available for", args.season)
        return
    sb = sb.sort_values("scrimmage_yards_season", ascending=False)
    sb.to_csv(out / f"season_projections_{args.season}.csv", index=False)

    cols = ["team", "player", "pos", "rank", "rec_yards_season", "rush_yards_season",
            "scrimmage_yards_season", "total_td_season", "anytime_td_pct_per_game"]
    print("\ntop 20 by projected scrimmage yards:")
    print(sb[[c for c in cols if c in sb.columns]].head(20).round(1).to_string(index=False))
    print(f"\nwrote {out}/season_projections_{args.season}.csv  ({len(sb)} players)")


if __name__ == "__main__":
    main()
