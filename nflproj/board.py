"""Build full projection boards: every relevant player, one row each."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import availability as av
from . import venues as V
from . import projections as pj
from . import usage as usage_mod
from . import schemes
from .config import LAST_COMPLETED_SEASON, PROJECTION_SEASON, TEAMS, normalize_team

# Positions worth projecting, and how far down the chart to go.
BOARD_DEPTH = {"QB": 1, "RB": 3, "WR": 5, "TE": 2, "FB": 1}


def schedule_for(games: pd.DataFrame, season: int, week: int | None = None) -> pd.DataFrame:
    g = games[(games["season"] == season) & (games["game_type"] == "REG")].copy()
    if week is not None:
        g = g[g["week"] == week]
    return g


def game_environments(games: pd.DataFrame, season: int, week: int) -> dict[str, dict]:
    """Scoring environment per team for a given week."""
    g = schedule_for(games, season, week)
    out: dict[str, dict] = {}
    for _, r in g.iterrows():
        env = V.environment({"roof": r.get("roof"), "wind": r.get("wind"),
                             "div_game": r.get("div_game", 0)})
        out[normalize_team(r["home_team"])] = env
        out[normalize_team(r["away_team"])] = env
    return out


def game_contexts(games: pd.DataFrame, season: int, week: int) -> dict[str, pj.GameContext]:
    """One context per team for a given week, carrying the market line."""
    g = schedule_for(games, season, week)
    out: dict[str, pj.GameContext] = {}
    for _, r in g.iterrows():
        spread = r.get("spread_line")
        total = r.get("total_line")
        home, away = normalize_team(r["home_team"]), normalize_team(r["away_team"])
        has_line = pd.notna(spread) and pd.notna(total)
        out[home] = pj.GameContext(
            team=home, opponent=away, is_home=True,
            total_line=float(total) if has_line else None,
            spread_line=float(spread) if has_line else None,
            week=week, neutral=not has_line,
        )
        out[away] = pj.GameContext(
            team=away, opponent=home, is_home=False,
            total_line=float(total) if has_line else None,
            spread_line=-float(spread) if has_line else None,
            week=week, neutral=not has_line,
        )
    return out


def project_team(
    team: str, ctx, projections_map: dict, usage_hist: pd.DataFrame,
    sampler: pj.TouchSampler, game_ctx: pj.GameContext,
    league_def: pd.Series, n_sims: int = 20000, seed: int | None = None,
    env: dict | None = None,
) -> list[pj.PlayerProjection]:
    """Project every rostered skill player for one team in one game."""
    rng = np.random.default_rng(seed)
    scheme = projections_map[team]["offense"]["projected"]

    opp = game_ctx.opponent
    opp_scheme = projections_map[opp]["offense"]["projected"] if opp in projections_map else None
    opp_def = projections_map[opp]["defense"]["projected"] if opp in projections_map else None

    volume = pj.team_volume(scheme, game_ctx, opp_scheme, env=env)
    adj = pj.defense_adjustment(opp_def, league_def)

    chart = ctx.chart[ctx.chart["team"] == team]
    results: list[pj.PlayerProjection] = []
    avail_hist = getattr(ctx, "availability", None)
    injuries = getattr(ctx, "injuries", {}) or {}

    qb_row = chart[chart["pos_abb"] == "QB"].sort_values("pos_rank")
    qb_id = qb_row.iloc[0].get("gsis_id") if len(qb_row) else None

    for _, row in chart.iterrows():
        pos, rank = row["pos_abb"], int(row["pos_rank"])
        if rank > BOARD_DEPTH.get(pos, 0):
            continue
        pid = row.get("gsis_id") or None
        name = row["player_name"]

        if pos == "QB":
            qb_id = pid
            results.append(pj.project_quarterback(
                player_id=pid, name=name, team=team, usage_hist=usage_hist,
                qb_hist=ctx.qb_profiles, volume=volume, sampler=sampler,
                scheme=scheme, def_adj=adj, n_sims=n_sims, rng=rng, env=env,
            ))
        else:
            shared, continuity = usage_mod.qb_continuity(ctx.plays, qb_id, pid)
            results.append(pj.project_skill_player(
                player_id=pid, name=name, team=team, position=pos, depth_rank=rank,
                usage_hist=usage_hist, volume=volume, sampler=sampler,
                scheme=scheme, def_adj=adj, n_sims=n_sims, rng=rng,
                current_team=team, qb_shared_targets=shared, qb_continuity=continuity,
                env=env,
            ))

    _rebalance(results, volume)

    # Availability is applied after rebalancing so that the shares still add up
    # among players who are on the field.
    for r in results:
        p_active, snap = av.project_availability(
            avail_hist, r.player_id, r.position, r.depth_rank,
            injury_status=injuries.get(r.player_id),
        )
        r.inputs["snap_share"] = snap
        r.apply_availability(p_active, rng)
    return results


def _rebalance(results: list[pj.PlayerProjection], volume: dict) -> None:
    """Keep a team's projected touches consistent with its projected volume.

    Shares are estimated independently per player, so they rarely sum to one.
    Rescaling the expectations - and the simulated samples with them - stops a
    deep receiving corps from inventing targets that were never thrown.
    """
    for kind, key, stats in (
        ("exp_targets", "team_targets", ("targets", "receptions", "rec_yards")),
        ("exp_carries", "rb_carries", ("carries", "rush_yards")),
    ):
        pool = [r for r in results if kind in r.inputs and r.position != "QB"]
        if not pool:
            continue
        total = sum(r.inputs[kind] for r in pool)
        if total <= 0:
            continue
        # The listed depth chart never covers every touch: deep reserves and
        # players not charted still absorb a slice, measured league-wide.
        coverage = usage_mod.CHARTED_COVERAGE["target" if kind == "exp_targets" else "carry"]
        target_total = volume[key] * coverage
        factor = target_total / total
        if not np.isfinite(factor) or abs(factor - 1.0) < 0.01:
            continue
        for r in pool:
            r.inputs[kind] *= factor
            for s in stats:
                if s in r.samples:
                    r.samples[s] = r.samples[s] * factor
        for r in pool:
            if "rec_yards" in r.samples and "rush_yards" in r.samples:
                r.samples["scrimmage_yards"] = r.samples["rec_yards"] + r.samples["rush_yards"]


def board_frame(results: list[pj.PlayerProjection], lines: dict | None = None) -> pd.DataFrame:
    """Flatten projections into a sortable board."""
    rows = [r.summary() for r in results]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if lines:
        for stat, line in lines.items():
            col = f"p_over_{stat}"
            df[col] = [
                next((r.prob_over(stat, line) for r in results if r.name == n), None)
                for n in df["player"]
            ]
    sort_col = "scrimmage_yards_mean" if "scrimmage_yards_mean" in df else "pass_yards_mean"
    return df.sort_values(sort_col, ascending=False)


def season_board(
    ctx, projections_map: dict, usage_hist: pd.DataFrame, sampler: pj.TouchSampler,
    season: int = PROJECTION_SEASON, n_sims: int = 4000, weeks: int = 17,
) -> pd.DataFrame:
    """Full-season per-game projections, averaged over each team's schedule.

    Every team's own slate is walked so that strength of schedule and the
    market's view of each game feed into the season line, rather than assuming
    a generic opponent seventeen times.
    """
    league_def = schemes.league_means(ctx.fingerprints, "defense", LAST_COMPLETED_SEASON)
    sched = schedule_for(ctx.games, season)
    if sched.empty:
        return pd.DataFrame()

    per_team_games: dict[str, list[pj.GameContext]] = {t: [] for t in TEAMS}
    for wk in sorted(sched["week"].unique()):
        for team, gc in game_contexts(ctx.games, season, int(wk)).items():
            if team in per_team_games:
                per_team_games[team].append(gc)

    frames = []
    for team, gcs in per_team_games.items():
        if not gcs:
            continue
        acc: dict[str, dict] = {}
        for i, gc in enumerate(gcs[:weeks]):
            res = project_team(team, ctx, projections_map, usage_hist, sampler, gc,
                               league_def, n_sims=n_sims, seed=1000 + i)
            for r in res:
                s = acc.setdefault(r.name, {"n": 0, "pos": r.position, "rank": r.depth_rank,
                                            "team": team, "totals": {}})
                s["n"] += 1
                for stat, vals in r.samples.items():
                    s["totals"][stat] = s["totals"].get(stat, 0.0) + float(np.mean(vals))
                s["totals"]["anytime_td_pct"] = s["totals"].get("anytime_td_pct", 0.0) + \
                    float((r.samples.get("total_td", np.zeros(1)) >= 1).mean() * 100)

        for name, s in acc.items():
            n = max(s["n"], 1)
            rec = {"team": team, "player": name, "pos": s["pos"], "rank": s["rank"], "games": n}
            for stat, tot in s["totals"].items():
                if stat == "anytime_td_pct":
                    rec["anytime_td_pct_per_game"] = tot / n
                else:
                    rec[f"{stat}_per_game"] = tot / n
                    rec[f"{stat}_season"] = (tot / n) * 17
            frames.append(rec)
    return pd.DataFrame(frames)
