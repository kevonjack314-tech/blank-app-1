"""Correlated simulation of a whole game at once.

The projection board simulates each player independently, which is correct for
any single player's line. It is wrong for anything that combines them.

A quarterback's passing yards and his receiver's receiving yards are the same
event counted twice: both rise when the offence throws a lot. A running back's
carries move the other way. Multiplying independent probabilities together —
which is what a naive parlay calculator does — understates correlated legs and
overstates opposing ones, and the error is largest for exactly the same-game
combinations people build most.

This module simulates the game once and reads every leg off the same draws, so
the dependence is carried rather than assumed away. Three layers of shared
randomness:

  * a game-level pace shock, moving both teams' play counts together;
  * a game-level scoring shock, the shootout-or-slugfest axis;
  * a team-level pass-rate shock, plus game script - the team trailing in a
    given simulation throws more, and its opponent runs more.

Player shares are drawn from a Dirichlet within each team, so a simulation where
the WR1 sees twenty targets is one where somebody else sees fewer. That is what
makes "WR1 over and WR2 over" correctly harder than the product of the two.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import availability as av
from . import blocking as bl
from . import projections as pj
from . import usage as usage_mod
from .board import BOARD_DEPTH

# Spread of the shared shocks, calibrated against within-team game-to-game
# dispersion measured over 2022-2025: play counts vary with a coefficient of
# variation of 0.129, pass attempts 0.206, carries 0.271. Guessing these low is
# what makes a joint model produce correlations near zero - the shared component
# has to be big enough to matter against per-player noise.
# Player touch counts are drawn as Poisson around the team volume, which
# already contributes sqrt(mean) of spread. These shocks supply only the
# systematic remainder, so that total simulated dispersion lands on the
# measured figures rather than overshooting them - over-dispersing team volume
# is what flattens the correlations this module exists to capture.
# --- shared randomness -----------------------------------------------------
# Calibrated against within-team game-to-game dispersion and same-game
# correlations measured over 2022-2025 (see docstring). Player touch counts are
# drawn as Poisson around team volume, which already supplies sqrt(mean) of
# spread, so these shocks provide only the systematic remainder.

PACE_SHOCK_SD = 0.060        # play count, shared by both teams
SCORING_SHOCK_SD = 0.280     # shootout versus slugfest, shared by both teams

# High-scoring games are longer games - more drives, more snaps for both
# offences. Without this the only link between the two teams is the margin,
# which pushes them apart, and opposing quarterbacks come out negatively
# correlated when in reality they rise together (real r = +0.135).
SHOOTOUT_PLAYS = 0.30

# Who wins must vary independently of how much scoring there is, or game script
# never fires: with only a shared scoring shock the margin is pinned to the
# fixed difference in implied points and nobody is ever blown out.
MARGIN_SHOCK_SD = 8.0

PASS_RATE_SHOCK_SD = 0.025   # residual pass-run variation beyond game script

# Early-down pass rate runs from 0.698 trailing by 14+ to 0.367 leading by 14+,
# about 0.012 per point at play level. Against a *final* margin - which a team
# does not carry all game - roughly half applies. This is what pushes a
# quarterback's yardage and his running back's in opposite directions.
SCRIPT_STRENGTH = 0.006      # pass-rate points per point of simulated margin

SHARE_CONCENTRATION = {"target": 26.0, "carry": 15.0}


@dataclass
class JointGame:
    """One game simulated jointly. Every array shares a simulation index."""
    home: str
    away: str
    n_sims: int
    players: dict = field(default_factory=dict)   # name -> {stat: samples}
    meta: dict = field(default_factory=dict)      # name -> position, team, p_active
    team_totals: dict = field(default_factory=dict)
    active: dict = field(default_factory=dict)     # name -> bool array per simulation

    def active_mask(self, *players: str) -> np.ndarray:
        """Simulations in which every named player was available."""
        m = np.ones(self.n_sims, dtype=bool)
        for name in players:
            a = self.active.get(name)
            if a is not None:
                m &= a
        return m

    def stat(self, player: str, stat: str) -> np.ndarray | None:
        return self.players.get(player, {}).get(stat)

    def prob_over(self, player: str, stat: str, line: float) -> float | None:
        s = self.stat(player, stat)
        if s is None or len(s) == 0:
            return None
        return float((s > line).mean())

    def prob_under(self, player: str, stat: str, line: float) -> float | None:
        p = self.prob_over(player, stat, line)
        return None if p is None else 1.0 - p

    def leg_mask(self, player: str, stat: str, line: float, side: str = "over") -> np.ndarray | None:
        """Boolean array: did this leg hit in each simulation?"""
        s = self.stat(player, stat)
        if s is None or len(s) == 0:
            return None
        return (s > line) if side == "over" else (s <= line)

    def roster(self) -> pd.DataFrame:
        rows = []
        for name, m in self.meta.items():
            rows.append({"player": name, **m})
        return pd.DataFrame(rows)


def _dirichlet_shares(rng, base: np.ndarray, concentration: float, n_sims: int) -> np.ndarray:
    """Per-simulation shares that vary but always sum to one.

    Returns an ``(n_sims, n_players)`` array. A Dirichlet is the natural choice
    because it keeps the constraint that a team's touches are finite: one
    player's good game is another's quiet one.
    """
    base = np.asarray(base, dtype=float)
    base = np.clip(base, 1e-4, None)
    base = base / base.sum()
    alpha = np.clip(base * concentration, 1e-3, None)
    return rng.dirichlet(alpha, size=n_sims)


def simulate_game(
    home: str, away: str, ctx, projections_map: dict, usage_hist: pd.DataFrame,
    sampler: pj.TouchSampler, contexts: dict, league_def: pd.Series,
    envs: dict | None = None, n_sims: int = 20000, seed: int | None = None,
    chart: pd.DataFrame | None = None,
) -> JointGame:
    """Simulate both teams in one game with shared randomness."""
    rng = np.random.default_rng(seed)
    envs = envs or {}

    # ---- shared game environment -----------------------------------------
    pace = rng.normal(1.0, PACE_SHOCK_SD, n_sims)
    scoring = rng.normal(1.0, SCORING_SHOCK_SD, n_sims)

    out = JointGame(home=home, away=away, n_sims=n_sims)
    source = ctx.chart if chart is None else chart

    # First pass: team volumes, so game script can see both sides.
    base = {}
    for team in (home, away):
        gc = contexts.get(team)
        if gc is None:
            continue
        scheme = projections_map[team]["offense"]["projected"]
        opp = home if team == away else away
        opp_scheme = projections_map[opp]["offense"]["projected"] if opp in projections_map else None
        protection = (getattr(ctx, "protection", None) or {}).get(team)
        vol = pj.team_volume(scheme, gc, opp_scheme, env=envs.get(team), protection=protection)
        base[team] = {"gc": gc, "scheme": scheme, "vol": vol,
                      "opp_def": projections_map[opp]["defense"]["projected"] if opp in projections_map else None}

    if len(base) < 2:
        return out

    # Two independent axes: how much scoring there is, and who gets it. The
    # first moves both teams together, the second pushes them apart.
    swing = rng.normal(0.0, MARGIN_SHOCK_SD, n_sims)
    pts = {
        home: np.maximum(base[home]["vol"]["implied_points"] * scoring + swing / 2, 0.0),
        away: np.maximum(base[away]["vol"]["implied_points"] * scoring - swing / 2, 0.0),
    } if len(base) == 2 else {t: base[t]["vol"]["implied_points"] * scoring for t in base}
    margin = {home: pts[home] - pts[away], away: pts[away] - pts[home]}

    for team, b in base.items():
        vol, scheme, gc = b["vol"], b["scheme"], b["gc"]
        env = envs.get(team)

        # High-scoring games are also longer games: more drives, more snaps for
        # both offences. Without this the only link between the two teams is the
        # margin, which pushes them apart, and opposing quarterbacks come out
        # negatively correlated when in reality they rise together.
        plays = vol["plays"] * pace * (1.0 + SHOOTOUT_PLAYS * (scoring - 1.0))
        # Trailing teams throw; leading teams run.
        pass_rate = np.clip(
            vol["pass_rate"] + rng.normal(0, PASS_RATE_SHOCK_SD, n_sims)
            - SCRIPT_STRENGTH * margin[team],
            0.30, 0.78,
        )
        dropbacks = plays * pass_rate
        sacks = dropbacks * float(scheme.get("sack_rate_allowed", 0.065))
        scrambles = dropbacks * float(scheme.get("scramble_rate", 0.07))
        attempts = np.maximum(dropbacks - sacks - scrambles, 1.0)
        designed_runs = plays * (1.0 - pass_rate)
        qb_runs = plays * float(scheme.get("qb_designed_run_rate", 0.03))
        rb_carries = np.maximum(designed_runs - qb_runs, 1.0)

        off_td = np.maximum(pj.expected_offensive_tds(0) +
                            pj.TD_FROM_POINTS[1] * pts[team], 0.2)
        pass_td_share = vol["pass_td"] / max(vol["expected_off_td"], 1e-6)
        team_pass_td = rng.poisson(np.clip(off_td * pass_td_share, 0.01, 6))
        team_rush_td = rng.poisson(np.clip(off_td * (1 - pass_td_share), 0.01, 6))

        out.team_totals[team] = {
            "plays": plays, "attempts": attempts, "rb_carries": rb_carries,
            "pass_td": team_pass_td, "rush_td": team_rush_td, "points": pts[team],
        }

        _allocate_team(out, team, b, ctx, projections_map, usage_hist, sampler,
                       league_def, source, attempts, rb_carries, qb_runs + scrambles,
                       team_pass_td, team_rush_td, env, rng, n_sims)

    return out


def _allocate_team(out: JointGame, team: str, b: dict, ctx, projections_map: dict,
                   usage_hist: pd.DataFrame, sampler: pj.TouchSampler,
                   league_def: pd.Series, source: pd.DataFrame,
                   attempts: np.ndarray, rb_carries: np.ndarray,
                   qb_carries: np.ndarray, team_pass_td: np.ndarray,
                   team_rush_td: np.ndarray, env: dict | None,
                   rng, n_sims: int) -> None:
    """Split one team's simulated volume among the players on its chart."""
    scheme = b["scheme"]
    adj = pj.defense_adjustment(b["opp_def"], league_def)
    chart = source[source["team"] == team]
    qb_row = chart[chart["pos_abb"] == "QB"].sort_values("pos_rank")
    qb_id = qb_row.iloc[0].get("gsis_id") if len(qb_row) else None

    skill, tgt_base, car_base, gl_base = [], [], [], []
    for _, row in chart.iterrows():
        pos, rank = row["pos_abb"], int(row["pos_rank"])
        if rank > BOARD_DEPTH.get(pos, 0) or pos == "QB":
            continue
        pid = row.get("gsis_id") or None
        pulls = {k: usage_mod.role_pull_for(usage_hist, pid, pos, rank, team, k)
                 for k in ("target", "carry", "goalline")}
        shared, cont = usage_mod.qb_continuity(ctx.plays, qb_id, pid)
        if pos in ("WR", "TE", "RB") and cont < 1.0:
            pulls["target"] = min(pulls["target"] + 0.22 * (1.0 - cont), 0.70)
        ts_, _ = usage_mod.project_share(usage_hist, pid, pos, rank, "target",
                                         current_team=team, role_pull=pulls["target"])
        cs_, _ = usage_mod.project_share(usage_hist, pid, pos, rank, "carry",
                                         current_team=team, role_pull=pulls["carry"])
        gs_, _ = usage_mod.project_share(usage_hist, pid, pos, rank, "goalline",
                                         current_team=team, role_pull=pulls["goalline"])
        skill.append({"id": pid, "name": row["player_name"], "pos": pos, "rank": rank})
        tgt_base.append(ts_); car_base.append(cs_); gl_base.append(gs_)

    if not skill:
        return

    # Availability first: an inactive player takes no share, and his work goes
    # to the team-mates who replace him rather than disappearing.
    p_active = []
    for s in skill:
        pa, _ = av.project_availability(
            getattr(ctx, "availability", None), s["id"], s["pos"], s["rank"],
            injury_status=(getattr(ctx, "injuries", None) or {}).get(s["id"]),
            practice_status=(getattr(ctx, "practice", None) or {}).get(s["id"]),
        )
        p_active.append(pa)
    p_active = np.array(p_active)
    active = rng.random((n_sims, len(skill))) < p_active[None, :]

    tgt_shares = _dirichlet_shares(rng, np.array(tgt_base), SHARE_CONCENTRATION["target"], n_sims)
    car_shares = _dirichlet_shares(rng, np.array(car_base), SHARE_CONCENTRATION["carry"], n_sims)
    gl_arr = np.clip(np.array(gl_base), 1e-4, None); gl_arr /= gl_arr.sum()

    # Zero out the inactive, then renormalise so team volume is conserved.
    def _renorm(shares: np.ndarray) -> np.ndarray:
        s = shares * active
        tot = s.sum(axis=1, keepdims=True)
        return np.divide(s, np.where(tot > 0, tot, 1.0))

    tgt_shares = _renorm(tgt_shares)
    car_shares = _renorm(car_shares)

    coverage_t = usage_mod.CHARTED_COVERAGE["target"]
    coverage_c = usage_mod.CHARTED_COVERAGE["carry"]

    for i, s in enumerate(skill):
        hist = usage_hist[usage_hist["player_id"] == s["id"]] if s["id"] else pd.DataFrame()
        catch_rate = pj._shrunk(hist, "catch_rate", "targets", 0.645,
                                pj.PRIOR_STRENGTH["rec_efficiency"])
        adot = pj._shrunk(hist, "adot", "targets", float(scheme.get("adot", 8.0)),
                          pj.PRIOR_STRENGTH["rec_efficiency"])
        ypr = pj._shrunk(hist, "yards_per_rec", "receptions", 11.2,
                         pj.PRIOR_STRENGTH["rec_efficiency"])
        sep = (getattr(ctx, "separation", None) or {}).get(s["id"])
        if sep is not None and np.isfinite(sep):
            catch_rate = float(np.clip(catch_rate + (sep - 2.85) * 0.030, 0.30, 0.90))

        rush_eff = None
        if s["pos"] in ("RB", "FB") and getattr(ctx, "team_blocking", None) is not None:
            rush_eff = bl.project_rushing_efficiency(team, s["id"], ctx.team_blocking,
                                                     ctx.player_elusiveness)
        ypc = float(rush_eff["ypc"]) if rush_eff else pj._shrunk(
            hist, "ypc", "carries", 4.30, pj.PRIOR_STRENGTH["rush_efficiency"])

        expected = sampler.expected_ypr(adot)
        rec_scale = float(np.clip(ypr / expected if expected > 0 else 1.0, 0.82, 1.22)) * adj["pass_yards"]
        rush_scale = float(np.clip(ypc / 4.30, 0.72, 1.35)) * adj["rush_yards"]
        eff_adot = adot
        if env:
            rec_scale *= float(env.get("pass_yards_mult", 1.0))
            rush_scale *= float(env.get("rush_yards_mult", 1.0))
            eff_adot = adot * float(env.get("deep_rate_mult", 1.0))

        targets = rng.poisson(np.maximum(attempts * coverage_t * tgt_shares[:, i], 0))
        carries = rng.poisson(np.maximum(rb_carries * coverage_c * car_shares[:, i], 0))
        receptions = rng.binomial(targets, float(np.clip(catch_rate, 0.05, 0.95)))
        rec_yards = sampler.sample_rec(receptions, adot=eff_adot, scale=rec_scale)
        rush_yards = sampler.sample_rush(carries, scale=rush_scale)

        # Touchdowns are drawn from the team's simulated total, so two players
        # cannot both be handed the same score.
        rz_share = pj._shrunk(hist, "rz_target_share", "targets", tgt_base[i],
                              pj.PRIOR_STRENGTH["td_rate"])
        rec_td_share = 0.60 * rz_share + 0.40 * tgt_base[i]
        rush_td_share = 0.70 * gl_arr[i] + 0.30 * car_base[i]
        rec_td = rng.binomial(team_pass_td, float(np.clip(rec_td_share, 0, 0.95))) * active[:, i]
        rush_td = rng.binomial(team_rush_td, float(np.clip(rush_td_share, 0, 0.95))) * active[:, i]

        out.players[s["name"]] = {
            "targets": targets.astype(float), "receptions": receptions.astype(float),
            "rec_yards": rec_yards, "carries": carries.astype(float),
            "rush_yards": rush_yards, "scrimmage_yards": rec_yards + rush_yards,
            "rec_td": rec_td.astype(float), "rush_td": rush_td.astype(float),
            "total_td": (rec_td + rush_td).astype(float),
        }
        out.meta[s["name"]] = {"position": s["pos"], "team": team,
                               "depth_rank": s["rank"], "p_active": float(p_active[i]),
                               "player_id": s["id"]}
        out.active[s["name"]] = active[:, i].copy()

    # A quarterback's passing line is not an independent event: his yards are
    # his receivers' yards, counted once at the other end of the throw. Summing
    # what was just allocated is both exactly right and the reason the
    # correlations come out at their measured strength rather than near zero.
    team_rec_yards = np.zeros(n_sims)
    team_receptions = np.zeros(n_sims)
    team_targets = np.zeros(n_sims)
    for s_ in skill:
        d = out.players.get(s_["name"])
        if d is None:
            continue
        team_rec_yards += d["rec_yards"]
        team_receptions += d["receptions"]
        team_targets += d["targets"]

    _add_quarterback(out, team, b, ctx, sampler, adj, attempts, qb_carries,
                     team_pass_td, team_rush_td, chart, env, rng, n_sims,
                     team_rec_yards, team_receptions, team_targets)


def _add_quarterback(out: JointGame, team: str, b: dict, ctx, sampler, adj: dict,
                     attempts: np.ndarray, qb_carries: np.ndarray,
                     team_pass_td: np.ndarray, team_rush_td: np.ndarray,
                     chart: pd.DataFrame, env: dict | None, rng, n_sims: int,
                     team_rec_yards: np.ndarray, team_receptions: np.ndarray,
                     team_targets: np.ndarray) -> None:
    """The quarterback's line is derived from his receivers', not drawn apart."""
    scheme = b["scheme"]
    qb = chart[chart["pos_abb"] == "QB"].sort_values("pos_rank")
    if qb.empty:
        return
    row = qb.iloc[0]
    pid = row.get("gsis_id") or None

    h = ctx.qb_profiles[ctx.qb_profiles["player_id"] == pid] if pid is not None else pd.DataFrame()
    cpoe = pj._shrunk_simple(h, "cpoe", 0.0, 300.0)
    comp = float(np.clip(0.655 + (cpoe / 100.0 if np.isfinite(cpoe) else 0.0), 0.55, 0.75))

    adot = float(scheme.get("adot", 7.8))
    scale = adj["pass_yards"]
    if env:
        adot *= float(env.get("deep_rate_mult", 1.0))
        scale *= float(env.get("pass_yards_mult", 1.0))
        comp = float(np.clip(comp + env.get("wind_excess", 0.0) * -0.0016, 0.50, 0.75))

    # Charted receivers do not absorb every target, so scale their totals back
    # up to the full offence before calling them the quarterback's line.
    inflate = 1.0 / usage_mod.CHARTED_COVERAGE["target"]
    att = np.maximum(np.rint(team_targets * inflate), 0)
    completions = np.minimum(np.rint(team_receptions * inflate), att)
    pass_yards = team_rec_yards * inflate

    # Throwaways and spikes are attempts with no intended receiver charted.
    att = att + rng.poisson(np.maximum(attempts * 0.04, 0))
    carries = rng.poisson(np.maximum(qb_carries, 0))
    rush_yards = sampler.sample_rush(carries, scale=adj["rush_yards"])
    qb_rush_share = float(np.clip(float(scheme.get("qb_designed_run_rate", 0.03)) * 6.0, 0.02, 0.35))
    rush_td = rng.binomial(team_rush_td, qb_rush_share)

    pa, _ = av.project_availability(
        getattr(ctx, "availability", None), pid, "QB", 1,
        injury_status=(getattr(ctx, "injuries", None) or {}).get(pid),
        practice_status=(getattr(ctx, "practice", None) or {}).get(pid))
    mask = (rng.random(n_sims) < pa).astype(float)

    out.players[row["player_name"]] = {
        "attempts": att.astype(float) * mask,
        "completions": completions.astype(float) * mask,
        "pass_yards": pass_yards * mask,
        "pass_td": team_pass_td.astype(float) * mask,
        "interceptions": rng.poisson(np.maximum(attempts * 0.023, 0)).astype(float) * mask,
        "carries": carries.astype(float) * mask,
        "rush_yards": rush_yards * mask,
        "rush_td": rush_td.astype(float) * mask,
        "total_td": rush_td.astype(float) * mask,
    }
    out.meta[row["player_name"]] = {"position": "QB", "team": team, "depth_rank": 1,
                                    "p_active": float(pa), "player_id": pid}
    out.active[row["player_name"]] = mask.astype(bool)
