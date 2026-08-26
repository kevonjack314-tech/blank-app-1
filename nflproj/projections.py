"""Yardage and touchdown projections with full distributions.

The chain runs: market line -> team points -> team plays and touchdowns ->
each player's share of that volume -> yards and scores. Everything is simulated
rather than reported as a single number, because the useful questions are
distributional - the chance a back clears 70 yards, the chance a receiver finds
the end zone - and a point estimate cannot answer them.

Per-touch yardage is bootstrapped from real play-by-play outcomes rather than
drawn from a fitted normal. Rushing and receiving gains are heavily skewed: most
carries gain three yards and a few go eighty, and that tail is exactly what
drives the over on a yardage prop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import LEAGUE_PLAYS_PER_GAME, POINTS_PER_TD, PRIOR_STRENGTH
from . import usage as usage_mod

# Fitted on 2022-2025 team-games: offensive touchdowns given points scored.
TD_FROM_POINTS = (-0.4628, 0.1255)
LEAGUE_PASS_TD_SHARE = 0.612

# Weekly volume is noisier than a season average implies - game script, injury,
# blowouts. Dispersion is expressed as variance = mean * factor (negative
# binomial style overdispersion), calibrated from weekly usage spread.
VOLUME_DISPERSION = {"target": 1.85, "carry": 2.10, "attempt": 1.35}


@dataclass
class GameContext:
    """One team's situation in one game."""
    team: str
    opponent: str | None = None
    is_home: bool = True
    total_line: float | None = None
    spread_line: float | None = None   # positive = this team favoured
    week: int | None = None
    neutral: bool = False              # ignore the market, project a generic game

    @property
    def implied_points(self) -> float:
        if self.neutral or self.total_line is None or self.spread_line is None:
            return 22.5
        return float(self.total_line) / 2.0 + float(self.spread_line) / 2.0

    @property
    def implied_against(self) -> float:
        if self.neutral or self.total_line is None or self.spread_line is None:
            return 22.5
        return float(self.total_line) / 2.0 - float(self.spread_line) / 2.0


def expected_offensive_tds(points: float) -> float:
    a, b = TD_FROM_POINTS
    return max(a + b * float(points), 0.35)


def team_volume(scheme: pd.Series, ctx: GameContext, opp_scheme: pd.Series | None = None,
                env: dict | None = None) -> dict:
    """Project a team's play volume and scoring for one game.

    ``env`` carries the scoring environment - wind, roof, divisional
    familiarity. Wind is the weather variable that actually matters: measured
    over 7,901 charted plays in 15+ mph conditions, offences throw less, throw
    shorter and complete fewer. Cold on its own barely moves efficiency.
    """
    plays = float(scheme.get("plays_per_game", LEAGUE_PLAYS_PER_GAME))
    if not np.isfinite(plays):
        plays = LEAGUE_PLAYS_PER_GAME

    # Tempo moves volume: a faster team runs more plays, and the opponent's
    # pace pulls the game total toward theirs.
    sec = float(scheme.get("sec_per_play", np.nan))
    if opp_scheme is not None and np.isfinite(sec):
        opp_sec = float(opp_scheme.get("sec_per_play", sec))
        if np.isfinite(opp_sec):
            blended = 0.5 * (sec + opp_sec)
            plays *= np.clip(sec / max(blended, 1e-6), 0.90, 1.10)

    pass_rate = float(scheme.get("early_down_pass_rate", 0.55))
    # Game script: a favoured team throws less, a trailing team throws more.
    # Roughly two points of spread is worth a point of pass rate.
    if not ctx.neutral and ctx.spread_line is not None:
        pass_rate -= 0.010 * float(ctx.spread_line)
    if env:
        pass_rate += float(env.get("pass_rate_delta", 0.0))
    pass_rate = float(np.clip(pass_rate, 0.34, 0.74))

    dropbacks = plays * pass_rate
    sack_rate = float(scheme.get("sack_rate_allowed", 0.065))
    scramble_rate = float(scheme.get("scramble_rate", 0.07))
    sacks = dropbacks * sack_rate
    scrambles = dropbacks * scramble_rate
    attempts = max(dropbacks - sacks - scrambles, 1.0)

    designed_runs = plays * (1.0 - pass_rate)
    qb_runs = plays * float(scheme.get("qb_designed_run_rate", 0.03))
    rb_carries = max(designed_runs - qb_runs, 1.0)

    points = ctx.implied_points
    if env:
        # The environment adjustment is a whole-game effect; this team owns half.
        points += float(env.get("total_delta", 0.0)) / 2.0
    off_td = expected_offensive_tds(points)

    # Split scoring between the pass and the run using the team's own red-zone
    # and goal-to-go tendencies rather than the league average.
    rz_pass = float(scheme.get("rz_pass_rate", 0.52))
    g2g_run = float(scheme.get("g2g_run_rate", 0.55))
    tilt = 0.5 * rz_pass + 0.5 * (1.0 - g2g_run)
    pass_td_share = float(np.clip(LEAGUE_PASS_TD_SHARE + (tilt - 0.5) * 0.80, 0.40, 0.80))

    return {
        "plays": plays,
        "pass_rate": pass_rate,
        "dropbacks": dropbacks,
        "attempts": attempts,
        "sacks": sacks,
        "scrambles": scrambles,
        "designed_runs": designed_runs,
        "qb_runs": qb_runs,
        "rb_carries": rb_carries,
        "team_targets": attempts,
        "implied_points": points,
        "expected_off_td": off_td,
        "pass_td": off_td * pass_td_share,
        "rush_td": off_td * (1.0 - pass_td_share),
    }


def defense_adjustment(opp_def: pd.Series | None, league_def: pd.Series | None) -> dict:
    """Multipliers describing how much a defence suppresses yards and points.

    Expressed relative to league average, and deliberately damped: defensive
    performance is less stable year to year than offensive usage, so a defence
    that was a standard deviation above average does not get full credit.
    """
    out = {"pass_yards": 1.0, "rush_yards": 1.0, "points": 1.0}
    if opp_def is None or league_def is None:
        return out
    damp = 0.55
    for key, col, scale in (
        ("pass_yards", "ypa_allowed", 1.0),
        ("rush_yards", "ypc_allowed", 1.0),
        ("points", "points_per_drive_allowed", 1.0),
    ):
        v, lg = opp_def.get(col, np.nan), league_def.get(col, np.nan)
        if np.isfinite(v) and np.isfinite(lg) and lg != 0:
            ratio = float(v) / float(lg)
            out[key] = float(np.clip(1.0 + damp * (ratio - 1.0) * scale, 0.80, 1.22))
    return out


# ---------------------------------------------------------------------------
# Empirical per-touch yardage
# ---------------------------------------------------------------------------

class TouchSampler:
    """Bootstrap pools of real per-play gains, split by position and role.

    Keeping the raw outcomes preserves the skew that matters: a running back's
    median carry and their 95th percentile carry are worlds apart, and a model
    built on means alone will systematically understate the long tail.
    """

    def __init__(self, plays: pd.DataFrame, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng(20260826)
        self.pools: dict[tuple, np.ndarray] = {}
        self._build(plays)

    def _build(self, plays: pd.DataFrame) -> None:
        runs = plays[plays["is_designed_run"] & plays["yards_gained"].notna()]
        self.pools[("rush", "ALL")] = runs["yards_gained"].to_numpy(dtype=float)

        rec = plays[plays["receiver_player_id"].notna()]
        comp = rec[rec["complete_pass"].fillna(0) > 0]
        self.pools[("rec", "ALL")] = comp["yards_gained"].to_numpy(dtype=float)

        # Depth of target changes the shape of a reception, so keep separate
        # pools for short, intermediate and deep work.
        for label, lo, hi in (("short", -99, 5), ("mid", 5, 15), ("deep", 15, 99)):
            m = comp["air_yards"].between(lo, hi, inclusive="left")
            arr = comp.loc[m, "yards_gained"].to_numpy(dtype=float)
            if len(arr) > 200:
                self.pools[("rec", label)] = arr

    def sample_rush(self, n: np.ndarray, scale: float = 1.0) -> np.ndarray:
        return self._sample(("rush", "ALL"), n, scale)

    def depth_mix(self, adot: float) -> dict[str, float]:
        """Share of a receiver's targets falling short, intermediate and deep.

        Fitted on 2022-2025: a player's average depth of target predicts his
        depth *mix* almost perfectly (r = 0.94-0.96), but even a 14-yard aDOT
        receiver runs only about a third of his routes deep. Treating a high
        aDOT as if every catch were a deep ball inflates yardage badly.
        """
        if not np.isfinite(adot):
            adot = 8.0
        raw = {
            "short": 0.8906 - 0.0473 * adot,
            "mid": 0.1139 + 0.0231 * adot,
            "deep": -0.0045 + 0.0242 * adot,
        }
        raw = {k: max(v, 0.01) for k, v in raw.items()}
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    def expected_ypr(self, adot: float) -> float:
        """Mean yards per reception implied by a receiver's depth mix."""
        mix = self.depth_mix(adot)
        num = 0.0
        for band, w in mix.items():
            pool = self.pools.get(("rec", band))
            if pool is None or not len(pool):
                pool = self.pools[("rec", "ALL")]
            num += w * float(pool.mean())
        return num

    def sample_rec(self, n: np.ndarray, adot: float = 8.0, scale: float = 1.0) -> np.ndarray:
        """Sample receiving yards, drawing each catch from the right depth band."""
        n = np.asarray(n, dtype=int)
        out = np.zeros(len(n), dtype=float)
        if n.sum() == 0:
            return out
        mix = self.depth_mix(adot)
        bands = [b for b in ("short", "mid", "deep") if ("rec", b) in self.pools]
        if not bands:
            return self._sample(("rec", "ALL"), n, scale)
        w = np.array([mix[b] for b in bands], dtype=float)
        w = w / w.sum()
        # Split each player's catches across depth bands, then draw from each.
        counts = self.rng.multinomial(n, w) if n.ndim == 0 else np.stack(
            [self.rng.multinomial(int(k), w) for k in n]
        )
        for i, b in enumerate(bands):
            out += self._sample(("rec", b), counts[:, i], scale)
        return out

    def _sample(self, key: tuple, n: np.ndarray, scale: float) -> np.ndarray:
        pool = self.pools.get(key)
        if pool is None or len(pool) == 0:
            return np.zeros(len(n))
        n = np.asarray(n, dtype=int)
        total = int(n.sum())
        if total == 0:
            return np.zeros(len(n), dtype=float)
        draws = self.rng.choice(pool, size=total, replace=True) * scale
        # Segment sums via one cumulative pass rather than a Python loop over
        # simulations - this runs tens of thousands of times per projection.
        idx = np.concatenate([[0], np.cumsum(n)])
        csum = np.concatenate([[0.0], np.cumsum(draws)])
        return csum[idx[1:]] - csum[idx[:-1]]


def _nb_sample(rng: np.random.Generator, mean: np.ndarray | float, dispersion: float,
               size: int) -> np.ndarray:
    """Negative-binomial counts with variance = mean * dispersion."""
    mean = np.asarray(mean, dtype=float)
    # A NaN mean means an upstream input was missing; fail loudly rather than
    # emitting a silently zeroed distribution.
    if not np.all(np.isfinite(mean)):
        raise ValueError(f"non-finite volume mean passed to sampler: {mean}")
    mean = np.maximum(mean, 1e-6)
    if dispersion <= 1.0:
        return rng.poisson(mean, size=size)
    r = mean / (dispersion - 1.0)
    p = r / (r + mean)
    return rng.negative_binomial(np.maximum(r, 1e-6), np.clip(p, 1e-6, 1 - 1e-6), size=size)


# ---------------------------------------------------------------------------
# Player projections
# ---------------------------------------------------------------------------

@dataclass
class PlayerProjection:
    """Simulated outcome distribution for one player in one game."""
    player_id: str | None
    name: str
    team: str
    position: str
    depth_rank: int
    inputs: dict = field(default_factory=dict)
    samples: dict = field(default_factory=dict)          # unconditional: includes weeks out
    conditional: dict = field(default_factory=dict)      # as-if active
    p_active: float = 1.0

    def apply_availability(self, p_active: float, rng: np.random.Generator) -> None:
        """Split the projection into an as-if-active view and an expected view.

        Everything simulated up to this point assumes the player suits up. A
        Bernoulli draw per simulation zeroes out the games he misses, which puts
        the right point mass at zero instead of quietly shrinking every number.
        """
        self.p_active = float(np.clip(p_active, 0.0, 1.0))
        self.conditional = {k: v.copy() for k, v in self.samples.items()}
        if self.p_active >= 1.0:
            return
        n = len(next(iter(self.samples.values()), np.zeros(1)))
        mask = (rng.random(n) < self.p_active).astype(float)
        self.samples = {k: v * mask for k, v in self.samples.items()}

    def summary(self) -> dict:
        out = {
            "player": self.name, "team": self.team, "pos": self.position,
            "rank": self.depth_rank,
        }
        for k, v in self.samples.items():
            if v is None or len(v) == 0:
                continue
            out[f"{k}_mean"] = float(np.mean(v))
            out[f"{k}_median"] = float(np.median(v))
            out[f"{k}_p10"] = float(np.percentile(v, 10))
            out[f"{k}_p90"] = float(np.percentile(v, 90))
        td = self.samples.get("total_td")
        if td is not None and len(td):
            out["anytime_td_pct"] = float((td >= 1).mean() * 100.0)
            out["two_plus_td_pct"] = float((td >= 2).mean() * 100.0)
        out["p_active"] = self.p_active * 100.0
        for k, v in self.conditional.items():
            if v is not None and len(v):
                out[f"{k}_if_active"] = float(np.mean(v))
        ctd = self.conditional.get("total_td")
        if ctd is not None and len(ctd):
            out["anytime_td_pct_if_active"] = float((ctd >= 1).mean() * 100.0)
        out.update({f"in_{k}": v for k, v in self.inputs.items()})
        return out

    def prob_over(self, stat: str, line: float, conditional: bool = False) -> float | None:
        """Chance of clearing a yardage line, the way a book would quote it.

        Defaults to the unconditional view, which is what a bet actually settles
        on - a player who does not dress does not clear the line.
        """
        pool = self.conditional if conditional else self.samples
        s = pool.get(stat) if pool else None
        if s is None or len(s) == 0:
            return None
        return float((s > line).mean() * 100.0)


def project_skill_player(
    *, player_id, name, team, position, depth_rank,
    usage_hist: pd.DataFrame, volume: dict, sampler: TouchSampler,
    scheme: pd.Series, def_adj: dict, n_sims: int = 20000,
    rng: np.random.Generator | None = None,
    current_team: str | None = None, qb_shared_targets: int = 0,
    qb_continuity: float = 1.0, env: dict | None = None,
) -> PlayerProjection:
    """Simulate a non-quarterback's receiving and rushing line for one game."""
    rng = rng or np.random.default_rng()

    pulls = {
        k: usage_mod.role_pull_for(usage_hist, player_id, position, depth_rank, current_team, k)
        for k in ("target", "carry", "goalline")
    }
    # An untested quarterback pairing is not evidence of a smaller role, but it
    # is a reason to trust the old target share less and the role prior more.
    if position in ("WR", "TE", "RB") and qb_continuity < 1.0:
        pulls["target"] = min(pulls["target"] + 0.22 * (1.0 - qb_continuity), 0.70)

    tgt_share, tgt_ev = usage_mod.project_share(
        usage_hist, player_id, position, depth_rank, "target",
        current_team=current_team, role_pull=pulls["target"])
    car_share, car_ev = usage_mod.project_share(
        usage_hist, player_id, position, depth_rank, "carry",
        current_team=current_team, role_pull=pulls["carry"])
    gl_share, _ = usage_mod.project_share(
        usage_hist, player_id, position, depth_rank, "goalline",
        current_team=current_team, role_pull=pulls["goalline"])

    exp_targets = volume["team_targets"] * tgt_share
    exp_carries = volume["rb_carries"] * car_share

    hist = usage_hist[usage_hist["player_id"] == player_id] if player_id else pd.DataFrame()
    catch_rate = _shrunk(hist, "catch_rate", "targets", 0.645, PRIOR_STRENGTH["rec_efficiency"])
    adot = _shrunk(hist, "adot", "targets", float(scheme.get("adot", 8.0)), PRIOR_STRENGTH["rec_efficiency"])
    ypc_player = _shrunk(hist, "ypc", "carries", 4.30, PRIOR_STRENGTH["rush_efficiency"])
    ypr_player = _shrunk(hist, "yards_per_rec", "receptions", 11.2, PRIOR_STRENGTH["rec_efficiency"])

    # Scale the bootstrap pools toward this player's own efficiency. For
    # receiving this is a residual only: the depth mix already accounts for how
    # far downfield he works, so the scale captures what he adds beyond that -
    # yards after the catch, contested-ball ability - and is clipped tightly to
    # avoid counting depth twice.
    rush_scale = float(np.clip(ypc_player / 4.30, 0.72, 1.35)) * def_adj["rush_yards"]
    expected = sampler.expected_ypr(adot)
    residual = ypr_player / expected if expected > 0 else 1.0
    rec_scale = float(np.clip(residual, 0.82, 1.22)) * def_adj["pass_yards"]
    if env:
        rec_scale *= float(env.get("pass_yards_mult", 1.0))
        rush_scale *= float(env.get("rush_yards_mult", 1.0))
        # Wind shortens the route tree as well as reducing yardage.
        adot *= float(env.get("deep_rate_mult", 1.0))

    targets = _nb_sample(rng, exp_targets, VOLUME_DISPERSION["target"], n_sims)
    carries = _nb_sample(rng, exp_carries, VOLUME_DISPERSION["carry"], n_sims)
    receptions = rng.binomial(targets, np.clip(catch_rate, 0.05, 0.95))

    rec_yards = sampler.sample_rec(receptions, adot=adot, scale=rec_scale)
    rush_yards = sampler.sample_rush(carries, scale=rush_scale)

    # Touchdowns. Scoring share is not the same as volume share: goal-line work
    # and red-zone targets carry most of the signal, so weight them heavily.
    rz_share = _shrunk(hist, "rz_target_share", "targets", tgt_share, PRIOR_STRENGTH["td_rate"])
    rec_td_share = 0.60 * rz_share + 0.40 * tgt_share
    rush_td_share = 0.70 * gl_share + 0.30 * car_share

    lam_rec_td = volume["pass_td"] * rec_td_share * def_adj["points"]
    lam_rush_td = volume["rush_td"] * rush_td_share * def_adj["points"]
    rec_td = rng.poisson(max(lam_rec_td, 0.0), n_sims)
    rush_td = rng.poisson(max(lam_rush_td, 0.0), n_sims)

    return PlayerProjection(
        player_id=player_id, name=name, team=team, position=position, depth_rank=depth_rank,
        inputs={
            "target_share": tgt_share, "carry_share": car_share, "goalline_share": gl_share,
            "exp_targets": exp_targets, "exp_carries": exp_carries,
            "catch_rate": catch_rate, "adot": adot, "ypc": ypc_player, "ypr": ypr_player,
            "usage_evidence": max(tgt_ev, car_ev),
            "qb_shared_targets": qb_shared_targets,
            "qb_continuity": qb_continuity,
            "role_pull": pulls["target"],
        },
        samples={
            "targets": targets, "receptions": receptions, "rec_yards": rec_yards,
            "carries": carries, "rush_yards": rush_yards,
            "scrimmage_yards": rec_yards + rush_yards,
            "rec_td": rec_td, "rush_td": rush_td, "total_td": rec_td + rush_td,
        },
    )


def project_quarterback(
    *, player_id, name, team, usage_hist: pd.DataFrame, qb_hist: pd.DataFrame,
    volume: dict, sampler: TouchSampler, scheme: pd.Series, def_adj: dict,
    n_sims: int = 20000, rng: np.random.Generator | None = None,
    env: dict | None = None,
) -> PlayerProjection:
    """Simulate a quarterback's passing line plus their own rushing."""
    rng = rng or np.random.default_rng()

    h = qb_hist[qb_hist["player_id"] == player_id] if player_id else pd.DataFrame()
    comp_pct = _shrunk_simple(h, "cpoe", 0.0, 300.0)
    base_comp = 0.655 + (comp_pct / 100.0 if np.isfinite(comp_pct) else 0.0)
    base_comp = float(np.clip(base_comp, 0.55, 0.75))

    attempts = _nb_sample(rng, volume["attempts"], VOLUME_DISPERSION["attempt"], n_sims)
    completions = rng.binomial(attempts, base_comp)

    adot = float(scheme.get("adot", 7.8))
    pass_scale = def_adj["pass_yards"]
    if env:
        adot *= float(env.get("deep_rate_mult", 1.0))
        pass_scale *= float(env.get("pass_yards_mult", 1.0))
        base_comp = float(np.clip(base_comp + env.get("wind_excess", 0.0) * -0.0016, 0.50, 0.75))
    pass_yards = sampler.sample_rec(completions, adot=adot, scale=pass_scale)

    # Quarterback rushing: designed calls plus scrambles, both already projected
    # at team level, so the QB simply absorbs them.
    qb_carries = _nb_sample(rng, volume["qb_runs"] + volume["scrambles"],
                            VOLUME_DISPERSION["carry"], n_sims)
    qb_rush_yards = sampler.sample_rush(qb_carries, scale=def_adj["rush_yards"])

    pass_td = rng.poisson(max(volume["pass_td"] * def_adj["points"], 0.0), n_sims)
    # Quarterbacks take a real slice of goal-line carries in mobile-QB offences.
    qb_rush_td_share = float(np.clip(scheme.get("qb_designed_run_rate", 0.03) * 6.0, 0.02, 0.35))
    qb_rush_td = rng.poisson(max(volume["rush_td"] * qb_rush_td_share, 0.0), n_sims)

    ints = rng.poisson(max(volume["attempts"] * 0.023, 0.0), n_sims)

    return PlayerProjection(
        player_id=player_id, name=name, team=team, position="QB", depth_rank=1,
        inputs={
            "exp_attempts": volume["attempts"], "completion_pct": base_comp,
            "adot": adot, "exp_qb_carries": volume["qb_runs"] + volume["scrambles"],
            "pass_td_lambda": volume["pass_td"],
        },
        samples={
            "attempts": attempts, "completions": completions, "pass_yards": pass_yards,
            "carries": qb_carries, "rush_yards": qb_rush_yards,
            "pass_td": pass_td, "rush_td": qb_rush_td, "interceptions": ints,
            "total_td": qb_rush_td,   # anytime-TD convention: QB rushing scores only
        },
    )


def _shrunk(hist: pd.DataFrame, col: str, weight_col: str, prior: float, strength: float) -> float:
    """Recency-weighted player rate, regressed toward a prior."""
    if hist is None or hist.empty or col not in hist.columns:
        return float(prior)
    w = 0.5 ** ((hist["season"].max() - hist["season"]) / 1.6)
    v, n = hist[col].astype(float), hist[weight_col].astype(float) * w
    m = v.notna() & (n > 0)
    if not m.any():
        return float(prior)
    obs = float((v[m] * n[m]).sum() / n[m].sum())
    n_eff = float(n[m].sum())
    return float((obs * n_eff + prior * strength) / (n_eff + strength))


def _shrunk_simple(hist: pd.DataFrame, col: str, prior: float, strength: float) -> float:
    if hist is None or hist.empty or col not in hist.columns:
        return float(prior)
    v = hist[col].astype(float).dropna()
    if v.empty:
        return float(prior)
    n = float(hist.get("dropbacks", pd.Series([100] * len(hist))).sum())
    return float((v.mean() * n + prior * strength) / (n + strength))
