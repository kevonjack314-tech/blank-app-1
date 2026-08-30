"""Player usage: how a team's touches get divided up.

Volume is most of a projection. A receiver's yardage is far more sensitive to
how many targets they draw than to how efficient they are with each one, and
touchdowns are largely a story about who is on the field inside the ten. This
module measures historical usage from play-by-play - including the red-zone and
goal-line splits that box-score data flattens away - and projects it forward
onto the current depth chart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LAST_COMPLETED_SEASON, PRIOR_STRENGTH

# Typical share of a team's targets or carries by depth-chart role. These are
# the priors a player regresses toward when their own sample is thin - a rookie
# WR1 with no NFL history still gets a WR1-shaped projection.
# Share of a team's targets or carries by depth-chart role. Measured from
# 2022-2025 play-by-play by ranking each team-season's players within position,
# so these are what NFL offences actually do rather than round-number guesses.
# A player regresses toward the prior for the role they hold, which is what lets
# a rookie WR1 or a veteran slotted into a new job get a sensible projection.
TARGET_SHARE_PRIOR = {
    ("WR", 1): 0.231, ("WR", 2): 0.150, ("WR", 3): 0.097, ("WR", 4): 0.056,
    ("WR", 5): 0.034, ("WR", 6): 0.026,
    ("TE", 1): 0.143, ("TE", 2): 0.053, ("TE", 3): 0.026,
    ("RB", 1): 0.104, ("RB", 2): 0.048, ("RB", 3): 0.019, ("RB", 4): 0.011,
    ("FB", 1): 0.023,
    ("QB", 1): 0.000,
}
CARRY_SHARE_PRIOR = {
    ("RB", 1): 0.539, ("RB", 2): 0.231, ("RB", 3): 0.087, ("RB", 4): 0.047,
    ("WR", 1): 0.021, ("WR", 2): 0.007, ("WR", 3): 0.003, ("WR", 4): 0.001,
    ("WR", 5): 0.001, ("WR", 6): 0.001,
    ("TE", 1): 0.007, ("TE", 2): 0.001, ("TE", 3): 0.001,
    ("FB", 1): 0.011,
}
# Goal-line work concentrates harder than overall carries: the lead back and
# the quarterback take the bulk of it, which is where rushing touchdowns live.
GOALLINE_SHARE_PRIOR = {
    ("RB", 1): 0.553, ("RB", 2): 0.200, ("RB", 3): 0.057, ("RB", 4): 0.019,
    ("FB", 1): 0.044, ("TE", 1): 0.014, ("TE", 2): 0.001,
    ("WR", 1): 0.013, ("WR", 2): 0.001, ("WR", 3): 0.001,
    ("WR", 4): 0.001, ("WR", 5): 0.001, ("WR", 6): 0.001,
}
# Share of team volume absorbed by players on the projected depth chart. The
# remainder goes to deep reserves and week-to-week call-ups who are not
# projected individually.
# Share of a team's *air yards* by role. Depth-weighted, so a deep threat
# indexes higher here than his raw target share suggests and a checkdown back
# indexes lower.
AIR_YARDS_SHARE_PRIOR = {
    ("WR", 1): 0.270, ("WR", 2): 0.175, ("WR", 3): 0.108, ("WR", 4): 0.060,
    ("WR", 5): 0.036, ("WR", 6): 0.027,
    ("TE", 1): 0.128, ("TE", 2): 0.047, ("TE", 3): 0.022,
    ("RB", 1): 0.045, ("RB", 2): 0.021, ("RB", 3): 0.008, ("RB", 4): 0.005,
    ("FB", 1): 0.008,
}
CHARTED_COVERAGE = {"target": 0.956, "carry": 0.907, "goalline": 0.881,
                    "air_yards": 0.96}

# How much of the target projection comes from the air-yards route rather than
# raw target share. Air-yards share is the single most stable usage metric
# measured (r = 0.727 year over year, against 0.55 for target share), because
# it encodes role and depth together - so it carries the majority of the blend.
AIR_YARDS_BLEND = 0.55


def player_usage(plays: pd.DataFrame) -> pd.DataFrame:
    """Per player-season usage and efficiency, measured off play-by-play."""
    df = plays
    team_totals = (
        df.groupby(["season", "posteam"])
        .agg(
            team_plays=("play_id", "size"),
            team_targets=("receiver_player_id", "count"),
            team_carries=("is_designed_run", "sum"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    rz = df[df["red_zone"]]
    gl = df[df["yardline_100"] <= 5]

    # Games a player actually appeared in, so shares can be put on a per-game
    # footing. A receiver who tore up four games and then got hurt held a
    # starter's role; season totals alone would score him as a reserve.
    appear = pd.concat([
        df[df["receiver_player_id"].notna()][["season", "posteam", "game_id", "receiver_player_id"]]
          .rename(columns={"receiver_player_id": "player_id"}),
        df[df["rusher_player_id"].notna()][["season", "posteam", "game_id", "rusher_player_id"]]
          .rename(columns={"rusher_player_id": "player_id"}),
    ])
    games_played = (
        appear.groupby(["season", "posteam", "player_id"])["game_id"].nunique()
        .rename("games_played").reset_index().rename(columns={"posteam": "team"})
    )
    team_games = (
        df.groupby(["season", "posteam"])["game_id"].nunique()
        .rename("team_games").reset_index().rename(columns={"posteam": "team"})
    )

    rec = (
        df[df["receiver_player_id"].notna()]
        .groupby(["season", "posteam", "receiver_player_id"])
        .agg(
            targets=("play_id", "size"),
            receptions=("complete_pass", "sum"),
            rec_yards=("yards_gained", lambda s: float(s.sum())),
            rec_air=("air_yards", "sum"),
            rec_adot=("air_yards", "mean"),
            rec_yac=("yards_after_catch", "sum"),
            rec_td=("pass_touchdown", "sum"),
        )
        .reset_index()
        .rename(columns={"receiver_player_id": "player_id", "posteam": "team"})
    )
    # Receiving yards must only count completions; sum over all targets would
    # fold in sacks and incompletions charted on the same play row.
    comp = df[(df["receiver_player_id"].notna()) & (df["complete_pass"].fillna(0) > 0)]
    comp_y = (
        comp.groupby(["season", "posteam", "receiver_player_id"])["yards_gained"]
        .sum().rename("rec_yards_true").reset_index()
        .rename(columns={"receiver_player_id": "player_id", "posteam": "team"})
    )
    rec = rec.merge(comp_y, on=["season", "team", "player_id"], how="left")
    rec["rec_yards"] = rec["rec_yards_true"].fillna(0.0)
    rec = rec.drop(columns=["rec_yards_true"])

    rz_rec = (
        rz[rz["receiver_player_id"].notna()]
        .groupby(["season", "posteam", "receiver_player_id"]).size()
        .rename("rz_targets").reset_index()
        .rename(columns={"receiver_player_id": "player_id", "posteam": "team"})
    )

    rush = (
        df[df["is_designed_run"] & df["rusher_player_id"].notna()]
        .groupby(["season", "posteam", "rusher_player_id"])
        .agg(
            carries=("play_id", "size"),
            rush_yards=("yards_gained", "sum"),
            rush_td=("rush_touchdown", "sum"),
        )
        .reset_index()
        .rename(columns={"rusher_player_id": "player_id", "posteam": "team"})
    )
    gl_rush = (
        gl[gl["is_designed_run"] & gl["rusher_player_id"].notna()]
        .groupby(["season", "posteam", "rusher_player_id"]).size()
        .rename("goalline_carries").reset_index()
        .rename(columns={"rusher_player_id": "player_id", "posteam": "team"})
    )

    out = rec.merge(rush, on=["season", "team", "player_id"], how="outer")
    out = out.merge(rz_rec, on=["season", "team", "player_id"], how="left")
    out = out.merge(gl_rush, on=["season", "team", "player_id"], how="left")
    out = out.merge(team_totals, on=["season", "team"], how="left")

    counts = ["targets", "receptions", "rec_yards", "rec_td", "carries",
              "rush_yards", "rush_td", "rz_targets", "goalline_carries", "rec_yac", "rec_air"]
    for c in counts:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    team_air = (
        df[df["receiver_player_id"].notna()]
        .groupby(["season", "posteam"])["air_yards"].sum()
        .rename("team_air_yards").reset_index().rename(columns={"posteam": "team"})
    )
    team_gl = gl[gl["is_designed_run"]].groupby(["season", "posteam"]).size().rename("team_goalline").reset_index().rename(columns={"posteam": "team"})
    team_rz = rz[rz["receiver_player_id"].notna()].groupby(["season", "posteam"]).size().rename("team_rz_targets").reset_index().rename(columns={"posteam": "team"})
    out = out.merge(team_gl, on=["season", "team"], how="left").merge(team_rz, on=["season", "team"], how="left")
    out = out.merge(team_air, on=["season", "team"], how="left")
    out["team_air_yards"] = out["team_air_yards"].fillna(1.0)
    out = out.merge(games_played, on=["season", "team", "player_id"], how="left")
    out = out.merge(team_games, on=["season", "team"], how="left")
    out["games_played"] = out["games_played"].fillna(0)
    out["team_games"] = out["team_games"].fillna(1)
    out["availability"] = (out["games_played"] / out["team_games"].clip(lower=1)).clip(0, 1)

    # Team volume over just the games this player appeared in.
    avail = out["availability"].replace(0, np.nan)
    for col, team_col in (("target_share", "team_targets"), ("carry_share", "team_carries"),
                          ("goalline_share", "team_goalline"), ("rz_target_share", "team_rz_targets"),
                          ("air_yards_share", "team_air_yards")):
        out[f"_{team_col}_avail"] = (out[team_col] * avail).clip(lower=1)
    out["air_yards_share"] = out["rec_air"] / out["_team_air_yards_avail"]
    out["target_share"] = out["targets"] / out["_team_targets_avail"]
    out["carry_share"] = out["carries"] / out["_team_carries_avail"]
    out["goalline_share"] = out["goalline_carries"] / out["_team_goalline_avail"]
    out["rz_target_share"] = out["rz_targets"] / out["_team_rz_targets_avail"]
    for c in [c for c in out.columns if c.startswith("_team_")]:
        del out[c]
    for c in ("target_share", "carry_share", "goalline_share", "rz_target_share",
              "air_yards_share"):
        out[c] = out[c].clip(upper=0.60)
    out["catch_rate"] = out["receptions"] / out["targets"].replace(0, np.nan)
    out["yards_per_target"] = out["rec_yards"] / out["targets"].replace(0, np.nan)
    out["yards_per_rec"] = out["rec_yards"] / out["receptions"].replace(0, np.nan)
    out["ypc"] = out["rush_yards"] / out["carries"].replace(0, np.nan)
    out["adot"] = out["rec_adot"]
    return out


def _recency_weights(seasons: pd.Series, halflife: float = 1.6) -> pd.Series:
    return 0.5 ** ((LAST_COMPLETED_SEASON - seasons) / halflife)


def player_history(usage: pd.DataFrame, player_id: str) -> pd.DataFrame:
    return usage[usage["player_id"] == player_id].sort_values("season")


# Once a season is under way, what a player is doing now outranks what he did
# last year - but not immediately, and not by a fixed amount. Fitted over
# 2022-2025 by predicting each player's week-N share from his season-to-date
# share and his prior-season share, the best blend weights the current season at
# n / (n + K) where n is the touches accumulated so far:
#
#   targets  K = 8   (after 5 targets 0.38, after 30 0.79, after 100 0.93)
#   carries  K = 4   (backfield roles resolve faster than target trees)
#
# Blending beats either source alone. For carries the gap is large - a
# root-mean-square error of 0.145 against 0.189 for prior-season-only, a 23%
# improvement - because a backfield can change hands completely between years.
INSEASON_K = {"target": 8.0, "carry": 4.0, "goalline": 4.0, "air_yards": 8.0}


# A share is team-relative: it describes how one offence divided its work, not
# a portable property of the player. When a player changes teams their past
# share is weak evidence for their new one, so it is discounted and the role
# they have actually been given carries more of the projection.
TEAM_CHANGE_DISCOUNT = 0.45


def project_share(
    usage: pd.DataFrame, player_id: str | None, pos: str, rank: int,
    kind: str = "target", current_team: str | None = None,
    role_pull: float = 0.0, current_season: int | None = None,
) -> tuple[float, float]:
    """Project one player's share of team volume.

    Blends their own recent usage with the prior for the depth-chart role they
    now occupy. Seasons spent elsewhere are discounted, since share reflects a
    specific offence and its other mouths to feed. ``role_pull`` adds extra
    regression toward the role prior for situations the history cannot speak
    to - a receiver paired with a quarterback he has barely played with.

    Returns the share and an evidence weight in [0, 1].
    """
    prior_map = {
        "target": TARGET_SHARE_PRIOR,
        "carry": CARRY_SHARE_PRIOR,
        "goalline": GOALLINE_SHARE_PRIOR,
        "air_yards": AIR_YARDS_SHARE_PRIOR,
    }[kind]
    col = {"target": "target_share", "carry": "carry_share",
           "goalline": "goalline_share", "air_yards": "air_yards_share"}[kind]
    denom_col = {"target": "targets", "carry": "carries",
                 "goalline": "goalline_carries", "air_yards": "targets"}[kind]

    prior = prior_map.get((pos, int(rank)), 0.01)
    if not player_id:
        return prior, 0.0

    h = usage[usage["player_id"] == player_id]
    if h.empty:
        return prior, 0.0

    # In season, split the player's own record into what he has done this year
    # and what he did before, and blend them by how much this year has
    # accumulated. Outside the season this branch is skipped entirely.
    if current_season is not None and (h["season"] == current_season).any():
        return _inseason_share(h, prior, kind, col, denom_col, current_season,
                               current_team, role_pull)

    w = _recency_weights(h["season"])
    # Weight seasons by how much the player actually did, so one injured cameo
    # does not overwrite a full workload season. Shares are already per-game,
    # so this governs confidence rather than the level itself.
    vol = h[denom_col].clip(lower=0)
    w = w * np.sqrt(vol / max(vol.max(), 1.0)).clip(lower=0.25)
    if current_team is not None and "team" in h.columns:
        w = w * np.where(h["team"] == current_team, 1.0, TEAM_CHANGE_DISCOUNT)
    if w.sum() <= 0:
        return prior, 0.0

    observed = float((h[col].fillna(0.0) * w).sum() / w.sum())
    strength = PRIOR_STRENGTH["usage_share"]
    # Effective sample carries the same team-change discount, so a newcomer is
    # projected with less confidence as well as less weight.
    n_eff = float((vol * _recency_weights(h["season"]) *
                   (np.where(h["team"] == current_team, 1.0, TEAM_CHANGE_DISCOUNT)
                    if current_team is not None and "team" in h.columns else 1.0)).sum())
    shrunk = (observed * n_eff + prior * strength) / (n_eff + strength)
    if role_pull > 0:
        shrunk = (1 - role_pull) * shrunk + role_pull * prior
    return float(shrunk), float(min(n_eff / strength, 1.0) * (1 - role_pull))


def normalize_shares(shares: dict[str, float], total: float = 1.0) -> dict[str, float]:
    """Rescale shares so a team's projected volume adds up."""
    s = sum(v for v in shares.values() if np.isfinite(v))
    if s <= 0:
        return shares
    return {k: (v / s) * total for k, v in shares.items()}


def _inseason_share(h: pd.DataFrame, prior: float, kind: str, col: str,
                    denom_col: str, current_season: int,
                    current_team: str | None, role_pull: float) -> tuple[float, float]:
    """Blend season-to-date usage against prior seasons and the role prior."""
    cur = h[h["season"] == current_season]
    past = h[h["season"] < current_season]

    n_now = float(cur[denom_col].clip(lower=0).sum())
    now = float((cur[col].fillna(0.0) * cur[denom_col].clip(lower=0)).sum() /
                max(n_now, 1e-9)) if n_now > 0 else np.nan

    # The prior-season estimate is the same shrunk figure used out of season.
    if past.empty:
        before = prior
        n_before = 0.0
    else:
        w = _recency_weights(past["season"])
        vol = past[denom_col].clip(lower=0)
        w = w * np.sqrt(vol / max(vol.max(), 1.0)).clip(lower=0.25)
        if current_team is not None and "team" in past.columns:
            w = w * np.where(past["team"] == current_team, 1.0, TEAM_CHANGE_DISCOUNT)
        before = float((past[col].fillna(0.0) * w).sum() / w.sum()) if w.sum() > 0 else prior
        n_before = float((vol * _recency_weights(past["season"])).sum())
        strength = PRIOR_STRENGTH["usage_share"]
        before = (before * n_before + prior * strength) / (n_before + strength)

    if not np.isfinite(now) or n_now <= 0:
        blended, evidence = before, min(n_before / PRIOR_STRENGTH["usage_share"], 1.0)
    else:
        K = INSEASON_K.get(kind, 8.0)
        w_now = n_now / (n_now + K)
        blended = w_now * now + (1.0 - w_now) * before
        evidence = float(min((n_now + n_before) / PRIOR_STRENGTH["usage_share"], 1.0))

    if role_pull > 0:
        blended = (1 - role_pull) * blended + role_pull * prior
        evidence *= (1 - role_pull)
    return float(blended), float(evidence)


def qb_continuity(plays: pd.DataFrame, qb_id: str | None, receiver_id: str | None) -> tuple[int, float]:
    """Shared history between a quarterback and a receiver.

    Returns the number of targets they have connected on and a 0-1 confidence
    score. A new pairing is not necessarily worse - it is simply unmeasured,
    and the projection should say so rather than borrow another passer's
    distribution of the ball wholesale.
    """
    if not qb_id or not receiver_id:
        return 0, 0.0
    n = int(((plays["passer_player_id"] == qb_id) &
             (plays["receiver_player_id"] == receiver_id)).sum())
    # Roughly a season of shared work before the pairing is treated as known.
    return n, float(min(n / 60.0, 1.0))


def role_pull_for(usage: pd.DataFrame, player_id: str | None, pos: str, rank: int,
                  current_team: str | None, kind: str = "target") -> float:
    """How hard to regress a player toward the prior for their assigned role.

    Two things make past usage a poor guide. Changing teams means the share was
    earned in a different offence with different competition for touches. Being
    slotted at a different depth than the one their history reflects means the
    staff has assigned them a new job - a back who carried the ball on 55% of
    his old team's runs and is now listed second is being told something, and
    the depth chart is the most direct statement of intent available before
    Week 1.
    """
    if not player_id:
        return 0.0
    h = usage[usage["player_id"] == player_id]
    if h.empty:
        return 0.0

    pull = 0.0
    if current_team is not None and "team" in h.columns:
        recent = h.sort_values("season").iloc[-1]
        if recent.get("team") != current_team:
            pull = max(pull, 0.45)

    # Compare the role their usage implies against the one they now hold.
    prior_map = {"target": TARGET_SHARE_PRIOR, "carry": CARRY_SHARE_PRIOR,
                 "goalline": GOALLINE_SHARE_PRIOR,
                 "air_yards": AIR_YARDS_SHARE_PRIOR}[kind]
    col = {"target": "target_share", "carry": "carry_share",
           "goalline": "goalline_share", "air_yards": "air_yards_share"}[kind]
    ranks = sorted(r for (p_, r) in prior_map if p_ == pos)
    if not ranks:
        return pull
    w = _recency_weights(h["season"])
    if w.sum() <= 0:
        return pull
    observed = float((h[col].fillna(0.0) * w).sum() / w.sum())
    implied = min(ranks, key=lambda r: abs(prior_map.get((pos, r), 0.0) - observed))
    mismatch = abs(int(implied) - int(rank))
    if mismatch >= 1:
        pull = max(pull, min(0.30 * mismatch, 0.60))
    return float(min(pull, 0.65))


def targets_from_air_yards(air_yards_share: float, team_air_yards: float,
                           adot: float) -> float:
    """Recover a target count from a projected air-yards share.

    Air yards are targets multiplied by depth, so dividing a receiver's
    projected share of team air yards by his own average depth of target gives
    an independent estimate of his target count - one anchored to the most
    stable role signal in the data rather than to target share alone.
    """
    if not np.isfinite(adot) or adot <= 1.0:
        return np.nan
    return float(air_yards_share * team_air_yards / adot)


# A rookie has no NFL sample, so the role prior for his depth-chart slot is all
# the model has. Draft capital is the one extra signal available, and it is a
# strong one: teams give early picks the ball. These multipliers scale the role
# prior by where a player was taken, measured against the same slot's league
# average usage.
DRAFT_CAPITAL_MULTIPLIER = [
    (1, 10, 1.28),      # top-ten picks are handed a role immediately
    (11, 32, 1.15),
    (33, 64, 1.05),
    (65, 105, 0.96),
    (106, 200, 0.88),
    (201, 262, 0.82),
]
UNDRAFTED_MULTIPLIER = 0.78


def draft_multiplier(draft: pd.DataFrame, player_id: str | None,
                     rookie_season: int | None = None) -> float:
    """Usage multiplier implied by where a player was drafted.

    Applies only while a player has no meaningful NFL sample of his own. Once
    he has played, what he actually did outranks where he was picked.
    """
    if not player_id or draft is None or draft.empty or "gsis_id" not in draft.columns:
        return 1.0
    row = draft[draft["gsis_id"] == player_id]
    if row.empty:
        return UNDRAFTED_MULTIPLIER
    pick = row["pick"].iloc[0]
    if pd.isna(pick):
        return UNDRAFTED_MULTIPLIER
    pick = int(pick)
    for lo, hi, mult in DRAFT_CAPITAL_MULTIPLIER:
        if lo <= pick <= hi:
            return mult
    return UNDRAFTED_MULTIPLIER
