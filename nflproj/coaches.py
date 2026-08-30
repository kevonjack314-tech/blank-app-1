"""Coaching registry and the scheme-transfer model.

A staff change breaks the usual assumption that last season predicts this one.
This module answers a narrower question: when a coordinator moves, how much of
their old scheme comes with them, and how much bends to the new roster?

The blend has three sources:

  * the play caller's measured fingerprint at their previous stop,
  * the team's own fingerprint last season (continuity of players and holdovers),
  * the league mean (regression for anything thinly sampled).

Weights depend on who calls plays, how much NFL evidence that coach has, and how
well corroborated the hire is. On top of the blend sit personnel constraints -
a scheme bends around the quarterback it inherits, which is why a run-first
coach with a pocket passer does not simply reproduce their old run rate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import DATA, LAST_COMPLETED_SEASON, TEAMS, normalize_team
from .schemes import (DEFENSE_IDENTITY, DEFENSE_QUALITY_PERSISTENCE,
                      OFFENSE_IDENTITY, OFFENSE_QUALITY_PERSISTENCE)

log = logging.getLogger(__name__)

REGISTRY_PATH = DATA / "coaching_2026.yaml"

# How far a projection is allowed to travel from team continuity, by how well
# the staff is corroborated. A low-confidence hire barely moves the baseline.
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.75, "low": 0.45}

# A head coach who calls plays imposes more of their scheme than a coordinator
# working under someone else's structure.
ROLE_WEIGHT = {"HC": 1.0, "OC": 0.9, "DC": 0.9}

# A prior stop as coordinator is a stronger signature than one as a position
# coach, and a lineage proxy is weaker still.
PRIOR_ROLE_WEIGHT = {
    "HC": 1.0, "OC": 1.0, "DC": 1.0,
    "passing game coordinator": 0.6, "run game coordinator": 0.6,
    "position coach": 0.35,
}
LINEAGE_WEIGHT = 0.45

# Seasons decay: a coach's most recent stop describes them better than one four
# years ago.
RECENCY_HALFLIFE = 2.0


@dataclass
class Staff:
    """One team's 2026 staff and how confident we are in it."""
    team: str
    play_caller: str = "HC"
    def_play_caller: str = "DC"
    hc: dict = field(default_factory=dict)
    oc: dict = field(default_factory=dict)
    dc: dict = field(default_factory=dict)
    confidence: str = "high"
    source: str = ""
    notes: str = ""
    continuity: bool = True

    def offensive_leader(self) -> dict:
        return self.hc if self.play_caller == "HC" else (self.oc or self.hc)

    def defensive_leader(self) -> dict:
        return self.hc if self.def_play_caller == "HC" else (self.dc or self.hc)

    @property
    def offense_is_new(self) -> bool:
        return bool(self.offensive_leader().get("new"))

    @property
    def defense_is_new(self) -> bool:
        return bool(self.defensive_leader().get("new"))


def load_registry(path: Path | None = None) -> dict[str, Staff]:
    """Read the YAML registry; teams absent from it are treated as continuity."""
    path = path or REGISTRY_PATH
    staffs: dict[str, Staff] = {}
    raw = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
    entries = raw.get("teams", {}) or {}

    for team in TEAMS:
        e = entries.get(team)
        if not e:
            staffs[team] = Staff(team=team, continuity=True, confidence="high",
                                 notes="No staff change on record; treated as continuity.")
            continue
        staffs[team] = Staff(
            team=team,
            play_caller=e.get("play_caller", "HC"),
            def_play_caller=e.get("def_play_caller", "DC"),
            hc=e.get("hc") or {},
            oc=e.get("oc") or {},
            dc=e.get("dc") or {},
            confidence=e.get("confidence", "medium"),
            source=e.get("source", ""),
            notes=(e.get("notes") or "").strip(),
            continuity=False,
        )
    return staffs


def _fp_row(fp: pd.DataFrame, team: str, season: int, side: str) -> pd.Series | None:
    m = fp[(fp["team"] == normalize_team(team)) & (fp["season"] == season) & (fp["side"] == side)]
    return m.iloc[0] if len(m) else None


def coach_prior_fingerprint(
    coach: dict, fp: pd.DataFrame, side: str, traits: list[str],
    anchor_season: int = LAST_COMPLETED_SEASON,
) -> tuple[pd.Series | None, float, list[str]]:
    """Weighted average of a coach's fingerprints across their previous stops.

    Returns the fingerprint, a 0-1 evidence score reflecting how much NFL
    sample stands behind it, and a human-readable description of the sources.
    """
    stops, weights, labels = [], [], []

    for prior in coach.get("prior") or []:
        team = normalize_team(prior.get("team"))
        role_w = PRIOR_ROLE_WEIGHT.get(str(prior.get("role", "")).strip(), 0.7)
        for season in prior.get("seasons") or []:
            row = _fp_row(fp, team, season, side)
            if row is None:
                continue
            if season > anchor_season:
                continue          # never let a projection see the future
            recency = 0.5 ** ((anchor_season - season) / RECENCY_HALFLIFE)
            stops.append(row[traits].astype(float))
            weights.append(role_w * recency)
            labels.append(f"{team} {season} ({prior.get('role', '?')})")

    lineage_used = False
    if not stops and coach.get("lineage"):
        lin = coach["lineage"]
        team = normalize_team(lin.get("team"))
        for season in lin.get("seasons") or []:
            row = _fp_row(fp, team, season, side)
            if row is None:
                continue
            if season > anchor_season:
                continue
            recency = 0.5 ** ((anchor_season - season) / RECENCY_HALFLIFE)
            stops.append(row[traits].astype(float))
            weights.append(LINEAGE_WEIGHT * recency)
            labels.append(f"{team} {season} (lineage)")
            lineage_used = True

    if not stops:
        return None, 0.0, []

    W = np.array(weights, dtype=float)
    M = pd.concat(stops, axis=1).astype(float)
    # Column-wise weighted mean that tolerates missing traits in a given season.
    vals, mask = M.to_numpy(), M.notna().to_numpy()
    num = np.nansum(np.where(mask, vals, 0.0) * W, axis=1)
    den = (mask * W).sum(axis=1)
    blended = pd.Series(np.where(den > 0, num / np.maximum(den, 1e-9), np.nan), index=M.index)

    # Evidence saturates at roughly two full recent seasons of coordinator work.
    evidence = float(min(W.sum() / 2.0, 1.0))
    if lineage_used:
        evidence *= 0.7
    return blended, evidence, labels


def project_fingerprint(
    team: str,
    staff: Staff,
    fp: pd.DataFrame,
    side: str,
    league: pd.Series,
    anchor_season: int = LAST_COMPLETED_SEASON,
) -> dict:
    """Blend coach signature, team continuity and league mean for one side."""
    traits = OFFENSE_IDENTITY if side == "offense" else DEFENSE_IDENTITY
    traits = [t for t in traits if t in fp.columns]

    base = _fp_row(fp, team, anchor_season, side)
    base_vec = base[traits].astype(float) if base is not None else pd.Series(np.nan, index=traits)
    league_vec = league.reindex(traits).astype(float)

    leader = staff.offensive_leader() if side == "offense" else staff.defensive_leader()
    is_new = staff.offense_is_new if side == "offense" else staff.defense_is_new

    coach_vec, evidence, labels = (None, 0.0, [])
    if is_new and leader:
        coach_vec, evidence, labels = coach_prior_fingerprint(
            leader, fp, side, traits, anchor_season=anchor_season)

    if coach_vec is None or evidence <= 0:
        # No usable coach signature: stay on team continuity, but regress harder
        # when we know the staff changed and simply cannot measure the newcomer.
        w_team = 0.70 if is_new else 0.88
        w_coach, w_league = 0.0, 1.0 - w_team
        coach_vec = pd.Series(np.nan, index=traits)
    else:
        role = staff.play_caller if side == "offense" else staff.def_play_caller
        conf = CONFIDENCE_WEIGHT.get(staff.confidence, 0.7)
        # Maximum share a brand-new play caller can claim, before confidence.
        w_coach = 0.62 * ROLE_WEIGHT.get(role, 0.9) * evidence * conf
        w_team = (1.0 - w_coach) * 0.80
        w_league = 1.0 - w_coach - w_team

    parts = pd.DataFrame({"coach": coach_vec, "team": base_vec, "league": league_vec})
    w = pd.Series({"coach": w_coach, "team": w_team, "league": w_league})
    mask = parts.notna()
    weighted = (parts.fillna(0.0) * w).sum(axis=1)
    denom = (mask * w).sum(axis=1)
    projected = weighted / denom.replace(0, np.nan)
    # Anything still missing falls back to the league mean.
    projected = projected.fillna(league_vec)

    if side == "defense":
        projected = _add_defense_quality(projected, base, league, anchor_season)
    else:
        projected = _add_quality(projected, base, league, OFFENSE_QUALITY_PERSISTENCE)

    return {
        "team": team,
        "side": side,
        "projected": projected,
        "baseline": base_vec,
        "weights": {"coach": w_coach, "team": w_team, "league": w_league},
        "coach_evidence": evidence,
        "coach_sources": labels,
        "coach_name": leader.get("name") if leader else None,
        "is_new": is_new,
        "confidence": staff.confidence,
    }


def _add_quality(projected: pd.Series, base: pd.Series | None,
                 league: pd.Series, persistence: dict) -> pd.Series:
    """Regress a result column toward the league mean by its own persistence."""
    out = projected.copy()
    for trait, r in persistence.items():
        lg = league.get(trait, np.nan)
        if not np.isfinite(lg):
            continue
        observed = float(base.get(trait, np.nan)) if base is not None else np.nan
        out[trait] = float(lg) if not np.isfinite(observed) else \
            float(lg) + (observed - float(lg)) * float(r)
    return out


def _add_defense_quality(projected: pd.Series, base: pd.Series | None,
                         league: pd.Series, anchor_season: int) -> pd.Series:
    """Attach how good a defence is, alongside how it is built.

    Quality is a result, not an identity, so it does not follow a coordinator.
    Each trait is pulled toward the league mean by its own measured persistence -
    yards allowed carry at about r = 0.11, explosive plays allowed at 0.29 - so
    the surviving spread is deliberately narrow. Narrow is not zero: without
    these columns the matchup adjustment finds nothing to read and treats every
    opponent as average.
    """
    out = projected.copy()
    for trait, persistence in DEFENSE_QUALITY_PERSISTENCE.items():
        lg = league.get(trait, np.nan)
        if not np.isfinite(lg):
            continue
        prior = float(lg)
        observed = float(base.get(trait, np.nan)) if base is not None else np.nan
        out[trait] = prior if not np.isfinite(observed) else \
            prior + (observed - prior) * float(persistence)
    return out
