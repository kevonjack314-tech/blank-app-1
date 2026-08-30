"""Who is actually on the field, and how they bend a coach's scheme.

A fingerprint carried over from a coach's previous stop describes what they did
with different players. Some of that signature is genuinely portable - tempo,
motion usage, how often they go on fourth down. Some of it is not: Todd Monken's
Baltimore offense ran designed quarterback runs at a high rate because Lamar
Jackson was taking the snaps, and none of that transfers to a different roster.

This module projects the depth chart into usable roles, measures each player's
own tendencies from history, and then constrains the scheme projection so that
personnel-bound traits follow the players rather than the coach.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LAST_COMPLETED_SEASON, PRIOR_STRENGTH, normalize_team

# Skill roles the projection cares about, and how deep to read the chart.
ROLE_DEPTH = {"QB": 2, "RB": 4, "WR": 6, "TE": 3, "FB": 1}
SKILL_POSITIONS = ("QB", "RB", "WR", "TE", "FB")

# For each personnel-bound trait: how much of the projection is dictated by the
# player rather than the play caller. 1.0 means the coach has essentially no say.
PERSONNEL_BOUND = {
    "qb_designed_run_rate": 0.85,   # a coach cannot scheme mobility into a statue
    "scramble_rate": 0.80,          # almost entirely a QB trait
    "adot": 0.40,                   # scheme sets the menu, the QB picks from it
    "deep_rate": 0.40,
    "yac_share": 0.30,              # depends on who is catching it
    "sack_rate_allowed": 0.45,      # QB pocket habits plus the line in front
}


def latest_depth_chart(depth: pd.DataFrame) -> pd.DataFrame:
    """Skill-position depth chart, one row per player per role."""
    if depth is None or depth.empty:
        return pd.DataFrame()
    df = depth.copy()
    df["team"] = df["team"].map(normalize_team)
    df = df[df["pos_abb"].isin(SKILL_POSITIONS)]
    df = df[df["player_name"].notna()]
    df["pos_rank"] = pd.to_numeric(df["pos_rank"], errors="coerce")
    df = df[df["pos_rank"].notna()]

    keep = []
    for pos, depth_n in ROLE_DEPTH.items():
        keep.append(df[(df["pos_abb"] == pos) & (df["pos_rank"] <= depth_n)])
    out = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()
    if out.empty:
        return out
    out = out.sort_values(["team", "pos_abb", "pos_rank"])
    return out[["team", "player_name", "gsis_id", "pos_abb", "pos_rank", "pos_grp"]].reset_index(drop=True)


def base_defensive_front(depth: pd.DataFrame) -> pd.DataFrame:
    """Each team's listed base front (3-4 / 4-3), straight off the depth chart."""
    if depth is None or depth.empty or "pos_grp" not in depth.columns:
        return pd.DataFrame(columns=["team", "base_front"])
    d = depth.copy()
    d["team"] = d["team"].map(normalize_team)
    d = d[d["pos_grp"].astype(str).str.contains("Base", na=False)]
    if d.empty:
        return pd.DataFrame(columns=["team", "base_front"])
    front = (
        d.groupby("team")["pos_grp"]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("base_front")
        .reset_index()
    )
    front["base_front"] = (
        front["base_front"].astype(str).str.replace("Base ", "", regex=False).str.replace(" D", "", regex=False)
    )
    return front


def qb_profiles(plays: pd.DataFrame) -> pd.DataFrame:
    """Per-quarterback tendencies that a new coordinator has to work around."""
    df = plays.copy()
    dropbacks = df[df["is_dropback"]]
    if dropbacks.empty:
        return pd.DataFrame()

    # Attribute team snaps to whoever was the passer, so rates share a denominator.
    snaps = (
        dropbacks.groupby(["season", "passer_player_id"])
        .agg(
            dropbacks=("play_id", "size"),
            team=("posteam", lambda s: s.value_counts().idxmax()),
            adot=("air_yards", "mean"),
            deep_rate=("air_yards", lambda s: float((s >= 20).mean()) if s.notna().any() else np.nan),
            sacks=("sack", "sum"),
            cpoe=("cpoe", "mean"),
            epa=("epa", "mean"),
            comp_yards=("yards_gained", "sum"),
        )
        .reset_index()
        .rename(columns={"passer_player_id": "player_id"})
    )

    runs = df[df["is_designed_run"]]
    qb_runs = (
        runs.groupby(["season", "rusher_player_id"])
        .agg(designed_runs=("play_id", "size"), rush_yards=("yards_gained", "sum"),
             rush_tds=("rush_touchdown", "sum"))
        .reset_index()
        .rename(columns={"rusher_player_id": "player_id"})
    )

    # A scramble is charted as a run, so it carries a rusher and no passer.
    # Attribute it through the ball carrier or it disappears from the profile.
    scrambles = (
        df[df["qb_scramble"].fillna(0) > 0]
        .groupby(["season", "rusher_player_id"])
        .size()
        .rename("scrambles")
        .reset_index()
        .rename(columns={"rusher_player_id": "player_id"})
    )

    completions = dropbacks[dropbacks["complete_pass"].fillna(0) > 0]
    yac = (
        completions.groupby(["season", "passer_player_id"])
        .agg(_yac=("yards_after_catch", "sum"), _rec_yards=("yards_gained", "sum"))
        .reset_index()
        .rename(columns={"passer_player_id": "player_id"})
    )
    yac["yac_share"] = yac["_yac"] / yac["_rec_yards"].replace(0, np.nan)

    team_plays = df.groupby(["season", "posteam"]).size().rename("team_plays").reset_index()
    team_plays = team_plays.rename(columns={"posteam": "team"})

    prof = snaps.merge(qb_runs, on=["season", "player_id"], how="left")
    prof = prof.merge(scrambles, on=["season", "player_id"], how="left")
    prof = prof.merge(yac[["season", "player_id", "yac_share"]], on=["season", "player_id"], how="left")
    prof["scrambles"] = prof["scrambles"].fillna(0.0)
    prof = prof.merge(team_plays, on=["season", "team"], how="left")
    prof["designed_runs"] = prof["designed_runs"].fillna(0.0)

    # Only count seasons where the player actually ran the offense.
    prof = prof[prof["dropbacks"] >= 100].copy()
    prof["qb_designed_run_rate"] = prof["designed_runs"] / prof["team_plays"].clip(lower=1)
    # Scrambles are dropbacks that became runs; the denominator is all dropbacks.
    prof["scramble_rate"] = prof["scrambles"] / (prof["dropbacks"] + prof["scrambles"]).clip(lower=1)
    prof["sack_rate_allowed"] = prof["sacks"] / prof["dropbacks"].clip(lower=1)

    # Passing efficiency, so a quarterback's own yards per attempt can be
    # separated from his offence's. Attempts exclude sacks and scrambles, which
    # are dropbacks that never became a throw.
    throws = dropbacks[(dropbacks["pass"].fillna(0) > 0) & (dropbacks["sack"].fillna(0) == 0)]
    eff = (
        throws.groupby(["season", "passer_player_id"])
        .agg(attempts=("play_id", "size"),
             pass_yards=("yards_gained", lambda s: float(s[s > -100].sum())))
        .reset_index()
        .rename(columns={"passer_player_id": "player_id"})
    )
    # Only completions carry passing yards.
    comp_yards = (
        throws[throws["complete_pass"].fillna(0) > 0]
        .groupby(["season", "passer_player_id"])["yards_gained"].sum()
        .rename("pass_yards").reset_index()
        .rename(columns={"passer_player_id": "player_id"})
    )
    eff = eff.drop(columns=["pass_yards"]).merge(comp_yards, on=["season", "player_id"], how="left")
    eff["pass_yards"] = eff["pass_yards"].fillna(0.0)
    eff["ypa"] = eff["pass_yards"] / eff["attempts"].clip(lower=1)
    games = (
        throws.groupby(["season", "passer_player_id"])["game_id"].nunique()
        .rename("qb_games").reset_index().rename(columns={"passer_player_id": "player_id"})
    )
    eff = eff.merge(games, on=["season", "player_id"], how="left")
    eff["attempts_per_game"] = eff["attempts"] / eff["qb_games"].clip(lower=1)
    prof = prof.merge(eff[["season", "player_id", "attempts", "ypa", "qb_games",
                           "attempts_per_game"]],
                      on=["season", "player_id"], how="left")
    return prof


# How much of a deviation in yards per attempt survives to next season, fitted
# jointly on 191 quarterback season pairs from 2018-2025:
#
#     ypa_next = league + 0.181 * (qb_ypa - league) + 0.239 * (team_ypa - league)
#
# The two carry about equally, which is the whole point. The model previously
# took passing efficiency entirely from the team's scheme fingerprint, at full
# strength and with no regression at all. That ranked quarterbacks backwards on
# the 2025 holdout - correlation -0.09 between projected and actual passing
# yards, because volume and efficiency trade off: the offences that throw most
# are usually the ones doing it badly and from behind.
YPA_WEIGHT_QB = 0.181
YPA_WEIGHT_TEAM = 0.239
# aDOT is stickier and leans the other way, toward the passer:
#     adot_next = league + 0.270 * (qb_adot - league) + 0.192 * (team_adot - league)
ADOT_WEIGHT_QB = 0.270
ADOT_WEIGHT_TEAM = 0.192
# Attempts before a quarterback's own history counts at full weight.
PASSING_EVIDENCE_ATTEMPTS = 350.0
LEAGUE_YPA = 7.12
LEAGUE_ADOT = 7.85


# How much of a passer's volume is his own rather than his offence's. Fitted on
# 448 quarterback games from 2025 against 2022-2024 history:
#
#     attempts = league + 0.772 * (his attempts/game - league)
#                       + 0.304 * (his offence's attempts/game - league)
#
# His own history carries most of it, and dropping it is expensive. Taking
# volume from the team alone - which is what the model did - projected passing
# yards with no skill at all on the 2025 holdout: correlation -0.08 against
# +0.18 for the trivial baseline of the passer's own prior yards per game. A
# quarterback is not interchangeable with the offence he is standing in.
ATTEMPT_WEIGHT_QB = 0.772
ATTEMPT_WEIGHT_TEAM = 0.304
LEAGUE_ATTEMPTS_PER_GAME = 31.2
ATTEMPT_MULTIPLIER_CLIP = (0.72, 1.32)


def qb_volume_multiplier(prof: pd.DataFrame, player_id: str | None,
                         team_attempts_per_game: float | None = None) -> float:
    """How far this passer's own history moves attempts off the team's number.

    Returned as a multiplier so the caller keeps its own game-level variation -
    opponent, weather, projected game script - and only the level moves.
    """
    lg = LEAGUE_ATTEMPTS_PER_GAME
    team = float(team_attempts_per_game) if team_attempts_per_game else lg
    if not np.isfinite(team) or team <= 0:
        team = lg

    own = blend_player_history(prof, player_id, ["attempts_per_game"]) if player_id else None
    if own is None or not np.isfinite(own.get("attempts_per_game", np.nan)):
        return 1.0
    d = prof[prof["player_id"] == player_id]
    att = float(d["attempts"].fillna(0.0).sum()) if "attempts" in d else 0.0
    evidence = float(min(att / PASSING_EVIDENCE_ATTEMPTS, 1.0))
    if evidence <= 0:
        return 1.0

    qb_att = float(own["attempts_per_game"])
    blended = (lg + ATTEMPT_WEIGHT_QB * evidence * (qb_att - lg)
               + ATTEMPT_WEIGHT_TEAM * (team - lg))
    # The denominator is what the team's number alone would have said, so the
    # multiplier is exactly the correction and nothing else.
    return float(np.clip(blended / team, *ATTEMPT_MULTIPLIER_CLIP))


def qb_passing_line(prof: pd.DataFrame, player_id: str | None, scheme: pd.Series,
                    league: pd.Series | None = None) -> dict:
    """A quarterback's projected aDOT and yards per attempt.

    Two claims are being separated. How far the ball travels and how much it
    gains are partly the offence - the same coordinator calls the same routes
    for whoever is under centre - and partly the passer, and the split is not
    the same for the two: depth of target follows the quarterback more than the
    scheme, efficiency slightly less. Both are regressed toward the league mean
    by the weights above, so neither the team's number nor the player's is used
    at full strength.

    A quarterback with no history at all is projected as his offence, regressed;
    that is the right answer for a rookie, whose own evidence is zero.
    """
    league = league if league is not None else pd.Series(dtype=float)
    lg_ypa = float(league.get("ypa", LEAGUE_YPA))
    lg_adot = float(league.get("adot", LEAGUE_ADOT))
    if not np.isfinite(lg_ypa):
        lg_ypa = LEAGUE_YPA
    if not np.isfinite(lg_adot):
        lg_adot = LEAGUE_ADOT

    team_ypa = float(scheme.get("ypa", lg_ypa))
    team_adot = float(scheme.get("adot", lg_adot))
    if not np.isfinite(team_ypa):
        team_ypa = lg_ypa
    if not np.isfinite(team_adot):
        team_adot = lg_adot

    own = blend_player_history(prof, player_id, ["ypa", "adot"]) if player_id else None
    qb_ypa, qb_adot, evidence = lg_ypa, lg_adot, 0.0
    if own is not None:
        d = prof[prof["player_id"] == player_id]
        att = float(d["attempts"].fillna(0.0).sum()) if "attempts" in d else 0.0
        evidence = float(min(att / PASSING_EVIDENCE_ATTEMPTS, 1.0))
        if np.isfinite(own.get("ypa", np.nan)):
            qb_ypa = float(own["ypa"])
        else:
            evidence = 0.0
        if np.isfinite(own.get("adot", np.nan)):
            qb_adot = float(own["adot"])

    ypa = lg_ypa + YPA_WEIGHT_QB * evidence * (qb_ypa - lg_ypa) \
        + YPA_WEIGHT_TEAM * (team_ypa - lg_ypa)
    adot = lg_adot + ADOT_WEIGHT_QB * evidence * (qb_adot - lg_adot) \
        + ADOT_WEIGHT_TEAM * (team_adot - lg_adot)
    return {"ypa": float(np.clip(ypa, 5.4, 9.0)),
            "adot": float(np.clip(adot, 5.5, 10.5)),
            "evidence": evidence,
            "ypa_multiplier": float(np.clip(ypa / lg_ypa, 0.80, 1.22))}


def blend_player_history(prof: pd.DataFrame, player_id: str, cols: list[str],
                         halflife: float = 1.8) -> pd.Series | None:
    """Recency-weighted average of one player's seasons."""
    d = prof[prof["player_id"] == player_id]
    if d.empty:
        return None
    w = 0.5 ** ((LAST_COMPLETED_SEASON - d["season"]) / halflife)
    w = w * np.sqrt(d["dropbacks"].clip(lower=1) / 400.0).clip(upper=1.5)
    out = {}
    for c in cols:
        if c not in d.columns:
            out[c] = np.nan          # trait not measured for this position
            continue
        v = d[c].astype(float)
        m = v.notna()
        out[c] = float((v[m] * w[m]).sum() / w[m].sum()) if m.any() and w[m].sum() > 0 else np.nan
    s = pd.Series(out)
    s["_evidence"] = float(min(w.sum(), 1.0))
    return s


def projected_qb(chart: pd.DataFrame, team: str) -> dict | None:
    """The QB1 listed on the most recent depth chart."""
    d = chart[(chart["team"] == team) & (chart["pos_abb"] == "QB")].sort_values("pos_rank")
    if d.empty:
        return None
    row = d.iloc[0]
    return {"name": row["player_name"], "gsis_id": row.get("gsis_id")}


def apply_personnel_constraints(
    projected: pd.Series,
    team: str,
    chart: pd.DataFrame,
    qb_prof: pd.DataFrame,
    league: pd.Series,
) -> tuple[pd.Series, dict]:
    """Pull personnel-bound traits toward the players actually rostered.

    Returns the adjusted fingerprint and a note describing what moved and why.
    """
    out = projected.copy()
    info: dict = {"qb": None, "adjusted": {}, "qb_evidence": 0.0}

    qb = projected_qb(chart, team)
    if qb is None:
        return out, info
    info["qb"] = qb["name"]

    traits = [t for t in PERSONNEL_BOUND if t in out.index]
    hist = None
    if qb.get("gsis_id"):
        hist = blend_player_history(qb_prof, qb["gsis_id"], traits)

    if hist is None:
        # An unproven starter - rookie or a career backup with no starting
        # sample. Regress the personnel-bound traits toward the league mean
        # rather than inheriting a previous quarterback's athletic profile.
        info["qb_evidence"] = 0.0
        for t in traits:
            if t in league.index and np.isfinite(league.get(t, np.nan)):
                before = float(out[t])
                pull = 0.5 * PERSONNEL_BOUND[t]
                out[t] = (1 - pull) * before + pull * float(league[t])
                info["adjusted"][t] = {"from": before, "to": float(out[t]), "driver": "unproven starter"}
        return out, info

    evidence = float(hist.get("_evidence", 0.0))
    info["qb_evidence"] = evidence
    for t in traits:
        pv = hist.get(t, np.nan)
        if not np.isfinite(pv):
            continue
        w = PERSONNEL_BOUND[t] * evidence
        before = float(out[t])
        out[t] = (1 - w) * before + w * float(pv)
        info["adjusted"][t] = {"from": before, "to": float(out[t]), "driver": qb["name"]}
    return out, info
