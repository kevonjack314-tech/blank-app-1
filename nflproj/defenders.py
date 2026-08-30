"""Individual defenders: what the coverage data can and cannot tell you.

This module exists because the obvious idea does not work, and it is worth
having the reason written down where the next person will find it.

The obvious idea is a receiver-versus-cornerback adjustment: find out which
defenders cover well, and mark down the receivers who have to face them. Pro
Football Reference charts every defender's targets, completions, yards and
yards after catch allowed, which looks like exactly the input that needs. It
is not, and the measurement is unambiguous.

Over 806 consecutive defender-seasons from 2018-2025 (minimum 40 targets and
eight games), year-over-year persistence looks encouraging at first:

| | raw | with coverage role removed |
| --- | --- | --- |
| Depth of target allowed | **0.714** | — |
| Targets per game | 0.586 | — |
| Completion % allowed | 0.455 | **0.092** |
| YAC per completion allowed | 0.364 | **0.144** |
| Yards per target allowed | 0.169 | **0.127** |
| Missed tackle % | 0.325 | — |

The left column is almost entirely *role*. A boundary corner sees deep throws
and a nickel sees shallow ones, and depth of target drives completion rate
mechanically - the aDOT curve alone explains 41% of the variance in completion
percentage allowed. Fit that curve out, and what is left, which is the part
that would actually mean "this defender covers well", persists at 0.09 to 0.14.
That is noise.

Two escape routes were tried and both closed. Within a single season, split-half
reliability tells the same story - depth of target 0.675, over-expected
completion rate 0.089 - so in-season mode cannot rescue it either. And missed
tackle rate, the one genuine individual skill in the table at 0.325, does not
predict the thing it should: a team's missed tackle rate this season correlates
**-0.07** with its yards after catch allowed next season.

So nothing here is wired into the projection. What this module produces is
descriptive: who a defense asks to cover, how deep they work, and who
quarterbacks actually throw at. That is real, it is stable, and it is the sort
of thing a coordinator would say out loud - it is simply not a yardage
multiplier, and presenting it as one would be inventing precision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import normalize_team

# Measured persistence, kept next to the code that would be tempted to use it.
DEFENDER_PERSISTENCE = {
    "adot_allowed": 0.714,
    "targets_per_game": 0.586,
    "completion_pct_allowed": 0.455,
    "yac_per_completion_allowed": 0.364,
    "missed_tackle_pct": 0.325,
    "yards_per_target_allowed": 0.169,
}
# The same measures with coverage role fitted out. These are what a genuine
# coverage-skill adjustment would have to run on, and they are all noise.
DEFENDER_SKILL_PERSISTENCE = {
    "completion_pct_over_expected": 0.092,
    "yac_per_completion_over_expected": 0.144,
    "yards_per_target_over_expected": 0.127,
}
# Nothing above clears the bar, so no defender term reaches a projection.
APPLIED_TO_PROJECTION = False

# Completion percentage allowed as a function of depth of target, fitted on
# 1,515 qualified defender-seasons weighted by targets. Used only to separate
# role from skill, never to project.
COMPLETION_VS_ADOT = (0.817, -0.0231, 0.00036)
YPT_VS_ADOT = (6.067, 0.1534, -0.00102)
YAC_VS_ADOT = (7.378, -0.4548, 0.01580)

MIN_TARGETS = 25
MIN_GAMES = 4

# Coverage roles, cut by how deep a defender is thrown at. The boundaries come
# from the league distribution: the middle half of qualified defenders works
# between about 6.5 and 10.5 yards downfield.
ROLE_BANDS = ((6.5, "underneath"), (10.5, "intermediate"), (99.0, "deep"))


def _curve(adot: pd.Series, coef: tuple) -> pd.Series:
    a = pd.to_numeric(adot, errors="coerce")
    return coef[0] + coef[1] * a + coef[2] * a ** 2


def defender_profiles(adv_def: pd.DataFrame, seasons: tuple | None = None,
                      min_targets: int = MIN_TARGETS) -> pd.DataFrame:
    """One row per defender: coverage role, workload and tackling.

    The over-expected columns are reported because leaving them out would
    invite someone to recompute them; they carry a persistence of about 0.1 and
    should be read as description of a season that happened, not as a forecast.
    """
    if adv_def is None or adv_def.empty:
        return pd.DataFrame()
    d = adv_def.copy()
    if seasons:
        d = d[d["season"].isin(seasons)]
    if d.empty:
        return pd.DataFrame()

    g = (
        d.groupby(["player_id", "player", "team"], dropna=False)
        .agg(games=("targets", "size"), targets=("targets", "sum"),
             completions=("completions", "sum"), yards=("yards", "sum"),
             yac=("yac", "sum"), touchdowns=("td", "sum"),
             interceptions=("ints", "sum"), adot=("adot", "mean"),
             tackles=("tackles", "sum"), missed=("missed", "sum"))
        .reset_index()
    )
    g = g[(g["targets"] >= min_targets) & (g["games"] >= MIN_GAMES)]
    if g.empty:
        return pd.DataFrame()

    tgt = g["targets"].clip(lower=1)
    g["targets_per_game"] = g["targets"] / g["games"].clip(lower=1)
    g["completion_pct"] = g["completions"] / tgt
    g["yards_per_target"] = g["yards"] / tgt
    g["yac_per_completion"] = g["yac"] / g["completions"].clip(lower=1)
    g["missed_tackle_pct"] = g["missed"] / (g["tackles"] + g["missed"]).clip(lower=1)

    g["completion_pct_oe"] = g["completion_pct"] - _curve(g["adot"], COMPLETION_VS_ADOT)
    g["yards_per_target_oe"] = g["yards_per_target"] - _curve(g["adot"], YPT_VS_ADOT)
    g["yac_per_completion_oe"] = g["yac_per_completion"] - _curve(g["adot"], YAC_VS_ADOT)

    g["role"] = [_role(a) for a in g["adot"]]
    # Share of the defense's charted coverage work that runs through him.
    total = g.groupby("team")["targets"].transform("sum").clip(lower=1)
    g["coverage_share"] = g["targets"] / total
    return g.sort_values(["team", "targets"], ascending=[True, False]).reset_index(drop=True)


def _role(adot: float) -> str:
    if not np.isfinite(adot):
        return "unknown"
    for cut, name in ROLE_BANDS:
        if adot < cut:
            return name
    return "deep"


def secondary_map(profiles: pd.DataFrame, team: str, top_n: int = 6) -> pd.DataFrame:
    """Who a defense asks to cover, ranked by how often they are thrown at."""
    if profiles is None or profiles.empty:
        return pd.DataFrame()
    d = profiles[profiles["team"] == normalize_team(team)]
    cols = ["player", "role", "adot", "targets", "targets_per_game",
            "coverage_share", "completion_pct", "yards_per_target",
            "yac_per_completion", "missed_tackle_pct"]
    cols = [c for c in cols if c in d.columns]
    return d.sort_values("targets", ascending=False).head(top_n)[cols].reset_index(drop=True)


def coverage_load(profiles: pd.DataFrame, team: str) -> dict:
    """How a defense distributes and shapes its coverage work.

    Concentration is the share carried by the single most-targeted defender.
    A defense that funnels everything to one player is one whose plan changes
    materially when that player is unavailable - which is a real thing to know
    even though the size of the effect is not projectable from this data.
    """
    d = secondary_map(profiles, team, top_n=99)
    if d.empty:
        return {}
    total = float(d["targets"].sum())
    by_role = d.groupby("role")["targets"].sum() / max(total, 1.0)
    return {
        "team": normalize_team(team),
        "defenders_charted": int(len(d)),
        "most_targeted": str(d.iloc[0]["player"]),
        "concentration": float(d.iloc[0]["targets"] / max(total, 1.0)),
        "mean_adot_allowed": float((d["adot"] * d["targets"]).sum() / max(total, 1.0)),
        "underneath_share": float(by_role.get("underneath", 0.0)),
        "intermediate_share": float(by_role.get("intermediate", 0.0)),
        "deep_share": float(by_role.get("deep", 0.0)),
    }


def matchup_note(profiles: pd.DataFrame, defense: str, receiver_adot: float | None) -> str:
    """A sentence about where a receiver's routes meet a defense's coverage.

    Descriptive by construction. It says who works at the depth this receiver
    lives at, not what it will cost him, because the second claim is not
    supported - see the module docstring.
    """
    load = coverage_load(profiles, defense)
    if not load:
        return ""
    d = secondary_map(profiles, defense, top_n=99)
    parts = [
        f"{load['team']} run their charted coverage through {load['most_targeted']} "
        f"more than anyone ({load['concentration']*100:.0f}% of targets), and face "
        f"{load['mean_adot_allowed']:.1f} yards of average target depth."
    ]
    if receiver_adot is not None and np.isfinite(receiver_adot):
        band = _role(float(receiver_adot))
        same = d[d["role"] == band]
        if not same.empty:
            who = ", ".join(same.head(2)["player"].astype(str))
            parts.append(
                f"A receiver working at {receiver_adot:.1f} yards is a {band} target; "
                f"{who} carry that depth for them."
            )
        share = load.get(f"{band}_share", 0.0)
        parts.append(f"They spend {share*100:.0f}% of their charted coverage there.")
    parts.append(
        "Read as personnel description only: individual coverage quality does not "
        "persist once role is removed (r = 0.09-0.14), so no adjustment is applied."
    )
    return " ".join(parts)
