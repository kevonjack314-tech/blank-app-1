"""The staff meeting: how we attack, how we stop them, and what beats us.

``coach.game_call`` produces a side and the two or three reasons behind it -
what a head coach says when asked who wins. This module produces what the staff
says in the room before he decides: the offensive coordinator's plan to move the
ball, the defensive coordinator's plan to take away what the other side does
best, and the head coach's own reading of which of the two matters more, plus
the thing that would make him wrong.

The last of those is the part worth having. A staff that has been together
twenty years does not argue about whether it is right; it argues about what
would prove it wrong, and then watches for that. So every game here carries an
explicit falsifier - a named, checkable condition that should change the read -
and they are drawn from measured quantities rather than written to sound humble:
a projected margin inside a single score, a swing player who may not dress, a
structural collision the model itself flags as high-variance, a disagreement
with the closing line.

Division of labour, kept honest:

* The **offensive coordinator** only ever speaks from collisions the scheme
  layer resolved in his offence's favour, coverage charting on the defence he is
  facing, and the down-and-distance bands where his offence's EPA meets their
  defence's worst. He does not get to claim an edge the fingerprints did not
  produce.
* The **defensive coordinator** speaks from the other side's simulated ceiling -
  the player whose 90th-percentile game actually swings a result - and from what
  his own structure concedes when it takes that away. Taking something away
  always concedes something; a coordinator who will not say what is selling a
  plan rather than describing one.
* The **head coach** weighs the two, states the call, and names the falsifier.
  He also carries the record. The model picks 64.7% of winners to the closing
  line's 68.4%, so a staff built on it should sound like a staff that knows it
  is the second-best forecaster in the building and is still worth listening to
  for the reasoning.

Nothing in here invents football. Every sentence traces to a number computed in
``schemes``, ``coverage``, ``gameplan``, ``playbook`` or the simulation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .availability import AVAILABILITY_PRIOR
from .config import TEAM_NAMES

# A one-score game is decided by a single possession, and no projection
# distinguishes a three-point favourite from a pick'em with any reliability:
# the model's own margin standard deviation is 4.2 points, so anything inside
# this is noise wearing a number.
ONE_SCORE = 4.0

# When a swing player is a question rather than a plan.
#
# This cannot be an absolute probability, which was the first mistake here. The
# availability layer's baseline for a *healthy* starter is already 0.82 (RB1) to
# 0.89 (TE1), because being an active participant in a given NFL game is simply
# not a certainty for anyone; and its baseline for a fifth receiver is 0.52 on
# role alone. An absolute bar of 0.85 therefore flagged healthy starters as
# injury concerns and would flag every backup forever, which is role information
# wearing an injury label.
#
# So the test is relative to the player's own role baseline. A discount this
# size or larger against ``availability.AVAILABILITY_PRIOR`` for his position
# and depth rank means something has actually happened to him: the injury
# report's "Questionable" multiplier is 0.72 and a week of not practising is
# 0.55, so both trip it, while a healthy player at any depth does not.
AVAILABILITY_DISCOUNT = 0.85

# A floor for the case where the role baseline is unknown - an unusual position
# or depth rank the prior table does not carry. Below the lowest starter
# baseline in that table, so a healthy starter still never trips it.
AVAILABILITY_FLOOR = 0.70

# A ceiling this far above a player's own projection is a genuine upside tail
# rather than ordinary variance, and is what makes a player worth planning for
# specifically instead of covering with structure.
#
# Measured rather than chosen. Across 864 qualifying player-seasons from 2022-25
# (RB/WR/TE, eight or more games, 20+ scrimmage yards per game) the ratio of a
# player's 90th-percentile week to his own mean has a median of 1.78 and a 90th
# percentile of 2.29. A first guess of 1.60 therefore sat *below the median* and
# flagged eleven of twelve threats on a test slate, which is a flag that says
# nothing. The bar is the measured 90th percentile: a tail in the top decile of
# tail lengths, which is what "we have to take him away by name" should mean.
#
# This is deliberately read against the real-world distribution rather than the
# simulator's own. The lotto work established that the simulation's upper tail
# runs roughly twice as fat as reality above 40 projected yards, so a bar set
# inside the simulation would drift with that bias; set against real weeks it
# stays interpretable, at the cost of firing slightly more often than it should.
CEILING_MULTIPLE = 2.29

# Play-level EPA has a standard deviation of 1.259 over the 48,201 scored plays
# of 2025, which is what makes down-and-distance splits so much weaker than they
# look. A "best band" of 20 offensive plays against a defence's 20 in the same
# band carries a two-standard-error bar of 2 * 1.259 * sqrt(2/20) = 0.80 EPA on
# the difference. Most bands on a real slate come in under that, so reporting
# the top row of the table as an edge is reporting noise with a label on it.
# Both gates below have to clear before a coordinator says anything about a band.
EPA_PLAY_SD = 1.259
MIN_BAND_PLAYS = 25

# The matchup has to move a player by at least this many yards, in the
# simulation, before a coordinator should change anything for it. Below it the
# difference is smaller than the week-to-week noise in his own usage.
MATCHUP_SWING_YARDS = 8.0

# How many lines each coordinator gets. A coordinator who says six things in a
# staff meeting has not prioritised, and the point of a staff is prioritisation.
MAX_PER_COACH = 3


def _name(team: str) -> str:
    return TEAM_NAMES.get(team, team)


def _role_baseline(pos: str | None, rank) -> float | None:
    """What the availability layer expects of a healthy player in this role."""
    if not pos or rank is None or rank != rank:
        return None
    try:
        return AVAILABILITY_PRIOR.get((str(pos), int(rank)))
    except (TypeError, ValueError):
        return None


def _side_blocks(plan: dict | None, team: str) -> tuple[dict, dict]:
    """The block where ``team`` has the ball, and the one where it defends."""
    if not plan:
        return {}, {}
    ours, theirs = {}, {}
    for key in ("home_offense", "away_offense"):
        blk = plan.get(key) or {}
        if blk.get("offense") == team:
            ours = blk
        elif blk.get("defense") == team:
            theirs = blk
    return ours, theirs


def _band_bar(plays: float) -> float:
    """The two-standard-error bar on a net EPA difference from this many plays.

    Assumes the defence's sample in the same band is about the size of the
    offence's, which is what the schedule produces. Understating the defensive
    sample would make the bar too generous, so the symmetric assumption is the
    conservative one available from the columns the table carries.
    """
    if not plays or plays <= 0:
        return float("inf")
    return 2.0 * EPA_PLAY_SD * float(np.sqrt(2.0 / float(plays)))


def _situational_read(situational: pd.DataFrame | None) -> tuple[dict | None, dict | None]:
    """Best and worst down-and-distance band - only where the sample supports one.

    ``playbook.situational_tendencies`` gives offensive EPA by band and
    ``playbook.defensive_profile`` gives EPA allowed; ``gameplan.situational_edges``
    adds them into ``net``. The top row of that table is not automatically an
    edge: these bands run 12 to 60 plays inside one season and play-level EPA is
    enormously noisy, so a band is returned only when it clears both a minimum
    sample and two standard errors on the difference. On most games that means
    nothing is returned, which is the correct answer rather than a missing one.
    """
    if situational is None or not isinstance(situational, pd.DataFrame) or situational.empty:
        return None, None
    if "net" not in situational.columns or "situation" not in situational.columns:
        return None, None
    df = situational.dropna(subset=["net"]).sort_values("net", ascending=False)
    if df.empty:
        return None, None

    def _row(r) -> dict | None:
        plays = float(r["off plays"]) if "off plays" in r.index and r["off plays"] == r["off plays"] else 0.0
        net = float(r["net"])
        if plays < MIN_BAND_PLAYS or abs(net) < _band_bar(plays):
            return None
        out = {"situation": str(r["situation"]), "net": net,
               "plays": plays, "bar": _band_bar(plays)}
        for src, dst in (("off EPA", "off_epa"), ("def EPA allowed", "def_epa_allowed"),
                         ("def blitz", "blitz")):
            if src in r.index and r[src] == r[src]:
                out[dst] = float(r[src])
        return out

    best = _row(df.iloc[0])
    worst = _row(df.iloc[-1]) if len(df) > 1 else None
    return best, worst


def offensive_plan(team: str, opponent: str, plan: dict | None,
                   x_rows: list | None = None,
                   situational: pd.DataFrame | None = None,
                   already_said: list | None = None) -> dict:
    """The offensive coordinator's plan to move the ball on ``opponent``.

    ``already_said`` is whatever the call above the staff block has already
    printed. The two draw on the same collisions, so without it the expander
    repeats the card word for word and the reader stops opening it.
    """
    ours, _ = _side_blocks(plan, team)
    collisions = ours.get("collisions") or []
    coverage = ours.get("coverage") or []

    attack, problems, variance = [], [], []
    for c in collisions:
        line = f"**{c['axis']}** — {c['edge']}. {c.get('detail', '')}".strip()
        favours = c.get("favours")
        if favours == team:
            attack.append(line)
        elif favours == opponent:
            problems.append(line)
        elif favours == "variance":
            variance.append(line)

    # Coverage charting comes back as plain prose with no side attached, and it
    # cuts both ways: "we are markedly better against zone, and they play man on
    # 40% of snaps" is a problem, not a plan. Filing it under "where we attack"
    # put exactly that sentence there. Since the note carries no field saying
    # whom it favours, it gets its own heading rather than a guess.
    coverage_notes = list(coverage[:3])

    said = set(already_said or [])
    if said:
        attack = [a for a in attack if a not in said]
        problems = [a for a in problems if a not in said]
        coverage_notes = [a for a in coverage_notes if a not in said]

    feature = None
    if x_rows:
        lifted = [r for r in x_rows
                  if r.get("matchup_swing") is not None
                  and r.get("matchup_swing") == r.get("matchup_swing")
                  and float(r["matchup_swing"]) >= MATCHUP_SWING_YARDS]
        pool = lifted or x_rows
        best = max(pool, key=lambda r: float(r.get("x_factor_score") or 0.0))
        swing = best.get("matchup_swing")
        swing = float(swing) if swing is not None and swing == swing else None
        # A swing smaller than the bar is not reported at all. Saying "this
        # matchup is worth -1 yards to him" is noise dressed as a read, and a
        # coordinator who says it loses the room.
        swing_note = None
        if swing is not None and abs(swing) >= MATCHUP_SWING_YARDS:
            swing_note = (
                f"The matchup is worth {swing:+.0f} yards to him against a "
                "league-average defence." if swing > 0 else
                f"This defence costs him {abs(swing):.0f} yards against a "
                "league-average one, so the volume has to come from somewhere else.")
        feature = {
            "player": best["player"],
            "pos": best.get("pos"),
            # Carried because the availability read is relative to the baseline
            # for a player's position and depth, not an absolute probability.
            "rank": best.get("rank"),
            "projection": float(best.get("projection") or 0.0),
            "ceiling": float(best.get("ceiling_90th") or 0.0),
            "boom_pct": best.get("boom_pct"),
            "p_active": best.get("p_active"),
            "matchup_swing": swing,
            "swing_note": swing_note,
            "lifted_by_matchup": bool(lifted) and best in lifted,
        }

    best_band, worst_band = _situational_read(situational)
    return {
        "unit": f"{_name(team)} offence",
        "opponent": _name(opponent),
        "attack": attack[:MAX_PER_COACH],
        "problems": problems[:MAX_PER_COACH],
        "coverage": coverage_notes,
        "variance": variance[:1],
        "feature": feature,
        "best_situation": best_band,
        "worst_situation": worst_band,
    }


def defensive_plan(team: str, opponent: str, plan: dict | None,
                   opp_x_rows: list | None = None,
                   situational: pd.DataFrame | None = None) -> dict:
    """The defensive coordinator's plan to take away what ``opponent`` does best."""
    _, theirs = _side_blocks(plan, team)
    collisions = theirs.get("collisions") or []

    take_away, concede, variance = [], [], []
    for c in collisions:
        line = f"**{c['axis']}** — {c['edge']}. {c.get('detail', '')}".strip()
        favours = c.get("favours")
        if favours == team:
            take_away.append(line)
        elif favours == opponent:
            concede.append(line)
        elif favours == "variance":
            variance.append(line)

    threat = None
    if opp_x_rows:
        best = max(opp_x_rows, key=lambda r: float(r.get("x_factor_score") or 0.0))
        proj = float(best.get("projection") or 0.0)
        ceiling = float(best.get("ceiling_90th") or 0.0)
        threat = {
            "player": best["player"],
            "pos": best.get("pos"),
            "rank": best.get("rank"),
            "projection": proj,
            "ceiling": ceiling,
            "boom_pct": best.get("boom_pct"),
            "p_active": best.get("p_active"),
            # A tail this long is the case for taking a player away by name
            # rather than trusting the structure to cover him.
            "bracket_worthy": bool(proj > 0 and ceiling >= proj * CEILING_MULTIPLE),
        }

    # The offence here is theirs, so the situational frame is their offence
    # against our defence: their best band is our exposure.
    their_best, their_worst = _situational_read(situational)
    return {
        "unit": f"{_name(team)} defence",
        "opponent": _name(opponent),
        "take_away": take_away[:MAX_PER_COACH],
        "concede": concede[:MAX_PER_COACH],
        "variance": variance[:1],
        "threat": threat,
        "exposure": their_best,
        "strength": their_worst,
    }


def _falsifiers(call: dict, row: pd.Series, off: dict, dfn: dict) -> list[str]:
    """What would have to be true for this call to be wrong.

    Each entry is tied to a quantity that was measured, and each is checkable
    before kickoff or in the first quarter. A falsifier nobody can check is a
    disclaimer, and disclaimers are not coaching.
    """
    out = []
    margin = abs(float(row.get("model_margin") or 0.0))
    if margin < ONE_SCORE:
        out.append(
            f"**It is a one-score game.** The projected margin is {margin:.1f} "
            "points, inside the 4.2-point standard deviation of the model's own "
            "margins. A single turnover or a missed kick is the whole edge, so "
            "treat the side as a preference rather than a read.")

    threat = dfn.get("threat")
    feature = off.get("feature")

    def _doubtful(blk) -> float | None:
        """His chance of dressing, if it is depressed against his own role."""
        if not blk:
            return None
        p = blk.get("p_active")
        if p is None or p != p:
            return None
        p = float(p)
        baseline = _role_baseline(blk.get("pos"), blk.get("rank"))
        if baseline is None:
            return p if p < AVAILABILITY_FLOOR else None
        return p if p <= baseline * AVAILABILITY_DISCOUNT else None

    threat_doubt = _doubtful(threat)

    # A ceiling only beats us if he plays, so availability supersedes the tail
    # rather than being listed next to it. Two falsifiers about one player read
    # as padding and bury the ones about other players.
    if threat and threat.get("bracket_worthy") and threat_doubt is None:
        out.append(
            f"**{threat['player']} has the tail that beats us.** Projected "
            f"{threat['projection']:.0f} with a 90th-percentile game of "
            f"{threat['ceiling']:.0f} - a range in the top decile of tail "
            "lengths. If he gets to the top of it the structural read does not "
            "matter.")

    for who, blk in (("ours", feature), ("theirs", threat)):
        p = _doubtful(blk)
        if p is None:
            continue
        tail = ""
        if who == "theirs" and blk.get("bracket_worthy"):
            tail = (f" His {blk['ceiling']:.0f}-yard ceiling is the thing we "
                    "would otherwise have to bracket, so this is the single "
                    "biggest unknown on the game.")
        out.append(
            f"**{blk['player']} may not dress** ({p*100:.0f}% to be active on "
            "the injury report and practice participation). "
            + ("That is our swing player, so the call softens without him."
               if who == "ours" else
               "He is their swing player, so the call strengthens without him.")
            + tail)

    for blk in (off, dfn):
        for v in blk.get("variance") or []:
            out.append(f"**High-variance exchange.** {v}")

    disagreement = call.get("disagreement")
    if disagreement and "off the board" in disagreement:
        out.append(
            "**The line disagrees.** The market has the other side, and the "
            "market picks winners better than this model does. The usual "
            "explanation is personnel news the model has not seen, so find out "
            "what it is before acting on either read.")
    return out[:4]


def _management(row: pd.Series) -> str | None:
    """The game-management note: how this one actually gets won on the sideline."""
    total = float(row.get("model_total") or 0.0)
    wind = row.get("wind")
    windy = wind is not None and wind == wind and float(wind) >= 15
    roof = row.get("roof")
    indoors = roof in ("dome", "closed")
    margin = abs(float(row.get("model_margin") or 0.0))

    if windy and total:
        return (f"{float(wind):.0f} mph of wind and a {total:.0f}-point projected "
                "total. This is a field-position game: the kicking game shortens, "
                "fourth-down maths moves toward going for it outside the "
                "35-yard line, and the deep ball is a worse bet than the "
                "possession throw in both directions.")
    if indoors and total >= 47:
        return (f"Indoors with a {total:.0f}-point total. Possessions matter more "
                "than field position — trading field goals loses this game, and "
                "the team that runs out of downs first loses the shootout.")
    if margin < ONE_SCORE:
        return ("A one-score projection means the fourth-quarter decisions are "
                "the game: timeout discipline, the two-minute drill, and whether "
                "you take the points or the down. Nothing in the projection "
                "separates these teams by more than a possession.")
    if total and total <= 41:
        return (f"A {total:.0f}-point total is a low-possession game. Each "
                "turnover is worth more than usual, and the first team to "
                "score twice usually controls the rest of it.")
    return None


def staff_report(call: dict, row: pd.Series, plan: dict | None = None,
                 x_factors: dict | None = None,
                 situational: dict | None = None) -> dict:
    """The whole room on one game: OC, DC, and the head coach's verdict.

    ``call`` is a ``coach.game_call`` result, ``row`` the ``predict_slate``
    record behind it, ``plan`` a ``gameplan.weekly_plan`` package, ``x_factors``
    a ``{team: [records]}`` map, and ``situational`` an optional
    ``{team: DataFrame}`` map of ``gameplan.situational_edges`` keyed by the team
    with the ball. Any of them may be missing; the report shrinks rather than
    failing, because a coordinator with no charting still has a view.
    """
    pick, opponent = call["pick"], call["opponent"]
    x_factors = x_factors or {}
    situational = situational or {}

    # The card above already prints the call's reasons, which are drawn from
    # these same collisions; the coordinator does not repeat them.
    off = offensive_plan(pick, opponent, plan, x_factors.get(pick),
                         situational.get(pick), call.get("reasons"))
    dfn = defensive_plan(pick, opponent, plan, x_factors.get(opponent),
                         situational.get(opponent))

    # Which coordinator is carrying the call. The offensive case is the count of
    # resolved collisions in our favour against the defensive case of structural
    # answers; it decides the emphasis, not the side.
    threat = dfn.get("threat") or {}
    off_case = len(off["attack"]) + (1 if off.get("feature") else 0)
    # A threat we can cover with structure rather than bracket by name is a
    # point in the defence's column; one with a tail that long is not.
    def_case = len(dfn["take_away"]) + (1 if threat and not threat.get("bracket_worthy") else 0)
    if off_case > def_case:
        emphasis = ("offence", "We win this by scoring on them, not by stopping them.")
    elif def_case > off_case:
        emphasis = ("defence", "We win this by taking away what they do best.")
    else:
        emphasis = ("balance", "No single unit decides this one.")

    falsifiers = _falsifiers(call, row, off, dfn)
    return {
        "game": f"{_name(call['away'])} at {_name(call['home'])}",
        "verdict": call["headline"],
        "confidence": call["confidence"],
        "emphasis": emphasis[0],
        "emphasis_line": emphasis[1],
        "offense": off,
        "defense": dfn,
        "falsifiers": falsifiers,
        "management": _management(row),
    }


def staff_reports(calls: list, slate: pd.DataFrame, plans: dict | None = None,
                  x_factors: dict | None = None,
                  situational: dict | None = None) -> dict:
    """``{(away, home): staff_report}`` for a whole week."""
    if not calls or slate is None or slate.empty:
        return {}
    plans, x_factors = plans or {}, x_factors or {}
    situational = situational or {}
    rows = {(r["away"], r["home"]): r for _, r in slate.iterrows()}
    out = {}
    for c in calls:
        key = (c["away"], c["home"])
        row = rows.get(key)
        if row is None:
            continue
        out[key] = staff_report(c, row, plans.get(key), x_factors.get(key),
                                situational.get(key))
    return out
