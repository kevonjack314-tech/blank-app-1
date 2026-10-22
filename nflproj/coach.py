"""The call: who wins, and the football reason why.

Everything else in this package produces numbers. This produces the thing a
staff actually leaves the room with - a side, a confidence, and the two or
three reasons that decided it - by reading the measured pieces the way a
coordinator would rather than presenting them as a table.

These are the model's own calls, stated plainly. Where one differs from the
closing line the difference is reported as what it is - a disagreement worth
understanding - rather than folded into a hedge. A model that defers to the
price on every game has no opinion, and an opinion is the thing being asked for.

The honest limit belongs somewhere, so it is stated once rather than stapled to
every game: over 544 held-out games the model picks 64.7% of winners against the
closing line's 68.4%, and betting its disagreements lost money (48.7% against the
spread, under the 52.4% break-even). That is real and the caller should surface
it once. It is not a reason to refuse to have a view - the reasoning behind a
call is what a price cannot give you, and it is worth having even where the
price forecasts better.

Every reason returned is derived from a measurement made elsewhere in the
package. Nothing here writes a narrative the data did not produce.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TEAM_NAMES, normalize_team

# Win probability bands. A coin flip is worth saying out loud: most NFL games
# genuinely are close, and a model that sounds certain about all sixteen is
# telling you about itself rather than about the games.
STRONG = 0.68
LEAN = 0.575

# Where the model and the closing line disagree by more than this, the staff
# should be able to say why. It is a prompt to look, not a signal to bet.
DISAGREEMENT_POINTS = 3.0

# Below this the two teams are level on opponent-adjusted EPA and saying
# otherwise would be reading noise. Ratings are shrunk toward the mean by their
# own measured persistence, so Week 1 gaps are small by construction.
MEANINGFUL_EPA_GAP = 0.030

# A staff leaves the room with a handful of reasons, not a dossier.
MAX_REASONS = 4

# Held-out accuracy, carried here so the caller cannot present a call as
# something better than it is.
MODEL_WINNER_ACCURACY = 0.647
MARKET_WINNER_ACCURACY = 0.684

# Against the spread, over 539 held-out games with a result (pushes dropped).
# At -110 the break-even is 52.38%.
#
# There is no edge here and no threshold rescues it. Sliced by the size of the
# disagreement the record runs 41.8% / 50.0% / 50.0% / 50.8% / 45.8% / 52.4%
# from one point out to seven-plus, which is a coin flip with the vig taken out
# of it. Two cuts appear to clear the bar - disagreements of seven or more
# points (52.4%) and games with a line of seven or more (52.8%) - and both sit
# inside their own error bars at n = 21 and n = 144. They are what turns up when
# you try fifteen cuts, not findings.
ATS_RECORD = 0.482
ATS_SAMPLE = 539
ATS_BREAKEVEN = 0.5238

# And here is *why*, which is more useful than the percentage.
#
# The model sides with the underdog 78.7% of the time. That is not a view about
# underdogs; it is a property of its own arithmetic. Ratings are regressed hard
# toward the mean - defence carries at only r = 0.12 year to year - so projected
# margins come out compressed: a standard deviation of 4.18 points against the
# market's 5.91, about 71% as far from a pick'em as the line is willing to go.
# A number that is systematically closer to zero than the line will sit on the
# dog almost every week by construction.
#
# So a spread pick here is mostly a reading of the model's own compression
# rather than of the game, and taking the dog four games in five returns exactly
# what that should: 48.4%. Scaling the margins up to match the market's spread
# would remove the tell without adding any information - the ranking underneath
# is unchanged - so it is not done.
ATS_UNDERDOG_RATE = 0.787
MODEL_MARGIN_SD = 4.18
MARKET_MARGIN_SD = 5.91

# Live 2026: every game projected from only the weeks that preceded it. The
# first real out-of-sample run, and it is not flattering.
#
# Through week 6, rebuilt each week from only the weeks that preceded it:
#
#   straight up   51.3% (39-37)   the closing line went 60.5% on the same games
#   against spread 56.9% (41-31)  above break-even, but n = 72 and +-11.4
#   totals        40.3% (31-46)
#   margin error  9.05 against the market's 8.90
#
# The spread figure is the one to be careful about, because it looks like an
# edge and is not. One week makes it: week 4 went 11-3 (78.6%) and the other
# four weeks went 50.0%, 50.0%, 50.0% and 57.1%. Drop week 4 and the record is
# 51.7% over 60 games, under break-even. The 95% interval on the full 72 games
# runs 45.5% to 68.3%, so break-even sits comfortably inside it. This is a
# coin flip with one good Sunday in it, which is exactly what the 539-game
# backtest at 48.2% predicted it would look like.
#
# Straight up is the honest headline and it got worse, not better: picking
# winners nine points behind the closing line, and below the 64.7% the model
# managed in backtest. Margin error is now within 0.15 points of the market's,
# which says the model's *numbers* are close - it is the sides it turns them
# into that are not.
LIVE_2026_SU = 0.513
LIVE_2026_MARKET_SU = 0.605
LIVE_2026_ATS = 0.569
LIVE_2026_TOTALS = 0.403
LIVE_2026_SAMPLE = 77
LIVE_2026_THROUGH_WEEK = 6
# What the spread record looks like with its one big week removed. Carried as a
# constant because quoting 56.9% without it would be the most misleading true
# number in this file.
LIVE_2026_ATS_EX_BEST_WEEK = 0.517
LIVE_2026_ATS_EX_BEST_SAMPLE = 60

# Totals: the band that looked promising is dead, and this is how it died.
#
# On the held-out seasons, disagreements of two points or more hit 55.7% over
# 255 games and held in both of them (53.1%, 58.3%). That was flagged here as
# worth watching rather than worth betting, because z was only +1.82 and it was
# one of about fifteen cuts tried to find it.
#
# It was then watched, live, through the first six weeks of 2026. It went
# 10-30 over five weeks and has not recovered since; totals overall sit at
# 40.3% on the season. The nine games published in the week 1 board went 0-9 on
# their own. Pooling the live run with the backtest gives 152 of 295, which is
# 51.5% - under the break-even it needed to clear in the first place.
#
# That is what a cut found by searching looks like when it meets new data, and
# it is the reason the flag is gone rather than being quietly reweighted.
TOTALS_ALL = 0.495            # backtest, and never better than a coin flip
TOTALS_EDGE2 = 0.515          # backtest + live 2026, pooled
TOTALS_EDGE2_SAMPLE = 295
TOTALS_BAND_RETIRED = True
TOTALS_MIN_EDGE = 2.0


def _confidence(p: float) -> str:
    p = max(p, 1.0 - p)
    if p >= STRONG:
        return "strong"
    if p >= LEAN:
        return "lean"
    return "coin flip"


def _ratings_reason(pick: str, other: str, ratings: pd.DataFrame) -> str | None:
    """Which side of the ball actually separates these two teams."""
    if ratings is None or ratings.empty or "team" not in ratings.columns:
        return None
    idx = ratings.set_index("team")
    if pick not in idx.index or other not in idx.index:
        return None
    a, b = idx.loc[pick], idx.loc[other]
    off_gap = float(a.get("off_rating", 0)) - float(b.get("off_rating", 0))
    def_gap = float(a.get("def_rating", 0)) - float(b.get("def_rating", 0))
    # Projected ratings are shrunk hard - defence carries at only r = 0.12 year
    # to year - so most Week 1 gaps are genuinely tiny. Reporting a hundredth of
    # an EPA point as "the separation" would be inventing confidence the number
    # does not carry, so below this the honest reading is that the teams are level.
    if max(abs(off_gap), abs(def_gap)) < MEANINGFUL_EPA_GAP:
        return None
    if abs(off_gap) >= abs(def_gap):
        side, gap = "offence", off_gap
        detail = ("moves the ball better than they do" if gap > 0
                  else "is the weaker unit here")
    else:
        side, gap = "defence", def_gap
        detail = ("is the harder unit to play against" if gap > 0
                  else "is the softer unit here")
    return (f"The separation is on {side} — the {TEAM_NAMES.get(pick, pick)} "
            f"{side} {detail}, {abs(gap):.3f} EPA per play "
            f"{'better' if gap > 0 else 'worse'} on opponent-adjusted numbers.")


def _x_factor_reason(pick: str, other: str, x_factors: dict) -> str | None:
    """The player who can take the game somewhere the projection will not."""
    best = {}
    for team in (pick, other):
        rows = x_factors.get(team)
        if isinstance(rows, list) and rows:
            best[team] = rows[0]
    if pick not in best:
        return None
    mine = best[pick]
    line = (f"{mine['player']} is the swing: projected {mine['projection']:.0f} "
            f"with a 90th-percentile game of {mine['ceiling_90th']:.0f} and a "
            f"{mine['boom_pct']:.0f}% chance of a genuinely game-breaking line")
    if other in best:
        theirs = best[other]
        if mine["ceiling_90th"] > theirs["ceiling_90th"] * 1.10:
            line += (f", a bigger ceiling than anything {TEAM_NAMES.get(other, other)} "
                     f"has ({theirs['player']}, {theirs['ceiling_90th']:.0f})")
        elif theirs["ceiling_90th"] > mine["ceiling_90th"] * 1.10:
            line += (f" - though {theirs['player']} is the larger ceiling on the "
                     f"other side ({theirs['ceiling_90th']:.0f}), which is how "
                     "this one gets away from them")
    return line + "."


def _collision_reasons(plan: dict, pick: str, limit: int = 2) -> list[str]:
    """Structural mismatches, phrased from the picked team's point of view."""
    out = []
    for side in ("away_offense", "home_offense"):
        blk = plan.get(side) or {}
        off, dfn = blk.get("offense"), blk.get("defense")
        for c in (blk.get("collisions") or [])[:limit]:
            who = c.get("favours")
            tag = "in their favour" if who == off and off == pick else (
                "against them" if who == off and dfn == pick else "")
            out.append(f"**{c['axis']}** — {c['edge']}" + (f" ({tag})" if tag else ""))
        for note in (blk.get("coverage") or [])[:limit]:
            out.append(note)
    return out


def game_call(row: pd.Series, plan: dict | None, x_factors: dict | None,
              ratings: pd.DataFrame | None) -> dict:
    """One game, decided and explained.

    ``row`` is a record from ``gamemodel.predict_slate``; the rest are the
    coaching layers, any of which may be missing.
    """
    home, away = row["home"], row["away"]
    p_home = float(row["home_win_pct"]) / 100.0
    pick = home if p_home >= 0.5 else away
    other = away if pick == home else home
    win_prob = p_home if pick == home else 1.0 - p_home

    # Ordered the way a staff would say them: who is better, where the schemes
    # collide, then who can break it open. Trimmed, because ten reasons is a
    # dossier rather than a call.
    reasons = []
    r = _ratings_reason(pick, other, ratings)
    if r:
        reasons.append(r)
    if plan:
        reasons.extend(_collision_reasons(plan, pick))
    x = _x_factor_reason(pick, other, x_factors or {})
    if x:
        reasons.append(x)
    seen, trimmed = set(), []
    for item in reasons:
        if item not in seen:
            seen.add(item)
            trimmed.append(item)
    reasons = trimmed[:MAX_REASONS]

    env_pts = float(row.get("env_pts") or 0.0)
    if abs(env_pts) >= 0.75:
        wind = row.get("wind")
        bits = []
        if row.get("roof") in ("dome", "closed"):
            bits.append("indoors")
        if wind is not None and wind == wind and float(wind) >= 12:
            bits.append(f"{float(wind):.0f} mph wind")
        if row.get("divisional"):
            bits.append("divisional")
        if bits:
            reasons.append(f"Conditions move the total {env_pts:+.1f} points "
                           f"({', '.join(bits)}).")

    market = row.get("market_spread")
    disagreement = None
    if market is not None and market == market:
        # Both numbers are home-relative and positive when the home side is
        # favoured: over the 2025 season spread_line correlates +0.50 with the
        # actual home margin, and home teams average +6.9 when it is positive
        # against -4.6 when it is negative. Reading the sign the other way made
        # every game on the slate look like a disagreement with the market.
        model_margin = float(row["model_margin"])
        gap = model_margin - float(market)
        market_pick = home if float(market) > 0 else away
        if market_pick != pick:
            disagreement = (
                f"We are off the board here — the line has "
                f"{TEAM_NAMES.get(market_pick, market_pick)}. Worth knowing why "
                "before acting on either read; the usual explanation is personnel "
                "news the model has not seen.")
        elif abs(gap) >= DISAGREEMENT_POINTS:
            disagreement = (
                f"Same side as the line, {abs(gap):.1f} points stronger about it.")

    return {
        "away": away, "home": home,
        "pick": pick, "pick_name": TEAM_NAMES.get(pick, pick),
        "opponent": other,
        "win_prob": win_prob,
        "confidence": _confidence(p_home),
        "margin": abs(float(row["model_margin"])),
        "model_total": float(row["model_total"]),
        "market_spread": market,
        "reasons": reasons,
        "disagreement": disagreement,
        "spread": spread_call(row),
        "total": total_call(row),
        "headline": _headline(pick, other, win_prob, abs(float(row["model_margin"]))),
    }


def _headline(pick: str, other: str, win_prob: float, margin: float) -> str:
    name, opp = TEAM_NAMES.get(pick, pick), TEAM_NAMES.get(other, other)
    conf = _confidence(win_prob)
    if conf == "coin flip":
        return f"{name} by a hair over {opp} — genuinely a coin flip ({win_prob*100:.0f}%)."
    if conf == "lean":
        return f"{name} over {opp}, leaning ({win_prob*100:.0f}%, by {margin:.1f})."
    return f"{name} over {opp}, confidently ({win_prob*100:.0f}%, by {margin:.1f})."


# A receiver working this far downfield or beyond is a defence's deep problem.
DEEP_ADOT = 11.0
SHALLOW_ADOT = 6.5
# Explosive index is a ratio against the league rate, so 1.0 is average.
EXPLOSIVE_NOTABLE = 1.25
# Below this, availability is worth saying out loud rather than burying.
AVAILABILITY_CONCERN = 0.85


def player_read(player: str, game, opponent: str, defenders: pd.DataFrame | None = None,
                explosive: pd.DataFrame | None = None,
                usage_hist: pd.DataFrame | None = None) -> list[str]:
    """Why this projection is what it is, in the language of a position coach.

    Built from the same measured pieces as everything else: his simulated range,
    how his touches have actually gone, what the defence he is facing is built
    to take away, and whether he is expected to be on the field. Where a piece
    is missing it is left out rather than guessed at.
    """
    meta = (game.meta or {}).get(player)
    if meta is None:
        return []
    pos, rank = meta.get("position"), int(meta.get("depth_rank", 1) or 1)
    pid = meta.get("player_id")
    out = []

    # --- role -------------------------------------------------------------
    primary = {"WR": "rec_yards", "TE": "rec_yards", "RB": "rush_yards",
               "FB": "rush_yards", "QB": "pass_yards"}.get(pos, "rec_yards")
    vals = game.stat(player, primary)
    if vals is not None and len(vals):
        mask = game.active_mask(player)
        v = vals[mask] if mask is not None and mask.any() else vals
        mean, ceiling = float(np.mean(v)), float(np.percentile(v, 90))
        seat = {1: "the primary", 2: "the second", 3: "the third"}.get(rank, f"the {rank}th")
        label = primary.replace("_", " ")
        out.append(
            f"He is {seat} {pos} in this offence and the range is wide: "
            f"{mean:.0f} {label} in the middle, {ceiling:.0f} at the top of it. "
            f"The gap between those two is the whole question.")

    # --- how he gets his yards -------------------------------------------
    if explosive is not None and not explosive.empty and pid:
        kind = "rush" if pos in ("RB", "FB") else "rec"
        row = explosive[(explosive["player_id"] == pid) & (explosive["kind"] == kind)]
        if not row.empty:
            idx = float(row.iloc[0]["explosive_index"])
            longest = row.iloc[0].get("longest")
            if idx >= EXPLOSIVE_NOTABLE:
                out.append(
                    f"He gets them in chunks — his touches break for twenty-plus "
                    f"at {idx:.1f}x the league rate"
                    + (f", with a long of {int(longest)}" if longest == longest else "")
                    + ". That is the profile that wins a game on one snap.")
            elif idx <= 0.75:
                out.append(
                    f"He grinds rather than breaks — explosive touches at {idx:.1f}x "
                    "the league rate. Volume is the path to a big day, not one play.")

    # --- what he is running into -----------------------------------------
    if defenders is not None and not defenders.empty and opponent:
        load = coverage_shape(defenders, opponent)
        if load:
            adot = _player_adot(usage_hist, pid)
            if adot is not None and adot >= DEEP_ADOT and load["deep_share"] >= 0.40:
                out.append(
                    f"He works deep ({adot:.1f} yards downfield) into a defence that "
                    f"spends {load['deep_share']*100:.0f}% of its charted coverage "
                    "there. This is the matchup, and it is not a soft one.")
            elif adot is not None and adot <= SHALLOW_ADOT and load["underneath_share"] >= 0.35:
                out.append(
                    f"He lives underneath ({adot:.1f} yards) against a defence built "
                    f"to sit on it — {load['underneath_share']*100:.0f}% of their "
                    "charted coverage is short. Expect the catches, not the yards.")
            else:
                out.append(
                    f"{TEAM_NAMES.get(opponent, opponent)} run their coverage through "
                    f"{load['most_targeted']} more than anyone and face "
                    f"{load['mean_adot_allowed']:.1f} yards of average target depth.")

    # --- is he going to play ---------------------------------------------
    p_active = float(meta.get("p_active", 1.0))
    if p_active < AVAILABILITY_CONCERN:
        out.append(
            f"Availability is the caveat: {p_active*100:.0f}% to dress, so the "
            "unconditional number is lower than the one shown.")
    return out


def coverage_shape(defenders: pd.DataFrame, team: str) -> dict:
    """A defence's coverage distribution, without importing the defenders module."""
    from .defenders import coverage_load
    return coverage_load(defenders, team)


def _player_adot(usage_hist: pd.DataFrame | None, player_id: str | None) -> float | None:
    if usage_hist is None or usage_hist.empty or not player_id:
        return None
    # player_usage names it rec_adot; accept either so this keeps working if
    # that ever changes. Without the right name the depth matchup - the most
    # useful line in the read - silently never fires.
    col = next((c for c in ("rec_adot", "adot") if c in usage_hist.columns), None)
    if col is None or "player_id" not in usage_hist.columns:
        return None
    d = usage_hist[usage_hist["player_id"] == player_id][col].dropna()
    return float(d.mean()) if len(d) else None


def spread_call(row: pd.Series) -> dict | None:
    """Which side of the posted number the model's own number takes.

    Both figures are home-relative and positive when the home side is favoured,
    so the difference between them is simply which way the model leans. Reported
    in the notation a book uses - a side and a price - because that is the
    question being asked, and reported alongside the fact that this is the one
    thing the model was measured doing badly.
    """
    market = row.get("market_spread")
    if market is None or market != market:
        return None
    market = float(market)
    home, away = row["home"], row["away"]
    edge = float(row["model_margin"]) - market

    if edge > 0:                       # model likes the home side
        side, line = home, -market
    else:
        side, line = away, market
    return {
        "side": side,
        "side_name": TEAM_NAMES.get(side, side),
        "line": line,
        "label": f"{side} {line:+.1f}",
        "edge_points": abs(edge),
        "model_margin": float(row["model_margin"]),
        "market_spread": market,
    }


def total_call(row: pd.Series) -> dict | None:
    """Over or under the posted total, and by how much the model differs."""
    market = row.get("market_total")
    if market is None or market != market:
        return None
    market, model = float(market), float(row["model_total"])
    edge = model - market
    return {
        "side": "Over" if edge > 0 else "Under",
        "line": market,
        "label": f"{'Over' if edge > 0 else 'Under'} {market:.1f}",
        "edge_points": abs(edge),
        "model_total": model,
        # This was once flagged as the cut worth watching. It was watched, it
        # went 10-30 live, and the flag is retired - so the field is kept only
        # so a caller can say the disagreement is large, never that it is good.
        # ``TOTALS_BAND_RETIRED`` is what a caller should read before dressing
        # this up as a signal.
        "large_disagreement": abs(edge) >= TOTALS_MIN_EDGE,
        "band_retired": TOTALS_BAND_RETIRED,
    }


def slate_calls(slate: pd.DataFrame, plans: dict | None = None,
                x_factors: dict | None = None,
                ratings: pd.DataFrame | None = None) -> list[dict]:
    """Every game on a week, called and explained, most confident first."""
    if slate is None or slate.empty:
        return []
    plans, x_factors = plans or {}, x_factors or {}
    calls = []
    for _, row in slate.iterrows():
        key = (row["away"], row["home"])
        calls.append(game_call(row, plans.get(key), x_factors.get(key), ratings))
    return sorted(calls, key=lambda c: -c["win_prob"])
