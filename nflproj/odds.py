"""Real book prices, and what they turn confidence into.

Every pick this model produces has carried the same caveat: there is no odds
feed, so "best" can only mean "what the model is most sure of", which is not
the same as value. A 90% leg at -1200 is a bad bet. This module is the missing
half - it takes prices from a book, lines them up against the simulation, and
turns a probability into expected value and a stake.

Three jobs, and the second is the one that actually breaks.

**Getting prices.** Two routes, deliberately. A file loader that reads whatever
you can export - CSV or JSON - which works with no key, no network and no
dependency on one book staying free. And a client for The Odds API, which is
the usual source for NFL player props, used when a key is present. The file
route is first because it always works; the API is a convenience, not a
foundation.

**Matching names.** A book says "Kenneth Walker III", the model says "Kenneth
Walker". A book says "Marvin Harrison Jr.", the roster says "Marvin Harrison
Jr." this year and "Marvin Harrison" last. Get this wrong and you do not get an
error, you get a confident price attached to the wrong player - the worst
failure mode available, because it looks like it worked. So matching is
conservative: normalise hard, require the team to agree where a team is given,
and **refuse an ambiguous match rather than guess**. Every rejection is
reported, because a silent 60% match rate looks identical to a working one.

**Pricing.** Once a price is attached, the arithmetic already exists in
``picks``: implied probability, expected value, Kelly. What this adds is doing
it across a whole board and sorting by edge rather than by confidence.

One honest note on what any of it is worth. The model's own calibration is the
ceiling: receiving yards correlate 0.55-0.57 with what happens, touchdowns land
within about a point. A positive expected value computed on top of a projection
that is 10% off is a positive expected value in a spreadsheet only. Treat the
number as a filter for which of your own ideas to look at, not as a signal to
follow blindly.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TEAMS, normalize_team
from .picks import (american_to_probability, expected_value, kelly_fraction,
                    probability_to_american)

log = logging.getLogger(__name__)

# The canonical shape everything is normalised into. A loader's only job is to
# produce this; nothing downstream knows which book a row came from.
ODDS_COLUMNS = ["player", "team", "market", "line", "over_odds", "under_odds",
                "book", "fetched_at"]

# Book market keys -> the model's statistic names. The Odds API spellings on
# the left; most books that export CSV use something close to one of these.
MARKET_ALIASES = {
    "player_rush_yds": "rush_yards", "player_rushing_yards": "rush_yards",
    "rush_yds": "rush_yards", "rushing yards": "rush_yards",
    "player_reception_yds": "rec_yards", "player_receiving_yards": "rec_yards",
    "rec_yds": "rec_yards", "receiving yards": "rec_yards",
    "player_receptions": "receptions", "receptions": "receptions",
    "player_pass_yds": "pass_yards", "player_passing_yards": "pass_yards",
    "pass_yds": "pass_yards", "passing yards": "pass_yards",
    "player_pass_tds": "pass_td", "passing tds": "pass_td",
    "player_pass_attempts": "attempts", "player_pass_completions": "completions",
    "player_pass_interceptions": "interceptions",
    "player_rush_attempts": "carries", "carries": "carries",
    "player_anytime_td": "total_td", "anytime td": "total_td",
    "player_rush_reception_yds": "scrimmage_yards",
    "scrimmage yards": "scrimmage_yards",
}

# Suffixes a book may or may not print. Stripped from both sides before
# comparing, never used to tell two players apart.
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

THE_ODDS_API = "https://api.the-odds-api.com/v4"
DEFAULT_MARKETS = ("player_rush_yds", "player_reception_yds", "player_receptions",
                   "player_pass_yds", "player_anytime_td")


# ------------------------------------------------------------------ names
def normalize_name(name: str) -> str:
    """A player's name reduced to what two sources can agree on.

    Lowercased, punctuation dropped, generational suffix removed. "Kenneth
    Walker III", "Kenneth Walker" and "kenneth  walker" all land on the same
    string; "Michael Carter" and "Michael Carter II" deliberately do too, which
    is why matching also checks the team and refuses ties.
    """
    if not isinstance(name, str):
        return ""
    s = name.lower().replace("&", " and ")
    s = re.sub(r"[.'`’]", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    parts = [p for p in s.split() if p]
    while parts and parts[-1] in NAME_SUFFIXES:
        parts.pop()
    return " ".join(parts)


def _initial_last(name: str) -> str:
    """"K.Walker" form, for the second matching pass."""
    parts = normalize_name(name).split()
    if len(parts) < 2:
        return ""
    return f"{parts[0][:1]}.{parts[-1]}"


def match_players(odds: pd.DataFrame, roster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach each priced row to a simulated player, or refuse to.

    ``roster`` needs ``player`` and ``team``. Returns the matched rows with a
    ``model_player`` column, and the rejects with a ``reason`` - because a
    matcher that silently drops a third of the board looks exactly like one that
    works, and the only way to tell is to count what it threw away.

    Two passes. Exact on the normalised name, then first-initial-and-surname.
    Both require the team to agree when the book supplies one. A name that
    matches more than one roster entry is rejected rather than guessed.
    """
    if odds is None or odds.empty:
        return pd.DataFrame(), pd.DataFrame()
    if roster is None or roster.empty:
        return pd.DataFrame(), odds.assign(reason="no roster to match against")

    r = roster.copy()
    r["_norm"] = r["player"].map(normalize_name)
    r["_init"] = r["player"].map(_initial_last)
    r["_team"] = r["team"].map(normalize_team) if "team" in r else None

    matched, rejected = [], []
    for row in odds.to_dict("records"):
        want_team = normalize_team(row["team"]) if row.get("team") else None
        for key, col in (("_norm", normalize_name), ("_init", _initial_last)):
            probe = col(row.get("player", ""))
            if not probe:
                continue
            hits = r[r[key] == probe]
            if want_team is not None and "_team" in hits:
                narrowed = hits[hits["_team"] == want_team]
                # Only trust the team filter when it finds something; books
                # disagree about which team a just-traded player is on.
                hits = narrowed if len(narrowed) else hits
            if len(hits) == 1:
                matched.append({**row, "model_player": hits.iloc[0]["player"]})
                break
            if len(hits) > 1:
                rejected.append({**row, "reason":
                                 f"ambiguous: {len(hits)} roster entries match"})
                break
        else:
            rejected.append({**row, "reason": "no roster entry with that name"})
            continue
        if matched and matched[-1].get("model_player") is None:
            rejected.append({**row, "reason": "no match"})
    return pd.DataFrame(matched), pd.DataFrame(rejected)


# ---------------------------------------------------------------- loading
def _finalize(df: pd.DataFrame, book: str | None = None) -> pd.DataFrame:
    """Coerce any loader's output into the canonical schema."""
    if df is None or df.empty:
        return pd.DataFrame(columns=ODDS_COLUMNS)
    d = df.copy()
    d.columns = [str(c).strip().lower().replace(" ", "_") for c in d.columns]
    rename = {"name": "player", "player_name": "player", "athlete": "player",
              "stat": "market", "market_key": "market", "prop": "market",
              "over": "over_odds", "under": "under_odds",
              "over_price": "over_odds", "under_price": "under_odds",
              "point": "line", "threshold": "line", "total": "line"}
    d = d.rename(columns={k: v for k, v in rename.items() if k in d.columns})
    for c in ODDS_COLUMNS:
        if c not in d.columns:
            d[c] = np.nan
    d["market"] = (d["market"].astype(str).str.strip().str.lower()
                   .map(lambda m: MARKET_ALIASES.get(m, m)))
    for c in ("line", "over_odds", "under_odds"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if book:
        d["book"] = d["book"].fillna(book)
    d["fetched_at"] = d["fetched_at"].fillna(datetime.now(timezone.utc).isoformat())
    d["team"] = d["team"].map(lambda t: normalize_team(t) if isinstance(t, str) else t)
    # A row with no line and no price cannot be used for anything.
    d = d[d["player"].notna() & d["market"].notna()]
    keep = d["over_odds"].notna() | d["under_odds"].notna()
    return d[keep][ODDS_COLUMNS].reset_index(drop=True)


def load_file(path: str | Path, book: str | None = None) -> pd.DataFrame:
    """Read prices exported from anywhere: CSV, TSV or JSON.

    The route that always works. No key, no network, no dependence on one
    provider's free tier surviving the season.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no odds file at {p}")
    if p.suffix.lower() in (".json", ".ndjson"):
        raw = json.loads(p.read_text())
        df = pd.json_normalize(raw if isinstance(raw, list) else raw.get("data", raw))
    else:
        df = pd.read_csv(p, sep=None, engine="python")
    return _finalize(df, book or p.stem)


def parse_the_odds_api(payload: list) -> pd.DataFrame:
    """Flatten The Odds API's nested event/bookmaker/market/outcome shape.

    Separated from the HTTP call so it can be tested against a saved response
    without a key or a network.
    """
    rows = []
    for event in payload or []:
        home = event.get("home_team")
        away = event.get("away_team")
        for bk in event.get("bookmakers", []) or []:
            for mkt in bk.get("markets", []) or []:
                key = MARKET_ALIASES.get(mkt.get("key", ""), mkt.get("key", ""))
                sides = {}
                for out in mkt.get("outcomes", []) or []:
                    who = out.get("description") or out.get("participant")
                    if not who:
                        continue
                    slot = sides.setdefault((who, out.get("point")), {})
                    name = str(out.get("name", "")).lower()
                    if name.startswith("over") or name == "yes":
                        slot["over_odds"] = out.get("price")
                    elif name.startswith("under") or name == "no":
                        slot["under_odds"] = out.get("price")
                for (who, point), prices in sides.items():
                    rows.append({"player": who, "team": None, "market": key,
                                 "line": point, "book": bk.get("title"),
                                 "fetched_at": bk.get("last_update"),
                                 "event": f"{away} @ {home}", **prices})
    return _finalize(pd.DataFrame(rows))


def fetch_the_odds_api(api_key: str | None = None, markets=DEFAULT_MARKETS,
                       regions: str = "us", timeout: int = 20) -> pd.DataFrame:
    """Live prices, when a key is available and the host is reachable.

    Returns an empty frame rather than raising when there is no key, so a
    caller can always try this and fall back to a file.
    """
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        log.info("no ODDS_API_KEY set; skipping the live odds fetch")
        return pd.DataFrame(columns=ODDS_COLUMNS)
    import requests

    try:
        events = requests.get(
            f"{THE_ODDS_API}/sports/americanfootball_nfl/events",
            params={"apiKey": key}, timeout=timeout)
        events.raise_for_status()
        out = []
        for ev in events.json():
            r = requests.get(
                f"{THE_ODDS_API}/sports/americanfootball_nfl/events/{ev['id']}/odds",
                params={"apiKey": key, "regions": regions,
                        "markets": ",".join(markets), "oddsFormat": "american"},
                timeout=timeout)
            if r.status_code == 200:
                out.append(r.json())
        return parse_the_odds_api(out)
    except Exception as exc:
        log.warning("live odds fetch failed (%s); use load_file instead", exc)
        return pd.DataFrame(columns=ODDS_COLUMNS)


# ---------------------------------------------------------------- pricing
def price_board(board: pd.DataFrame, odds: pd.DataFrame,
                player_col: str = "player", stat_col: str = "stat",
                line_col: str = "line", prob_col: str = "probability",
                side_col: str | None = "side") -> pd.DataFrame:
    """Attach real prices to a board of model picks and compute the edge.

    Matches on player, market and line. A book's line rarely sits exactly where
    the model generated one, so only an exact line match is priced - shading a
    price across half a point is how a 2% edge gets invented.
    """
    if board is None or board.empty or odds is None or odds.empty:
        out = (board.copy() if board is not None else pd.DataFrame())
        for c in ("price", "ev_per_100", "kelly", "prob_edge", "book"):
            out[c] = np.nan
        return out

    idx = {}
    for r in odds.to_dict("records"):
        who = r.get("model_player") or r.get("player")
        idx[(normalize_name(who), r["market"], float(r["line"]))] = r

    price, ev, kel, edge, bk = [], [], [], [], []
    for r in board.to_dict("records"):
        key = (normalize_name(r.get(player_col, "")), r.get(stat_col),
               float(r.get(line_col, np.nan)) if r.get(line_col) == r.get(line_col) else np.nan)
        hit = idx.get(key)
        side = str(r.get(side_col, "over")).lower() if side_col else "over"
        p = hit.get("under_odds" if side.startswith("under") else "over_odds") if hit else None
        if hit is None or p is None or p != p:
            price.append(np.nan); ev.append(np.nan); kel.append(np.nan)
            edge.append(np.nan); bk.append(None)
            continue
        prob = float(r.get(prob_col, np.nan))
        price.append(float(p))
        ev.append(expected_value(prob, float(p)))
        kel.append(kelly_fraction(prob, float(p)))
        edge.append(prob - american_to_probability(float(p)))
        bk.append(hit.get("book"))
    out = board.copy()
    out["price"], out["ev_per_100"] = price, ev
    out["kelly"], out["prob_edge"], out["book"] = kel, edge, bk
    return out


def value_board(games: list, matched: pd.DataFrame, conditional: bool = True,
                min_prob: float = 0.02) -> pd.DataFrame:
    """Ask the simulation about **the book's** lines, not its own.

    This is the right way round, and the first attempt had it backwards. The
    model generates candidate lines from each player's own distribution, so a
    board of them almost never coincides with what a book actually hung - the
    first end-to-end run priced zero rows out of two hundred. Matching on the
    model's lines and then shading a price across half a point would have
    "fixed" it by inventing the edge it was looking for.

    So instead every priced line is taken as given, the simulation is asked for
    the probability at exactly that number, and both sides are costed. The book
    sets the question; the model only answers it.

    Priced conditional on the player being active, which is how a book grades a
    prop - a scratch voids rather than loses.
    """
    if matched is None or matched.empty or not games:
        return pd.DataFrame()

    where = {}
    for g in games:
        for name in g.players:
            where.setdefault(name, g)

    rows = []
    for r in matched.to_dict("records"):
        who = r.get("model_player") or r.get("player")
        game = where.get(who)
        stat, line = r.get("market"), r.get("line")
        if game is None or stat is None or line is None or line != line:
            continue
        samples = game.stat(who, stat)
        if samples is None or len(samples) == 0:
            continue
        if conditional:
            mask = game.active_mask(who)
            samples = samples[mask] if mask is not None and mask.any() else samples
        if len(samples) == 0:
            continue

        line = float(line)
        p_over = float((samples > line).mean())
        meta = game.meta.get(who, {})
        for side, prob, price in (("Over", p_over, r.get("over_odds")),
                                  ("Under", 1.0 - p_over, r.get("under_odds"))):
            if price is None or price != price:
                continue
            if not (min_prob <= prob <= 1.0 - min_prob):
                continue        # a near-certainty is where the model's error is largest
            price = float(price)
            rows.append({
                "player": who, "team": meta.get("team"),
                "position": meta.get("position"), "market": stat,
                "line": line, "side": side,
                "projection": float(np.mean(samples)),
                "probability": prob,
                "fair_odds": probability_to_american(prob),
                "price": price,
                "implied": american_to_probability(price),
                "prob_edge": prob - american_to_probability(price),
                "ev_per_100": expected_value(prob, price),
                "kelly": kelly_fraction(prob, price),
                "p_active": float(meta.get("p_active", 1.0)),
                "book": r.get("book"),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ev_per_100", ascending=False).reset_index(drop=True)


def best_value(priced: pd.DataFrame, min_ev: float = 0.0,
               max_kelly: float = 0.05, top_n: int = 25) -> pd.DataFrame:
    """The priced board sorted by edge rather than by confidence.

    ``max_kelly`` caps the suggested stake. Full Kelly on a projection this
    model's accuracy is a good way to find out how wrong it is, so the cap is
    the default rather than an option someone has to remember.
    """
    if priced is None or priced.empty or "ev_per_100" not in priced:
        return pd.DataFrame()
    d = priced[priced["ev_per_100"].notna() & (priced["ev_per_100"] > min_ev)].copy()
    if d.empty:
        return d
    d["suggested_stake_pct"] = (d["kelly"].clip(lower=0, upper=max_kelly) * 100).round(2)
    return d.sort_values("ev_per_100", ascending=False).head(top_n).reset_index(drop=True)
