"""Independent game predictions: margin, total, and win probability.

The player projections take the market's total and spread as inputs. This module
forms its own opinion instead, so the two can be compared and disagreements
surfaced.

Team strength is estimated by ridge regression on play-level EPA, which
separates a team's own quality from the schedule it happened to face - beating
up on bad defences and losing to good ones look identical in raw points. Ratings
are then regressed toward the mean before being carried into the next season,
with offense and defense treated very differently: measured over 2022-2025,
offensive EPA carries year to year at r = 0.44 while defensive EPA carries at
r = 0.12. A defence is close to a coin flip from one season to the next, and the
model says so rather than projecting last year's ranking forward.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

from . import venues as V
from .config import TEAMS, normalize_team

# Year-over-year persistence of opponent-adjusted EPA, measured 2022-2025.
# These are the shrinkage weights applied when carrying a rating forward.
PERSISTENCE = {"offense": 0.44, "defense": 0.12}

# Ridge penalty. Chosen so a team-season of ~1,000 plays moves meaningfully off
# zero without letting a small sample dominate.
RIDGE_LAMBDA = 90.0

# Recency weighting across the seasons feeding the rating.
SEASON_HALFLIFE = 1.1


@dataclass
class GamePrediction:
    home: str
    away: str
    home_points: float
    away_points: float
    home_win_prob: float
    margin: float          # positive favours the home team
    total: float
    market_spread: float | None = None
    market_total: float | None = None

    @property
    def spread_edge(self) -> float | None:
        """Model margin minus the market's. Positive means the model likes home."""
        if self.market_spread is None:
            return None
        return self.margin - self.market_spread

    @property
    def total_edge(self) -> float | None:
        if self.market_total is None:
            return None
        return self.total - self.market_total


def adjusted_ratings(plays: pd.DataFrame, seasons=None) -> pd.DataFrame:
    """Opponent-adjusted offensive and defensive EPA per play, by team-season.

    Solves ``epa ~ offense[team] + defense[opponent] + home`` by ridge
    regression over every play, so a rating reflects performance against the
    schedule actually faced.
    """
    df = plays.dropna(subset=["epa", "posteam", "defteam"]).copy()
    if seasons is not None:
        df = df[df["season"].isin(seasons)]
    if df.empty:
        return pd.DataFrame()

    out = []
    for season, grp in df.groupby("season"):
        teams = sorted(set(grp["posteam"]) | set(grp["defteam"]))
        idx = {t: i for i, t in enumerate(teams)}
        n, k = len(grp), len(teams)

        rows = np.arange(n)
        off_col = grp["posteam"].map(idx).to_numpy()
        def_col = grp["defteam"].map(idx).to_numpy() + k
        home = (grp["posteam"] == grp.get("home_team", grp["posteam"])).astype(float).to_numpy()

        # Design matrix: one offense column and one defense column per team,
        # plus a home-field term.
        data = np.concatenate([np.ones(n), np.ones(n), home])
        r = np.concatenate([rows, rows, rows])
        c = np.concatenate([off_col, def_col, np.full(n, 2 * k)])
        X = sparse.csr_matrix((data, (r, c)), shape=(n, 2 * k + 1))
        y = grp["epa"].to_numpy(dtype=float)

        sol = lsqr(X, y, damp=np.sqrt(RIDGE_LAMBDA), atol=1e-8, btol=1e-8, iter_lim=800)[0]
        for t, i in idx.items():
            out.append({
                "season": int(season), "team": t,
                # Defensive coefficient is EPA allowed: lower is better, so flip
                # the sign to make "higher is better" hold on both sides.
                "off_rating": float(sol[i]),
                "def_rating": float(-sol[i + k]),
                "plays": int((grp["posteam"] == t).sum()),
            })
    return pd.DataFrame(out)


def project_ratings(ratings: pd.DataFrame, anchor_season: int,
                    coach_penalty: dict[str, float] | None = None) -> pd.DataFrame:
    """Carry ratings into the next season, regressed toward the mean.

    ``coach_penalty`` optionally shrinks a team further toward average when its
    staff has turned over, expressed as a multiplier in [0, 1] applied on top of
    the usual persistence.
    """
    hist = ratings[ratings["season"] <= anchor_season].copy()
    if hist.empty:
        return pd.DataFrame()

    hist["w"] = 0.5 ** ((anchor_season - hist["season"]) / SEASON_HALFLIFE)
    rows = []
    for team, g in hist.groupby("team"):
        w = g["w"].to_numpy()
        rec = {"team": team}
        for side, col in (("offense", "off_rating"), ("defense", "def_rating")):
            blended = float((g[col].to_numpy() * w).sum() / w.sum())
            keep = PERSISTENCE[side]
            if coach_penalty:
                keep *= float(coach_penalty.get(team, 1.0))
            rec[col] = blended * keep      # league mean is ~0 by construction
        rows.append(rec)
    out = pd.DataFrame(rows)
    # Re-centre so the league averages to zero after shrinkage.
    for col in ("off_rating", "def_rating"):
        out[col] = out[col] - out[col].mean()
    return out


def fit_scoring_map(plays: pd.DataFrame, games: pd.DataFrame,
                    ratings: pd.DataFrame) -> dict:
    """Calibrate rating differences into points.

    Regresses each team-game's actual points on the rating edge it brought into
    that game, giving the conversion from EPA per play to points on the board.
    """
    pts = pd.concat([
        games.rename(columns={"home_team": "team", "home_score": "points",
                              "away_team": "opp", "away_score": "opp_points"})
             .assign(home=1)[["season", "game_id", "team", "opp", "points", "home"]],
        games.rename(columns={"away_team": "team", "away_score": "points",
                              "home_team": "opp", "home_score": "opp_points"})
             .assign(home=0)[["season", "game_id", "team", "opp", "points", "home"]],
    ]).dropna(subset=["points"])

    r = ratings.rename(columns={"off_rating": "off", "def_rating": "def"})
    m = pts.merge(r[["season", "team", "off"]], on=["season", "team"], how="inner")
    m = m.merge(r[["season", "team", "def"]].rename(columns={"team": "opp", "def": "opp_def"}),
                on=["season", "opp"], how="inner")
    if m.empty:
        return {"intercept": 22.5, "slope": 60.0, "hfa": 1.2}

    m["edge"] = m["off"] - m["opp_def"]
    A = np.column_stack([np.ones(len(m)), m["edge"], m["home"]])
    coef, *_ = np.linalg.lstsq(A, m["points"].to_numpy(dtype=float), rcond=None)
    resid = m["points"].to_numpy() - A @ coef
    out = {"intercept": float(coef[0]), "slope": float(coef[1]),
           "hfa": float(coef[2]), "sigma": float(resid.std(ddof=1))}

    # Margin and total need their own spreads. Deriving them from the per-team
    # residual understates margin variance badly, because the two teams' scores
    # in a game are correlated through pace and script - and that shared
    # component cancels when you subtract. Fit both directly on game outcomes.
    pair = m.pivot_table(index="game_id", columns="home", values="points", aggfunc="first")
    pred = pd.Series(A @ coef, index=m.index)
    m2 = m.assign(_pred=pred)
    pp = m2.pivot_table(index="game_id", columns="home", values=["points", "_pred"], aggfunc="first")
    try:
        act_margin = pp[("points", 1)] - pp[("points", 0)]
        pred_margin = pp[("_pred", 1)] - pp[("_pred", 0)]
        act_total = pp[("points", 1)] + pp[("points", 0)]
        pred_total = pp[("_pred", 1)] + pp[("_pred", 0)]
        out["margin_sigma"] = float((act_margin - pred_margin).std(ddof=1))
        out["total_sigma"] = float((act_total - pred_total).std(ddof=1))
    except KeyError:
        out["margin_sigma"] = out["sigma"] * np.sqrt(2)
        out["total_sigma"] = out["sigma"] * np.sqrt(2)
    return out


def predict_game(home: str, away: str, proj: pd.DataFrame, scoring: dict,
                 n_sims: int = 40000, rng: np.random.Generator | None = None,
                 market_spread: float | None = None,
                 market_total: float | None = None,
                 rest_diff: float = 0.0,
                 env: dict | None = None) -> GamePrediction | None:
    """Predict one game from projected ratings."""
    rng = rng or np.random.default_rng(7)
    r = proj.set_index("team")
    if home not in r.index or away not in r.index:
        return None

    b, s, hfa = scoring["intercept"], scoring["slope"], scoring["hfa"]
    margin_sigma = scoring.get("margin_sigma", 13.1)
    total_sigma = scoring.get("total_sigma", 10.4)

    home_pts = b + s * (r.loc[home, "off_rating"] - r.loc[away, "def_rating"]) + hfa
    away_pts = b + s * (r.loc[away, "off_rating"] - r.loc[home, "def_rating"])
    # Extra rest is worth a fraction of a point per day of advantage.
    home_pts += 0.08 * rest_diff

    # Scoring environment: wind, roof and divisional familiarity move the total
    # without favouring either side, so split the adjustment between them.
    if env and np.isfinite(env.get("total_delta", np.nan)):
        half = float(env["total_delta"]) / 2.0
        home_pts += half
        away_pts += half

    # Simulate margin and total as independent axes with their own calibrated
    # spreads, then recover the two scores. Drawing each team's points
    # separately would understate how far real margins scatter.
    sim_margin = (home_pts - away_pts) + rng.normal(0, margin_sigma, n_sims)
    sim_total = (home_pts + away_pts) + rng.normal(0, total_sigma, n_sims)
    h = (sim_total + sim_margin) / 2.0
    a = (sim_total - sim_margin) / 2.0
    margin = sim_margin

    return GamePrediction(
        home=home, away=away,
        home_points=float(home_pts), away_points=float(away_pts),
        home_win_prob=float((margin > 0).mean() + 0.5 * (margin == 0).mean()),
        margin=float(home_pts - away_pts),
        total=float(home_pts + away_pts),
        market_spread=market_spread, market_total=market_total,
    )


def coaching_penalties(staffs: dict, strength: float = 0.80) -> dict[str, float]:
    """Shrink teams with new play callers further toward the mean.

    A staff change is genuine uncertainty about how good a unit will be, so its
    rating is trusted less. Applied per side via whichever leader is new.
    """
    out = {}
    for team, staff in staffs.items():
        if getattr(staff, "continuity", True):
            continue
        if staff.offense_is_new or staff.defense_is_new:
            out[team] = strength
    return out


def predict_slate(games: pd.DataFrame, season: int, proj: pd.DataFrame,
                  scoring: dict, week: int | None = None,
                  n_sims: int = 20000) -> pd.DataFrame:
    """Predict every game on a season or a single week."""
    g = games[(games["season"] == season) & (games["game_type"] == "REG")].copy()
    if week is not None:
        g = g[g["week"] == week]
    rows = []
    for i, r in enumerate(g.itertuples()):
        home, away = normalize_team(r.home_team), normalize_team(r.away_team)
        spread = float(r.spread_line) if pd.notna(r.spread_line) else None
        total = float(r.total_line) if pd.notna(r.total_line) else None
        rest = (float(r.home_rest) - float(r.away_rest)) if pd.notna(getattr(r, "home_rest", np.nan)) else 0.0
        env = V.environment({"roof": getattr(r, "roof", None), "wind": getattr(r, "wind", None),
                             "div_game": getattr(r, "div_game", 0)})
        p = predict_game(home, away, proj, scoring, n_sims=n_sims,
                         rng=np.random.default_rng(1000 + i),
                         market_spread=spread, market_total=total, rest_diff=rest, env=env)
        if p is None:
            continue
        rows.append({
            "week": int(r.week), "game_id": r.game_id, "away": away, "home": home,
            "away_pts": round(p.away_points, 1), "home_pts": round(p.home_points, 1),
            "model_margin": round(p.margin, 1), "model_total": round(p.total, 1),
            "home_win_pct": round(p.home_win_prob * 100, 1),
            "market_spread": spread, "market_total": total,
            "spread_edge": round(p.spread_edge, 1) if p.spread_edge is not None else None,
            "total_edge": round(p.total_edge, 1) if p.total_edge is not None else None,
            "wind": getattr(r, "wind", None), "roof": getattr(r, "roof", None),
            "divisional": bool(getattr(r, "div_game", 0) == 1),
            "env_pts": round(env["total_delta"], 2),
            "travel_miles": round(V.haversine(away, home)),
            "tz_shift": V.timezone_shift(away, home),
        })
    return pd.DataFrame(rows)


# How quickly current-season evidence should displace the preseason prior. A
# team-season reaches roughly this many plays by mid-season, at which point the
# current year carries about half the weight.
INSEASON_HALF_PLAYS = 420.0


def progressive_ratings(plays: pd.DataFrame, season: int, through_week: int,
                        prior_proj: pd.DataFrame) -> pd.DataFrame:
    """Blend the preseason projection with what has happened so far this year.

    Predicting a Week 12 game from preseason ratings alone throws away eleven
    weeks of evidence, which is most of what a closing line knows. Current-season
    ratings are computed from games already played and blended against the
    preseason prior by how much has been seen.
    """
    played = plays[(plays["season"] == season) & (plays["week"] < through_week)]
    if played.empty or through_week <= 1:
        return prior_proj.copy()

    cur = adjusted_ratings(played.assign(season=season))
    if cur.empty:
        return prior_proj.copy()
    cur = cur.set_index("team")

    rows = []
    for r in prior_proj.itertuples():
        team = r.team
        if team not in cur.index:
            rows.append({"team": team, "off_rating": r.off_rating, "def_rating": r.def_rating})
            continue
        n = float(cur.loc[team, "plays"])
        w = n / (n + INSEASON_HALF_PLAYS)     # weight on the current season
        rows.append({
            "team": team,
            "off_rating": (1 - w) * r.off_rating + w * float(cur.loc[team, "off_rating"]),
            "def_rating": (1 - w) * r.def_rating + w * float(cur.loc[team, "def_rating"]),
        })
    out = pd.DataFrame(rows)
    for col in ("off_rating", "def_rating"):
        out[col] = out[col] - out[col].mean()
    return out
