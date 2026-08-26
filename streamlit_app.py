"""NFL projection model - yardage, touchdowns and scheme intelligence for 2026."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from nflproj import (board, gamemodel as gm, pipeline, playbook,
                     projections as pj, report, schemes, usage as um, venues as V)
from nflproj.config import PROJECTION_SEASON, TEAM_NAMES, TEAMS

st.set_page_config(page_title="NFL Projection Model", page_icon="🏈", layout="wide")

YARD_LINES = {
    "rec_yards": [29.5, 39.5, 49.5, 59.5, 69.5, 79.5],
    "rush_yards": [29.5, 39.5, 49.5, 59.5, 69.5, 79.5],
    "scrimmage_yards": [39.5, 49.5, 59.5, 69.5, 79.5, 99.5],
    "pass_yards": [199.5, 224.5, 249.5, 274.5, 299.5],
}


@st.cache_resource(show_spinner="Loading play-by-play, charting and depth charts…")
def load():
    ctx = pipeline.build_context()
    pm = pipeline.project_team_schemes(ctx)
    usage_hist = um.player_usage(ctx.plays)
    sampler = pj.TouchSampler(ctx.plays)
    anchor = int(ctx.fingerprints["season"].max())
    lg_off = schemes.league_means(ctx.fingerprints, "offense", anchor)
    lg_def = schemes.league_means(ctx.fingerprints, "defense", anchor)
    ratings = gm.adjusted_ratings(ctx.plays)
    penalties = gm.coaching_penalties(ctx.staffs)
    team_proj = gm.project_ratings(ratings, anchor_season=anchor, coach_penalty=penalties)
    scoring = gm.fit_scoring_map(ctx.plays, ctx.games[ctx.games["season"] <= anchor], ratings)
    return ctx, pm, usage_hist, sampler, lg_off, lg_def, anchor, ratings, team_proj, scoring


ctx, pm, usage_hist, sampler, lg_off, lg_def, anchor, ratings, team_proj, scoring = load()

st.title("🏈 NFL Projection Model")
st.caption(
    f"Projecting {PROJECTION_SEASON} from {anchor} and earlier · play-by-play and FTN charting via nflverse · "
    "scheme carried across coaching changes, constrained by personnel"
)

tab_board, tab_games, tab_scout, tab_matchup, tab_scheme, tab_method = st.tabs(
    ["Projection board", "Game predictions", "Team scouting", "Matchup",
     "Scheme explorer", "Method & limits"]
)

# ---------------------------------------------------------------- projections
with tab_board:
    c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1.2])
    team = c1.selectbox("Team", TEAMS, index=TEAMS.index("NYG"),
                        format_func=lambda t: f"{t} — {TEAM_NAMES[t]}")
    weeks = sorted(board.schedule_for(ctx.games, PROJECTION_SEASON)["week"].unique())
    week = c2.selectbox("Week", weeks, index=0)
    view = c3.radio("View", ["Expected", "If active"], horizontal=True,
                    help="Expected prices in the chance a player is inactive. "
                         "If active shows the line assuming he dresses and plays.")
    n_sims = c4.select_slider("Simulations", [2000, 5000, 10000, 20000, 40000], value=10000)

    gcs = board.game_contexts(ctx.games, PROJECTION_SEASON, int(week))
    envs = board.game_environments(ctx.games, PROJECTION_SEASON, int(week))
    if team not in gcs:
        st.warning(f"{team} is on bye in week {week}.")
    else:
        gc = gcs[team]
        env = dict(envs.get(team, {}))

        sched_row = board.schedule_for(ctx.games, PROJECTION_SEASON, int(week))
        row = sched_row[(sched_row.home_team == team) | (sched_row.away_team == team)]
        row = row.iloc[0] if len(row) else None

        with st.expander("Conditions and travel", expanded=False):
            e1, e2 = st.columns([1, 1.4])
            known_wind = row is not None and pd.notna(row.get("wind"))
            wind_val = float(row.get("wind")) if known_wind else 0.0
            forecast = e1.slider(
                "Wind (mph)", 0, 35, int(wind_val),
                help="Weather is only published once a game is played, so a season-ahead "
                     "projection has none. Enter a forecast to see its effect.",
            )
            roof = row.get("roof") if row is not None else None
            div = int(row.get("div_game", 0)) if row is not None else 0
            env = V.environment({"roof": roof, "wind": float(forecast), "div_game": div})

            if row is not None:
                away, home = row["away_team"], row["home_team"]
                tv = V.travel_context(away, home, V.parse_kickoff(row.get("gametime")))
                bc = tv.get("away_body_clock")
                e2.markdown(
                    f"**Venue** {roof or 'unknown'} · "
                    f"**Divisional** {'yes' if div else 'no'}  \n"
                    f"**{away} travel** {tv['travel_miles']:,.0f} miles, "
                    f"{abs(tv['tz_shift'])} time zone(s) {tv['direction'] if tv['tz_shift'] else ''}"
                    + (f" · body clock {bc:.0f}:00" if bc is not None and pd.notna(bc) else "")
                    + f"  \n**Scoring environment** {env['total_delta']:+.2f} pts"
                )
                e2.caption(
                    "Travel is shown for context only. Distance, time zones and body clock "
                    "were tested against 7,276 games and are **not** applied — the "
                    "east-to-west effect was worth -2.8 points against the spread in "
                    "1999-2009 and has since decayed to zero. Wind, roof and divisional "
                    "status **are** applied."
                )

        res = board.project_team(team, ctx, pm, usage_hist, sampler, gc, lg_def,
                                 n_sims=int(n_sims), seed=int(week), env=env)
        scheme = pm[team]["offense"]["projected"]
        vol = pj.team_volume(scheme, gc,
                             pm.get(gc.opponent, {}).get("offense", {}).get("projected"), env=env)

        m1, m2, m3, m4, m5 = st.columns(5)
        opp = gc.opponent or "—"
        m1.metric("Matchup", f"{'vs' if gc.is_home else '@'} {opp}")
        m2.metric("Implied points", f"{gc.implied_points:.1f}",
                  help="From the market total and spread" if not gc.neutral else "No line posted; generic game")
        m3.metric("Plays", f"{vol['plays']:.0f}")
        m4.metric("Pass attempts", f"{vol['attempts']:.0f}")
        m5.metric("Expected off. TDs", f"{vol['expected_off_td']:.2f}")

        suffix = "_if_active" if view == "If active" else ""
        df = board.board_frame(res)

        def col(name):
            c = f"{name}{suffix}" if f"{name}{suffix}" in df.columns else f"{name}_mean"
            return c if c in df.columns else None

        skill = df[df["pos"] != "QB"].copy()
        show = {
            "player": "Player", "pos": "Pos", "rank": "Dpt", "p_active": "Active %",
            col("targets"): "Tgt", col("receptions"): "Rec", col("rec_yards"): "Rec yds",
            col("carries"): "Car", col("rush_yards"): "Rush yds",
            col("scrimmage_yards"): "Scrim yds",
            ("anytime_td_pct_if_active" if suffix else "anytime_td_pct"): "Anytime TD %",
        }
        show = {k: v for k, v in show.items() if k and k in skill.columns}
        st.subheader("Skill players")
        st.dataframe(
            skill[list(show)].rename(columns=show).round(1),
            hide_index=True, use_container_width=True,
        )

        qbs = df[df["pos"] == "QB"]
        if not qbs.empty:
            st.subheader("Quarterback")
            qshow = {"player": "Player", "p_active": "Active %",
                     col("attempts"): "Att", col("completions"): "Cmp",
                     col("pass_yards"): "Pass yds", col("pass_td"): "Pass TD",
                     col("interceptions"): "INT", col("carries"): "Car",
                     col("rush_yards"): "Rush yds"}
            qshow = {k: v for k, v in qshow.items() if k and k in qbs.columns}
            st.dataframe(qbs[list(qshow)].rename(columns=qshow).round(1),
                         hide_index=True, use_container_width=True)

        st.subheader("Probability of clearing a line")
        stat = st.selectbox("Stat", list(YARD_LINES), index=2,
                            format_func=lambda s: s.replace("_", " ").title())
        rows = []
        for r in res:
            if stat not in r.samples:
                continue
            row = {"Player": r.name, "Pos": r.position,
                   "Projection": float(np.mean(r.samples[stat]))}
            for line in YARD_LINES[stat]:
                p = r.prob_over(stat, line, conditional=(view == "If active"))
                row[f"o{line}"] = p
            rows.append(row)
        if rows:
            pdf = pd.DataFrame(rows).sort_values("Projection", ascending=False)
            st.dataframe(
                pdf.round(1), hide_index=True, use_container_width=True,
                column_config={c: st.column_config.NumberColumn(c, format="%.1f%%")
                               for c in pdf.columns if c.startswith("o")},
            )

# ------------------------------------------------------------- game model
with tab_games:
    st.caption(
        "An independent view of each game, formed from opponent-adjusted EPA "
        "ratings rather than the market. The player board takes the market line "
        "as an input; this does not, so the two can disagree."
    )
    st.warning(
        "**The closing line beats this model.** Over 544 held-out games it predicts "
        "margin to 10.3 points against the market's 9.7, and picks 64.7% of winners "
        "against the market's 68.4%. Betting its disagreements lost money in "
        "backtest (48.7% against the spread, below the 52.4% break-even). Treat a "
        "large edge as a flag that the model is missing something - usually injury "
        "or personnel news - rather than as a signal.",
        icon="⚠️",
    )

    gw = sorted(board.schedule_for(ctx.games, PROJECTION_SEASON)["week"].unique())
    c1, c2 = st.columns([1, 3])
    gweek = c1.selectbox("Week", gw, index=0, key="gm_week")
    slate = gm.predict_slate(ctx.games, PROJECTION_SEASON, team_proj, scoring,
                             week=int(gweek), n_sims=20000)
    if slate.empty:
        st.info("No games scheduled for this week.")
    else:
        show = slate.rename(columns={
            "away": "Away", "home": "Home", "away_pts": "Away pts", "home_pts": "Home pts",
            "model_margin": "Margin", "model_total": "Total", "home_win_pct": "Home win %",
            "market_spread": "Mkt spread", "market_total": "Mkt total",
            "spread_edge": "Spread edge", "total_edge": "Total edge",
            "env_pts": "Env pts", "divisional": "Div", "roof": "Roof",
            "travel_miles": "Away miles", "tz_shift": "TZ",
        })
        st.dataframe(
            show.drop(columns=["week", "game_id"]), hide_index=True, use_container_width=True,
            column_config={"Home win %": st.column_config.NumberColumn(format="%.1f%%")},
        )
        st.caption(
            "Margin is home-relative. Edge is the model minus the market; a positive "
            "spread edge means the model likes the home side more than the market does. "
            "Blank market columns mean no line is posted for that game yet. "
            "**Env pts** is the scoring-environment adjustment applied to the total from "
            "roof, wind and divisional status. **Away miles** and **TZ** are shown for "
            "context and are deliberately not applied — see Method & limits."
        )

    st.subheader(f"Projected {PROJECTION_SEASON} team strength")
    st.caption(
        "Opponent-adjusted EPA per play, regressed toward the mean. Offense carries "
        "year to year at r = 0.44 and defense at only r = 0.12, so defensive ratings "
        "are pulled far harder toward average - a defence is close to a coin flip "
        "from one season to the next. Teams with a new play caller are shrunk further."
    )
    tp = team_proj.copy()
    tp["net"] = tp["off_rating"] + tp["def_rating"]
    tp = tp.sort_values("net", ascending=False)
    tp["new staff"] = tp["team"].map(lambda t: not ctx.staffs[t].continuity)
    st.dataframe(
        tp.rename(columns={"team": "Team", "off_rating": "Offense",
                           "def_rating": "Defense", "net": "Net"}).round(4),
        hide_index=True, use_container_width=True,
    )

# ------------------------------------------------------------------ scouting
with tab_scout:
    t = st.selectbox("Team", TEAMS, index=TEAMS.index("NYG"),
                     format_func=lambda x: f"{x} — {TEAM_NAMES[x]}", key="scout_team")
    rep = report.team_report(t, pm, ctx, lg_off, lg_def, seasons=(anchor,))
    staff = rep["staff"]

    st.header(rep["name"])
    a, b = st.columns(2)
    with a:
        st.subheader("Offense")
        st.write(rep["coaching_note"])
        if rep["personnel_note"]:
            st.write(rep["personnel_note"])
        if not rep["offense_identity"].empty:
            o = rep["offense_identity"][["trait", "display", "league_display", "phrase"]]
            st.dataframe(o.rename(columns={"trait": "Trait", "display": "Projected",
                                           "league_display": "League", "phrase": "Read"}),
                         hide_index=True, use_container_width=True)
    with b:
        st.subheader("Defense")
        st.write(rep["defense_note"])
        if rep["base_front"]:
            st.write(f"Base front: **{rep['base_front']}**")
        if not rep["defense_identity"].empty:
            d = rep["defense_identity"][["trait", "display", "league_display", "phrase"]]
            st.dataframe(d.rename(columns={"trait": "Trait", "display": "Projected",
                                           "league_display": "League", "phrase": "Read"}),
                         hide_index=True, use_container_width=True)

    if rep["notes"]:
        st.info(rep["notes"])

    st.subheader(f"Signature concepts ({anchor})")
    st.caption("Calls this team ran more than the league did, with what they produced. "
               "Lift is usage relative to league rate.")
    con = rep["concepts"]
    if not con.empty:
        st.dataframe(
            con[["concept", "plays", "share", "league_share", "lift", "yards", "epa", "success"]]
            .rename(columns={"concept": "Concept", "plays": "Plays", "share": "Team rate",
                             "league_share": "League rate", "lift": "Lift",
                             "yards": "Yds/play", "epa": "EPA/play", "success": "Success"})
            .round(3), hide_index=True, use_container_width=True)
    else:
        st.caption("Not enough charted plays for a stable concept profile.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Situational tendencies")
        sit = rep["situational"]
        if not sit.empty:
            cols = [c for c in ["situation", "plays", "pass_rate", "shotgun_rate", "motion_rate",
                                "play_action_rate", "epa"] if c in sit.columns]
            st.dataframe(sit[cols].round(3), hide_index=True, use_container_width=True)
    with c2:
        st.subheader("Defensive situational")
        ds = rep["def_situational"]
        if not ds.empty:
            st.dataframe(ds.round(3), hide_index=True, use_container_width=True)

    st.subheader("Run direction")
    rd = rep["run_directions"]
    if not rd.empty:
        st.dataframe(rd.head(12).round(3), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------- matchup
with tab_matchup:
    weeks = sorted(board.schedule_for(ctx.games, PROJECTION_SEASON)["week"].unique())
    wk = st.selectbox("Week", weeks, index=0, key="mu_week")
    sched = board.schedule_for(ctx.games, PROJECTION_SEASON, int(wk))
    if sched.empty:
        st.warning("No games scheduled.")
    else:
        labels = {f"{r.away_team} @ {r.home_team}": (r.away_team, r.home_team)
                  for r in sched.itertuples()}
        pick = st.selectbox("Game", list(labels))
        away, home = labels[pick]
        row = sched[(sched.away_team == away) & (sched.home_team == home)].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", f"{row.total_line}" if pd.notna(row.total_line) else "no line")
        c2.metric("Spread (home)", f"{row.spread_line}" if pd.notna(row.spread_line) else "no line")
        c3.metric("Kickoff", str(row.gameday))

        ra = report.team_report(away, pm, ctx, lg_off, lg_def, seasons=(anchor,))
        rh = report.team_report(home, pm, ctx, lg_off, lg_def, seasons=(anchor,))
        for off, dfn in ((ra, rh), (rh, ra)):
            st.subheader(f"{TEAM_NAMES[off['team']]} offense vs {TEAM_NAMES[dfn['team']]} defense")
            edges = report.matchup_edges(off, dfn)
            if edges:
                for e in edges:
                    st.markdown(f"- {e}")
            else:
                st.caption("No standout structural edge; both sides near league norms.")

# ------------------------------------------------------------ scheme explorer
with tab_scheme:
    side = st.radio("Side", ["offense", "defense"], horizontal=True)
    tbl = pipeline.scheme_table(pm, side)
    st.caption(f"Projected {PROJECTION_SEASON} identity for all 32 teams. "
               "`coach_weight` is how much of each row comes from the play caller's "
               "previous stop rather than team continuity.")
    numeric = tbl.select_dtypes(include=[np.number]).columns.tolist()
    default = [c for c in (schemes.OFFENSE_IDENTITY if side == "offense" else schemes.DEFENSE_IDENTITY)
               if c in numeric][:8]
    chosen = st.multiselect("Columns", numeric, default=default)
    meta = [c for c in ["qb", "coach", "base_front", "new_staff", "confidence", "coach_weight"]
            if c in tbl.columns]
    st.dataframe(tbl[meta + chosen].round(3), use_container_width=True)

# ------------------------------------------------------------------- method
with tab_method:
    st.markdown((pipeline.ROOT / "METHOD.md").read_text() if (pipeline.ROOT / "METHOD.md").exists()
                else "See METHOD.md in the repository.")
