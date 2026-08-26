"""NFL projection model - yardage, touchdowns and scheme intelligence for 2026."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from nflproj import board, pipeline, playbook, projections as pj, report, schemes, usage as um
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
    return ctx, pm, usage_hist, sampler, lg_off, lg_def, anchor


ctx, pm, usage_hist, sampler, lg_off, lg_def, anchor = load()

st.title("🏈 NFL Projection Model")
st.caption(
    f"Projecting {PROJECTION_SEASON} from {anchor} and earlier · play-by-play and FTN charting via nflverse · "
    "scheme carried across coaching changes, constrained by personnel"
)

tab_board, tab_scout, tab_matchup, tab_scheme, tab_method = st.tabs(
    ["Projection board", "Team scouting", "Matchup", "Scheme explorer", "Method & limits"]
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
    if team not in gcs:
        st.warning(f"{team} is on bye in week {week}.")
    else:
        gc = gcs[team]
        res = board.project_team(team, ctx, pm, usage_hist, sampler, gc, lg_def,
                                 n_sims=int(n_sims), seed=int(week))
        scheme = pm[team]["offense"]["projected"]
        vol = pj.team_volume(scheme, gc, pm.get(gc.opponent, {}).get("offense", {}).get("projected"))

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
