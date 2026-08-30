"""NFL projection model - yardage, touchdowns and scheme intelligence for 2026.

The app is deliberately shallow at the front. Three tabs answer the three
questions people actually arrive with - who is projected for what, what is
worth betting, and give me a parlay - and everything else lives one click
deeper. Nothing was removed to get there; the coordinator-level tools moved
behind a single selector, and every explanation moved into an expander so it
is available without being in the way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import altair as alt

from nflproj import (board, coverage as cvg, data as ndata, gamemodel as gm,
                     defenders as dfn, gameplan as gp, joint as jnt, kicking as kk,
                     leaderboard as lb, routes as rt,
                     lotto, news, parlay as play, picks, pipeline, playbook,
                     projections as pj, report, schemes, usage as um, venues as V)

# Single categorical hue; a bar chart of one measure is one series, so no
# legend is needed and values are labelled directly.
SERIES_1 = "#2a78d6"
INK_MUTED = "#52514e"
INK_GRID = "#e1e0d9"
from nflproj.config import PROJECTION_SEASON, TEAM_NAMES, TEAMS

st.set_page_config(page_title="NFL Projection Model", page_icon="🏈", layout="wide")

YARD_LINES = {
    "rec_yards": [29.5, 39.5, 49.5, 59.5, 69.5, 79.5],
    "rush_yards": [29.5, 39.5, 49.5, 59.5, 69.5, 79.5],
    "scrimmage_yards": [39.5, 49.5, 59.5, 69.5, 79.5, 99.5],
    "pass_yards": [199.5, 224.5, 249.5, 274.5, 299.5],
}

# Plain-English definitions, shown wherever the term appears rather than filed
# away in a glossary nobody opens.
GLOSSARY = {
    "Projection": "The average outcome across every simulation of the game — "
                  "the single number, if you only get one.",
    "Floor": "The 10th percentile: he beats this in nine games out of ten.",
    "Ceiling": "The 90th percentile: he reaches this in one game out of ten. "
               "This is the number that matters for longshots.",
    "Median": "The middle outcome. Below the projection whenever the "
              "distribution is right-skewed, which for yardage it always is.",
    "Active %": "The chance he dresses and plays. Projections are shown "
                "assuming he does, because that is how a book prices a prop — "
                "a scratch voids the bet rather than losing it.",
    "Model %": "The model's probability that the line is cleared.",
    "Fair odds": "The American price at which that probability breaks even, "
                 "with no bookmaker margin added. If a book offers you longer "
                 "than this, the model thinks you have the better of it.",
    "vs typical": "How much likelier this player is to reach a longshot line "
                  "than a generic player projected for the same total.",
    "Correlation lift": "How much likelier the parlay is than multiplying the "
                        "legs together would say. Above 1.00 the legs help each "
                        "other; below 1.00 they fight.",
}


def _glossary(*terms: str) -> None:
    """Explain the columns on screen, without shouting."""
    with st.expander("What do these columns mean?"):
        for t in terms:
            if t in GLOSSARY:
                st.markdown(f"**{t}** — {GLOSSARY[t]}")


def _first_run_sync() -> None:
    """Fetch the nflverse cache when the app starts on an empty container.

    A fresh host has no data directory, so the first visitor would otherwise sit
    through a silent multi-minute download. This shows what is happening and
    only runs when the cache is genuinely missing.
    """
    from nflproj.config import RAW, HISTORY_SEASONS, PROJECTION_SEASON
    if list(RAW.glob("pbp_*.parquet")):
        return
    with st.status("First run: downloading public nflverse data (~140 MB)…",
                   expanded=True) as status:
        st.write("This happens once. Subsequent starts read the local cache.")
        from nflproj import data as ndata_boot
        results = ndata_boot.sync_all(seasons=HISTORY_SEASONS,
                                      projection_season=PROJECTION_SEASON)
        ok = sum(1 for v in results.values() if v)
        status.update(label=f"Downloaded {ok} of {len(results)} datasets.",
                      state="complete", expanded=False)


@st.cache_resource(show_spinner="Loading play-by-play, charting and depth charts…")
def load(through_week: int | None = None):
    ctx = pipeline.build_context(through_week=through_week)
    pm = pipeline.project_team_schemes(ctx)
    usage_hist = um.player_usage(ctx.plays)
    sampler = pj.TouchSampler(ctx.plays)
    anchor = int(ctx.fingerprints["season"].max())
    lg_off = schemes.league_means(ctx.fingerprints, "offense", anchor)
    lg_def = schemes.league_means(ctx.fingerprints, "defense", anchor)
    cov_plays = cvg.attach(ctx.plays, ctx.participation)
    cov_fp = cvg.coverage_fingerprint(cov_plays, seasons=(anchor,))
    ratings = gm.adjusted_ratings(ctx.plays)
    penalties = gm.coaching_penalties(ctx.staffs)
    team_proj = gm.project_ratings(ratings, anchor_season=anchor, coach_penalty=penalties)
    scoring = gm.fit_scoring_map(ctx.plays, ctx.games[ctx.games["season"] <= anchor], ratings)
    return (ctx, pm, usage_hist, sampler, lg_off, lg_def, anchor, ratings,
            team_proj, scoring, cov_plays, cov_fp)


_first_run_sync()

# In-season mode. Once the season is under way, rebuilding from games already
# played is worth far more than any single modelling refinement: usage settles,
# and defensive quality goes from barely knowable (r = 0.11 across seasons) to
# genuinely informative (r = 0.32 within one). It lives in the sidebar because
# it is a setting, not a question anyone needs to answer to use the app.
_avail = ndata.fetch("pbp", PROJECTION_SEASON)
_played = int(_avail["week"].max()) if _avail is not None and len(_avail) else 0
with st.sidebar:
    st.markdown("### Model state")
    if _played:
        _live = st.toggle("In-season mode", value=True,
                          help=f"Rebuild from {PROJECTION_SEASON} games already played "
                               "rather than projecting from prior seasons alone.")
        _wk = st.number_input("Project as of week", 2, 18, min(_played + 1, 18),
                              disabled=not _live)
        _through = int(_wk) if _live else None
        st.caption(f"{PROJECTION_SEASON} play-by-play available through week {_played}.")
    else:
        _through = None
        st.caption(
            f"No {PROJECTION_SEASON} play-by-play published yet, so the model is "
            "running preseason: everything comes from prior seasons and the current "
            "depth charts. In-season mode switches on automatically once games are "
            "played."
        )

(ctx, pm, usage_hist, sampler, lg_off, lg_def, anchor, ratings,
 team_proj, scoring, cov_plays, cov_fp) = load(_through)

if ctx.current_season:
    st.sidebar.success(f"In-season: built through week {ctx.through_week - 1}")

st.title("🏈 NFL Projection Model")
st.caption(
    f"Projecting {PROJECTION_SEASON} from {anchor} and earlier · play-by-play and FTN "
    "charting via nflverse · scheme carried across coaching changes, constrained by personnel"
)

WEEKS = sorted(board.schedule_for(ctx.games, PROJECTION_SEASON)["week"].unique())

# One week selector for the whole app, so changing the week does not mean
# changing it again in every tab.
_wsel = st.sidebar.selectbox("Week", WEEKS, index=0, key="global_week",
                             help="Applies to every tab.")
WEEK = int(_wsel)

(tab_players, tab_picks, tab_parlay, tab_games, tab_deep, tab_learn) = st.tabs(
    ["📊 Players", "🎯 Picks", "🎰 Parlay", "🏈 Games", "🔍 Deep dive", "📖 Learn"]
)


@st.cache_resource(show_spinner="Attaching charted routes…")
def _routed_plays():
    """Charted route on each target, canonicalised across seasons."""
    return rt.attach_routes(ctx.plays, ctx.participation)


@st.cache_resource(show_spinner="Measuring explosive rates…")
def _explosive_profile():
    """How often each player's touches break for twenty yards. Used to shape
    the tail on longshot props, so it is worth computing once."""
    return gp.explosive_profile(ctx.plays)


@st.cache_resource(show_spinner="Simulating the slate…")
def simulate_week(week: int, n_sims: int = 20000):
    """Simulate every game in a week jointly, so correlated legs price right."""
    sched = board.schedule_for(ctx.games, PROJECTION_SEASON, int(week))
    gcs = board.game_contexts(ctx.games, PROJECTION_SEASON, int(week))
    envs = board.game_environments(ctx.games, PROJECTION_SEASON, int(week))
    out = []
    for i, r in enumerate(sched.itertuples()):
        try:
            out.append(jnt.simulate_game(
                r.home_team, r.away_team, ctx, pm, usage_hist, sampler, gcs,
                lg_def, envs=envs, n_sims=n_sims, seed=1000 + i))
        except Exception:
            continue
    return out


def _distribution_chart(values, line: float, title: str):
    """Histogram of a simulated stat with the line marked."""
    d = pd.DataFrame({"value": values})
    hist = (
        alt.Chart(d)
        .mark_bar(color=SERIES_1, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=40), title=title,
                    axis=alt.Axis(grid=False, labelColor=INK_MUTED, titleColor=INK_MUTED)),
            y=alt.Y("count()", title="simulations",
                    axis=alt.Axis(grid=True, labelColor=INK_MUTED, titleColor=INK_MUTED)),
            tooltip=[alt.Tooltip("count()", title="simulations"),
                     alt.Tooltip("value:Q", bin=True, title=title)],
        )
    )
    rule = (alt.Chart(pd.DataFrame({"line": [line]}))
            .mark_rule(color="#0b0b0b", strokeWidth=2, strokeDash=[4, 3])
            .encode(x="line:Q", tooltip=alt.Tooltip("line:Q", title="line")))
    return (hist + rule).properties(height=240)


def _ranked_bar(df: pd.DataFrame, value_col: str, label: str, n: int = 20):
    """Players in a listed order, longest bar first, with the range on it.

    The bar is the projection and the whisker is the 10th-to-90th percentile
    range, so a steady player and a volatile one at the same projection do not
    look identical - which, for anyone shopping for a ceiling, is the whole
    point of looking at a chart instead of a column of numbers.
    """
    d = df.head(n).copy()
    d["label"] = d["player"] + "  (" + d["team"] + ")"
    order = d["label"].tolist()
    base = alt.Chart(d).encode(
        y=alt.Y("label:N", sort=order, title=None,
                axis=alt.Axis(labelColor=INK_MUTED, labelLimit=200)),
    )
    tips = [alt.Tooltip("player:N", title="Player"),
            alt.Tooltip("team:N", title="Team"),
            alt.Tooltip("matchup:N", title="Game"),
            alt.Tooltip(f"{value_col}:Q", title=label, format=".1f"),
            alt.Tooltip("floor:Q", title="Floor", format=".1f"),
            alt.Tooltip("ceiling:Q", title="Ceiling", format=".1f")]
    # Headroom so the value label at the end of each range has somewhere to sit.
    top = float(d["ceiling"].max()) * 1.10
    bars = base.mark_bar(color=SERIES_1, cornerRadiusEnd=4, height=13).encode(
        x=alt.X(f"{value_col}:Q", title=label, scale=alt.Scale(domain=[0, top]),
                axis=alt.Axis(grid=True, gridColor=INK_GRID,
                              labelColor=INK_MUTED, titleColor=INK_MUTED)),
        tooltip=tips,
    )
    # The 10th-to-90th range, drawn as a thin rule above the bar rather than
    # through it: a steady player and a volatile one at the same projection
    # should not look identical, and the two marks should not fight.
    span = base.mark_rule(color="#0b0b0b", strokeWidth=1, opacity=0.35,
                          yOffset=-9).encode(x="floor:Q", x2="ceiling:Q",
                                             tooltip=tips)
    caps = base.mark_tick(color="#0b0b0b", opacity=0.35, thickness=1, size=6,
                          yOffset=-9).encode(x="ceiling:Q")
    text = base.mark_text(align="left", dx=5, color=INK_MUTED, fontSize=11).encode(
        x=f"{value_col}:Q", text=alt.Text(f"{value_col}:Q", format=".0f"))
    return (bars + span + caps + text).properties(height=max(30 * len(d), 120))


# =============================================================== PLAYERS
with tab_players:
    st.subheader(f"Week {WEEK} projections")
    c1, c2, c3 = st.columns([1.2, 1.6, 1])
    group = c1.selectbox("Position", list(picks.POSITION_GROUPS), key="lb_grp")
    positions = picks.POSITION_GROUPS[group]
    stat_menu = lb.stats_for(positions)
    stat_label = c2.selectbox("Ranked by", [lab for _, lab in stat_menu], key="lb_stat")
    stat = dict((lab, k) for k, lab in stat_menu)[stat_label]
    show_n = c3.slider("Players shown", 10, 60, 20, 5, key="lb_n")

    slate = simulate_week(WEEK)
    table = lb.leaderboard(slate, positions, stat)

    if table.empty:
        st.info("No games simulated for this week.")
    else:
        st.altair_chart(_ranked_bar(table, "projection", stat_label, n=int(show_n)),
                        width="stretch")

        cols = {"player": "Player", "team": "Team", "pos": "Pos",
                "matchup": "Game", "projection": "Projection", "floor": "Floor",
                "median": "Median", "ceiling": "Ceiling"}
        disp = table.head(int(show_n)).copy()
        if "p_milestone" in disp:
            ms = disp["milestone"].iloc[0]
            cols["p_milestone"] = f"{ms:g}+ %"
            disp["p_milestone"] *= 100
        if "p_anytime_td" in disp:
            cols["p_anytime_td"] = "Anytime TD %"
            disp["p_anytime_td"] *= 100
        cols["p_active"] = "Active %"
        disp["p_active"] *= 100
        st.dataframe(disp[list(cols)].rename(columns=cols).round(1),
                     hide_index=True, width="stretch")
        _glossary("Projection", "Floor", "Median", "Ceiling", "Active %")

        st.divider()
        st.markdown("#### One player, in detail")
        st.caption("Everything above is already on screen — this is only for when "
                   "you want the shape of a single player's range.")
        who = st.selectbox("Player", table["player"].head(int(show_n)).tolist(),
                           key="lb_player", label_visibility="collapsed")
        card = lb.player_card(slate, who)
        if card:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Team", card["meta"].get("team", "—"))
            m2.metric("Game", card["matchup"])
            m3.metric("Role", f"{card['meta'].get('position')}"
                              f"{int(card['meta'].get('depth_rank', 1))}")
            m4.metric("Active %", f"{card['meta'].get('p_active', 1) * 100:.0f}%")

            d1, d2 = st.columns([1.3, 1])
            with d1:
                ln = card["lines"].copy()
                ln["p_milestone"] *= 100
                st.dataframe(
                    ln.rename(columns={"market": "Market", "projection": "Projection",
                                       "floor": "Floor", "median": "Median",
                                       "ceiling": "Ceiling", "milestone": "Line",
                                       "p_milestone": "Over %",
                                       "fair_milestone": "Fair odds"})
                      .drop(columns=["stat"]).round(1),
                    hide_index=True, width="stretch")
            with d2:
                vals = card["samples"].get(stat)
                if vals is not None and len(vals):
                    line = st.number_input(
                        "Line", value=float(np.round(np.median(vals) * 2) / 2),
                        step=0.5, key="lb_line")
                    p_over = float((vals > line).mean())
                    st.metric(f"Over {line:g}", f"{p_over * 100:.1f}%",
                              delta=f"fair {picks.probability_to_american(p_over):+.0f}",
                              delta_color="off")
                    st.altair_chart(_distribution_chart(vals, line, stat_label),
                                    width="stretch")


# ================================================================= PICKS
with tab_picks:
    mode = st.radio("Show", ["Best picks", "Lotto plays"], horizontal=True,
                    key="pk_mode", label_visibility="collapsed")
    slate = simulate_week(WEEK)

    if mode == "Best picks":
        st.subheader(f"Week {WEEK} — what the model is most sure of")
        with st.expander("Read this before betting any of it"):
            st.markdown(
                "**Confidence is not value.** There is no odds feed here, so this "
                "ranks what the model is most sure of — not what a book has "
                "mispriced. A 90% leg at -1200 is still a bad bet. The only number "
                "worth acting on is the expected value you get after entering a "
                "real price below.\n\n"
                "Near-certainties are left off deliberately: that is where a book's "
                "margin is heaviest and where the model's own error is "
                "proportionally largest."
            )
        c1, c2 = st.columns([2, 1])
        band = c1.select_slider(
            "How confident", options=["Wide", "Standard", "Only the strongest"],
            value="Standard", key="bp_band",
            help="Standard is 60–90% — the range where a model's edge is usable.")
        bands = {"Wide": (0.55, 0.95), "Standard": (0.60, 0.90),
                 "Only the strongest": (0.72, 0.90)}
        lo, hi = bands[band]
        top_n = c2.slider("Show", 10, 60, 25, 5, key="bp_n")

        bp = picks.best_picks(slate, min_prob=lo, max_prob=hi, top_n=int(top_n))
        if bp.empty:
            st.info("Nothing in that band this week. Widen it.")
        else:
            show = bp.rename(columns={
                "matchup": "Game", "player": "Player", "team": "Team", "pos": "Pos",
                "market": "Market", "line": "Line", "side": "Side",
                "projection": "Projection", "probability": "Model %",
                "fair_odds": "Fair odds", "p_active": "Active %"})
            show["Model %"] *= 100
            show["Active %"] *= 100
            show["Fair odds"] = show["Fair odds"].round(0)
            st.dataframe(show.drop(columns=["edge_score", "stat"]).round(1),
                         hide_index=True, width="stretch")
            _glossary("Model %", "Fair odds", "Projection", "Active %")

            st.markdown("#### Price one")
            pc1, pc2, pc3 = st.columns([3, 1, 1])
            idx = pc1.selectbox(
                "Pick", range(len(bp)),
                format_func=lambda i: f"{bp.iloc[i].player} {bp.iloc[i].side} "
                                      f"{bp.iloc[i].line:g} {bp.iloc[i].market}",
                key="bp_row", label_visibility="collapsed")
            offered = pc2.number_input("Odds", value=-110, step=5, key="bp_odds")
            stake = pc3.number_input("Stake", value=100, step=10, key="bp_stake")
            p = float(bp.iloc[int(idx)]["probability"])
            ev = picks.expected_value(p, offered, stake)
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Model probability", f"{p * 100:.1f}%")
            e2.metric("Break-even", f"{picks.american_to_probability(offered) * 100:.1f}%")
            e3.metric("Expected value", f"{ev:+.2f}", delta=f"on {stake:.0f}",
                      delta_color="off")
            k = picks.kelly_fraction(p, offered)
            e4.metric("Kelly stake", f"{k * 100:.1f}%" if k > 0 else "no bet")
            if ev <= 0:
                st.caption("Negative at this price — the model does not like it "
                           "enough to overcome the vig.")

    else:
        st.subheader(f"Week {WEEK} — longshots")
        st.caption("Lines a player clears only when the game goes his way: "
                   "projected for fifty yards, bet to reach a hundred.")
        with st.expander("How longshots are priced here"):
            st.markdown(
                "Two things about the far tail were measured rather than assumed.\n\n"
                "**Doubling a projection gets much harder as the projection grows.** "
                "A receiver projected around 27 yards doubled 18.3% of the time on "
                "the 2025 holdout; one projected around 78 yards did it 0.9% of the "
                "time — +446 against +9900 in fair-odds terms. So \"twice his "
                "number\" is never one market.\n\n"
                "**Explosive players have the fatter tail, but only once you hold "
                "the projection fixed.** The raw comparison says the opposite, "
                "because explosive players have higher averages and so a larger bar "
                "to clear. At 20–40 projected yards they double 22.5% of the time "
                "against 13.1% for everyone else.\n\n"
                "**The simulation's own tail runs too fat out here**, so above the "
                "projection it is blended with a curve fitted to what actually "
                "happened, and the curve takes over as the line moves out.\n\n"
                "Quarterback passing longshots are deliberately not offered — there "
                "is no fitted curve for them and the passing projection underneath "
                "only just matches a trivial baseline."
            )
        l1, l2 = st.columns(2)
        lt_min_p = l1.slider("Minimum chance", 0.03, 0.25, 0.05, 0.01, key="lt_minp",
                             help="Below about 3% a longshot is a raffle ticket.")
        lt_n = l2.slider("Show", 10, 60, 25, 5, key="lt_n")

        lt_board = lotto.lotto_board(slate, explosive=_explosive_profile(), top_n=200)
        if not lt_board.empty:
            lt_board = lt_board[lt_board["p_model"] >= lt_min_p].head(int(lt_n))
        if lt_board.empty:
            st.info("Nothing clears that floor this week. Lower it.")
        else:
            show = lt_board.copy()
            show["Model %"] = show["p_model"] * 100
            show["vs typical"] = show["edge_ratio"].map(
                lambda x: f"{x:.2f}x" if np.isfinite(x) else "—")
            show = show.rename(columns={
                "player": "Player", "team": "Team", "position": "Pos",
                "market": "Market", "line": "Line", "projection": "Projection",
                "multiple": "Multiple", "fair_odds": "Fair odds"})
            show["Fair odds"] = show["Fair odds"].round(0)
            st.dataframe(
                show[["Player", "Team", "Pos", "Market", "Line", "Projection",
                      "Multiple", "Model %", "Fair odds", "vs typical"]].round(2),
                hide_index=True, width="stretch")
            _glossary("vs typical", "Model %", "Fair odds", "Projection")
            st.caption(
                "Sorted by **vs typical**, not by probability. Sorting by "
                "probability would just put whoever has the smallest projection on "
                "top, because a small number is the easiest one to double."
            )

            st.markdown("#### Price one")
            lp1, lp2, lp3 = st.columns([3, 1, 1])
            lidx = lp1.selectbox(
                "Longshot", range(len(lt_board)),
                format_func=lambda i: f"{lt_board.iloc[i].player} over "
                                      f"{lt_board.iloc[i].line:g} {lt_board.iloc[i].market}",
                key="lt_row", label_visibility="collapsed")
            lt_odds = lp2.number_input("Odds", value=600, step=25, key="lt_odds")
            lt_stake = lp3.number_input("Stake", value=20, step=5, key="lt_stake")
            lrow = lt_board.iloc[int(lidx)]
            lp = float(lrow["p_model"])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Model probability", f"{lp * 100:.1f}%")
            m2.metric("Fair price", f"{lrow['fair_odds']:+.0f}")
            m3.metric("Break-even", f"{picks.american_to_probability(lt_odds) * 100:.1f}%")
            m4.metric("Expected value",
                      f"{picks.expected_value(lp, lt_odds, lt_stake):+.2f}",
                      delta=f"on {lt_stake:.0f}", delta_color="off")
            if lrow.get("calibrated") and np.isfinite(lrow.get("p_sim", np.nan)):
                st.caption(
                    f"Simulation alone said {lrow['p_sim'] * 100:.1f}%; the fitted "
                    f"tail curve says {lrow['p_fitted'] * 100:.1f}%; the blend is "
                    f"{lp * 100:.1f}%."
                )
            if lrow.get("extrapolated"):
                st.warning("This line sits outside the 1.25x–3x range the curve was "
                           "fitted on, so it is an extrapolation.", icon="⚠️")


# ================================================================ PARLAY
with tab_parlay:
    st.subheader(f"Week {WEEK} parlay")
    g1, g2, g3, g4 = st.columns([1, 1.2, 1.4, 1])
    n_legs = g1.selectbox("Legs", [2, 3, 4, 5], index=1, key="pg_legs")
    style = g2.selectbox("Style", list(play.STYLES), index=1, key="pg_style",
                         help="Safer legs hit more often and pay less. Longshot "
                              "legs are the other way round.")
    stack = g3.selectbox("Shape", ["Same-team stack", "Spread across games"],
                         key="pg_shape",
                         help="A stack leans on legs helping each other; spreading "
                              "across games lowers variance instead.")
    g4.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
    go = g4.button("🎲 Generate", type="primary", width="stretch",
                   key="pg_go")

    if "pg_seed" not in st.session_state:
        st.session_state.pg_seed = 0
    if go:
        st.session_state.pg_seed += 1

    slate = simulate_week(WEEK)
    result = play.generate(slate, n_legs=int(n_legs), style=style,
                           correlated=(stack == "Same-team stack"),
                           seed=st.session_state.pg_seed)

    if "error" in result:
        st.info(result["error"])
    else:
        s = result["summary"]
        legs = result["legs"]
        st.markdown(f"##### {s['matchups']}")
        show = legs.copy()
        show["probability"] *= 100
        show["fair_odds"] = show["fair_odds"].round(0)
        st.dataframe(
            show[["leg", "probability", "fair_odds", "p_active"]].rename(columns={
                "leg": "Leg", "probability": "Model %", "fair_odds": "Fair odds",
                "p_active": "Active %"}).assign(
                **{"Active %": lambda d: d["Active %"] * 100}).round(1),
            hide_index=True, width="stretch")

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Parlay hits", f"{s['probability'] * 100:.1f}%")
        p2.metric("Fair price", f"{s['fair_odds']:+.0f}")
        p3.metric("Naive price", f"{s['naive_fair_odds']:+.0f}",
                  help="What a parlay calculator that assumes independence would say.")
        lift = s["correlation_lift"]
        p4.metric("Correlation lift", f"{lift:.2f}x",
                  delta="legs help each other" if lift > 1.02
                  else ("legs fight" if lift < 0.98 else "roughly independent"))

        pr1, pr2 = st.columns([1, 3])
        offered = pr1.number_input("Offered price", value=int(s["fair_odds"]),
                                   step=25, key="pg_odds")
        stake = pr1.number_input("Stake", value=20, step=5, key="pg_stake")
        ev = picks.expected_value(float(s["probability"]), offered, stake)
        with pr2:
            e1, e2 = st.columns(2)
            e1.metric("Break-even",
                      f"{picks.american_to_probability(offered) * 100:.1f}%")
            e2.metric("Expected value", f"{ev:+.2f}", delta=f"on {stake:.0f}",
                      delta_color="off")

        with st.expander("Why this differs from a parlay calculator"):
            st.markdown(
                "Multiplying leg probabilities assumes the legs are independent. "
                "Same-game legs never are. A quarterback over his passing yards and "
                "his receiver over his receiving yards are nearly the same bet, and "
                "the naive number is too low. A quarterback and his own running back "
                "pull against each other, and the naive number is too high.\n\n"
                "Every leg here is priced against one shared simulation of the game, "
                "so the correlation is carried rather than assumed away. Both figures "
                "are shown above.\n\n"
                "**None of this makes a parlay a good bet.** A parlay is worse than "
                "its legs at every setting; correlation only changes by how much."
            )
            _glossary("Correlation lift", "Model %", "Fair odds", "Active %")

        with st.expander("Other slips the model rates"):
            alt_df = result["alternatives"]
            a = alt_df.copy()
            a["probability"] *= 100
            st.dataframe(
                a[["legs", "matchups", "probability", "fair_odds",
                   "correlation_lift"]].rename(columns={
                    "legs": "Legs", "matchups": "Games", "probability": "Model %",
                    "fair_odds": "Fair odds", "correlation_lift": "Lift"}).round(2),
                hide_index=True, width="stretch")

        with st.expander("Build one by hand instead"):
            st.caption("Pick your own legs and price them against the same simulation.")
            manual_on = st.checkbox("Build my own slip", key="mn_on")
        if manual_on:
            roster = pd.concat([g.roster() for g in slate], ignore_index=True)
            names = sorted(roster["player"].unique())
            manual, ok = [], True
            n_manual = st.number_input("Legs", 2, 6, 2, key="mn_n")
            for i in range(int(n_manual)):
                c1, c2, c3, c4 = st.columns([2.2, 1.6, 1, 1])
                # Each leg starts on a different player, so the default slip is
                # a valid one rather than the same selection repeated.
                who = c1.selectbox("Player", names, index=min(i, len(names) - 1),
                                   key=f"mn_p{i}")
                g = next((x for x in slate if who in x.players), None)
                pos = g.meta[who]["position"] if g else "WR"
                opts = [(k, lab) for k, lab in picks.STAT_MENU.get(pos, [])
                        if g and g.stat(who, k) is not None]
                if not opts:
                    ok = False
                    continue
                lab = c2.selectbox("Market", [l for _, l in opts], key=f"mn_s{i}")
                stat_key = dict((l, k) for k, l in opts)[lab]
                vals = g.stat(who, stat_key)
                default = float(np.round(np.median(vals) * 2) / 2)
                line = c3.number_input("Line", value=default, step=0.5, key=f"mn_l{i}")
                side = c4.selectbox("Side", ["over", "under"], key=f"mn_d{i}")
                manual.append(play.Leg(player=who, stat=stat_key, line=float(line),
                                       side=side, label=lab))
            if ok and len(manual) >= 2:
                res = play.evaluate(manual, slate)
                if "error" in res:
                    st.warning(res["error"])
                else:
                    q1, q2, q3 = st.columns(3)
                    q1.metric("Parlay hits", f"{res['probability'] * 100:.2f}%")
                    q2.metric("Fair price", f"{res['fair_odds']:+.0f}")
                    q3.metric("Correlation lift", f"{res['correlation_lift']:.2f}x")
                    cm = play.correlation_matrix(manual, slate)
                    if not cm.empty:
                        st.dataframe(cm.round(3), width="stretch")


# ================================================================= GAMES
with tab_games:
    st.subheader(f"Week {WEEK} game predictions")
    with st.expander("The closing line beats this model — read before betting it"):
        st.markdown(
            "Over 544 held-out games it predicts margin to 10.3 points against the "
            "market's 9.7, and picks 64.7% of winners against the market's 68.4%. "
            "Betting its disagreements **lost money** in backtest: 48.7% against the "
            "spread, below the 52.4% break-even.\n\n"
            "Treat a large edge as a flag that the model is missing something — "
            "usually injury or personnel news — rather than as a signal. It is an "
            "independent second opinion, formed from opponent-adjusted EPA rather "
            "than from the market, which is what makes it worth having at all."
        )
    gslate = gm.predict_slate(ctx.games, PROJECTION_SEASON, team_proj, scoring,
                              week=WEEK, n_sims=20000)
    if gslate.empty:
        st.info("No games scheduled for this week.")
    else:
        show = gslate.rename(columns={
            "away": "Away", "home": "Home", "away_pts": "Away pts",
            "home_pts": "Home pts", "model_margin": "Margin", "model_total": "Total",
            "home_win_pct": "Home win %", "market_spread": "Mkt spread",
            "market_total": "Mkt total", "spread_edge": "Spread edge",
            "total_edge": "Total edge", "env_pts": "Env pts", "divisional": "Div",
            "roof": "Roof", "travel_miles": "Away miles", "tz_shift": "TZ"})
        st.dataframe(
            show.drop(columns=["week", "game_id"]), hide_index=True,
            width="stretch",
            column_config={"Home win %": st.column_config.NumberColumn(format="%.1f%%")})
        with st.expander("What these columns mean"):
            st.markdown(
                "**Margin** is home-relative — positive means the home side wins by "
                "that much.\n\n"
                "**Edge** is the model minus the market. A positive spread edge means "
                "the model likes the home side more than the market does.\n\n"
                "**Env pts** is the scoring-environment adjustment applied to the "
                "total from roof, wind and divisional status.\n\n"
                "**Away miles** and **TZ** are shown for context and are deliberately "
                "**not** applied. Travel was tested against 7,276 games: the "
                "east-to-west effect was worth -2.8 points against the spread in "
                "1999-2009 and has since decayed to zero.\n\n"
                "Blank market columns mean no line is posted for that game yet."
            )

    st.markdown(f"#### Projected {PROJECTION_SEASON} team strength")
    tp = team_proj.copy()
    tp["net"] = tp["off_rating"] + tp["def_rating"]
    tp = tp.sort_values("net", ascending=False)
    tp["new staff"] = tp["team"].map(lambda t: not ctx.staffs[t].continuity)
    st.dataframe(
        tp.rename(columns={"team": "Team", "off_rating": "Offense",
                           "def_rating": "Defense", "net": "Net"}).round(4),
        hide_index=True, width="stretch")
    st.caption(
        "Opponent-adjusted EPA per play, regressed toward the mean. Offense carries "
        "year to year at r = 0.44 and defense at only r = 0.12, so defensive ratings "
        "are pulled far harder toward average — a defence is close to a coin flip "
        "from one season to the next. Teams with a new play caller are shrunk further."
    )


# ============================================================= DEEP DIVE
# The coordinator-level tools. Everything that used to be its own tab lives
# here behind one selector: nothing was removed, it just stopped competing
# for attention with the three questions most visits are about.
def _deep_team_board():
    """Full projection board for one team, with conditions and travel."""
    c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1.2])
    team = c1.selectbox("Team", TEAMS, index=TEAMS.index("NYG"),
                        format_func=lambda t: f"{t} — {TEAM_NAMES[t]}")
    weeks = sorted(board.schedule_for(ctx.games, PROJECTION_SEASON)["week"].unique())
    week = c2.selectbox("Week", weeks, index=weeks.index(WEEK) if WEEK in weeks else 0,
                        key="tb_week")
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
            hide_index=True, width="stretch",
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
                         hide_index=True, width="stretch")

        kick = board.project_kicker_for(team, ctx, pm, gc, env=env,
                                        n_sims=int(n_sims), seed=int(week))
        if kick:
            st.subheader("Kicker")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Kicker", kick["player"])
            k2.metric("FG attempts", f"{np.mean(kick['samples']['fg_attempts']):.2f}")
            k3.metric("FG made", f"{np.mean(kick['samples']['fg_made']):.2f}")
            k4.metric("Kicking points", f"{np.mean(kick['samples']['kicking_points']):.2f}")
            st.caption(
                "Make probability is a logistic in distance fitted on 4,325 attempts. "
                "Wind is deliberately not applied to accuracy - controlling for distance "
                "the effect is not distinguishable from noise - but it suppresses scoring, "
                "which reduces trips into range."
            )

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
                pdf.round(1), hide_index=True, width="stretch",
                column_config={c: st.column_config.NumberColumn(c, format="%.1f%%")
                               for c in pdf.columns if c.startswith("o")},
            )

# ------------------------------------------------------------------ players

def _deep_war_room():
    """Scheme collisions, X-factors and what changed this week."""
    st.caption(
        "The weekly coordinator view: where two schemes actually collide, which "
        "players the matchup swings, and what has changed since last week."
    )
    ww = sorted(board.schedule_for(ctx.games, PROJECTION_SEASON)["week"].unique())
    w1, w2 = st.columns([1, 3])
    w_week = w1.selectbox("Week", ww, index=ww.index(WEEK) if WEEK in ww else 0,
                          key="wr_week")
    w_sched = board.schedule_for(ctx.games, PROJECTION_SEASON, int(w_week))
    if w_sched.empty:
        st.info("No games scheduled.")
    else:
        labels = {f"{r.away_team} @ {r.home_team}": (r.away_team, r.home_team)
                  for r in w_sched.itertuples()}
        pick_g = w2.selectbox("Game", list(labels), key="wr_game")
        away, home = labels[pick_g]

        plan = gp.weekly_plan(home, away, ctx, pm, cov_plays, lg_off, lg_def,
                              seasons=(anchor,))
        st.subheader("Scheme collisions")
        for side in ("away_offense", "home_offense"):
            sd = plan[side]
            st.markdown(f"**{TEAM_NAMES[sd['offense']]} offense vs "
                        f"{TEAM_NAMES[sd['defense']]} defense**")
            any_note = False
            for c in sd["collisions"]:
                who = {"neutral": "even", "variance": "high variance"}.get(
                    c["favours"], c["favours"])
                st.markdown(f"- **{c['axis']}** — {c['edge']}  ·  _favours {who}_  \n"
                            f"  {c['detail']}")
                any_note = True
            for n in sd["coverage"]:
                st.markdown(f"- **Coverage** — {n}")
                any_note = True
            if not any_note:
                st.caption("Both sides near league norms on every axis measured.")

        st.subheader("X-factors")
        st.caption(
            "Players who can change the trajectory of a game — ranked on the top "
            "of their range, not the middle. **Ceiling** is the 90th-percentile "
            "outcome, **boom %** is how often the simulation produces a genuinely "
            "game-breaking line, and **explosive index** is how often the player's "
            "own touches have actually gone 20+ yards against league average. A "
            "back projected for a steady 70 is valuable and is not an X-factor; a "
            "receiver projected for 55 with a real chance of 140 and two scores is. "
            "**Matchup swing** is carried alongside because it is informative, but "
            "facing a soft defence does not by itself make a player an X-factor."
        )
        gcs_w = board.game_contexts(ctx.games, PROJECTION_SEASON, int(w_week))
        envs_w = board.game_environments(ctx.games, PROJECTION_SEASON, int(w_week))
        xa, xb = st.columns(2)
        for col, team in ((xa, away), (xb, home)):
            with col:
                st.markdown(f"**{team}**")
                if team not in gcs_w:
                    st.caption("On bye.")
                    continue
                xf = gp.x_factors(team, gcs_w[team].opponent, ctx, pm, usage_hist,
                                  sampler, gcs_w[team], lg_def,
                                  env=envs_w.get(team), n_sims=6000, top_n=5)
                if xf.empty:
                    st.caption("No meaningful swing.")
                else:
                    st.dataframe(xf[["player", "pos", "projection", "ceiling_90th",
                                     "boom_pct", "explosive_index", "matchup_swing",
                                     "p_active"]].rename(columns={
                        "player": "Player", "pos": "Pos", "projection": "Proj",
                        "ceiling_90th": "Ceiling", "boom_pct": "Boom %",
                        "explosive_index": "Explosive", "matchup_swing": "Swing",
                        "p_active": "Active"}).round(2),
                        hide_index=True, width="stretch")

        st.subheader("Situational edges")
        se = gp.situational_edges(away, home, ctx.plays, seasons=(anchor,))
        if not se.empty:
            st.dataframe(se.round(3), hide_index=True, width="stretch")

        st.divider()
        st.subheader("What changed this week")
        st.caption(
            "Role changes the model can see for itself. Known narratives — revenge "
            "games, bye weeks, primetime, National Tight Ends Day — were tested "
            "against closing lines back to 2006 and none of them move outcomes. "
            "Information that has not propagated yet is a different matter, and "
            "that is what this watches."
        )
        brief = news.briefing(PROJECTION_SEASON, int(w_week), ctx)
        n1, n2 = st.columns(2)
        with n1:
            st.markdown("**Depth-chart moves**")
            dm = brief["depth_moves"]
            if dm is not None and not dm.empty:
                st.dataframe(dm[["team", "player_name", "pos_abb", "pos_rank_before",
                                 "pos_rank_after", "direction"]].head(20).rename(columns={
                    "team": "Team", "player_name": "Player", "pos_abb": "Pos",
                    "pos_rank_before": "Was", "pos_rank_after": "Now",
                    "direction": "Move"}), hide_index=True, width="stretch")
            else:
                st.caption("No movement detected between these weeks.")
        with n2:
            st.markdown("**Injury report**")
            inj = brief["injuries"]
            if inj is not None and not inj.empty:
                st.dataframe(inj.head(20), hide_index=True, width="stretch")
            else:
                st.caption("No report published for this week yet.")

        st.markdown("**Snap-share trend** — snaps move before targets do")
        stt = brief["snap_trend"]
        if stt is not None and not stt.empty:
            skill = stt[stt["position"].isin(["QB", "RB", "WR", "TE", "FB"])]
            st.dataframe(skill.head(15).round(3), hide_index=True,
                         width="stretch")
        else:
            st.caption("Not enough of the season played to compute a trend.")

        notes_df = brief["notes"]
        st.markdown("**Your notes**")
        if notes_df is not None and not notes_df.empty:
            st.dataframe(notes_df, hide_index=True, width="stretch")
        else:
            st.caption(
                "None entered. Add beat reporting to `data/news_2026.yaml` — first-team "
                "reps, a snap-count plan, a coordinator saying he intends to feature "
                "someone. Those move projections because they have not propagated yet."
            )

# ------------------------------------------------------------- game model

def _deep_scouting():
    """One team's projected identity, personnel and coaching note."""
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
                         hide_index=True, width="stretch")
    with b:
        st.subheader("Defense")
        st.write(rep["defense_note"])
        if rep["base_front"]:
            st.write(f"Base front: **{rep['base_front']}**")
        if not rep["defense_identity"].empty:
            d = rep["defense_identity"][["trait", "display", "league_display", "phrase"]]
            st.dataframe(d.rename(columns={"trait": "Trait", "display": "Projected",
                                           "league_display": "League", "phrase": "Read"}),
                         hide_index=True, width="stretch")

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
            .round(3), hide_index=True, width="stretch")
    else:
        st.caption("Not enough charted plays for a stable concept profile.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Situational tendencies")
        sit = rep["situational"]
        if not sit.empty:
            cols = [c for c in ["situation", "plays", "pass_rate", "shotgun_rate", "motion_rate",
                                "play_action_rate", "epa"] if c in sit.columns]
            st.dataframe(sit[cols].round(3), hide_index=True, width="stretch")
    with c2:
        st.subheader("Defensive situational")
        ds = rep["def_situational"]
        if not ds.empty:
            st.dataframe(ds.round(3), hide_index=True, width="stretch")

    st.subheader("Run direction")
    rd = rep["run_directions"]
    if not rd.empty:
        st.dataframe(rd.head(12).round(3), hide_index=True, width="stretch")

# ------------------------------------------------------------------ coverage

def _deep_coverage():
    """What each defence actually plays, and who it suits."""
    st.caption(
        "What defences play behind the front, from nflverse participation charting: "
        "man or zone, and which shell. Coverage is charted on pass plays only, so rates "
        "are shares of charted dropbacks rather than of all snaps."
    )
    if cov_fp.empty:
        st.info("Coverage charting unavailable for this season.")
    else:
        lg_man = cov_fp["man_rate"].mean()
        lg_single = cov_fp["single_high_rate"].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("League man rate", f"{lg_man*100:.1f}%")
        c2.metric("League single-high", f"{lg_single*100:.1f}%")
        c3.metric("League Cover 0", f"{cov_fp['cover0_rate'].mean()*100:.1f}%")

        show = cov_fp.rename(columns={
            "team": "Team", "charted_plays": "Charted", "man_rate": "Man",
            "zone_rate": "Zone", "single_high_rate": "Single-high",
            "two_high_rate": "Two-high", "cover0_rate": "Cover 0",
            "cover1_rate": "Cover 1", "cover2_rate": "Cover 2",
            "cover3_rate": "Cover 3", "cover4_rate": "Cover 4",
            "cover6_rate": "Cover 6", "2man_rate": "2-man",
            "pressure_rate": "Pressure", "epa_vs_man": "EPA vs man",
            "epa_vs_zone": "EPA vs zone",
        })
        st.dataframe(show.sort_values("Man", ascending=False).round(3),
                     hide_index=True, width="stretch")

        ct = st.selectbox("Team detail", TEAMS, index=TEAMS.index("NYG"),
                          format_func=lambda x: f"{x} — {TEAM_NAMES[x]}", key="cov_team")
        d1, d2 = st.columns(2)
        with d1:
            st.subheader(f"{ct} defense: coverage mix")
            prof = cvg.defense_coverage_profile(cov_plays, ct, seasons=(anchor,))
            if not prof.empty:
                st.dataframe(
                    prof[["shell", "plays", "rate", "epa", "ypa", "comp_rate", "adot"]]
                    .rename(columns={"shell": "Shell", "plays": "Plays", "rate": "Rate",
                                     "epa": "EPA/play", "ypa": "Yds/att",
                                     "comp_rate": "Comp%", "adot": "aDOT"}).round(3),
                    hide_index=True, width="stretch")
        with d2:
            st.subheader(f"{ct} offense vs coverage")
            ovc = cvg.offense_vs_coverage(cov_plays, ct, seasons=(anchor,))
            if not ovc.empty:
                st.dataframe(ovc.round(3), hide_index=True, width="stretch")
            st.subheader("Personnel groupings")
            pers = cvg.personnel_profile(cov_plays, ct, seasons=(anchor,))
            if not pers.empty:
                st.dataframe(pers.round(3), hide_index=True, width="stretch")

        st.subheader(f"{ct} route menu")
        st.caption("Routes run more or less than the league does, with what they produced.")
        rt = cvg.route_profile(cov_plays, ct, seasons=(anchor,))
        if not rt.empty:
            st.dataframe(
                rt[["route", "plays", "rate", "league_rate", "lift", "yards", "epa", "comp_rate"]]
                .rename(columns={"route": "Route", "plays": "Plays", "rate": "Team rate",
                                 "league_rate": "League rate", "lift": "Lift",
                                 "yards": "Yds/play", "epa": "EPA/play",
                                 "comp_rate": "Comp%"}).round(3),
                hide_index=True, width="stretch")

# ------------------------------------------------------------------- matchup

def _deep_matchup():
    """Two teams side by side on every measured axis."""
    weeks = sorted(board.schedule_for(ctx.games, PROJECTION_SEASON)["week"].unique())
    wk = st.selectbox("Week", weeks, index=weeks.index(WEEK) if WEEK in weeks else 0,
                      key="mu_week")
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
            edges += cvg.coverage_matchup(off["team"], dfn["team"], cov_plays,
                                          seasons=(anchor,))
            if edges:
                for e in edges:
                    st.markdown(f"- {e}")
            else:
                st.caption("No standout structural edge; both sides near league norms.")

# ------------------------------------------------------------ scheme explorer

def _deep_scheme_explorer():
    """All 32 teams on whichever scheme columns you choose."""
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
    st.dataframe(tbl[meta + chosen].round(3), width="stretch")

# ------------------------------------------------------------------- method

def _deep_defenders_routes():
    """Who a defense asks to cover, and the concepts an offense calls.

    Both layers are description. The measurements that kept them out of the
    projection are stated on screen rather than buried, because a reader
    looking at a cornerback's yards-allowed will reasonably assume it is being
    used, and it is not.
    """
    st.info(
        "**Neither of these adjusts a projection, and the reason is measured.** "
        "Individual coverage quality does not survive removing coverage role: "
        "completion rate allowed persists at r = 0.46 year to year, but at "
        "**0.09** once the depth-of-target curve is fitted out, and yards per "
        "target at **0.13**. Within a single season it is the same — 0.09 on "
        "split halves — so in-season mode cannot rescue it. Route mix is "
        "genuinely persistent (r = 0.86) but is a near-substitute for depth of "
        "target, which the model already carries: adding it to a projection of "
        "next-season yards per target moves the multiple correlation from 0.534 "
        "to 0.541.\n\n"
        "What is real and stable here is **role** — how deep a defender is asked "
        "to work, who quarterbacks throw at, and which concepts an offense "
        "actually calls. That is worth reading before a game. It is not a "
        "yardage multiplier.",
        icon="🧭",
    )
    d1, d2 = st.columns(2)
    dteam = d1.selectbox("Defense", TEAMS, index=TEAMS.index("BUF"),
                         format_func=lambda t: f"{t} — {TEAM_NAMES[t]}", key="df_team")
    oteam = d2.selectbox("Offense", TEAMS, index=TEAMS.index("NYG"),
                         format_func=lambda t: f"{t} — {TEAM_NAMES[t]}", key="rt_team")

    st.markdown(f"#### {TEAM_NAMES[dteam]} coverage")
    prof = getattr(ctx, "defenders", None)
    smap = dfn.secondary_map(prof, dteam) if prof is not None else pd.DataFrame()
    if smap.empty:
        st.caption("No charted coverage for this defense in the anchor season.")
    else:
        show = smap.rename(columns={
            "player": "Defender", "role": "Role", "adot": "Target depth",
            "targets": "Targets", "targets_per_game": "Tgt/game",
            "coverage_share": "Share of coverage", "completion_pct": "Comp % allowed",
            "yards_per_target": "Yds/target", "yac_per_completion": "YAC/catch",
            "missed_tackle_pct": "Missed tackle %"})
        for c in ("Share of coverage", "Comp % allowed", "Missed tackle %"):
            if c in show:
                show[c] = show[c] * 100
        st.dataframe(show.round(1), hide_index=True, width="stretch")
        load = dfn.coverage_load(prof, dteam)
        if load:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Most targeted", load["most_targeted"])
            c2.metric("His share", f"{load['concentration'] * 100:.0f}%")
            c3.metric("Avg target depth", f"{load['mean_adot_allowed']:.1f} yds")
            c4.metric("Deep coverage share", f"{load['deep_share'] * 100:.0f}%")

    st.markdown(f"#### {TEAM_NAMES[oteam]} concept menu")
    routed = _routed_plays()
    menu = rt.team_route_menu(routed, oteam, seasons=(anchor,)) if not routed.empty \
        else pd.DataFrame()
    if menu.empty:
        st.caption("No charted routes for this offense in the anchor season.")
    else:
        m = menu.rename(columns={
            "route": "Concept", "targets": "Targets", "share": "Share",
            "league_share": "League share", "vs_league": "vs league",
            "yards_per_target": "Yds/target", "epa": "EPA", "adot": "Depth"})
        m["Share"] = m["Share"] * 100
        m["League share"] = m["League share"] * 100
        st.dataframe(m.round(2), hide_index=True, width="stretch")
        st.caption("**vs league** is how many times more often than the rest of "
                   "the league this offense is thrown on that concept.")

    if not routed.empty:
        st.markdown("#### One receiver's menu")
        pool = routed[routed["posteam"] == oteam]
        names = (pool.groupby("receiver_player_id").size().sort_values(ascending=False)
                 .head(12).index.tolist())
        lookup = ctx.chart.set_index("gsis_id")["player_name"].to_dict() \
            if "gsis_id" in ctx.chart.columns else {}
        labels = {lookup.get(p, p): p for p in names}
        if labels:
            who = st.selectbox("Receiver", list(labels), key="rt_player")
            rp = rt.player_route_profile(routed, labels[who], seasons=None)
            if rp.empty:
                st.caption("Too few charted targets for a menu.")
            else:
                rr = rp.rename(columns={
                    "route": "Concept", "targets": "Targets", "share": "Share",
                    "catch_rate": "Catch %", "adot": "Depth",
                    "yards_per_target": "Yds/target", "league_ypt": "League",
                    "vs_league": "vs league", "epa": "EPA"})
                rr["Share"] = rr["Share"] * 100
                rr["Catch %"] = rr["Catch %"] * 100
                st.dataframe(rr.drop(columns=["yards"]).round(2), hide_index=True,
                             width="stretch")
                st.caption(
                    f"Route-implied depth {rt.route_implied_depth(rp):.1f} yards "
                    "against a measured depth of "
                    f"{rp['adot'].mul(rp['share']).sum():.1f}. A gap says the "
                    "offense is using him at a different depth than the concepts "
                    "alone would suggest."
                )

    st.markdown("#### Offensive line continuity")
    lc = getattr(ctx, "line_continuity", None)
    if lc is None or lc.empty:
        st.caption("No snap counts loaded.")
    else:
        latest = lc[lc["season"] == lc["season"].max()].copy()
        latest = latest.sort_values("returning_starters", ascending=False)
        st.dataframe(
            latest.rename(columns={
                "season": "Season", "team": "Team", "line_snaps": "Line snaps",
                "snap_continuity": "Snap continuity",
                "returning_starters": "Returning starters"}).round(3),
            hide_index=True, width="stretch")
        st.caption(
            "Measured and not applied. Over 223 team-seasons a returning starter "
            "is worth −0.0017 of sack rate (t = −1.54) and nothing at all on "
            "rushing — the whole range from one returning starter to five is "
            "about two thirds of a percentage point of sack rate, inside the "
            "noise. Kept because the sign is consistent: if it clears "
            "significance in a later season, applying it is a one-line change."
        )


DEEP_SECTIONS = {
    "Team projection board": _deep_team_board,
    "War room (scheme collisions & X-factors)": _deep_war_room,
    "Team scouting report": _deep_scouting,
    "Coverage charting": _deep_coverage,
    "Defenders, routes & line continuity": _deep_defenders_routes,
    "Head-to-head matchup": _deep_matchup,
    "Scheme explorer (all 32 teams)": _deep_scheme_explorer,
}

with tab_deep:
    st.caption(
        "The coordinator-level view. Nothing here is needed to use the first "
        "three tabs — it is what sits behind them."
    )
    _sec = st.selectbox("Tool", list(DEEP_SECTIONS), key="deep_sec",
                        label_visibility="collapsed")
    DEEP_SECTIONS[_sec]()


# ================================================================= LEARN
with tab_learn:
    st.subheader("What everything means")
    st.caption("Every term the app uses, in plain English, followed by the full "
               "method write-up.")

    for term, meaning in GLOSSARY.items():
        st.markdown(f"**{term}** — {meaning}")

    st.markdown(
        "**Projection vs If active** — every number in the app is shown assuming the "
        "player dresses, because that is how a book prices a prop: a scratch voids "
        "the bet rather than losing it. The chance he does not play is reported "
        "separately as **Active %** rather than folded silently into the projection.\n\n"
        "**Simulation** — a projection here is not a formula, it is the average of "
        "twenty thousand simulated versions of the game. Every player in a game is "
        "simulated together, so a receiver's big day and his quarterback's big day "
        "are the same simulation rather than two independent guesses. That is what "
        "makes the parlay maths work.\n\n"
        "**Scheme fingerprint** — a coaching staff reduced to numbers: how often "
        "they pass on early downs, how much motion and play-action they use, how "
        "deep they throw, how much they blitz. When a coordinator changes teams the "
        "model carries his fingerprint with him, then constrains it by the personnel "
        "he now has — a quarterback's designed-run rate follows the quarterback, not "
        "the coach.\n\n"
        "**Shrinkage** — a small sample is pulled toward the league average, by an "
        "amount fitted from how much that particular statistic actually persists. "
        "Team passing efficiency carries at r = 0.42 season to season, so most of it "
        "survives. Defensive yards allowed carries at r = 0.11, so almost none does."
    )

    st.divider()
    st.subheader("How good is it, honestly")
    st.markdown(
        "Held out the 2025 season and projected it from 2022-2024 only, over 5,671 "
        "skill-player games:\n\n"
        "| | preseason | in-season |\n"
        "|---|---|---|\n"
        "| Targets | 0.61 | 0.63 |\n"
        "| Carries | 0.82 | 0.83 |\n"
        "| Receiving yards | 0.55 | 0.57 |\n"
        "| Rushing yards | 0.74 | 0.75 |\n"
        "| Scrimmage yards | 0.58 | 0.61 |\n"
        "| Passing yards | 0.07 | 0.17 |\n\n"
        "Anytime-touchdown probability lands within about a point across most of its "
        "range. Passing yards are the weak column: before a season starts they only "
        "match the trivial baseline of a passer's own prior yards per game (+0.18), "
        "which is why longshots are not offered on that market. Once games are "
        "played they improve to +0.17 with lower error than that baseline.\n\n"
        "On games, the closing line beats the model — 10.3 points of margin error "
        "against the market's 9.7 — and betting the disagreements lost money in "
        "backtest."
    )

    st.divider()
    with st.expander("Full method write-up", expanded=False):
        st.markdown((pipeline.ROOT / "METHOD.md").read_text()
                    if (pipeline.ROOT / "METHOD.md").exists()
                    else "See METHOD.md in the repository.")
