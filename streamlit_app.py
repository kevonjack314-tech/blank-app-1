"""NFL projection model - yardage, touchdowns and scheme intelligence for 2026."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import altair as alt

from nflproj import (board, coverage as cvg, data as ndata, gamemodel as gm,
                     gameplan as gp, joint as jnt, kicking as kk, news,
                     parlay as play, picks, pipeline, playbook,
                     projections as pj, report, schemes, usage as um, venues as V)

# Single categorical hue; the distribution is one series, so no legend is
# needed and the probability is stated directly rather than read off shading.
SERIES_1 = "#2a78d6"
INK_MUTED = "#52514e"
from nflproj.config import PROJECTION_SEASON, TEAM_NAMES, TEAMS

st.set_page_config(page_title="NFL Projection Model", page_icon="🏈", layout="wide")

YARD_LINES = {
    "rec_yards": [29.5, 39.5, 49.5, 59.5, 69.5, 79.5],
    "rush_yards": [29.5, 39.5, 49.5, 59.5, 69.5, 79.5],
    "scrimmage_yards": [39.5, 49.5, 59.5, 69.5, 79.5, 99.5],
    "pass_yards": [199.5, 224.5, 249.5, 274.5, 299.5],
}


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
def load():
    ctx = pipeline.build_context()
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

(ctx, pm, usage_hist, sampler, lg_off, lg_def, anchor, ratings,
 team_proj, scoring, cov_plays, cov_fp) = load()

st.title("🏈 NFL Projection Model")
st.caption(
    f"Projecting {PROJECTION_SEASON} from {anchor} and earlier · play-by-play and FTN charting via nflverse · "
    "scheme carried across coaching changes, constrained by personnel"
)

(tab_board, tab_players, tab_picks, tab_parlay, tab_war, tab_games, tab_scout,
 tab_coverage, tab_matchup, tab_scheme, tab_method) = st.tabs(
    ["Projection board", "Players", "Best picks", "Parlay builder", "War room",
     "Game predictions", "Team scouting", "Coverage", "Matchup",
     "Scheme explorer", "Method & limits"]
)


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
                pdf.round(1), hide_index=True, use_container_width=True,
                column_config={c: st.column_config.NumberColumn(c, format="%.1f%%")
                               for c in pdf.columns if c.startswith("o")},
            )

# ------------------------------------------------------------------ players
with tab_players:
    st.caption(
        "Every projected player, grouped by position. Numbers come from a joint "
        "simulation of the whole game, so a player's line here is consistent with "
        "his team-mates' — which is what makes the parlay tab work."
    )
    pw = sorted(board.schedule_for(ctx.games, PROJECTION_SEASON)["week"].unique())
    c1, c2, c3 = st.columns([1, 1.4, 2])
    p_week = c1.selectbox("Week", pw, index=0, key="pl_week")
    slate = simulate_week(int(p_week))

    roster = pd.concat([g.roster() for g in slate], ignore_index=True) if slate else pd.DataFrame()
    if roster.empty:
        st.info("No games simulated for this week.")
    else:
        group = c2.selectbox("Position group", list(picks.POSITION_GROUPS), key="pl_grp")
        allowed = picks.POSITION_GROUPS[group]
        pool = roster[roster["position"].isin(allowed)].copy()
        pool = pool.sort_values(["team", "depth_rank", "player"])
        labels = {f"{r.player}  ({r.team} {r.position}{int(r.depth_rank)})": r.player
                  for r in pool.itertuples()}
        if not labels:
            st.info("No players at this position this week.")
        else:
            chosen = c3.selectbox(f"Player  ·  {len(labels)} available", list(labels),
                                  key="pl_player")
            player = labels[chosen]
            game = next((g for g in slate if player in g.players), None)
            meta = game.meta[player]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Team", meta["team"])
            m2.metric("Matchup", f"{game.away} @ {game.home}")
            m3.metric("Role", f"{meta['position']}{int(meta['depth_rank'])}")
            m4.metric("Active %", f"{meta['p_active']*100:.0f}%")

            view = st.radio("View", ["If active", "Expected"], horizontal=True,
                            key="pl_view",
                            help="If active assumes he dresses — how a book prices a "
                                 "prop. Expected prices in the chance he does not.")
            cond = view == "If active"
            mask = game.active_mask(player) if cond else None

            st.subheader("Projection")
            stats = picks.STAT_MENU.get(meta["position"], [])
            rows = []
            for stat, label in stats:
                v = game.stat(player, stat)
                if v is None:
                    continue
                vv = v[mask] if mask is not None else v
                if len(vv) == 0:
                    continue
                rows.append({"Market": label, "Projection": float(np.mean(vv)),
                             "Median": float(np.median(vv)),
                             "10th": float(np.percentile(vv, 10)),
                             "90th": float(np.percentile(vv, 90))})
            if rows:
                st.dataframe(pd.DataFrame(rows).round(1), hide_index=True,
                             use_container_width=True)

            st.subheader("Distribution")
            d1, d2 = st.columns([2, 1])
            stat_opts = {lab: st for st, lab in stats
                         if game.stat(player, st) is not None}
            pick_stat = d1.selectbox("Statistic", list(stat_opts), key="pl_stat")
            stat_key = stat_opts[pick_stat]
            vals = game.stat(player, stat_key)
            vals = vals[mask] if mask is not None else vals
            default_line = float(np.round(np.median(vals) * 2) / 2)
            line = d2.number_input("Line", value=default_line, step=0.5, key="pl_line")
            p_over = float((vals > line).mean())
            st.altair_chart(_distribution_chart(vals, line, pick_stat),
                            use_container_width=True)
            o1, o2, o3 = st.columns(3)
            o1.metric(f"Over {line:g}", f"{p_over*100:.1f}%")
            o2.metric(f"Under {line:g}", f"{(1-p_over)*100:.1f}%")
            o3.metric("Fair price (over)", f"{picks.probability_to_american(p_over):+.0f}")

            st.subheader("All lines")
            pl = picks.player_lines(game, player, conditional=cond)
            if not pl.empty:
                show = pl[["market", "line", "projection", "p_over", "p_under",
                           "fair_over", "fair_under"]].rename(columns={
                    "market": "Market", "line": "Line", "projection": "Projection",
                    "p_over": "Over %", "p_under": "Under %",
                    "fair_over": "Fair over", "fair_under": "Fair under"})
                show["Over %"] *= 100; show["Under %"] *= 100
                st.dataframe(show.round(1), hide_index=True, use_container_width=True)
            else:
                st.caption("No lines in a plausible range for this player.")

# --------------------------------------------------------------- best picks
with tab_picks:
    st.caption(
        "The model's highest-confidence sides across a week, from the same joint "
        "simulation. Enter a price on any row to turn confidence into expected value."
    )
    st.warning(
        "**Confidence is not value.** There is no odds feed here, so this ranks what "
        "the model is most sure of — not what a book has mispriced. A 90% leg at "
        "-1200 is still a bad bet. The only number worth acting on is the expected "
        "value you get after entering a real price.",
        icon="⚠️",
    )
    k1, k2, k3, k4 = st.columns(4)
    k_week = k1.selectbox("Week", pw, index=0, key="bp_week")
    lo, hi = k2.slider("Probability band", 0.50, 0.98, (0.60, 0.90), 0.01, key="bp_band")
    min_act = k3.slider("Min active %", 0.0, 1.0, 0.70, 0.05, key="bp_act")
    top_n = k4.slider("Show", 10, 80, 30, 5, key="bp_n")

    bp_slate = simulate_week(int(k_week))
    bp = picks.best_picks(bp_slate, min_prob=lo, max_prob=hi,
                          min_active=min_act, top_n=int(top_n))
    if bp.empty:
        st.info("Nothing in that band. Widen the probability range.")
    else:
        show = bp.rename(columns={
            "matchup": "Game", "player": "Player", "team": "Team", "pos": "Pos",
            "market": "Market", "line": "Line", "side": "Side",
            "projection": "Projection", "probability": "Model %",
            "fair_odds": "Fair odds", "p_active": "Active %"})
        show["Model %"] *= 100; show["Active %"] *= 100
        st.dataframe(show.drop(columns=["edge_score", "stat"]).round(1),
                     hide_index=True, use_container_width=True)

        st.subheader("Price a pick")
        pc1, pc2, pc3 = st.columns([3, 1, 1])
        idx = pc1.selectbox("Row", range(len(bp)),
                            format_func=lambda i: f"{bp.iloc[i].player} {bp.iloc[i].side} "
                                                  f"{bp.iloc[i].line:g} {bp.iloc[i].market}",
                            key="bp_row")
        offered = pc2.number_input("Odds (American)", value=-110, step=5, key="bp_odds")
        stake = pc3.number_input("Stake", value=100, step=10, key="bp_stake")
        row = bp.iloc[int(idx)]
        p = float(row["probability"])
        ev = picks.expected_value(p, offered, stake)
        kel = picks.kelly_fraction(p, offered)
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Model probability", f"{p*100:.1f}%")
        e2.metric("Break-even", f"{picks.american_to_probability(offered)*100:.1f}%")
        e3.metric("Expected value", f"{ev:+.2f}", delta=f"on {stake:.0f}")
        e4.metric("Kelly", f"{kel*100:.1f}%" if kel > 0 else "no bet")
        if ev <= 0:
            st.caption("Negative expected value at this price — the model does not "
                       "like it enough to overcome the vig.")

# ------------------------------------------------------------ parlay builder
with tab_parlay:
    st.caption(
        "Legs are priced against one shared simulation of the game, so correlation "
        "is carried rather than assumed away."
    )
    st.info(
        "**Why this differs from a parlay calculator.** Multiplying leg probabilities "
        "assumes independence. Same-game legs are never independent: a quarterback "
        "over his passing yards and his receiver over his receiving yards are nearly "
        "the same bet, and the naive number is too low. A quarterback and his own "
        "running back pull against each other, and the naive number is too high. "
        "Both figures are shown below.",
        icon="🔗",
    )
    pr_week = st.selectbox("Week", pw, index=0, key="pr_week")
    pr_slate = simulate_week(int(pr_week))
    pr_roster = pd.concat([g.roster() for g in pr_slate], ignore_index=True) if pr_slate else pd.DataFrame()

    if pr_roster.empty:
        st.info("No games simulated for this week.")
    else:
        st.subheader("Build a slip")
        n_legs = st.number_input("Legs", 2, 6, 2, key="pr_n")
        legs, ok = [], True
        for i in range(int(n_legs)):
            c1, c2, c3, c4, c5 = st.columns([2.4, 1.8, 1, 1, 1])
            pool = pr_roster.sort_values(["team", "position", "depth_rank"])
            names = {f"{r.player} ({r.team} {r.position})": r.player for r in pool.itertuples()}
            who = c1.selectbox(f"Leg {i+1} player", list(names), key=f"pr_p{i}",
                               index=min(i, len(names) - 1))
            pl_name = names[who]
            gm_ = next((g for g in pr_slate if pl_name in g.players), None)
            pos = gm_.meta[pl_name]["position"]
            opts = {lab: stt for stt, lab in picks.STAT_MENU.get(pos, [])
                    if gm_.stat(pl_name, stt) is not None}
            if not opts:
                ok = False
                continue
            mkt = c2.selectbox("Market", list(opts), key=f"pr_m{i}")
            stt = opts[mkt]
            vals = gm_.stat(pl_name, stt)[gm_.active_mask(pl_name)]
            ln = c3.number_input("Line", value=float(np.round(np.median(vals) * 2) / 2),
                                 step=0.5, key=f"pr_l{i}")
            side = c4.selectbox("Side", ["over", "under"], key=f"pr_s{i}")
            od = c5.number_input("Odds", value=-110, step=5, key=f"pr_o{i}")
            legs.append(play.Leg(player=pl_name, stat=stt, line=float(ln),
                                 side=side, odds=float(od), label=mkt))

        if ok and len(legs) >= 2:
            conditional = st.checkbox(
                "Assume all players active", value=True, key="pr_cond",
                help="Books usually void a leg when a player does not dress. Untick "
                     "to price a scratch as a loss instead.")
            res = play.evaluate(legs, pr_slate, conditional=conditional)
            if "error" in res:
                st.error(res["error"])
            else:
                st.subheader("Legs")
                lg_df = pd.DataFrame(res["legs"])[["leg", "probability", "fair_odds", "odds", "p_active"]]
                lg_df["probability"] *= 100
                lg_df["p_active"] = (lg_df["p_active"].astype(float) * 100).round(0)
                st.dataframe(lg_df.rename(columns={
                    "leg": "Leg", "probability": "Model %", "fair_odds": "Fair",
                    "odds": "Your odds", "p_active": "Active %"}).round(1),
                    hide_index=True, use_container_width=True)

                a, b, c, d = st.columns(4)
                a.metric("Correlated probability", f"{res['probability']*100:.2f}%")
                b.metric("If independent", f"{res['naive_probability']*100:.2f}%")
                c.metric("Fair price", f"{res['fair_odds']:+.0f}",
                         delta=f"naive {res['naive_fair_odds']:+.0f}")
                lift = res["correlation_lift"]
                d.metric("Correlation lift", f"{lift:.2f}x",
                         help="Above 1 means the legs help each other and a naive "
                              "calculator understates the slip. Below 1 means they fight.")

                if "expected_value_per_100" in res:
                    e1, e2, e3 = st.columns(3)
                    e1.metric("Offered price", f"{res['offered_american']:+.0f}")
                    e2.metric("Break-even", f"{res['breakeven_probability']*100:.2f}%")
                    e3.metric("EV per $100", f"{res['expected_value_per_100']:+.2f}")
                    if res["expected_value_per_100"] <= 0:
                        st.caption("Negative expected value at this price.")
                if conditional:
                    st.caption(f"All named players active in "
                               f"{res['all_active_probability']*100:.0f}% of simulations.")

                cm = play.correlation_matrix(legs, pr_slate)
                if not cm.empty:
                    st.subheader("How the legs move together")
                    st.dataframe(cm.round(3), use_container_width=True)

        st.divider()
        st.subheader("Suggested slips")
        s1, s2, s3 = st.columns(3)
        sug_legs = s1.slider("Legs per slip", 2, 4, 2, key="pr_sn")
        flavour = s2.selectbox("Style", ["correlated", "independent"], key="pr_flav",
                               format_func=lambda x: "Same-game stack" if x == "correlated"
                               else "Spread across games")
        sug_n = s3.slider("Show", 3, 15, 6, key="pr_sc")
        if st.button("Generate", key="pr_go"):
            sug = play.suggest(pr_slate, n_legs=int(sug_legs), target=flavour,
                               top_n=int(sug_n))
            if sug.empty:
                st.info("No combinations met the filters.")
            else:
                sug["probability"] *= 100
                sug["naive"] *= 100
                st.dataframe(sug.rename(columns={
                    "legs": "Slip", "matchups": "Game(s)", "probability": "Model %",
                    "naive": "If independent %", "correlation_lift": "Lift",
                    "fair_odds": "Fair", "naive_fair_odds": "Naive fair"}).round(2),
                    hide_index=True, use_container_width=True)
                st.caption(
                    "Ranked by the model's correlated probability, not by value — "
                    "there are no prices here. A parlay is a worse bet than its legs "
                    "however the correlation falls."
                )

# --------------------------------------------------------------- war room
with tab_war:
    st.caption(
        "The weekly coordinator view: where two schemes actually collide, which "
        "players the matchup swings, and what has changed since last week."
    )
    ww = sorted(board.schedule_for(ctx.games, PROJECTION_SEASON)["week"].unique())
    w1, w2 = st.columns([1, 3])
    w_week = w1.selectbox("Week", ww, index=0, key="wr_week")
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
                        hide_index=True, use_container_width=True)

        st.subheader("Situational edges")
        se = gp.situational_edges(away, home, ctx.plays, seasons=(anchor,))
        if not se.empty:
            st.dataframe(se.round(3), hide_index=True, use_container_width=True)

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
                    "direction": "Move"}), hide_index=True, use_container_width=True)
            else:
                st.caption("No movement detected between these weeks.")
        with n2:
            st.markdown("**Injury report**")
            inj = brief["injuries"]
            if inj is not None and not inj.empty:
                st.dataframe(inj.head(20), hide_index=True, use_container_width=True)
            else:
                st.caption("No report published for this week yet.")

        st.markdown("**Snap-share trend** — snaps move before targets do")
        stt = brief["snap_trend"]
        if stt is not None and not stt.empty:
            skill = stt[stt["position"].isin(["QB", "RB", "WR", "TE", "FB"])]
            st.dataframe(skill.head(15).round(3), hide_index=True,
                         use_container_width=True)
        else:
            st.caption("Not enough of the season played to compute a trend.")

        notes_df = brief["notes"]
        st.markdown("**Your notes**")
        if notes_df is not None and not notes_df.empty:
            st.dataframe(notes_df, hide_index=True, use_container_width=True)
        else:
            st.caption(
                "None entered. Add beat reporting to `data/news_2026.yaml` — first-team "
                "reps, a snap-count plan, a coordinator saying he intends to feature "
                "someone. Those move projections because they have not propagated yet."
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

# ------------------------------------------------------------------ coverage
with tab_coverage:
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
                     hide_index=True, use_container_width=True)

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
                    hide_index=True, use_container_width=True)
        with d2:
            st.subheader(f"{ct} offense vs coverage")
            ovc = cvg.offense_vs_coverage(cov_plays, ct, seasons=(anchor,))
            if not ovc.empty:
                st.dataframe(ovc.round(3), hide_index=True, use_container_width=True)
            st.subheader("Personnel groupings")
            pers = cvg.personnel_profile(cov_plays, ct, seasons=(anchor,))
            if not pers.empty:
                st.dataframe(pers.round(3), hide_index=True, use_container_width=True)

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
                hide_index=True, use_container_width=True)

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
            edges += cvg.coverage_matchup(off["team"], dfn["team"], cov_plays,
                                          seasons=(anchor,))
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
