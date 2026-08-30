"""Invariant tests for the projection chain.

These guard the properties that make the model correct rather than merely
runnable: that scheme transfer respects personnel, that shares add up, that
availability shifts the distribution the right way, and that nothing reads a
season it is supposed to be predicting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflproj import availability as av
from nflproj import coaches, playbook, projections as pj, schemes, usage as um


# --------------------------------------------------------------------- rates
def test_shrinkage_moves_toward_prior_on_small_samples():
    prior = 0.5
    strong = schemes._rate(90, 100, prior, strength=10)   # plenty of evidence
    weak = schemes._rate(9, 10, prior, strength=100)      # almost none
    assert strong > weak
    assert abs(weak - prior) < abs(strong - prior)


def test_rate_falls_back_to_prior_without_denominator():
    assert schemes._rate(0, 0, 0.42, strength=50) == pytest.approx(0.42)


# ------------------------------------------------------------------- volume
def test_expected_tds_rise_with_implied_points():
    assert pj.expected_offensive_tds(31) > pj.expected_offensive_tds(17)


def test_expected_tds_never_negative_for_shutout_scripts():
    assert pj.expected_offensive_tds(0) > 0


def test_favoured_teams_are_projected_to_throw_less():
    scheme = pd.Series({
        "plays_per_game": 63, "early_down_pass_rate": 0.55, "sack_rate_allowed": 0.06,
        "scramble_rate": 0.06, "qb_designed_run_rate": 0.03, "rz_pass_rate": 0.52,
        "g2g_run_rate": 0.55, "sec_per_play": 31, "adot": 8.0,
    })
    favoured = pj.team_volume(scheme, pj.GameContext("X", total_line=45, spread_line=+9))
    trailing = pj.team_volume(scheme, pj.GameContext("X", total_line=45, spread_line=-9))
    assert favoured["pass_rate"] < trailing["pass_rate"]
    # A favourite is still expected to score more.
    assert favoured["expected_off_td"] > trailing["expected_off_td"]


def test_sampler_rejects_non_finite_volume():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        pj._nb_sample(rng, np.nan, 1.5, 10)


# ------------------------------------------------------------ scheme transfer
def _fake_fingerprints():
    traits = {t: 0.5 for t in schemes.OFFENSE_IDENTITY}
    rows = []
    for season in (2024, 2025):
        for team, bump in (("AAA", 0.0), ("BBB", 0.3)):
            r = {"season": season, "team": team, "side": "offense"}
            r.update({k: v + bump for k, v in traits.items()})
            rows.append(r)
    return pd.DataFrame(rows)


def test_new_coach_pulls_projection_toward_their_previous_team():
    fp = _fake_fingerprints()
    league = fp.select_dtypes(np.number).mean()
    staff = coaches.Staff(
        team="AAA", play_caller="OC", continuity=False, confidence="high",
        oc={"name": "Someone", "new": True,
            "prior": [{"team": "BBB", "seasons": [2024, 2025], "role": "OC"}]},
    )
    out = coaches.project_fingerprint("AAA", staff, fp, "offense", league, anchor_season=2025)
    baseline = fp[(fp.team == "AAA") & (fp.season == 2025)]["motion_rate"].iloc[0]
    source = fp[(fp.team == "BBB") & (fp.season == 2025)]["motion_rate"].iloc[0]
    assert baseline < out["projected"]["motion_rate"] < source
    assert out["weights"]["coach"] > 0


def test_continuity_staff_stays_on_its_own_baseline():
    fp = _fake_fingerprints()
    league = fp.select_dtypes(np.number).mean()
    staff = coaches.Staff(team="AAA", continuity=True)
    out = coaches.project_fingerprint("AAA", staff, fp, "offense", league, anchor_season=2025)
    assert out["weights"]["coach"] == 0
    assert out["weights"]["team"] > out["weights"]["league"]


def test_coach_prior_never_reads_a_future_season():
    """A backtest anchored on 2024 must not see 2025, even if it is in the table."""
    fp = _fake_fingerprints()
    coach = {"prior": [{"team": "BBB", "seasons": [2024, 2025], "role": "OC"}]}
    _, _, labels = coaches.coach_prior_fingerprint(
        coach, fp, "offense", schemes.OFFENSE_IDENTITY, anchor_season=2024)
    assert labels and all("2025" not in l for l in labels)


# ------------------------------------------------------------------- usage
def _usage_frame(team="AAA", carry_share=0.55, seasons=(2024, 2025)):
    return pd.DataFrame([
        {"player_id": "P1", "season": s, "team": team, "carry_share": carry_share,
         "target_share": 0.10, "goalline_share": 0.5, "carries": 250, "targets": 50,
         "goalline_carries": 20}
        for s in seasons
    ])


def test_share_regresses_to_role_prior_when_player_is_new_to_the_team():
    u = _usage_frame(team="OLD")
    same, _ = um.project_share(u, "P1", "RB", 1, "carry", current_team="OLD")
    moved, _ = um.project_share(u, "P1", "RB", 1, "carry", current_team="NEW",
                                role_pull=um.role_pull_for(u, "P1", "RB", 1, "NEW", "carry"))
    prior = um.CARRY_SHARE_PRIOR[("RB", 1)]
    assert abs(moved - prior) < abs(same - prior)


def test_demotion_on_the_depth_chart_lowers_projected_share():
    u = _usage_frame(team="AAA")
    as_rb1, _ = um.project_share(u, "P1", "RB", 1, "carry", current_team="AAA")
    pull = um.role_pull_for(u, "P1", "RB", 2, "AAA", "carry")
    as_rb2, _ = um.project_share(u, "P1", "RB", 2, "carry", current_team="AAA", role_pull=pull)
    assert as_rb2 < as_rb1


def test_unknown_player_falls_back_to_the_role_prior():
    share, evidence = um.project_share(pd.DataFrame(), None, "WR", 1, "target")
    assert share == pytest.approx(um.TARGET_SHARE_PRIOR[("WR", 1)])
    assert evidence == 0.0


def test_qb_continuity_counts_shared_targets():
    plays = pd.DataFrame({
        "passer_player_id": ["QB1"] * 30 + ["QB2"] * 30,
        "receiver_player_id": ["WR1"] * 60,
    })
    n, conf = um.qb_continuity(plays, "QB1", "WR1")
    assert n == 30
    assert 0 < conf < 1
    assert um.qb_continuity(plays, None, "WR1") == (0, 0.0)


# ------------------------------------------------------------- availability
def test_availability_prior_applies_without_player_history():
    p, snap = av.project_availability(pd.DataFrame(), None, "WR", 1)
    assert p == pytest.approx(av.AVAILABILITY_PRIOR[("WR", 1)])
    assert 0 < snap <= 1


def test_out_designation_zeroes_availability():
    p, _ = av.project_availability(pd.DataFrame(), None, "RB", 1, injury_status="Out")
    assert p == 0.0


def test_availability_lowers_expectation_but_not_the_active_line():
    rng = np.random.default_rng(3)
    proj = pj.PlayerProjection(
        player_id="P", name="P", team="AAA", position="WR", depth_rank=1,
        samples={"rec_yards": np.full(20000, 60.0), "total_td": np.ones(20000)},
    )
    proj.apply_availability(0.5, rng)
    assert proj.conditional["rec_yards"].mean() == pytest.approx(60.0)
    assert proj.samples["rec_yards"].mean() == pytest.approx(30.0, rel=0.05)
    # The zeros are a real point mass, not a uniform shrink.
    assert (proj.samples["rec_yards"] == 0).mean() == pytest.approx(0.5, abs=0.02)


def test_prob_over_distinguishes_the_two_views():
    rng = np.random.default_rng(4)
    proj = pj.PlayerProjection(
        player_id="P", name="P", team="AAA", position="WR", depth_rank=1,
        samples={"rec_yards": np.full(10000, 60.0)},
    )
    proj.apply_availability(0.5, rng)
    assert proj.prob_over("rec_yards", 50, conditional=True) == pytest.approx(100.0)
    assert proj.prob_over("rec_yards", 50) == pytest.approx(50.0, abs=2)


# ---------------------------------------------------------------- sampling
def test_touch_sampler_preserves_skew_of_real_outcomes():
    plays = pd.DataFrame({
        "is_designed_run": [True] * 1000,
        "yards_gained": [2.0] * 950 + [60.0] * 50,
        "receiver_player_id": [None] * 1000,
        "complete_pass": [0] * 1000,
        "air_yards": [np.nan] * 1000,
    })
    s = pj.TouchSampler(plays, rng=np.random.default_rng(1))
    totals = s.sample_rush(np.full(400, 10))
    # A normal approximation would never produce this much right-tail mass.
    assert totals.max() > totals.mean() * 1.8
    assert totals.mean() == pytest.approx(10 * (2 * .95 + 60 * .05), rel=0.12)


def test_zero_touches_produce_zero_yards():
    plays = pd.DataFrame({
        "is_designed_run": [True] * 50, "yards_gained": [4.0] * 50,
        "receiver_player_id": [None] * 50, "complete_pass": [0] * 50,
        "air_yards": [np.nan] * 50,
    })
    s = pj.TouchSampler(plays, rng=np.random.default_rng(2))
    assert s.sample_rush(np.zeros(10, dtype=int)).sum() == 0


# ---------------------------------------------------------------- playbook
def test_signature_concepts_ignore_sacks_and_scrambles():
    n = 120
    plays = pd.DataFrame({
        "posteam": ["AAA"] * n, "season": [2025] * n, "play_id": range(n),
        "shotgun": [1] * n, "is_dropback": [True] * n,
        "sack": [1] * 60 + [0] * 60, "qb_scramble": [0] * n,
        "pass_length": ["short"] * n, "pass_location": ["left"] * n,
        "run_location": [None] * n, "run_gap": [None] * n,
        "epa": [-2.0] * 60 + [0.5] * 60, "yards_gained": [-7.0] * 60 + [8.0] * 60,
        "success": [0] * 60 + [1] * 60,
        "is_motion": [False] * n, "is_play_action": [False] * n,
        "is_rpo": [False] * n, "is_screen_pass": [False] * n,
    })
    out = playbook.signature_concepts(plays, "AAA", seasons=(2025,), min_plays=10)
    assert not out.empty
    # Only the 60 non-sack plays should survive.
    assert out["plays"].sum() == 60
    assert (out["epa"] > 0).all()


# --------------------------------------------------------------- game model
def _rating_frame():
    return pd.DataFrame([
        {"season": s, "team": t, "off_rating": o, "def_rating": d, "plays": 1000}
        for s in (2024, 2025)
        for t, o, d in [("AAA", 0.10, 0.05), ("BBB", -0.10, -0.05), ("CCC", 0.0, 0.0)]
    ])


def test_ratings_regress_toward_mean_and_recentre():
    from nflproj import gamemodel as gmod
    proj = gmod.project_ratings(_rating_frame(), anchor_season=2025)
    # Offense persists more than defense, so it should retain more of its edge.
    a = proj[proj.team == "AAA"].iloc[0]
    assert 0 < a.off_rating < 0.10
    assert 0 < a.def_rating < 0.05
    assert abs(a.off_rating) / 0.10 > abs(a.def_rating) / 0.05
    assert proj["off_rating"].mean() == pytest.approx(0.0, abs=1e-9)


def test_new_staff_shrinks_a_team_further():
    from nflproj import gamemodel as gmod
    base = gmod.project_ratings(_rating_frame(), anchor_season=2025)
    pen = gmod.project_ratings(_rating_frame(), anchor_season=2025,
                               coach_penalty={"AAA": 0.5})
    b = base[base.team == "AAA"].iloc[0].off_rating
    p = pen[pen.team == "AAA"].iloc[0].off_rating
    assert abs(p) < abs(b)


def test_home_field_favours_the_home_team():
    from nflproj import gamemodel as gmod
    proj = pd.DataFrame([{"team": "AAA", "off_rating": 0.0, "def_rating": 0.0},
                         {"team": "BBB", "off_rating": 0.0, "def_rating": 0.0}])
    sc = {"intercept": 22.0, "slope": 40.0, "hfa": 2.0,
          "margin_sigma": 13.0, "total_sigma": 13.0}
    g = gmod.predict_game("AAA", "BBB", proj, sc, n_sims=40000)
    assert g.margin == pytest.approx(2.0, abs=0.01)
    assert 0.52 < g.home_win_prob < 0.58


def test_margin_variance_is_not_understated_by_shared_scoring():
    """The two teams' scores are correlated; margin spread must stay calibrated."""
    from nflproj import gamemodel as gmod
    proj = pd.DataFrame([{"team": "AAA", "off_rating": 0.0, "def_rating": 0.0},
                         {"team": "BBB", "off_rating": 0.0, "def_rating": 0.0}])
    sc = {"intercept": 22.0, "slope": 40.0, "hfa": 0.0,
          "margin_sigma": 13.0, "total_sigma": 10.0}
    rng = np.random.default_rng(11)
    g = gmod.predict_game("AAA", "BBB", proj, sc, n_sims=200000, rng=rng)
    # A coin-flip game should sit at even money, not drift confident.
    assert g.home_win_prob == pytest.approx(0.50, abs=0.01)


def test_spread_edge_is_signed_against_the_market():
    from nflproj import gamemodel as gmod
    proj = pd.DataFrame([{"team": "AAA", "off_rating": 0.0, "def_rating": 0.0},
                         {"team": "BBB", "off_rating": 0.0, "def_rating": 0.0}])
    sc = {"intercept": 22.0, "slope": 40.0, "hfa": 6.0,
          "margin_sigma": 13.0, "total_sigma": 10.0}
    g = gmod.predict_game("AAA", "BBB", proj, sc, n_sims=2000, market_spread=3.0)
    assert g.spread_edge == pytest.approx(3.0, abs=0.01)   # model likes home by 3 more
    assert gmod.predict_game("AAA", "BBB", proj, sc, n_sims=2000).spread_edge is None


# ------------------------------------------------------------- environment
def test_wind_only_counts_outdoors_and_above_the_calm_threshold():
    from nflproj import venues as ven
    assert ven.environment({"roof": "dome", "wind": 30, "div_game": 0})["wind_excess"] == 0
    assert ven.environment({"roof": "outdoors", "wind": 5, "div_game": 0})["wind_excess"] == 0
    assert ven.environment({"roof": "outdoors", "wind": 20, "div_game": 0})["wind_excess"] == 12


def test_wind_suppresses_scoring_and_passing():
    from nflproj import venues as ven
    calm = ven.environment({"roof": "outdoors", "wind": 4, "div_game": 0})
    windy = ven.environment({"roof": "outdoors", "wind": 25, "div_game": 0})
    assert windy["total_delta"] < calm["total_delta"]
    assert windy["pass_yards_mult"] < 1.0
    assert windy["deep_rate_mult"] < windy["pass_yards_mult"]   # deep shots suffer most
    assert windy["pass_rate_delta"] < 0                          # teams throw less


def test_unknown_weather_produces_no_adjustment():
    """Future games have no weather; the model must not invent one."""
    from nflproj import venues as ven
    e = ven.environment({"roof": "outdoors", "wind": np.nan, "div_game": 0})
    assert e["total_delta"] == pytest.approx(0.0)
    assert e["pass_yards_mult"] == 1.0


def test_roof_and_divisional_apply_without_weather():
    from nflproj import venues as ven
    dome = ven.environment({"roof": "dome", "wind": np.nan, "div_game": 0})
    div = ven.environment({"roof": "outdoors", "wind": np.nan, "div_game": 1})
    assert dome["total_delta"] > 0      # domes score more
    assert div["total_delta"] < 0       # divisional games score less


def test_travel_is_reported_but_never_applied():
    from nflproj import venues as ven
    t = ven.travel_context("SEA", "MIA", 13.0)
    assert t["travel_miles"] > 2500
    assert t["tz_shift"] == -3            # Seattle travels east... to Eastern time
    assert t["away_body_clock"] == 10.0   # 1pm ET is 10am to a Pacific body clock
    assert t["applied_to_projection"] is False
    assert "total_delta" not in t         # carries no scoring weight


def test_wind_shifts_volume_toward_the_run():
    from nflproj import venues as ven
    scheme = pd.Series({
        "plays_per_game": 63, "early_down_pass_rate": 0.55, "sack_rate_allowed": 0.06,
        "scramble_rate": 0.06, "qb_designed_run_rate": 0.03, "rz_pass_rate": 0.52,
        "g2g_run_rate": 0.55, "sec_per_play": 31, "adot": 8.0,
    })
    gc = pj.GameContext("X", total_line=45, spread_line=0)
    calm = pj.team_volume(scheme, gc, env=ven.environment({"roof": "outdoors", "wind": 3, "div_game": 0}))
    windy = pj.team_volume(scheme, gc, env=ven.environment({"roof": "outdoors", "wind": 25, "div_game": 0}))
    assert windy["pass_rate"] < calm["pass_rate"]
    assert windy["rb_carries"] > calm["rb_carries"]
    assert windy["expected_off_td"] < calm["expected_off_td"]


# --------------------------------------------------------- season-type policy
def _mixed_season_plays():
    n = 4
    return pd.DataFrame({
        "season_type": ["PRE", "REG", "POST", "PRE"],
        "posteam": ["AAA"] * n, "defteam": ["BBB"] * n,
        "play_type": ["run"] * n, "special": [0] * n,
        "qb_kneel": [0] * n, "qb_spike": [0] * n,
        "wp": [0.5] * n, "down": [1] * n, "pass": [0] * n, "rush": [1] * n,
        "qb_dropback": [0] * n, "qb_scramble": [0] * n,
        "yards_gained": [4] * n, "yardline_100": [50] * n,
        "game_id": [f"g{i}" for i in range(n)], "play_id": range(n),
    })


def test_preseason_is_always_excluded():
    """Preseason must never reach the model, even if a feed starts shipping it."""
    out = schemes.prepare_plays(_mixed_season_plays())
    assert "PRE" not in set(out["season_type"])
    # And it stays excluded even when postseason is explicitly requested.
    out2 = schemes.prepare_plays(_mixed_season_plays(), include_postseason=True)
    assert "PRE" not in set(out2["season_type"])


def test_postseason_is_off_by_default_and_opt_in():
    default = schemes.prepare_plays(_mixed_season_plays())
    assert set(default["season_type"]) == {"REG"}
    opted_in = schemes.prepare_plays(_mixed_season_plays(), include_postseason=True)
    assert set(opted_in["season_type"]) == {"REG", "POST"}


def test_allowed_season_types_never_includes_preseason():
    from nflproj import config as cfg
    for flag in (True, False, None):
        allowed = cfg.allowed_season_types(flag)
        assert not any(x in allowed for x in cfg.EXCLUDED_SEASON_TYPES)
        assert "REG" in allowed


def test_availability_ignores_non_regular_season_snaps():
    snaps = pd.DataFrame({
        "game_type": ["REG", "REG", "PRE", "WC"],
        "season": [2025] * 4, "team": ["AAA"] * 4,
        "game_id": ["a", "b", "c", "d"],
        "pfr_player_id": ["P1"] * 4,
        "offense_snaps": [50, 50, 50, 50], "offense_pct": [0.9] * 4,
    })
    players = pd.DataFrame({"gsis_id": ["00-1"], "pfr_id": ["P1"]})
    out = av.player_availability(snaps, players)
    # Two regular-season games out of two counted; the other rows are dropped.
    assert int(out["games"].iloc[0]) == 2
    assert int(out["team_games"].iloc[0]) == 2


# ---------------------------------------------------------------- coverage
def test_coverage_fingerprint_separates_man_and_zone():
    from nflproj import coverage as cvg
    n = 200
    plays = pd.DataFrame({
        "season": [2025] * n, "defteam": ["AAA"] * n, "posteam": ["BBB"] * n,
        "play_id": range(n), "epa": np.linspace(-1, 1, n),
        "defense_man_zone_type": ["MAN_COVERAGE"] * 80 + ["ZONE_COVERAGE"] * 120,
        "defense_coverage_type": ["COVER_1"] * 80 + ["COVER_3"] * 60 + ["COVER_2"] * 60,
        "yards_gained": [8.0] * n, "complete_pass": [1] * n, "air_yards": [9.0] * n,
        "was_pressure": [False] * n,
    })
    fp = cvg.coverage_fingerprint(plays, seasons=(2025,))
    r = fp.iloc[0]
    assert r["man_rate"] == pytest.approx(0.40)
    assert r["zone_rate"] == pytest.approx(0.60)
    # Cover 1 and Cover 3 are single-high; Cover 2 is two-high.
    assert r["single_high_rate"] == pytest.approx(0.70)
    assert r["two_high_rate"] == pytest.approx(0.30)


def test_coverage_requires_a_minimum_sample():
    from nflproj import coverage as cvg
    tiny = pd.DataFrame({
        "season": [2025] * 5, "defteam": ["AAA"] * 5, "posteam": ["BBB"] * 5,
        "play_id": range(5), "epa": [0.0] * 5,
        "defense_man_zone_type": ["MAN_COVERAGE"] * 5,
        "defense_coverage_type": ["COVER_1"] * 5,
        "yards_gained": [5.0] * 5, "complete_pass": [1] * 5, "air_yards": [8.0] * 5,
    })
    assert cvg.coverage_fingerprint(tiny, seasons=(2025,)).empty


# ---------------------------------------------------------------- blocking
def test_rushing_splits_into_blocking_and_back():
    from nflproj import blocking as blk
    tb = pd.DataFrame([{"season": 2025, "team": "AAA", "carries": 400,
                        "ybc": 400 * 3.2, "yac": 400 * 1.9,
                        "ybc_per_carry": 3.2, "yac_per_carry": 1.9}])
    pe = pd.DataFrame([{"season": 2025, "player_id": "P1", "carries": 250,
                        "yac_per_carry": 2.4, "broken_per_carry": 0.15}])
    good = blk.project_rushing_efficiency("AAA", "P1", tb, pe)
    assert good["yards_before_contact"] > blk.LEAGUE_YBC   # good line shows up
    assert good["ypc"] == pytest.approx(good["yards_before_contact"] + good["yards_after_contact"])


def test_blocking_persists_more_than_elusiveness():
    """The measured asymmetry must be reflected in how far each is regressed."""
    from nflproj import blocking as blk
    assert blk.TEAM_BLOCKING_PERSISTENCE > blk.PLAYER_ELUSIVENESS_PERSISTENCE
    tb = pd.DataFrame([{"season": 2025, "team": "AAA", "carries": 400,
                        "ybc_per_carry": 3.45, "yac_per_carry": 1.85}])
    pe = pd.DataFrame([{"season": 2025, "player_id": "P1", "carries": 400,
                        "yac_per_carry": 2.45}])
    r = blk.project_rushing_efficiency("AAA", "P1", tb, pe)
    # Both are a full point above league; blocking should retain more of it.
    assert r["blocking_vs_league"] > r["elusiveness_vs_league"]


def test_unknown_team_falls_back_to_league_blocking():
    from nflproj import blocking as blk
    r = blk.project_rushing_efficiency("ZZZ", None, pd.DataFrame(), pd.DataFrame())
    assert r["yards_before_contact"] == pytest.approx(blk.LEAGUE_YBC)
    assert r["has_player_sample"] is False


# ----------------------------------------------------------------- kicking
def test_field_goal_probability_falls_with_distance():
    from nflproj import kicking as kik
    assert kik.make_probability(25) > kik.make_probability(45) > kik.make_probability(60)
    assert 0.95 < kik.make_probability(25) < 1.0
    assert 0.45 < kik.make_probability(58) < 0.65


def test_kicker_skill_is_heavily_regressed():
    from nflproj import kicking as kik
    hist = pd.DataFrame([{"season": 2025, "player_id": "K1", "attempts": 30,
                          "fg_over_expected": 0.10}])
    skill = kik.project_kicker_accuracy(hist, "K1")
    assert 0 < skill < 0.10          # a strong season moves it only part way
    assert kik.project_kicker_accuracy(hist, "UNKNOWN") == 0.0


def test_no_field_goal_wind_coefficient_exists():
    """Wind on FG accuracy was tested and not supported; it must stay out."""
    from nflproj import venues as ven
    assert not hasattr(ven, "WIND_FG_PCT_PER_MPH")
    assert "fg_pct_mult" not in ven.environment({"roof": "outdoors", "wind": 25, "div_game": 0})


# ------------------------------------------------------------ draft capital
def test_draft_capital_orders_by_pick():
    from nflproj import usage as usg
    draft = pd.DataFrame({"gsis_id": ["A", "B", "C"], "pick": [3, 50, 240]})
    a = usg.draft_multiplier(draft, "A")
    b = usg.draft_multiplier(draft, "B")
    c = usg.draft_multiplier(draft, "C")
    assert a > b > c
    assert usg.draft_multiplier(draft, "UNKNOWN") == usg.UNDRAFTED_MULTIPLIER


# --------------------------------------------------- point-in-time charts
def test_depth_chart_respects_as_of_cutoff():
    """A Week 3 projection must not see a Week 18 chart."""
    from nflproj import data as dat
    early = dat.depth_chart(2025, as_of="2025-09-15")
    late = dat.depth_chart(2025)
    if early.empty or late.empty:
        pytest.skip("depth chart cache unavailable")
    assert early["dt"].max() <= pd.Timestamp("2025-09-15", tz="UTC")
    assert early["dt"].max() < late["dt"].max()


# ------------------------------------------------------ practice status
def test_practice_participation_lowers_availability():
    p_full, s_full = av.project_availability(pd.DataFrame(), None, "WR", 1,
                                             practice_status="Full Participation in Practice")
    p_dnp, s_dnp = av.project_availability(pd.DataFrame(), None, "WR", 1,
                                           practice_status="Did Not Participate In Practice")
    assert p_dnp < p_full
    assert s_dnp < s_full


def test_pass_protection_moves_the_sack_rate():
    scheme = pd.Series({
        "plays_per_game": 63, "early_down_pass_rate": 0.55, "sack_rate_allowed": 0.07,
        "scramble_rate": 0.06, "qb_designed_run_rate": 0.03, "rz_pass_rate": 0.52,
        "g2g_run_rate": 0.55, "sec_per_play": 31, "adot": 8.0,
    })
    gc = pj.GameContext("X", total_line=45, spread_line=0)
    good = pj.team_volume(scheme, gc, protection=0.70)   # allows little pressure
    bad = pj.team_volume(scheme, gc, protection=1.40)
    assert good["sacks"] < bad["sacks"]
    assert good["attempts"] > bad["attempts"]


# ------------------------------------------------------------- odds arithmetic
def test_american_odds_round_trip():
    from nflproj import picks as pk
    for odds in (-250, -110, 100, 150, 400):
        p = pk.american_to_probability(odds)
        assert 0 < p < 1
        assert pk.probability_to_american(p) == pytest.approx(odds, rel=1e-6)


def test_favourite_and_underdog_prices_have_the_right_sign():
    from nflproj import picks as pk
    assert pk.probability_to_american(0.75) < 0     # favourite is negative
    assert pk.probability_to_american(0.25) > 0     # underdog is positive
    assert pk.american_to_probability(-110) > 0.5


def test_expected_value_is_zero_at_the_fair_price():
    from nflproj import picks as pk
    p = 0.62
    fair = pk.probability_to_american(p)
    assert pk.expected_value(p, fair) == pytest.approx(0.0, abs=1e-6)
    # Beating the fair price is positive, paying worse than it is negative.
    assert pk.expected_value(p, fair + 60) > 0
    assert pk.expected_value(0.50, -110) < 0


def test_kelly_declines_a_negative_edge():
    from nflproj import picks as pk
    assert pk.kelly_fraction(0.50, -110) < 0
    assert pk.kelly_fraction(0.70, +100) > 0


def test_lines_track_each_player_own_distribution():
    """A backup must not be handed a star's line and scored a 90% under."""
    from nflproj import picks as pk
    rng = np.random.default_rng(0)
    star = pk.candidate_lines(rng.gamma(4, 21, 20000), "rush_yards")     # ~84 yds
    backup = pk.candidate_lines(rng.gamma(3, 7, 20000), "rush_yards")    # ~21 yds
    assert star and backup
    assert min(star) > max(backup)


def test_every_offered_line_sits_near_a_coin_flip():
    """Lines chosen by rounding produced 79% unders; selection is on probability."""
    from nflproj import picks as pk
    rng = np.random.default_rng(1)
    for stat, sample in (("rec_yards", rng.gamma(3, 20, 20000)),
                         ("receptions", rng.poisson(4.2, 20000).astype(float)),
                         ("carries", rng.poisson(11.0, 20000).astype(float))):
        for line in pk.candidate_lines(sample, stat):
            p_over = float((sample > line).mean())
            assert pk.TARGET_BAND[0] <= p_over <= pk.TARGET_BAND[1], (stat, line, p_over)


def test_markets_below_a_floor_are_not_offered():
    from nflproj import picks as pk
    assert pk.candidate_lines(np.full(500, 3.0), "rush_yards") == []
    assert pk.candidate_lines(np.zeros(500), "total_td") == []


# ------------------------------------------------------------------- parlay
class _FakeGame:
    """Two perfectly correlated legs and one independent, for exact arithmetic."""

    def __init__(self, n=20000, seed=0):
        rng = np.random.default_rng(seed)
        self.n_sims = n
        base = rng.random(n)
        self.players = {
            "A": {"yards": base * 100},
            "B": {"yards": base * 100},          # identical to A
            "C": {"yards": rng.random(n) * 100},  # independent
        }
        self.meta = {k: {"position": "WR", "team": "AAA", "depth_rank": 1,
                         "p_active": 1.0, "player_id": k} for k in self.players}
        self.active = {k: np.ones(n, dtype=bool) for k in self.players}
        self.home, self.away = "AAA", "BBB"

    def stat(self, p, s):
        return self.players.get(p, {}).get(s)

    def active_mask(self, *players):
        m = np.ones(self.n_sims, dtype=bool)
        for p in players:
            m &= self.active[p]
        return m

    def leg_mask(self, p, s, line, side="over"):
        v = self.stat(p, s)
        return None if v is None else ((v > line) if side == "over" else (v <= line))


def test_perfectly_correlated_legs_beat_the_naive_product():
    from nflproj import parlay as pl
    g = _FakeGame()
    legs = [pl.Leg("A", "yards", 50), pl.Leg("B", "yards", 50)]
    r = pl.evaluate(legs, [g])
    # Identical legs: the parlay is just the single leg, not its square.
    assert r["probability"] == pytest.approx(0.5, abs=0.02)
    assert r["naive_probability"] == pytest.approx(0.25, abs=0.02)
    assert r["correlation_lift"] == pytest.approx(2.0, rel=0.08)


def test_independent_legs_match_the_naive_product():
    from nflproj import parlay as pl
    g = _FakeGame()
    r = pl.evaluate([pl.Leg("A", "yards", 50), pl.Leg("C", "yards", 50)], [g])
    assert r["probability"] == pytest.approx(r["naive_probability"], abs=0.02)
    assert r["correlation_lift"] == pytest.approx(1.0, abs=0.08)


def test_opposing_legs_fall_below_the_naive_product():
    from nflproj import parlay as pl
    g = _FakeGame()
    # A over and B under are mutually exclusive, since B mirrors A exactly.
    r = pl.evaluate([pl.Leg("A", "yards", 50, "over"),
                     pl.Leg("B", "yards", 50, "under")], [g])
    assert r["probability"] == pytest.approx(0.0, abs=0.01)
    assert r["naive_probability"] > 0.2


def test_parlay_reports_a_fair_price_and_expected_value():
    from nflproj import parlay as pl
    g = _FakeGame()
    legs = [pl.Leg("A", "yards", 50, odds=-110), pl.Leg("C", "yards", 50, odds=-110)]
    r = pl.evaluate(legs, [g])
    assert "offered_american" in r and "expected_value_per_100" in r
    assert r["breakeven_probability"] == pytest.approx(1 / r["offered_decimal"])


def test_parlay_rejects_a_player_it_never_simulated():
    from nflproj import parlay as pl
    r = pl.evaluate([pl.Leg("Nobody", "yards", 50)], [_FakeGame()])
    assert "error" in r


def test_correlation_matrix_recovers_the_structure():
    from nflproj import parlay as pl
    g = _FakeGame()
    cm = pl.correlation_matrix([pl.Leg("A", "yards", 50), pl.Leg("B", "yards", 50),
                                pl.Leg("C", "yards", 50)], [g])
    assert cm.iloc[0, 1] == pytest.approx(1.0, abs=0.02)   # identical
    assert cm.iloc[0, 2] == pytest.approx(0.0, abs=0.05)   # independent


# -------------------------------------------------- joint simulation shape
def test_dirichlet_shares_sum_to_one_per_simulation():
    from nflproj import joint as jt
    rng = np.random.default_rng(0)
    shares = jt._dirichlet_shares(rng, np.array([0.4, 0.35, 0.25]), 30.0, 500)
    assert shares.shape == (500, 3)
    assert np.allclose(shares.sum(axis=1), 1.0)
    # The mean should track the base allocation it was given.
    assert shares.mean(axis=0)[0] == pytest.approx(0.4, abs=0.05)


def test_shared_shocks_are_configured_to_be_nonzero():
    """Zeroed shocks would silently collapse the model to independence."""
    from nflproj import joint as jt
    assert jt.PACE_SHOCK_SD > 0
    assert jt.SCORING_SHOCK_SD > 0
    assert jt.MARGIN_SHOCK_SD > 0
    assert jt.SHOOTOUT_PLAYS > 0


# ------------------------------------------------------- opponent adjustment
def test_defence_quality_is_projected_not_dropped():
    """Without these columns every opponent silently evaluates as average."""
    from nflproj import schemes as sch
    assert sch.DEFENSE_QUALITY, "no defensive quality traits declared"
    # Quality must not be mistaken for identity: it does not follow a coach.
    assert not (set(sch.DEFENSE_QUALITY) & set(sch.DEFENSE_IDENTITY))
    for trait, persistence in sch.DEFENSE_QUALITY_PERSISTENCE.items():
        assert 0.0 < persistence < 0.5, f"{trait} persistence looks wrong"


def test_defence_adjustment_separates_opponents():
    """A good and a bad defence must not produce the same multiplier."""
    league = pd.Series({"ypa_allowed": 6.4, "ypc_allowed": 4.3,
                        "points_per_drive_allowed": 2.0})
    good = pd.Series({"ypa_allowed": 5.6, "ypc_allowed": 3.9,
                      "points_per_drive_allowed": 1.6})
    bad = pd.Series({"ypa_allowed": 7.2, "ypc_allowed": 4.8,
                     "points_per_drive_allowed": 2.5})
    a = pj.defense_adjustment(good, league)
    b = pj.defense_adjustment(bad, league)
    assert a["pass_yards"] < 1.0 < b["pass_yards"]
    assert a["rush_yards"] < b["rush_yards"]


def test_missing_quality_columns_fall_back_to_neutral():
    league = pd.Series({"ypa_allowed": 6.4})
    assert pj.defense_adjustment(pd.Series(dtype=float), league)["pass_yards"] == 1.0
    assert pj.defense_adjustment(None, None)["pass_yards"] == 1.0


# ------------------------------------------------------------------- news
def test_notes_cap_how_far_they_can_move_a_projection():
    from nflproj import news as nw
    notes = pd.DataFrame([
        {"player": "X", "effect": "usage_up", "magnitude": 0.9},
        {"player": "X", "effect": "usage_up", "magnitude": 0.9},
    ])
    assert nw.usage_multiplier(notes, None, "X") <= nw.MAX_USAGE_MULTIPLIER
    down = pd.DataFrame([{"player": "X", "effect": "usage_down", "magnitude": 0.9}])
    assert nw.usage_multiplier(down, None, "X") >= nw.MIN_USAGE_MULTIPLIER


def test_note_matching_is_case_insensitive_and_scoped():
    from nflproj import news as nw
    notes = pd.DataFrame([{"player": "Malik Nabers", "effect": "usage_up",
                           "magnitude": 0.2}])
    assert nw.usage_multiplier(notes, None, "malik nabers") > 1.0
    assert nw.usage_multiplier(notes, None, "Someone Else") == 1.0


def test_out_note_overrides_availability():
    from nflproj import news as nw
    out = pd.DataFrame([{"player": "X", "effect": "out"}])
    q = pd.DataFrame([{"player": "X", "effect": "questionable"}])
    assert nw.availability_override(out, None, "X") == 0.0
    assert 0 < nw.availability_override(q, None, "X") < 1
    assert nw.availability_override(pd.DataFrame(), None, "X") is None


def test_expired_notes_are_dropped(tmp_path):
    from nflproj import news as nw
    import yaml as _yaml
    from datetime import date
    p = tmp_path / "n.yaml"
    p.write_text(_yaml.safe_dump({"notes": [
        {"player": "Old", "effect": "usage_up", "magnitude": 0.2, "expires": "2020-01-01"},
        {"player": "Live", "effect": "usage_up", "magnitude": 0.2, "expires": "2099-01-01"},
    ]}))
    loaded = nw.load_notes(p, as_of=date(2026, 9, 1))
    assert list(loaded["player"]) == ["Live"]


def test_unknown_effects_are_ignored(tmp_path):
    from nflproj import news as nw
    import yaml as _yaml
    p = tmp_path / "n.yaml"
    p.write_text(_yaml.safe_dump({"notes": [
        {"player": "A", "effect": "hot_hand", "magnitude": 0.5},
        {"player": "B", "effect": "usage_up", "magnitude": 0.1},
    ]}))
    assert list(nw.load_notes(p)["player"]) == ["B"]
