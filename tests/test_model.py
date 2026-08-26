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
