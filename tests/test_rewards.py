"""Tests for the reward system: signed points, old-format conversion, custom rules."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autodrive.rewards import (DEFAULT_TERMS, TERMS, RewardComputer, RewardConfigError,  # noqa: E402
                               load_reward_config, parse_config)


def base_m(**kw):
    """A neutral step measurement (nothing happens, car rolling at 5 m/s)."""
    m = dict(ds=0.5, parked=False, park_err=0.0, collision=False, ped_hit=False, ped_close=0.0,
             crosswalk_violation=False, speed=5.0, limit=14.0, red_violation=False, stop_violation=False,
             lat=0.0, on_street=True, offroad_frac=0.0, jerk=0.0, steer_rate=0.0, legit_wait=False,
             off_route=False, stuck=False, meas=None)
    m.update(kw)
    return m


def meas(**kw):
    d = dict(speed_kmh=18.0, accel=0.0, steering_pct=0.0, distance_m=10.0, to_spot_m=50.0, time_s=3.0,
             lane_offset_m=0.0, heading_err_deg=0.0, off_road=False, in_lot=False, nearest_car_m=999.0,
             nearest_pedestrian_m=999.0, obstacle_ahead_m=999.0, crashed=False, hit_pedestrian=False,
             parked=False, ran_red_light=False, rolled_stop_sign=False, went_off_route=False, got_stuck=False)
    d.update(kw)
    return d


def test_old_format_converts_to_signed_points():
    cfg = load_reward_config(ROOT / "experiments/balanced_45min/reward_config.yaml")
    for k, t in DEFAULT_TERMS.items():
        if t["enabled"]:
            assert cfg["terms"][k]["weight"] == pytest.approx(t["weight"], rel=0.01), k


def test_speed_terms_now_per_kmh_and_old_files_keep_their_behaviour():
    # format 2 counted per m/s: -0.05 per m/s == -0.05/3.6 per km/h
    old = parse_config({"format": 2, "terms": {"speeding": {"weight": -0.05}}})
    assert old["terms"]["speeding"]["weight"] == pytest.approx(-0.05 / 3.6)
    new = parse_config({"format": 3, "terms": {"speeding": {"weight": -0.05}}})
    assert new["terms"]["speeding"]["weight"] == pytest.approx(-0.05)
    # 10 km/h over the limit (+ tolerance) -> 10 x weight
    m = base_m(speed=(30 + 3 + 10) / 3.6, limit=30 / 3.6)
    total, parts, _ = RewardComputer(new)(m)
    assert parts["speeding"] == pytest.approx(-0.5, rel=1e-6)


def test_number_is_points_given_any_sign():
    for w in (-50, 50):
        cfg = parse_config({"format": 2, "terms": {"collision": {"weight": w}}})
        total, parts, _ = RewardComputer(cfg)(base_m(collision=True))
        assert total == pytest.approx(w) and parts["collision"] == pytest.approx(w)


def test_old_format_rejects_negative_with_hint():
    with pytest.raises(RewardConfigError, match="format: 2"):
        parse_config({"terms": {"collision": {"weight": -5}}})


def test_empty_config_error_or_allowed():
    with pytest.raises(RewardConfigError):
        parse_config({"format": 2, "terms": {}})
    cfg = parse_config({"format": 2, "terms": {}}, allow_empty=True)
    assert RewardComputer(cfg)(base_m())[0] == 0.0


def custom_cfg(**c):
    rule = dict(name="r", when=[], give=-1.0, times=None, paid="every_step")
    rule.update(c)
    return parse_config({"format": 2, "custom": [rule]})


def test_custom_threshold_every_step():
    cfg = custom_cfg(when=[[{"m": "speed_kmh", "op": ">", "v": 30}]], give=-1)
    rc = RewardComputer(cfg)
    assert rc(base_m(meas=meas(speed_kmh=20)))[0] == 0.0
    assert rc(base_m(meas=meas(speed_kmh=40)))[0] == -1.0
    assert rc(base_m(meas=meas(speed_kmh=45)))[0] == -1.0          # every step while true


def test_custom_on_change_pays_on_transitions_only():
    cfg = custom_cfg(when=[[{"m": "speed_kmh", "op": "<", "v": 4}]], give=-2, paid="on_change")
    rc = RewardComputer(cfg)
    seq = [10, 3, 2, 1, 10, 3]           # becomes true at index 1 and 5
    got = [rc(base_m(meas=meas(speed_kmh=v)))[0] for v in seq]
    assert got == [0, -2, 0, 0, 0, -2]
    rc.reset()                            # new episode: memory cleared
    assert rc(base_m(meas=meas(speed_kmh=1)))[0] == -2


def test_custom_groups_any_of_all():
    near_and_fast = [{"m": "nearest_pedestrian_m", "op": "<", "v": 6}, {"m": "speed_kmh", "op": ">", "v": 15}]
    cfg = custom_cfg(when=[near_and_fast, [{"m": "crashed", "op": "is", "v": True}]], give=-3)
    rc = RewardComputer(cfg)
    assert rc(base_m(meas=meas(nearest_pedestrian_m=4, speed_kmh=10)))[0] == 0     # only one of group 1
    assert rc(base_m(meas=meas(nearest_pedestrian_m=4, speed_kmh=20)))[0] == -3    # all of group 1
    assert rc(base_m(meas=meas(crashed=True)))[0] == -3                             # group 2


def test_custom_times_measurement_and_no_condition():
    cfg = custom_cfg(when=[], give=0.05, times="speed_kmh")
    total, parts, _ = RewardComputer(cfg)(base_m(meas=meas(speed_kmh=40)))
    assert total == pytest.approx(2.0) and parts["custom_r"] == pytest.approx(2.0)


@pytest.mark.parametrize("bad, msg", [
    (dict(when=[[{"m": "warp_speed", "op": ">", "v": 1}]]), "unknown measurement"),
    (dict(when=[[{"m": "speed_kmh", "op": "~", "v": 1}]]), "unknown operator"),
    (dict(give="lots"), "must be a number"),
    (dict(times="crashed"), "times"),
    (dict(paid="sometimes"), "paid"),
])
def test_custom_validation_messages(bad, msg):
    with pytest.raises(RewardConfigError, match=msg):
        custom_cfg(**bad)


def test_presets_and_template_load():
    for f in ["rewards.yaml", "rewards/balanced.yaml", "rewards/cautious.yaml", "rewards/aggressive.yaml"]:
        load_reward_config(ROOT / f)
    load_reward_config(ROOT / "rewards/template.yaml", allow_empty=True)


def test_approach_speed_plus_speeding_equals_old_speeding():
    w = -0.02
    cfg = parse_config({"format": 3, "terms": {"speeding": {"weight": w}, "approach_speed": {"weight": w}}})
    # game street (posted 108 km/h), advice 20 km/h before the spot, car at 43 km/h -> 20 km/h over (+3 free)
    m = base_m(speed=43 / 3.6, limit=30.0, advice=20 / 3.6)
    _, parts, _ = RewardComputer(cfg)(m)
    assert parts["speeding"] == 0
    assert parts["approach_speed"] == pytest.approx(w * 20, rel=1e-6)
    # school zone 30 km/h, advice 20, car 43 -> speeding 10 over, approach adds the other 10
    m = base_m(speed=43 / 3.6, limit=30 / 3.6, advice=20 / 3.6)
    _, parts, _ = RewardComputer(cfg)(m)
    assert parts["speeding"] + parts["approach_speed"] == pytest.approx(w * 20, rel=1e-6)


def test_old_files_get_approach_speed_from_speeding():
    old = parse_config({"format": 2, "terms": {"speeding": {"weight": -0.05}}})
    assert old["terms"]["approach_speed"]["enabled"]
    assert old["terms"]["approach_speed"]["weight"] == pytest.approx(-0.05 / 3.6)
