"""
rewards.py - the configurable reward function.

Every reward TERM is computed from plain measurements of what happened in
the last 0.1 s step (how far we progressed, did we hit something, how fast
were we going vs the limit, ...). The YAML config decides for each term
whether it is ENABLED and its WEIGHT:

    reward_t = sum over enabled terms of   weight * signal_t

Signals are defined so that a POSITIVE weight always means "this is good"
for bonuses and "this is bad" for penalties (we subtract penalties). In the
YAML you therefore always write positive numbers; the sign convention is
listed in TERMS below (+1 bonus, -1 penalty).

The per-term values are returned as a dict so the HUD, the logs and the
reward-hacking detector can show exactly where the reward came from.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

# name: (sign, kind, short description)
#   kind "step"  = paid every step proportional to a signal
#   kind "event" = paid once when something happens
TERMS = {
    "progress":             (+1, "step",  "metres advanced along the route"),
    "goal_reached":         (+1, "event", "parked successfully in the target slot"),
    "parking_precision":    (+1, "event", "bonus x (1 - parking error) on success"),
    "collision":            (-1, "event", "hit a car, parked car or building (episode ends)"),
    "pedestrian_hit":       (-1, "event", "hit a pedestrian (episode ends)"),
    "pedestrian_proximity": (-1, "step",  "driving fast close to a pedestrian"),
    "crosswalk_yield":      (-1, "event", "drove over a crosswalk a pedestrian was using"),
    "speeding":             (-1, "step",  "m/s above the speed limit (+ tolerance)"),
    "red_light":            (-1, "event", "crossed a stop line on red"),
    "stop_sign":            (-1, "event", "rolled through a stop sign without stopping"),
    "lane_deviation":       (-1, "step",  "metres away from the lane centre (beyond deadband)"),
    "off_road":             (-1, "step",  "any wheel on the sidewalk / off the asphalt"),
    "comfort":              (-1, "step",  "jerk + fast steering (passenger comfort)"),
    "stall":                (-1, "step",  "standing still with no reason to"),
    "time":                 (-1, "step",  "flat cost per step (hurry up)"),
    "off_route":            (-1, "event", "left the route by > off_route_distance (episode ends)"),
    "speed_bonus":          (+1, "step",  "raw speed, regardless of direction (HACKABLE!)"),
}

DEFAULT_SETTINGS = {
    "speed_tolerance_kmh": 3.0,
    "lane_deadband_m": 0.4,
    "ped_proximity_radius_m": 4.0,
    "stall_speed_ms": 0.5,
    "off_route_distance_m": 6.0,
    "terminate_on_red_light": False,
    "terminate_on_stop_sign": False,
}

DEFAULT_TERMS = {
    "progress": dict(enabled=True, weight=0.1),
    "goal_reached": dict(enabled=True, weight=30.0),
    "parking_precision": dict(enabled=True, weight=15.0),
    "collision": dict(enabled=True, weight=50.0),
    "pedestrian_hit": dict(enabled=True, weight=100.0),
    "pedestrian_proximity": dict(enabled=True, weight=0.1),
    "crosswalk_yield": dict(enabled=True, weight=10.0),
    "speeding": dict(enabled=True, weight=0.05),
    "red_light": dict(enabled=True, weight=20.0),
    "stop_sign": dict(enabled=True, weight=10.0),
    "lane_deviation": dict(enabled=True, weight=0.03),
    "off_road": dict(enabled=True, weight=0.2),
    "comfort": dict(enabled=True, weight=0.01),
    "stall": dict(enabled=True, weight=0.03),
    "time": dict(enabled=True, weight=0.005),
    "off_route": dict(enabled=True, weight=20.0),
    "speed_bonus": dict(enabled=False, weight=0.02),
}


def load_reward_config(path=None) -> dict:
    """Load a YAML reward config and fill in anything missing with defaults."""
    cfg = {"terms": copy.deepcopy(DEFAULT_TERMS), "settings": dict(DEFAULT_SETTINGS),
           "name": "defaults"}
    if path is not None:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        cfg["name"] = user.get("name", Path(path).stem)
        cfg["description"] = user.get("description", "")
        for k, v in (user.get("terms") or {}).items():
            if k not in TERMS:
                raise ValueError(f"Unknown reward term '{k}' in {path}. Known: {list(TERMS)}")
            cfg["terms"][k].update(v or {})
        for k, v in (user.get("settings") or {}).items():
            if k not in DEFAULT_SETTINGS:
                raise ValueError(f"Unknown setting '{k}' in {path}. Known: {list(DEFAULT_SETTINGS)}")
            cfg["settings"][k] = v
    for k, t in cfg["terms"].items():
        t["weight"] = float(t.get("weight", 0.0))
        t["enabled"] = bool(t.get("enabled", True))
        if t["weight"] < 0:
            raise ValueError(f"Weight of '{k}' is negative. Use positive weights; the sign is "
                             f"fixed per term (see TERMS in autodrive/rewards.py).")
    return cfg


class RewardComputer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.terms = cfg["terms"]
        self.s = cfg["settings"]

    def signals(self, m: dict) -> dict:
        """Turn raw step measurements into unweighted, non-negative signals."""
        st = self.s
        sig = {}
        sig["progress"] = m["ds"]                      # can be negative if going backwards
        sig["goal_reached"] = float(m["parked"])
        sig["parking_precision"] = (1.0 - m["park_err"]) if m["parked"] else 0.0
        sig["collision"] = float(m["collision"])
        sig["pedestrian_hit"] = float(m["ped_hit"])
        sig["pedestrian_proximity"] = m["ped_close"] * min(1.0, m["speed"] / 3.0)
        sig["crosswalk_yield"] = float(m["crosswalk_violation"])
        tol = st["speed_tolerance_kmh"] / 3.6
        sig["speeding"] = max(0.0, m["speed"] - (m["limit"] + tol))
        sig["red_light"] = float(m["red_violation"])
        sig["stop_sign"] = float(m["stop_violation"])
        sig["lane_deviation"] = max(0.0, abs(m["lat"]) - st["lane_deadband_m"]) if m["on_street"] else 0.0
        sig["off_road"] = m["offroad_frac"]
        sig["comfort"] = abs(m["jerk"]) / 10.0 + abs(m["steer_rate"])
        sig["stall"] = float(m["speed"] < st["stall_speed_ms"] and not m["legit_wait"])
        sig["time"] = 1.0
        sig["off_route"] = float(m["off_route"])
        sig["speed_bonus"] = m["speed"]
        return sig

    def __call__(self, m: dict):
        sig = self.signals(m)
        parts = {}
        total = 0.0
        for name, (sign, _kind, _d) in TERMS.items():
            t = self.terms[name]
            if not t["enabled"]:
                parts[name] = 0.0
                continue
            val = sign * t["weight"] * sig[name]
            parts[name] = val
            total += val
        return total, parts, sig
