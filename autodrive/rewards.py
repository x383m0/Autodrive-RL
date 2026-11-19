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


class RewardConfigError(ValueError):
    """A problem in a reward YAML file, explained in plain English."""


def load_reward_config(path=None) -> dict:
    """Load a YAML reward config.

    ONLY the terms written in your file count. Any term you leave out is
    switched OFF - nothing sneaks in behind your back. (Settings you leave
    out use the defaults in DEFAULT_SETTINGS.)

    With path=None you get the built-in "balanced" defaults (used by tests).
    """
    cfg = {"terms": copy.deepcopy(DEFAULT_TERMS), "settings": dict(DEFAULT_SETTINGS),
           "name": "defaults"}
    if path is not None:
        try:
            with open(path) as f:
                user = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise RewardConfigError(f"Reward file not found: {path}") from None
        except yaml.YAMLError as e:
            raise RewardConfigError(
                f"Could not read {path} - the layout is off somewhere.\n"
                f"Most common causes: a TAB instead of spaces, or lines not lined up "
                f"(term names start with 2 spaces, 'enabled'/'weight' with 4).\n"
                f"Details: {e}") from None
        if not isinstance(user, dict):
            raise RewardConfigError(f"{path} should start with 'terms:' - see rewards/template.yaml")
        cfg["name"] = user.get("name", Path(path).stem)
        cfg["description"] = user.get("description", "")
        # start with every term OFF, then switch on what the file lists
        for t in cfg["terms"].values():
            t["enabled"] = False
        for k, v in (user.get("terms") or {}).items():
            if k not in TERMS:
                close = [t for t in TERMS if t[:4] == str(k)[:4]]
                hint = f" Did you mean '{close[0]}'?" if close else ""
                raise RewardConfigError(
                    f"Unknown reward term '{k}' in {path}.{hint}\nValid names: {', '.join(TERMS)}")
            if v is None:
                v = {}
            if not isinstance(v, dict):
                raise RewardConfigError(
                    f"Term '{k}' in {path} should have 'enabled:' and 'weight:' lines under it, "
                    f"indented by 4 spaces.")
            for key in v:
                if key not in ("enabled", "weight"):
                    raise RewardConfigError(
                        f"Term '{k}' has an unknown line '{key}'. Only 'enabled' and 'weight' are allowed.")
            cfg["terms"][k]["enabled"] = v.get("enabled", True)   # listed = on unless you say false
            if "weight" in v:
                cfg["terms"][k]["weight"] = v["weight"]
        for k, v in (user.get("settings") or {}).items():
            if k not in DEFAULT_SETTINGS:
                raise RewardConfigError(
                    f"Unknown setting '{k}' in {path}.\nValid settings: {', '.join(DEFAULT_SETTINGS)}")
            cfg["settings"][k] = v
    for k, t in cfg["terms"].items():
        if not isinstance(t["enabled"], bool):
            raise RewardConfigError(f"'enabled' for '{k}' must be true or false, got {t['enabled']!r}")
        try:
            t["weight"] = float(t.get("weight", 0.0))
        except (TypeError, ValueError):
            raise RewardConfigError(f"'weight' for '{k}' must be a number, got {t['weight']!r}") from None
        if t["weight"] < 0:
            raise RewardConfigError(
                f"Weight of '{k}' is negative. Always use positive weights: whether a term is a "
                f"bonus or a penalty is fixed (see the [+]/[-] tags in rewards/template.yaml).")
    if path is not None and not any(t["enabled"] and t["weight"] > 0 for t in cfg["terms"].values()):
        raise RewardConfigError(
            f"No reward terms are switched on in {path}, so the car would learn nothing. "
            f"Set at least one term to 'enabled: true' with a weight above 0.")
    return cfg


def describe_config(cfg) -> str:
    """One line per switched-on term, printed at the start of training."""
    lines = []
    for k, (sign, _kind, desc) in TERMS.items():
        t = cfg["terms"][k]
        if t["enabled"] and t["weight"] > 0:
            lines.append(f"  {'+' if sign > 0 else '-'} {k:22s} weight {t['weight']:<8g} {desc}")
    off = [k for k in TERMS if not (cfg["terms"][k]["enabled"] and cfg["terms"][k]["weight"] > 0)]
    return "\n".join(lines) + (f"\n  (off: {', '.join(off)})" if off else "")


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
