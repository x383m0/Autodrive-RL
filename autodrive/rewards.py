"""
rewards.py - the configurable reward function.

Every 0.1 s the environment measures what happened (metres advanced, did we
hit something, speed vs limit, ...). Each reward TERM turns one measurement
into a SIGNAL, and your YAML config gives each term a number:

    reward_t = sum over enabled terms of   points * signal_t
             + sum of your custom rewards (see "Custom rewards" below)

THE NUMBER IS THE POINTS GIVEN (format 2). Positive = the car gains points,
negative = it loses points. `collision: -50` punishes crashing; `+50` would
reward it. Nothing is flipped behind your back.

    event terms   signal is 1 on the step it happens   -> points once
    step terms    signal is an amount (metres, m/s, ..) -> points x amount, every step

OLD FILES (format 1, no `format: 2` line): weights were always positive and
each term had a fixed direction (penalties were subtracted). Such files are
converted automatically when loaded, so old experiments behave exactly as
they did.

CUSTOM REWARDS (rule builder, `custom:` in the YAML):

    - name: slow near people
      when:                       # ANY group is true ...
        - - {m: nearest_pedestrian_m, op: "<", v: 6}     # ... and ALL conditions in a group
          - {m: speed_kmh, op: ">", v: 15}
      give: -0.2                  # points (any sign)
      times: null                 # or a measurement name: points x that measurement
      paid: every_step            # or on_change (each time `when` turns true)

Only whitelisted measurement names and operators are accepted (no code is
executed). See MEASUREMENTS for the full list.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

# Normalisers for the comfort term (see vehicle.py): the largest possible
# change of acceleration in one 0.1 s step, and the steering rack speed.
MAX_JERK = (3.5 + 8.0) / 0.1          # m/s^3
MAX_STEER_RATE = 2.094                # rad/s (120 deg/s)
FORMAT = 2

# name: (usual sign, kind, short description)
#   usual sign: whether people normally give it positive or negative points
#               (only used to convert old format-1 files and for help texts)
#   kind "step"  = paid every step x a measured amount
#   kind "event" = paid once when something happens
TERMS = {
    "progress":             (+1, "step",  "points per metre advanced along the route"),
    "goal_reached":         (+1, "event", "parked successfully in the target slot"),
    "parking_precision":    (+1, "event", "on parking: points x (1 - parking error)"),
    "collision":            (-1, "event", "hit a car, parked car or building (episode ends)"),
    "pedestrian_hit":       (-1, "event", "hit a pedestrian (episode ends)"),
    "pedestrian_proximity": (-1, "step",  "closeness (0-1) to a pedestrian while moving"),
    "crosswalk_yield":      (-1, "event", "drove over a crosswalk a pedestrian was using"),
    "speeding":             (-1, "step",  "per m/s above the speed limit (+ tolerance)"),
    "red_light":            (-1, "event", "crossed a stop line on red"),
    "stop_sign":            (-1, "event", "rolled through a stop sign without stopping"),
    "lane_deviation":       (-1, "step",  "per metre from the lane centre / planned path (beyond deadband)"),
    "off_road":             (-1, "step",  "share (0-1) of the car on the sidewalk / off the asphalt"),
    "comfort":              (-1, "step",  "jerk + fast steering, each scaled 0..1"),
    "stall":                (-1, "step",  "standing still with no reason to (1 per step)"),
    "time":                 (-1, "step",  "every step (1 per step)"),
    "off_route":            (-1, "event", "left the route by > off_route_distance (episode ends)"),
    "stuck":                (-1, "event", "stood still for stuck_seconds with no reason (episode ends)"),
    "speed_bonus":          (+1, "step",  "per m/s of raw speed, regardless of direction (HACKABLE!)"),
}

DEFAULT_SETTINGS = {
    "speed_tolerance_kmh": 3.0,
    "lane_deadband_m": 0.4,
    "ped_proximity_radius_m": 4.0,
    "stall_speed_ms": 0.5,
    "off_route_distance_m": 6.0,
    "stuck_seconds": 12.0,
    "terminate_on_red_light": False,
    "terminate_on_stop_sign": False,
}

# the "balanced" preset, in signed points
DEFAULT_TERMS = {
    "progress": dict(enabled=True, weight=0.1),
    "goal_reached": dict(enabled=True, weight=30.0),
    "parking_precision": dict(enabled=True, weight=15.0),
    "collision": dict(enabled=True, weight=-50.0),
    "pedestrian_hit": dict(enabled=True, weight=-100.0),
    "pedestrian_proximity": dict(enabled=True, weight=-0.1),
    "crosswalk_yield": dict(enabled=True, weight=-10.0),
    "speeding": dict(enabled=True, weight=-0.05),
    "red_light": dict(enabled=True, weight=-20.0),
    "stop_sign": dict(enabled=True, weight=-10.0),
    "lane_deviation": dict(enabled=True, weight=-0.03),
    "off_road": dict(enabled=True, weight=-0.2),
    "comfort": dict(enabled=True, weight=-0.01),
    "stall": dict(enabled=True, weight=-0.03),
    "time": dict(enabled=True, weight=-0.005),
    "off_route": dict(enabled=True, weight=-20.0),
    "stuck": dict(enabled=True, weight=-20.0),
    "speed_bonus": dict(enabled=False, weight=0.02),
}

# ---------------------------------------------------------------- custom rewards
# measurement name: (label, unit, type, group)
MEASUREMENTS = {
    "speed_kmh":            ("speed", "km/h", "num", "Driving"),
    "accel":                ("acceleration (negative = braking)", "m/s²", "num", "Driving"),
    "steering_pct":         ("steering (+ = right)", "%", "num", "Driving"),
    "distance_m":           ("distance driven this drive", "m", "num", "Driving"),
    "to_spot_m":            ("distance to the parking spot (along the route)", "m", "num", "Driving"),
    "time_s":               ("time since start", "s", "num", "Driving"),
    "lane_offset_m":        ("distance from lane centre / planned path", "m", "num", "Position"),
    "heading_err_deg":      ("heading error vs the route", "°", "num", "Position"),
    "off_road":             ("off road", "", "bool", "Position"),
    "in_lot":               ("in the parking lot", "", "bool", "Position"),
    "nearest_car_m":        ("nearest car seen by the sensors (999 = none)", "m", "num", "Surroundings"),
    "nearest_pedestrian_m": ("nearest pedestrian seen by the sensors (999 = none)", "m", "num", "Surroundings"),
    "obstacle_ahead_m":     ("nearest obstacle straight ahead (999 = none)", "m", "num", "Surroundings"),
    "crashed":              ("crashed", "", "bool", "Events"),
    "hit_pedestrian":       ("hit a pedestrian", "", "bool", "Events"),
    "parked":               ("parked", "", "bool", "Events"),
    "ran_red_light":        ("ran a red light", "", "bool", "Events"),
    "rolled_stop_sign":     ("rolled a stop sign", "", "bool", "Events"),
    "went_off_route":       ("went off route", "", "bool", "Events"),
    "got_stuck":            ("got stuck", "", "bool", "Events"),
}
OPERATORS = {
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b, "=": lambda a, b: a == b, "!=": lambda a, b: a != b,
}
PAID = ("every_step", "on_change")


def custom_key(name: str) -> str:
    """Column/breakdown key of a custom reward, e.g. 'custom_slow_near_people'."""
    return "custom_" + (re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_") or "reward")


class RewardConfigError(ValueError):
    """A problem in a reward YAML file, explained in plain English."""


def _check_custom(i, c, where):
    """Validate one custom reward; return a cleaned copy."""
    label = f"custom reward #{i + 1}" + (f" ('{c.get('name')}')" if isinstance(c, dict) and c.get("name") else "")
    if not isinstance(c, dict):
        raise RewardConfigError(f"{label} in {where} should be a block with name/when/give/paid.")
    name = str(c.get("name") or f"custom {i + 1}")
    when = c.get("when") or []
    if not isinstance(when, list) or any(not isinstance(g, list) for g in when):
        raise RewardConfigError(f"{label}: 'when' must be a list of groups, each a list of conditions.")
    groups = []
    for g in when:
        conds = []
        for cond in g:
            if not isinstance(cond, dict) or "m" not in cond:
                raise RewardConfigError(f"{label}: each condition needs m (measurement), op and v.")
            m, op = cond["m"], str(cond.get("op", "is"))
            if m not in MEASUREMENTS:
                raise RewardConfigError(f"{label}: unknown measurement '{m}'. "
                                        f"Valid: {', '.join(MEASUREMENTS)}")
            kind = MEASUREMENTS[m][2]
            if kind == "bool":
                v = cond.get("v", True)
                if op not in ("is", "=", "!="):
                    raise RewardConfigError(f"{label}: '{m}' is true/false - use 'is'.")
                conds.append({"m": m, "op": "=" if op == "is" else op, "v": bool(v)})
            else:
                if op not in OPERATORS:
                    raise RewardConfigError(f"{label}: unknown operator '{op}' (use {' '.join(OPERATORS)}).")
                try:
                    v = float(cond.get("v"))
                except (TypeError, ValueError):
                    raise RewardConfigError(f"{label}: the value for '{m}' must be a number.") from None
                conds.append({"m": m, "op": op, "v": v})
        if conds:
            groups.append(conds)
    try:
        give = float(c.get("give"))
    except (TypeError, ValueError):
        raise RewardConfigError(f"{label}: 'give' must be a number (the points to give).") from None
    times = c.get("times")
    if times is not None and (times not in MEASUREMENTS or MEASUREMENTS[times][2] != "num"):
        raise RewardConfigError(f"{label}: 'times' must be a number measurement "
                                f"({', '.join(k for k, v in MEASUREMENTS.items() if v[2] == 'num')}).")
    paid = c.get("paid", "every_step")
    if paid not in PAID:
        raise RewardConfigError(f"{label}: 'paid' must be every_step or on_change.")
    return {"name": name, "key": custom_key(name), "when": groups, "give": give, "times": times,
            "paid": paid, "enabled": bool(c.get("enabled", True))}


def parse_config(user: dict, where="config", allow_empty=False) -> dict:
    """Validate a reward config given as a dict (from YAML or from the game)."""
    cfg = {"terms": copy.deepcopy(DEFAULT_TERMS), "settings": dict(DEFAULT_SETTINGS),
           "name": user.get("name", "reward"), "description": user.get("description", ""),
           "custom": [], "format": FORMAT}
    fmt = int(user.get("format", 1))
    if fmt not in (1, 2):
        raise RewardConfigError(f"{where}: unknown format {fmt} (use 2).")
    for t in cfg["terms"].values():                       # only what you write counts
        t["enabled"] = False
    for k, v in (user.get("terms") or {}).items():
        if k not in TERMS:
            close = [t for t in TERMS if t[:4] == str(k)[:4]]
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise RewardConfigError(f"Unknown reward term '{k}' in {where}.{hint}\nValid names: {', '.join(TERMS)}")
        v = {} if v is None else v
        if not isinstance(v, dict):
            raise RewardConfigError(f"Term '{k}' in {where} should have 'enabled:' and 'weight:' lines "
                                    f"under it, indented by 4 spaces.")
        for key in v:
            if key not in ("enabled", "weight"):
                raise RewardConfigError(f"Term '{k}' has an unknown line '{key}'. Only 'enabled' and 'weight' are allowed.")
        enabled = v.get("enabled", True)
        if not isinstance(enabled, bool):
            raise RewardConfigError(f"'enabled' for '{k}' must be true or false, got {enabled!r}")
        try:
            w = float(v.get("weight", DEFAULT_TERMS[k]["weight"]))
        except (TypeError, ValueError):
            raise RewardConfigError(f"'weight' for '{k}' must be a number, got {v.get('weight')!r}") from None
        if fmt == 1:
            # old files: positive magnitude, the direction came from the term
            if w < 0:
                raise RewardConfigError(f"Weight of '{k}' is negative in an old-format file. Add "
                                        f"'format: 2' at the top to write signed points instead.")
            w = TERMS[k][0] * w
        cfg["terms"][k] = {"enabled": enabled, "weight": w}
    for k, v in (user.get("settings") or {}).items():
        if k not in DEFAULT_SETTINGS:
            raise RewardConfigError(f"Unknown setting '{k}' in {where}.\nValid settings: {', '.join(DEFAULT_SETTINGS)}")
        cfg["settings"][k] = v
    custom = user.get("custom") or []
    if not isinstance(custom, list):
        raise RewardConfigError(f"'custom' in {where} must be a list of custom rewards.")
    keys = set()
    for i, c in enumerate(custom):
        cc = _check_custom(i, c, where)
        while cc["key"] in keys:                          # unique column names
            cc["key"] += "_2"
        keys.add(cc["key"])
        cfg["custom"].append(cc)
    if not allow_empty and not active_names(cfg):
        raise RewardConfigError(
            f"No rewards are switched on in {where}, so the car would learn nothing. "
            f"Switch at least one on with a non-zero number.")
    return cfg


def load_reward_config(path=None, allow_empty=False) -> dict:
    """Load a YAML reward config (format 1 or 2). path=None -> balanced defaults."""
    if path is None:
        cfg = {"terms": copy.deepcopy(DEFAULT_TERMS), "settings": dict(DEFAULT_SETTINGS),
               "name": "defaults", "description": "", "custom": [], "format": FORMAT}
        return cfg
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
    user.setdefault("name", Path(path).stem)
    return parse_config(user, str(path), allow_empty=allow_empty)


def to_yaml_dict(cfg) -> dict:
    """A config as a format-2 dict (what we write next to every experiment)."""
    return {"format": FORMAT, "name": cfg.get("name", ""), "description": cfg.get("description", ""),
            "terms": {k: dict(t) for k, t in cfg["terms"].items() if t["enabled"]},
            "custom": [{k: c[k] for k in ("name", "when", "give", "times", "paid", "enabled")}
                       for c in cfg.get("custom", [])],
            "settings": dict(cfg["settings"])}


def active_names(cfg):
    """Keys of everything that can give points (enabled, non-zero)."""
    out = [k for k, t in cfg["terms"].items() if t["enabled"] and t["weight"] != 0]
    out += [c["key"] for c in cfg.get("custom", []) if c["enabled"] and c["give"] != 0]
    return out


def all_names(cfg):
    """Every breakdown key: all built-in terms + this config's custom rewards."""
    return list(TERMS) + [c["key"] for c in cfg.get("custom", [])]


def describe_config(cfg) -> str:
    """One line per active reward, printed at the start of training."""
    lines = []
    for k, (_s, kind, desc) in TERMS.items():
        t = cfg["terms"][k]
        if t["enabled"] and t["weight"] != 0:
            lines.append(f"  {k:22s} {t['weight']:+g} points  ({'once: ' if kind == 'event' else ''}{desc})")
    for c in cfg.get("custom", []):
        if c["enabled"] and c["give"] != 0:
            lines.append(f"  {c['key']:22s} {c['give']:+g} points{' x ' + c['times'] if c['times'] else ''}"
                         f"  (custom, {c['paid'].replace('_', ' ')})")
    off = [k for k in TERMS if not (cfg["terms"][k]["enabled"] and cfg["terms"][k]["weight"] != 0)]
    if not lines:
        lines.append("  (no rewards: the car will learn nothing)")
    return "\n".join(lines) + (f"\n  (off: {', '.join(off)})" if off else "")


class RewardComputer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.terms = cfg["terms"]
        self.s = cfg["settings"]
        self.custom = [c for c in cfg.get("custom", [])]
        self.names = all_names(cfg)
        self.reset()

    def reset(self):
        """Call at the start of every episode (on_change memory)."""
        self.prev_true = {c["key"]: False for c in self.custom}

    def signals(self, m: dict) -> dict:
        """Turn raw step measurements into the signal of each built-in term."""
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
        # distance from the planned path: the lane centre on streets, the
        # GPS path through the parking lot (this matters: without it nothing
        # tells the car to follow the curve into the slot until it's too late)
        sig["lane_deviation"] = max(0.0, abs(m["lat"]) - st["lane_deadband_m"])
        sig["off_road"] = m["offroad_frac"]
        # both parts normalised to [0, 1]: full-throttle -> full-brake in one step
        # is jerk 1.0, the steering wheel at its maximum turning speed is 1.0
        sig["comfort"] = min(abs(m["jerk"]) / MAX_JERK, 1.0) + min(abs(m["steer_rate"]) / MAX_STEER_RATE, 1.0)
        sig["stall"] = float(m["speed"] < st["stall_speed_ms"] and not m["legit_wait"])
        sig["time"] = 1.0
        sig["off_route"] = float(m["off_route"])
        sig["stuck"] = float(m["stuck"])
        sig["speed_bonus"] = m["speed"]
        return sig

    @staticmethod
    def _cond(c, meas):
        return OPERATORS[c["op"]](meas[c["m"]], c["v"])

    def custom_value(self, c, meas):
        """Points of one custom reward this step."""
        # no conditions = always true; otherwise ANY group with ALL conditions true
        true = (not c["when"]) or any(all(self._cond(x, meas) for x in g) for g in c["when"])
        if c["paid"] == "on_change":
            fire = true and not self.prev_true[c["key"]]
            self.prev_true[c["key"]] = true
        else:
            fire = true
        if not fire:
            return 0.0
        return c["give"] * (float(meas[c["times"]]) if c["times"] else 1.0)

    def __call__(self, m: dict):
        sig = self.signals(m)
        parts = {}
        total = 0.0
        for name in TERMS:
            t = self.terms[name]
            val = t["weight"] * sig[name] if t["enabled"] else 0.0
            parts[name] = val
            total += val
        meas = m.get("meas")
        for c in self.custom:
            val = self.custom_value(c, meas) if (c["enabled"] and meas is not None) else 0.0
            parts[c["key"]] = val
            total += val
        return total, parts, sig
