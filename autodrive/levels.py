"""
levels.py - the game levels of "Reward Designer" (see GAME_DESIGN.md).

Each level defines:
    world       what the test drives look like (a curriculum stage dict)
    curriculum  hidden training stages, easiest first (the player never sees
                these; they only design the reward)
    terms       reward terms the player can use in this level (new ones unlock
                as levels progress)
    score       "precision" (level 1) or "speed" (time to park); levels with
                rules also subtract rule-violation penalties
    stars       score thresholds for 1, 2 and 3 stars (1 star unlocks the next level)

Scoring of ONE test drive (0-1000 points):
    distance    up to 400 for the share of the route driven
    parking     400 for ending parked in the spot
    quality     up to 200 more when parked: precision (level 1) or speed (others)
    rules       levels 5+: -40 per red light / rolled stop sign / crosswalk
                violation, -10 per km/h over the limit in a school zone
So driving the whole way is worth about as much as parking itself, and a
fast, clean park scores the most.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- building blocks
_BASE = dict(grid=(3, 3), hops=(0, 1), controls=False, n_cars=0, n_peds=0, parked=0.0, n_kids=0,
             jaywalk=0.0, school=0.0, lot_occ=0.3, hard_brake=0.0, advance=0.65, no_limits=True)


def _stage(name, **kw):
    d = dict(_BASE)
    d.update(kw)
    d["name"] = name
    return d


def _park_lessons(bands, advance=0.6):
    return [_stage(f"park from {lo}-{hi} m", hops=(0, 0), start_togo=(float(lo), float(hi)),
                   advance=advance) for lo, hi in bands]


_ROAD = dict(world="road", lot_occ=0.3)

TERMS_1 = ["progress", "goal_reached", "parking_precision", "collision", "off_route", "stall",
           "stuck", "time", "speed_bonus"]
TERMS_2 = TERMS_1 + ["off_road", "lane_deviation", "comfort"]
TERMS_5 = TERMS_2 + ["speeding", "red_light", "stop_sign"]
TERMS_6 = TERMS_5 + ["pedestrian_hit", "pedestrian_proximity", "crosswalk_yield"]

LEVELS = [
    dict(id=1, name="Park here", available=True, score="precision", minutes=3,
         brief="The parking spot is a few metres away. Teach the car to stop inside it - "
               "the straighter and more centred, the more points.",
         terms=TERMS_1, stars=(600, 750, 880),
         curriculum=_park_lessons([(1, 6), (4, 15)]),
         test=_stage("Park here (test)", hops=(0, 0), start_togo=(6.0, 20.0))),
    dict(id=2, name="Straight road", available=True, score="speed", minutes=4,
         brief="An empty straight road with a parking bay at the end. No speed limit: "
               "the faster it parks, the more points. Don't overshoot!",
         terms=TERMS_2, stars=(550, 700, 850),
         curriculum=[_stage("straight road, last 25 m", start_togo=(5.0, 25.0), **_ROAD),
                     _stage("straight road, last 80 m", start_togo=(20.0, 80.0), **_ROAD),
                     _stage("straight road, whole road", **_ROAD)],
         test=_stage("Straight road (test)", **_ROAD)),
    dict(id=3, name="Bends", available=True, score="speed", minutes=6,
         brief="Same idea, but the road bends. Fast on the straights, careful in the turns.",
         terms=TERMS_2, stars=(550, 700, 850),
         curriculum=[_stage("bendy road, last 25 m", start_togo=(5.0, 25.0), bends=2, **_ROAD),
                     _stage("bendy road, last 90 m", start_togo=(20.0, 90.0), bends=2, **_ROAD),
                     _stage("bendy road, whole road", bends=2, **_ROAD)],
         test=_stage("Bends (test)", bends=3, **_ROAD)),
    dict(id=4, name="Into town", available=True, score="speed", minutes=10,
         brief="Real streets now: buildings, parked cars and turns at intersections, "
               "then into a parking lot. Still no speed limits.",
         terms=TERMS_2, stars=(550, 700, 850),
         curriculum=_park_lessons([(1, 6), (4, 15), (12, 30), (25, 70)])
         + [_stage("empty streets", hops=(0, 1), parked=0.3)],
         test=_stage("Into town (test)", hops=(1, 2), parked=0.3)),
    dict(id=5, name="Rules of the road", available=True, score="speed", rules=True, minutes=15,
         brief="Traffic lights, stop signs and school zones appear. School zones are the only "
               "places with a speed limit. Every rule you break costs points.",
         terms=TERMS_5, stars=(500, 650, 800),
         curriculum=_park_lessons([(1, 6), (4, 15), (12, 30), (25, 70)])
         + [_stage("empty streets", hops=(0, 1), parked=0.3),
            _stage("rules", hops=(1, 3), controls=True, parked=0.3, school=0.2)],
         test=_stage("Rules of the road (test)", hops=(1, 3), controls=True, parked=0.3, school=0.2)),
    # ---- not playable yet (see GAME_DESIGN.md build order) -------------------------
    dict(id=6, name="Rush hour", available=False, score="speed", rules=True, minutes=20,
         brief="Normal pedestrians and normal traffic.", terms=TERMS_6, stars=(500, 650, 800)),
    dict(id=7, name="Chaos", available=False, score="speed", rules=True, minutes=25,
         brief="Jaywalkers and reckless drivers.", terms=TERMS_6, stars=(500, 650, 800)),
    dict(id=8, name="Heavy traffic", available=False, score="speed", rules=True, minutes=25,
         brief="Heavier traffic and trucks.", terms=TERMS_6, stars=(500, 650, 800)),
    dict(id=9, name="Night storm", available=False, score="speed", rules=True, minutes=25,
         brief="Night and bad weather: the sensors see less.", terms=TERMS_6, stars=(500, 650, 800)),
    dict(id=10, name="Big roads", available=False, score="speed", rules=True, minutes=30,
         brief="Roundabouts and highways.", terms=TERMS_6, stars=(500, 650, 800)),
    dict(id=11, name="Roadworks", available=False, score="speed", rules=True, minutes=30,
         brief="Construction zones: cones, closed lanes, detours.", terms=TERMS_6,
         stars=(500, 650, 800)),
]
BY_ID = {lv["id"]: lv for lv in LEVELS}
REF_SPEED = 10.0          # m/s: an average speed of 10 m/s earns half the speed points
TEST_SEED0 = 90_000_000   # hidden test roads (never used in training)


def score_drive(level, st, dist=None):
    """Points (0-1000) for one test drive, from env.episode_stats().

    dist: metres the car had to drive in this test (route length from its start)."""
    pts = 400.0 * float(st["progress_frac"])
    if st["success"]:
        pts += 400.0
        if level["score"] == "precision":
            quality = 1.0 - float(st["park_err"] or 0.0)
        else:
            avg_speed = (dist or st["route_len"]) / max(st["time_s"], 0.1)
            quality = float(np.clip(avg_speed / (2 * REF_SPEED), 0.0, 1.0))
        pts += 200.0 * quality
    if level.get("rules"):
        pts -= 40.0 * (st["red_violations"] + st["stop_violations"] + st["crosswalk_violations"])
        pts -= 10.0 * max(0.0, st["school_over_limit"] * 3.6 - 3.0)       # km/h over in school zones
    return round(max(pts, 0.0), 1)


def stars_for(level, score):
    return sum(score >= t for t in level["stars"])


def run_test(exp_dir, level, n=12, on_drive=None):
    """Hidden test: n drives on never-seen roads, deterministic policy.

    Returns dict(score, stars, drives=[...]). on_drive(i, drive) is called after
    each drive (for progress bars)."""
    from .env import AutoDriveEnv
    from .video import load_policy
    policy, cfg = load_policy(exp_dir)
    env = AutoDriveEnv(reward_config=cfg, stages=[level["test"]], replay_prob=0.0)
    drives = []
    for i in range(n):
        obs, _ = env.reset(seed=TEST_SEED0 + level["id"] * 1000 + i, options={"stage": 0})
        dist = env.route.length - env.start_s
        while True:
            obs, r, term, trunc, info = env.step(policy(obs))
            if term or trunc:
                break
        st = info["episode_stats"]
        d = dict(status=st["status"], score=score_drive(level, st, dist), time=round(st["time_s"], 1),
                 progress=round(float(st["progress_frac"]), 3), red=st["red_violations"],
                 stop=st["stop_violations"], crosswalk=st["crosswalk_violations"],
                 school_over_kmh=round(max(0.0, st["school_over_limit"] * 3.6), 1),
                 park_err=st["park_err"], avg_speed_kmh=round(dist / max(st["time_s"], 0.1) * 3.6, 1),
                 stall_frac=round(st["stall_frac"], 3), turns_per_100m=round(st["turns_per_100m"], 2))
        drives.append(d)
        if on_drive:
            on_drive(i, d)
    score = round(float(np.mean([d["score"] for d in drives])), 1)
    return dict(score=score, stars=stars_for(level, score), drives=drives)
