"""
env.py - the Gymnasium environment: AutoDriveEnv.

One EPISODE:
  1. generate a random city (seeded) according to the curriculum stage
  2. pick a target parking slot and a start position; build the GPS route
  3. the agent drives: every 0.1 s it outputs [steer, accel] in [-1, 1]
  4. the episode ends when it parks (success), crashes, hits a pedestrian,
     leaves the route, gets stuck, or runs out of time

OBSERVATION (what the policy sees; see sensors.py) - one frame is:
    ray sensors          69 rays x [closeness, static, vehicle, pedestrian]   276
    ego state            speed, steering, accel, yaw rate, last action          6
    lane perception      lateral offset, heading error (sin,cos), off-road, in-lot 5
    GPS route            10 waypoints ahead in the car frame + distance to goal  21
    speed signs          current limit, limit 30 m ahead, school zone            3
    next control         light? G/Y/R, distance to stop line, stop sign? done?   7
    parking target       slot pose in car frame (only when < 40 m away)          5
                                                                        total  323
  and the env returns the last 4 frames stacked (4 x 323 = 1292 floats), so a
  plain MLP can infer motion (who is moving where, how fast).

ACTION: Box(2): steer in [-1,1] (+ = right), target speed in [-1,1]
        (-1 = stop, +1 = 16 m/s; a cruise controller tracks it, see vehicle.py).
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .city import CROSSWALK_W, City
from .curriculum import STAGES
from .geometry import (obb_circle_overlap, obb_corners, obb_overlap_many, to_local,
                       wrap_angle)
from .pedestrians import PedestrianManager
from .rewards import TERMS, RewardComputer, load_reward_config
from .sensors import N_RAYS, SensorSuite
from .traffic import TrafficManager
from .vehicle import LENGTH, MAX_BRAKE, MAX_STEER, WIDTH, BicycleCar

DT = 0.1
WP_OFFSETS = np.array([2, 4, 6, 8, 11, 14, 18, 23, 30, 40], dtype=float)
FRAME_DIM = N_RAYS * 4 + 6 + 5 + 21 + 3 + 7 + 5
PARK_LAT_TOL = 0.6
PARK_LON_TOL = 1.0
PARK_HEAD_TOL = np.deg2rad(15)
PARK_SPEED = 0.3


class AutoDriveEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, reward_config=None, stage=0, frame_stack=4, render_mode=None,
                 replay_prob=0.15, seed_range=(0, 10_000_000), audio_events=False):
        super().__init__()
        if reward_config is None or isinstance(reward_config, str):
            reward_config = load_reward_config(reward_config)
        self.reward_cfg = reward_config
        self.rewarder = RewardComputer(reward_config)
        self.stage = stage
        self.replay_prob = replay_prob
        self.frame_stack = frame_stack
        self.render_mode = render_mode
        self.seed_range = seed_range
        self.sensors = SensorSuite()
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(FRAME_DIM * frame_stack,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.renderer = None
        self.np_random_master = np.random.default_rng()
        self.frames = deque(maxlen=frame_stack)

    # ------------------------------------------------------------------ API
    def set_stage(self, k):
        self.stage = int(k)
        return self.stage

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if seed is not None:
            self.np_random_master = np.random.default_rng(seed)
        m = self.np_random_master
        stage = options.get("stage", self.stage)
        if "stage" not in options and stage > 0 and m.random() < self.replay_prob:
            stage = int(m.integers(0, stage))          # replay an easier stage
        self.ep_stage = stage
        city_seed = options.get("city_seed")
        if city_seed is None:
            city_seed = int(m.integers(*self.seed_range))
        self.city_seed = city_seed
        rng = np.random.default_rng(city_seed)
        self.rng = rng
        p = dict(STAGES[stage])
        p.update(options.get("overrides", {}))
        self.params = p
        scenario = options.get("scenario")
        self.scenario = None
        if scenario is not None:
            from .scenarios import SCENARIOS
            self.scenario = SCENARIOS[scenario](self, rng)
        else:
            self._build_world(p, rng)
        self._post_build()
        obs = self._observe()
        for _ in range(self.frame_stack):
            self.frames.append(obs)
        return self._stacked(), {}

    # ---------------------------------------------------------------- world
    def _build_world(self, p, rng, start_edge=None, start_s=None, between_cars=False,
                     city=None):
        """Create city, route, agent, traffic and pedestrians for stage params p."""
        nx, ny = p["grid"]
        self.city = city or City(rng, nx, ny, controls=p["controls"], parked_prob=p["parked"],
                                 school_prob=p["school"], lot_occupancy=p["lot_occ"])
        if self.city.target_slot is None:
            self.city.choose_target_slot(rng, between_cars=between_cars or rng.random() < 0.3)
        self.route, self.start_edge = self.city.build_route(rng, *p["hops"],
                                                            start_edge=start_edge,
                                                            start_s=start_s)
        p0 = self.route.pts[0]
        th = float(self.route.headings[0])
        v0 = float(p.get("v0", rng.uniform(0, 0.6) * self.route.limit(0.0)))
        self.car = BicycleCar(p0[0], p0[1], th, v0)
        self.traffic = TrafficManager(self.city, rng, p["n_cars"], hard_brake_scale=p["hard_brake"])
        self.peds = PedestrianManager(self.city, rng, p["n_peds"], jaywalk_prob=p["jaywalk"],
                                      n_kids=p["n_kids"])

    def _post_build(self):
        """Populate dynamic actors and reset all episode book-keeping."""
        self.traffic.populate(self.car.pos)
        self.peds.populate(self.car.pos)
        if self.scenario is None:
            self.peds.place_kids(self.route.pts, self.rng)
        self.t = 0.0
        self.steps = 0
        self.route_idx = 0
        self.s, self.lat, self.route_idx = self.route.project(self.car.pos, 0)
        self.best_s = self.s
        self.last_progress_step = 0
        self.max_steps = int((self.route.length / 4.0 + 40.0) / DT)
        self.last_action = np.zeros(2)
        self.prev_delta = 0.0
        self.ev_done = set()                 # processed control events
        self.stop_min_speed = {}             # event index -> min speed near the line
        self.cw_flagged = set()
        self.light_seen = {}
        self.ep = {k: 0.0 for k in TERMS}    # per-term episode sums
        self.stats = dict(red=0, stop=0, crosswalk=0, speeding_steps=0, offroad_steps=0,
                          stall_steps=0, lane_dev_sum=0.0, heading_change=0.0, comfort=0.0,
                          speed_sum=0.0, max_over=0.0, school_over=0.0)
        self.last_terms = {k: 0.0 for k in TERMS}
        self.last_reward = 0.0
        self.ep_return = 0.0
        self.events = []
        self.status = "driving"
        self.slot = self.city.slots[self.city.target_slot]
        self._scan()

    # ---------------------------------------------------------------- step
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        car = self.car
        self.events = []
        prev_theta = car.theta
        prev_delta = car.delta
        car.step(action[0], action[1], DT)
        self.t += DT
        self.steps += 1

        # --- world update ---------------------------------------------------
        legit = self._legit_wait()
        self.traffic.step(DT, self.t, agent=car, peds=self.peds.peds, agent_legit_stop=legit)
        cars_state = np.array([[c.x, c.y, c.v] for c in self.traffic.cars] + [[car.x, car.y, car.v]])
        all_boxes = np.concatenate([self.traffic.boxes(), car.box[None]], 0)
        self.peds.step(DT, self.t, cars_state, car, car_boxes=all_boxes)
        if self.scenario is not None and hasattr(self.scenario, "tick"):
            self.scenario.tick(self)
        if self.steps % 20 == 0 and self.scenario is None:
            self.traffic.recycle(car.pos)
            self.peds.recycle(car.pos)
        for e in self.traffic.events:
            self.events.append(e)

        # --- measurements -----------------------------------------------------
        m = self._measure(prev_theta, prev_delta, legit)
        reward, parts, sig = self.rewarder(m)
        self.last_terms = parts
        self.last_reward = reward
        self.ep_return += reward
        for k, v in parts.items():
            self.ep[k] += v

        terminated = bool(m["collision"] or m["ped_hit"] or m["off_route"] or m["parked"])
        st = self.reward_cfg["settings"]
        if m["red_violation"] and st["terminate_on_red_light"]:
            terminated = True
        if m["stop_violation"] and st["terminate_on_stop_sign"]:
            terminated = True
        stuck = (self.steps - self.last_progress_step) > int(30 / DT)
        truncated = (not terminated) and (self.steps >= self.max_steps or stuck)
        if m["parked"]:
            self.status = "parked"
            self.events.append(("chime", car.pos.copy()))
        elif m["collision"]:
            self.status = "collision"
            self.events.append(("crash", car.pos.copy()))
        elif m["ped_hit"]:
            self.status = "pedestrian_hit"
            self.events.append(("ped_hit", car.pos.copy()))
        elif m["off_route"]:
            self.status = "off_route"
        elif truncated:
            self.status = "stuck" if stuck else "timeout"

        self.last_action = action
        obs = self._observe()
        self.frames.append(obs)
        info = {"reward_terms": parts, "status": self.status}
        if terminated or truncated:
            info["episode_stats"] = self.episode_stats(m)
        return self._stacked(), float(reward), terminated, truncated, info

    # ---------------------------------------------------------------- helpers
    def _stacked(self):
        return np.concatenate(list(self.frames)).astype(np.float32)

    def _scan(self):
        car = self.car
        self.static_boxes = self.city.static_boxes()
        self.npc_boxes = self.traffic.boxes()
        self.ped_circ = self.peds.circles()
        d, c, o, dirs = self.sensors.scan(car.x, car.y, car.theta, self.static_boxes,
                                          self.npc_boxes, self.ped_circ)
        self.ray_dist, self.ray_cls, self.ray_org, self.ray_dir = d, c, o, dirs

    def _next_event(self, s_front, kinds=("light", "stop")):
        for i, ev in enumerate(self.route.events):
            if ev["kind"] in kinds and ev["s"] > s_front - 0.5 and i not in self.ev_done:
                return i, ev
        return None, None

    def _legit_wait(self):
        """Is the agent allowed to stand still right now? (used by the stall term
        and by NPCs deciding whether to honk). Uses ground truth - rewards may
        be privileged, observations are not."""
        car = self.car
        s_front = self.s + LENGTH / 2
        i, ev = self._next_event(s_front)
        if ev is not None:
            dist = ev["s"] - s_front
            if ev["kind"] == "light":
                stt = self.city.light_state(ev["node"], ev["axis"], self.t)
                if stt in ("R", "Y") and -1.0 < dist < 40:
                    return True
            elif ev["kind"] == "stop" and -1.0 < dist < 12:
                return True
        if self.route.length - self.s < 2.0:
            return True
        # something in the way ahead (car, pedestrian)?
        ahead = self.route.pos(self.s + np.arange(2.0, 16.0, 1.5))
        objs = []
        if len(self.traffic.cars):
            objs.append(np.array([c.pos for c in self.traffic.cars]))
        if len(self.peds.peds):
            objs.append(np.array([p.pos for p in self.peds.peds]))
        if objs:
            P = np.concatenate(objs)
            dd = np.linalg.norm(ahead[:, None, :] - P[None, :, :], axis=2)
            if (dd < 2.5).any():
                return True
        # waiting at an intersection because of cross traffic counts too
        return False

    def _measure(self, prev_theta, prev_delta, legit):
        car = self.car
        city = self.city
        route = self.route
        s_old = self.s
        s_new, lat, idx = route.project(car.pos, self.route_idx)
        ds = float(np.clip(s_new - s_old, -2.0, 2.0))
        self.s, self.lat, self.route_idx = s_new, lat, idx
        if s_new > self.best_s + 1.0:
            self.best_s = s_new
            self.last_progress_step = self.steps

        self._scan()
        box = car.box
        # --- collisions -------------------------------------------------------
        near = self.static_boxes[np.linalg.norm(self.static_boxes[:, :2] - car.pos, axis=1) < 40] \
            if len(self.static_boxes) else self.static_boxes
        coll = bool(obb_overlap_many(box, near).any()) if len(near) else False
        if len(self.npc_boxes):
            coll = coll or bool(obb_overlap_many(box, self.npc_boxes).any())
        ped_hit = bool(obb_circle_overlap(box, self.ped_circ).any()) if len(self.ped_circ) else False

        # --- pedestrian proximity ------------------------------------------
        ped_close = 0.0
        R = self.reward_cfg["settings"]["ped_proximity_radius_m"]
        if len(self.ped_circ):
            loc = to_local(self.ped_circ[:, :2], car.pos, car.theta)
            dx = np.maximum(np.abs(loc[:, 0]) - LENGTH / 2, 0)
            dy = np.maximum(np.abs(loc[:, 1]) - WIDTH / 2, 0)
            dist = np.sqrt(dx ** 2 + dy ** 2)
            ped_close = float(np.max(np.maximum(0.0, 1.0 - dist / R)))

        # --- crosswalk yield ---------------------------------------------------
        cw_violation = False
        for p in self.peds.peds:
            if p.state == "cross" and p.crossing is not None and id(p) not in self.cw_flagged:
                p0, p1 = p.crossing
                mid = (p0 + p1) / 2
                if np.linalg.norm(mid - car.pos) > 12:
                    continue
                v = p1 - p0
                cw = np.array([mid[0], mid[1], np.linalg.norm(v) / 2, CROSSWALK_W / 2,
                               np.arctan2(v[1], v[0])])
                if car.v > 1.0 and obb_overlap_many(box, cw[None])[0]:
                    cw_violation = True
                    self.cw_flagged.add(id(p))

        # --- traffic controls -----------------------------------------------------
        red_v = stop_v = False
        s_front_old = s_old + LENGTH / 2
        s_front = s_new + LENGTH / 2
        for i, ev in enumerate(route.events):
            if i in self.ev_done or ev["kind"] not in ("light", "stop"):
                continue
            if ev["kind"] == "light":
                stt = city.light_state(ev["node"], ev["axis"], self.t)
                # light-change "ping" when the next light ahead changes
                if 0 < ev["s"] - s_front < 50:
                    if self.light_seen.get(i) not in (None, stt):
                        self.events.append(("light_change", city.nodes[ev["node"]].copy()))
                    self.light_seen[i] = stt
                if s_front_old <= ev["s"] < s_front:
                    if stt == "R":
                        red_v = True
                    self.ev_done.add(i)
            else:
                if ev["s"] - 8.0 < s_front < ev["s"] + 1.0:
                    self.stop_min_speed[i] = min(self.stop_min_speed.get(i, 99.0), car.v)
                if s_front >= ev["s"] + 1.0 and s_front_old < ev["s"] + 1.0:
                    if self.stop_min_speed.get(i, 99.0) > 0.5:
                        stop_v = True
                    self.ev_done.add(i)
        if red_v:
            self.stats["red"] += 1
        if stop_v:
            self.stats["stop"] += 1
        if cw_violation:
            self.stats["crosswalk"] += 1

        # --- off road ---------------------------------------------------------
        corners = obb_corners(box)
        offroad = 1.0 - float(city.drivable(corners).mean())

        # --- parking ----------------------------------------------------------
        sl = self.slot
        loc = to_local(car.pos[None], sl["center"], sl["heading"])[0]
        he = abs(wrap_angle(car.theta - sl["heading"]))
        he = min(he, np.pi - he)
        parked = (abs(loc[0]) < PARK_LON_TOL and abs(loc[1]) < PARK_LAT_TOL and he < PARK_HEAD_TOL
                  and car.v < PARK_SPEED)
        park_err = float(np.clip(0.4 * abs(loc[1]) / PARK_LAT_TOL + 0.3 * abs(loc[0]) / PARK_LON_TOL
                                 + 0.3 * he / PARK_HEAD_TOL, 0, 1))
        self.park_loc = (loc, he)

        limit = route.limit(s_new)
        jerk = (car.a - car.prev_a) / DT
        steer_rate = abs(car.delta - prev_delta) / DT
        on_street = s_new < route.in_lot_s
        off_route = abs(lat) > self.reward_cfg["settings"]["off_route_distance_m"]

        # --- stats for the reward-hacking detector ----------------------------
        st = self.stats
        tol = self.reward_cfg["settings"]["speed_tolerance_kmh"] / 3.6
        over = car.v - limit
        if over > tol:
            st["speeding_steps"] += 1
        st["max_over"] = max(st["max_over"], over)
        if route.school[min(idx, len(route.school) - 1)]:
            st["school_over"] = max(st["school_over"], over)
        if offroad > 0.01:
            st["offroad_steps"] += 1
        if car.v < 0.5 and not legit:
            st["stall_steps"] += 1
        st["lane_dev_sum"] += abs(lat) if on_street else 0.0
        st["heading_change"] += abs(wrap_angle(car.theta - prev_theta))
        st["comfort"] += min(abs(jerk) / 115.0, 1) + min(steer_rate / 2.094, 1)
        st["speed_sum"] += car.v
        if car.v > 4 and car.a < -5.0:
            self.events.append(("skid", car.pos.copy()))

        return dict(ds=ds, parked=parked, park_err=park_err, collision=coll, ped_hit=ped_hit,
                    ped_close=ped_close, crosswalk_violation=cw_violation, speed=car.v,
                    limit=limit, red_violation=red_v, stop_violation=stop_v, lat=lat,
                    on_street=on_street, offroad_frac=offroad, jerk=jerk,
                    steer_rate=steer_rate, legit_wait=legit, off_route=off_route)

    def _observe(self):
        car = self.car
        route = self.route
        city = self.city
        rays = self.sensors.encode(self.ray_dist, self.ray_cls)
        ego = [car.v / 20.0, car.delta / MAX_STEER, car.a / MAX_BRAKE, car.yaw_rate,
               self.last_action[0], self.last_action[1]]
        he = wrap_angle(car.theta - route.heading(self.s))
        in_lot = float(self.s >= route.in_lot_s)
        corners = obb_corners(car.box)
        offroad = 1.0 - float(city.drivable(corners).mean())
        lane = [np.clip(self.lat / 3.0, -2, 2), np.sin(he), np.cos(he), offroad, in_lot]
        wps = route.pos(self.s + WP_OFFSETS)
        loc = to_local(wps, car.pos, car.theta) / 40.0
        togo = (route.length - self.s) / 400.0
        gps = list(loc.reshape(-1)) + [min(togo, 2.0)]
        lim_now = route.limit(self.s)
        lim_ahead = route.limit(self.s + 30)
        school = float(route.school[min(self.route_idx, len(route.school) - 1)])
        signs = [lim_now / 20.0, lim_ahead / 20.0, school]
        # next control (perceived only within 60 m: the camera reads it)
        s_front = self.s + LENGTH / 2
        ctrl = [0, 0, 0, 0, 1.0, 0, 0]
        i, ev = self._next_event(s_front)
        if ev is not None and ev["s"] - s_front < 60:
            dist = (ev["s"] - s_front) / 60.0
            if ev["kind"] == "light":
                stt = city.light_state(ev["node"], ev["axis"], self.t)
                ctrl = [1, stt == "G", stt == "Y", stt == "R", dist, 0, 0]
            else:
                done = self.stop_min_speed.get(i, 99) <= 0.5
                ctrl = [0, 0, 0, 0, dist, 1, float(done)]
        # parking target
        sl = self.slot
        rel = to_local(sl["center"][None], car.pos, car.theta)[0]
        if np.linalg.norm(rel) < 40:
            ra = wrap_angle(sl["heading"] - car.theta)
            park = [rel[0] / 30.0, rel[1] / 30.0, np.sin(ra), np.cos(ra), 1.0]
        else:
            park = [0, 0, 0, 0, 0]
        frame = np.concatenate([rays, np.array(ego + lane + gps + signs + ctrl + park, dtype=np.float32)])
        return np.clip(frame, -5, 5).astype(np.float32)

    def blinker(self):
        """The agent signals automatically from the navigation route (like a Tesla)."""
        for ev in self.route.events:
            d = ev["s"] - self.s
            if 0 < d < 30 and ev.get("turn") in ("L", "R"):
                return ev["turn"]
        if 0 < self.route.in_lot_s - self.s < 25:
            return "R"
        return None

    def episode_stats(self, m=None):
        n = max(self.steps, 1)
        st = self.stats
        km = max(self.car.odometer / 1000.0, 1e-3)
        return {
            "success": self.status == "parked",
            "status": self.status,
            "stage": self.ep_stage,
            "city_seed": self.city_seed,
            "steps": self.steps,
            "time_s": self.t,
            "route_len": self.route.length,
            "progress": float(self.best_s),
            "progress_frac": float(min(self.best_s / self.route.length, 1.0)),
            "distance": self.car.odometer,
            "mean_speed": st["speed_sum"] / n,
            "collision": self.status == "collision",
            "ped_hit": self.status == "pedestrian_hit",
            "off_route": self.status == "off_route",
            "red_violations": st["red"],
            "stop_violations": st["stop"],
            "crosswalk_violations": st["crosswalk"],
            "violations_per_km": (st["red"] + st["stop"] + st["crosswalk"]) / km,
            "speeding_frac": st["speeding_steps"] / n,
            "max_over_limit": st["max_over"],
            "school_over_limit": st["school_over"],
            "offroad_frac": st["offroad_steps"] / n,
            "stall_frac": st["stall_steps"] / n,
            "lane_dev_mean": st["lane_dev_sum"] / n,
            "heading_change": st["heading_change"],
            "turns_per_100m": st["heading_change"] / (2 * np.pi) / max(self.car.odometer / 100, 0.01),
            "comfort_mean": st["comfort"] / n,
            "park_err": float(m["park_err"]) if (m and m["parked"]) else None,
            "return": self.ep_return,
            "terms": dict(self.ep),
        }

    # ---------------------------------------------------------------- render
    def render(self):
        if self.renderer is None:
            from .render import Renderer
            self.renderer = Renderer()
        return self.renderer.draw(self)

    def close(self):
        self.renderer = None
