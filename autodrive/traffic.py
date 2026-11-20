"""
traffic.py - hard-coded, rule-following NPC cars with random personalities.

NPC cars do NOT learn. Each one moves along a `Path` (right lane, arcs
through intersections) and only decides its SPEED, using the classic
Intelligent Driver Model (IDM):

    a = a_max * [ 1 - (v/v0)^4 - (s*/gap)^2 ]
    s* = s0 + v*T + v*dv / (2*sqrt(a_max*b))

    v0  desired speed (speed limit x personality factor)
    T   desired time headway (s)       s0  minimum gap (m)
    gap distance to whatever is in front (car, pedestrian, red-light stop line)
    dv  closing speed towards it

Traffic rules are implemented as "virtual obstacles": a red light or an
un-served stop sign is a stopped car sitting on the stop line, so IDM
brakes for it naturally.

Personalities change the IDM parameters and add quirks:
    calm     - slow, long headway
    normal   - textbook
    rusher   - fast, short headway, runs yellow lights
    timid    - very slow, brakes early
    erratic  - random hard-brake events (the "car ahead slams on the brakes")
"""
from __future__ import annotations

import numpy as np

from .city import ROAD_HW, STOPLINE_D, City, Path

NPC_LEN = 4.5
NPC_WID = 1.85

PERSONALITIES = {
    #            speed  T     a_max b    s0   yellow_run  hard_brake/s
    "calm":    (0.85, 1.8, 1.6, 2.5, 2.5, 0.0, 0.000),
    "normal":  (1.00, 1.4, 2.0, 3.0, 2.0, 0.1, 0.000),
    "rusher":  (1.12, 1.0, 2.8, 3.5, 1.5, 0.8, 0.000),
    "timid":   (0.75, 2.2, 1.3, 2.0, 3.0, 0.0, 0.000),
    "erratic": (1.00, 1.3, 2.4, 3.0, 2.0, 0.3, 0.040),
}
PERSONALITY_P = {"calm": 0.25, "normal": 0.35, "rusher": 0.15, "timid": 0.1, "erratic": 0.15}


class NPCCar:
    def __init__(self, path: Path, s: float, v: float, personality: str, color: int, rng):
        self.path = path
        self.s = s
        self.v = v
        self.name = personality
        (self.speed_f, self.T, self.a_max, self.b, self.s0,
         self.p_yellow, self.hard_brake_rate) = PERSONALITIES[personality]
        self.color = color
        self.a = 0.0
        self.brake_timer = 0.0         # >0 while doing a sudden hard brake
        self.stop_state = {}           # event index -> "stopped time"/"passed"
        self.wait_time = 0.0
        self.honk_cd = 0.0
        self.yellow_decision = {}
        self.forced_brake = None       # scenario hook: (t_start, duration)
        self.id = int(rng.integers(0, 1 << 30))
        self._update_pose()

    def _update_pose(self):
        p = self.path.pos(self.s)
        self.x, self.y = float(p[0]), float(p[1])
        self.theta = float(self.path.heading(self.s + 1.0))

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def box(self):
        return np.array([self.x, self.y, NPC_LEN / 2, NPC_WID / 2, self.theta])

    @property
    def braking(self):
        return self.a < -1.0 or (self.v < 0.3)

    @property
    def blinker(self):
        """'L'/'R' if a turn is coming within 30 m, else None."""
        for ev in self.path.events:
            if 0 < ev["s"] - self.s < 30 and ev.get("turn") in ("L", "R"):
                return ev["turn"]
            if ev["s"] - self.s >= 30:
                break
        return None


class TrafficManager:
    def __init__(self, city: City, rng, n_cars: int, hard_brake_scale=1.0, reckless=1.0):
        self.city = city
        self.rng = rng
        self.n_cars = n_cars
        self.cars: list[NPCCar] = []
        self.hard_brake_scale = hard_brake_scale
        self.reckless = reckless      # 0 = only calm/normal/timid drivers, 1 = the full mix
        self.events = []          # sound/visual events emitted this step

    # ---------------------------------------------------------------- spawn
    def spawn(self, agent_pos, min_dist=30.0, max_dist=140.0, tries=40, path=None, s=None,
              personality=None, v=None):
        rng = self.rng
        for _ in range(tries):
            if path is None:
                e = self.city.random_directed_edge(rng)
                L = float(np.linalg.norm(self.city.nodes[e[1]] - self.city.nodes[e[0]]))
                s0 = float(rng.uniform(STOPLINE_D + 2, L - STOPLINE_D - 4))
                # cheap checks first, building the full path is expensive
                pos = self.city.lane_point(e[0], e[1], s0)
                if not (min_dist < np.linalg.norm(pos - agent_pos) < max_dist):
                    continue
                if any(np.linalg.norm(pos - c.pos) < 9.0 for c in self.cars):
                    continue
                seq = self.city.random_walk_nodes(rng, e, n_edges=7)
                p = self.city.build_path(seq, start_s=s0)
                ss = 0.0
            else:
                p, ss = path, (s or 0.0)
            pos = p.pos(ss)
            d_agent = np.linalg.norm(pos - agent_pos)
            if path is None and not (min_dist < d_agent < max_dist):
                continue
            if any(np.linalg.norm(pos - c.pos) < 9.0 for c in self.cars):
                if path is None:
                    continue
            if personality:
                name = personality
            else:
                pp = {k: v * (self.reckless if k in ("rusher", "erratic") else 1.0) for k, v in PERSONALITY_P.items()}
                tot = sum(pp.values())
                name = rng.choice(list(pp), p=[v / tot for v in pp.values()])
            if name == "erratic" and self.hard_brake_scale == 0:
                name = "normal"
            vv = v if v is not None else 0.7 * p.limit(ss)
            car = NPCCar(p, ss, vv, name, int(rng.integers(0, 1 << 16)), rng)   # any model of the fleet
            self.cars.append(car)
            return car
        return None

    def populate(self, agent_pos):
        while len(self.cars) < self.n_cars:
            if self.spawn(agent_pos) is None:
                break

    # ---------------------------------------------------------------- logic
    def _intersection_busy(self, node, me, agent_box):
        c = self.city.nodes[node]
        r = ROAD_HW + 1.0
        for o in self.cars:
            if o is me:
                continue
            if abs(o.x - c[0]) < r and abs(o.y - c[1]) < r:
                return True
        if agent_box is not None and abs(agent_box[0] - c[0]) < r and abs(agent_box[1] - c[1]) < r:
            return True
        return False

    def _earliest_waiting(self, node, me):
        """FIFO for all-way stops: am I the earliest car waiting at `node`?"""
        mine = me.stop_state.get(("arr", node), 1e9)
        for o in self.cars:
            if o is me:
                continue
            t = o.stop_state.get(("arr", node))
            if t is not None and o.stop_state.get(("go", node)) is None and t < mine - 1e-6:
                return False
        return True

    def step(self, dt, t, agent=None, peds=None, agent_legit_stop=False):
        """Advance all NPCs. agent: object with pos, theta, v, box (or None)."""
        self.events = []
        rng = self.rng
        city = self.city
        # obstacle list: all cars + agent  -> (pos, heading, speed, half-length)
        others_pos = [c.pos for c in self.cars]
        others_v = [c.v for c in self.cars]
        others_th = [c.theta for c in self.cars]
        if agent is not None:
            others_pos.append(agent.pos)
            others_v.append(agent.v)
            others_th.append(agent.theta)
        others_pos = np.array(others_pos) if others_pos else np.zeros((0, 2))
        ped_pos = np.array([p.pos for p in peds]) if peds else np.zeros((0, 2))
        agent_box = agent.box if agent is not None else None

        look = np.arange(1.0, 45.0, 1.5)
        for k, car in enumerate(self.cars):
            v0 = car.path.limit(car.s + 3.0) * car.speed_f
            gap, dv = 1e9, 0.0
            pts_ahead = car.path.pos(car.s + look)                       # (n,2)
            # --- vehicles in my lane ahead
            leader = -1
            if len(others_pos) > 1:
                d = np.linalg.norm(pts_ahead[:, None, :] - others_pos[None, :, :], axis=2)  # (n, m)
                d[:, k] = 1e9                     # that's me
                hit = d < 1.7
                for j in np.where(hit.any(0))[0]:
                    i0 = int(np.argmax(hit[:, j]))
                    # ignore cars travelling towards us (oncoming lane overlap at turns)
                    if np.cos(others_th[j] - car.theta) < -0.3:
                        continue
                    g = look[i0] - NPC_LEN
                    if g < gap:
                        gap = g
                        dv = car.v - others_v[j] * max(0.0, np.cos(others_th[j] - car.theta))
                        leader = j
                if agent is not None and leader == len(others_pos) - 1:
                    # stuck behind the agent: maybe honk
                    if car.v < 0.3 and agent.v < 0.3 and not agent_legit_stop:
                        car.wait_time += dt
                        if car.wait_time > 4.0 and car.honk_cd <= 0:
                            self.events.append(("horn", car.pos.copy()))
                            car.honk_cd = 8.0
                    else:
                        car.wait_time = 0.0
            # --- pedestrians in my path (yield!)
            if len(ped_pos):
                d = np.linalg.norm(pts_ahead[:20, None, :] - ped_pos[None, :, :], axis=2)
                hit = d < 2.2
                if hit.any():
                    i0 = int(np.argmax(hit.any(1)))
                    g = look[i0] - NPC_LEN / 2 - 2.0
                    if g < gap:
                        gap, dv = g, car.v
            # --- intersection controls (virtual obstacles at the stop line)
            for ei, ev in enumerate(car.path.events):
                dist = ev["s"] - (car.s + NPC_LEN / 2)
                if dist < -1.0 or dist > 60:
                    continue
                stop_here = False
                if ev["kind"] == "light":
                    st = city.light_state(ev["node"], ev["axis"], t)
                    if st == "R":
                        stop_here = dist > -0.5
                    elif st == "Y":
                        dec = car.yellow_decision.get(ei)
                        if dec is None:
                            can_stop = dist > car.v ** 2 / (2 * 3.5)
                            run = (not can_stop) or (rng.random() < car.p_yellow)
                            dec = car.yellow_decision[ei] = "go" if run else "stop"
                        stop_here = dec == "stop" and dist > -0.5
                    else:
                        car.yellow_decision.pop(ei, None)
                elif ev["kind"] == "stop":
                    if car.stop_state.get(("go", ev["node"])) is None:
                        stop_here = True
                        # IDM halts ~s0 metres before the (virtual) obstacle on the line
                        if dist < car.s0 + 2.0 and car.v < 0.3:
                            car.stop_state.setdefault(("arr", ev["node"]), t)
                            waited = t - car.stop_state[("arr", ev["node"])]
                            if (waited > 1.0 and self._earliest_waiting(ev["node"], car)
                                    and not self._intersection_busy(ev["node"], car, agent_box)):
                                car.stop_state[("go", ev["node"])] = t
                                stop_here = False
                if stop_here and dist < gap:
                    gap, dv = max(dist, 0.01), car.v
                break    # only the next control matters
            # --- IDM
            sstar = car.s0 + car.v * car.T + car.v * dv / (2 * np.sqrt(car.a_max * car.b))
            acc = car.a_max * (1 - (car.v / max(v0, 0.1)) ** 4)
            if gap < 1e8:
                acc -= car.a_max * (sstar / max(gap, 0.1)) ** 2
            acc = min(max(float(acc), -8.0), car.a_max)
            # --- quirks: sudden hard braking (erratic drivers / scenarios)
            if car.brake_timer <= 0 and car.v > 6 and rng.random() < car.hard_brake_rate * self.hard_brake_scale * dt:
                car.brake_timer = rng.uniform(1.0, 2.0)
            if car.forced_brake is not None:
                t0, dur = car.forced_brake
                if t0 <= t < t0 + dur:
                    car.brake_timer = max(car.brake_timer, dt)
            if car.brake_timer > 0:
                acc = -7.0
                car.brake_timer -= dt
            car.a = acc
            car.v = max(0.0, car.v + acc * dt)
            car.s += car.v * dt
            car.honk_cd -= dt
            car._update_pose()

    def recycle(self, agent_pos, far=170.0):
        """Remove cars that finished their path or wandered far away, respawn."""
        keep = []
        for c in self.cars:
            if c.s > c.path.length - 3 or np.linalg.norm(c.pos - agent_pos) > far:
                continue
            keep.append(c)
        self.cars = keep
        self.populate(agent_pos)

    def boxes(self):
        if not self.cars:
            return np.zeros((0, 5))
        return np.array([c.box for c in self.cars])

    def signals(self):
        """Turn signal of every car, same order as boxes(): -1 left, +1 right, 0 none."""
        return np.array([{"L": -1.0, "R": 1.0}.get(c.blinker, 0.0) for c in self.cars], dtype=np.float32)
