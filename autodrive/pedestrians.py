"""
pedestrians.py - people walking around the city.

Three behaviours:

  * WALKER / CROSSWALK-FOLLOWER
      Wanders the sidewalk graph (corner to corner). Crosses streets only
      on crosswalks: at traffic lights when the crossed street has RED,
      at stop/uncontrolled intersections when no car is close and fast.

  * JAYWALKER
      A walker that, somewhere mid-block, simply turns 90 degrees and
      walks straight across the street. "Careless" ones don't look.

  * EMERGER (the kid behind the parked car)
      Hidden in a gap between two parked cars, invisible to the agent's
      sensors because the parked cars occlude it. When the agent comes
      within a trigger distance it runs out across the lane.

Pedestrians are circles of radius 0.35 m for sensing and collisions.
"""
from __future__ import annotations

import math

import numpy as np

from .city import CORNER_OFF, PARKED_OFF, City, right_of

PED_R = 0.35


def _touch_matrix(boxes, pts, r):
    """For each point: does a circle of radius r there touch ANY oriented box?"""
    c, s = np.cos(boxes[:, 4])[None], np.sin(boxes[:, 4])[None]
    dx = pts[:, 0:1] - boxes[None, :, 0]
    dy = pts[:, 1:2] - boxes[None, :, 1]
    lx, ly = dx * c + dy * s, -dx * s + dy * c
    qx = np.clip(lx, -boxes[None, :, 2], boxes[None, :, 2])
    qy = np.clip(ly, -boxes[None, :, 3], boxes[None, :, 3])
    return (((lx - qx) ** 2 + (ly - qy) ** 2) < r * r).any(axis=1)


class Ped:
    def __init__(self, pos, kind, sprite, speed):
        self.pos = np.array(pos, dtype=float)
        self.kind = kind            # walker | jaywalker | emerger
        self.sprite = sprite
        self.speed = speed
        self.vel = np.zeros(2)
        self.state = "walk"         # walk | wait | cross | jay | hidden | run
        self.a = None               # sidewalk node we come from
        self.b = None               # sidewalk node we walk to
        self.target = None          # free target point (jaywalk/run)
        self.careless = False
        self.crossing = None        # (p0, p1) segment while on a crosswalk
        self.wait_t = 0.0
        self.trigger = 15.0
        self.anim = np.random.random() * 10
        self.heading = 0.0

    @property
    def in_road(self):
        return self.state in ("cross", "jay", "run")


class PedestrianManager:
    def __init__(self, city: City, rng, n_peds=10, jaywalk_prob=0.2, n_kids=0):
        self.city = city
        self.rng = rng
        self.n_peds = n_peds
        self.jaywalk_prob = jaywalk_prob
        self.n_kids = n_kids
        self.peds: list[Ped] = []

    # ---------------------------------------------------------------- spawn
    def _spawn_walker(self, agent_pos, near=(15, 110)):
        city, rng = self.city, self.rng
        for _ in range(30):
            a = int(rng.integers(0, len(city.sw_nodes)))
            if not city.sw_adj[a]:
                continue
            b = int(rng.choice(city.sw_adj[a]))
            if city.sw_edge[(a, b)][0] != "walk":
                continue
            t = rng.uniform(0.1, 0.9)
            pos = city.sw_nodes[a] * (1 - t) + city.sw_nodes[b] * t
            d = np.linalg.norm(pos - agent_pos)
            if not (near[0] < d < near[1]):
                continue
            kind = "jaywalker" if rng.random() < self.jaywalk_prob else "walker"
            p = Ped(pos, kind, int(rng.integers(0, 12)), rng.uniform(1.0, 1.6))
            p.a, p.b = a, b
            p.careless = kind == "jaywalker" and rng.random() < 0.5
            self.peds.append(p)
            return p
        return None

    def spawn_kid(self, gap, rng=None, trigger=None):
        """Hide a kid in a parked-car gap (dict from city.parked_gaps)."""
        rng = rng or self.rng
        p = Ped(gap["pos"], "emerger", int(rng.integers(0, 12)), rng.uniform(2.4, 3.2))
        p.state = "hidden"
        a, b = gap["edge"]
        d = self.city.edge_dir(a, b)
        p.cross_dir = -right_of(d)        # runs from the right curb towards the far side
        p.edge_dir = d
        p.trigger = trigger if trigger is not None else rng.uniform(13, 22)
        p.heading = float(np.arctan2(p.cross_dir[1], p.cross_dir[0]))
        self.peds.append(p)
        return p

    def populate(self, agent_pos, route=None):
        while sum(p.kind != "emerger" for p in self.peds) < self.n_peds:
            if self._spawn_walker(agent_pos) is None:
                break

    def place_kids(self, route_pts, rng):
        """Put kids into parked gaps that lie along the agent's route."""
        gaps = self.city.parked_gaps
        if not gaps or self.n_kids <= 0:
            return
        cands = []
        for g in gaps:
            d = np.min(np.linalg.norm(route_pts - g["pos"], axis=1))
            if d < PARKED_OFF + 0.5:      # gap is right next to our lane
                cands.append(g)
        rng.shuffle(cands)
        for g in cands[: self.n_kids]:
            self.spawn_kid(g, rng)

    # ---------------------------------------------------------------- logic
    def _cross_allowed(self, kind_meta, t, cars, p0, p1):
        node, axis = kind_meta
        st = self.city.light_state(node, axis, t)
        if st is not None:
            return st == "R"
        # unsignalised: wait for a gap (no fast car within 14 m of the crosswalk)
        mid = (p0 + p1) / 2
        d = np.hypot(cars[:, 0] - mid[0], cars[:, 1] - mid[1])
        return not bool(((d < 14) & (cars[:, 2] > 1.0)).any())

    def step(self, dt, t, cars_state, agent, car_boxes=None):
        """cars_state: (N,3) array of [x, y, speed] for NPCs + agent.
        car_boxes: (N,5) OBBs of all cars - pedestrians won't walk INTO a car
        that is already in front of them (they are reckless, not suicidal)."""
        city, rng = self.city, self.rng
        if not self.peds:
            return
        old = np.array([p.pos for p in self.peds])
        for p in self.peds:
            self._step_one(p, dt, t, cars_state, agent)
        if car_boxes is None or not len(car_boxes):
            return
        new = np.array([p.pos for p in self.peds])
        # would a pedestrian now touch a car (inflated by 0.5 m)? then it waits.
        hit_new = _touch_matrix(car_boxes, new, PED_R + 0.5)
        if hit_new.any():
            hit_old = _touch_matrix(car_boxes, old, PED_R + 0.5)
            for k in np.where(hit_new & ~hit_old)[0]:
                if self.peds[k].state != "hidden":
                    self.peds[k].pos = old[k]
                    self.peds[k].vel = np.zeros(2)

    def _step_one(self, p, dt, t, cars_state, agent):
        city, rng = self.city, self.rng
        if True:
            p.anim += dt * p.speed * 2.5
            if p.state == "hidden":
                rel = p.pos - agent.pos
                f = np.array([np.cos(agent.theta), np.sin(agent.theta)])
                along = rel @ f
                lat = rel @ np.array([-f[1], f[0]])
                # trigger distance grows with speed so the situation is (barely) solvable
                trig = max(p.trigger, agent.v * 1.9)
                if 0 < along < trig and abs(lat) < 7 and (p.edge_dir @ f) > 0.7:
                    p.state = "run"
                    p.target = p.pos + p.cross_dir * (2 * PARKED_OFF + 1.5)
                p.vel[:] = 0
                return
            if p.state in ("walk",):
                tgt = city.sw_nodes[p.b]
                vec = tgt - p.pos
                dist = math.hypot(vec[0], vec[1])
                # jaywalker decides to cross mid-block
                if p.kind == "jaywalker" and 8 < dist and rng.random() < 0.08 * dt:
                    e = city.sw_nodes[p.b] - city.sw_nodes[p.a]
                    if np.linalg.norm(e) > 20:
                        side = 1 if rng.random() < 0.5 else -1
                        n = right_of(e / np.linalg.norm(e))
                        # cross towards the street (the street is on one side of us)
                        # pick the direction whose target is a sidewalk point
                        for sgn in (side, -side):
                            tgt2 = p.pos + n * sgn * 2 * CORNER_OFF
                            if city.on_sidewalk(tgt2)[0] and city.drivable(p.pos + n * sgn * 5)[0]:
                                if p.careless or self._gap_ok(p.pos, cars_state):
                                    p.state = "jay"
                                    p.target = tgt2
                                break
                        if p.state == "jay":
                            return
                if dist < 0.3:
                    # arrived at a corner: choose the next edge (avoid going back)
                    opts = [q for q in city.sw_adj[p.b] if q != p.a] or [p.a]
                    nb = int(rng.choice(opts))
                    p.a, p.b = p.b, nb
                    if city.sw_edge[(p.a, p.b)][0] == "cross":
                        p.state = "wait"
                        p.wait_t = 0.0
                    return
                p.vel = vec / dist * p.speed
                p.pos = p.pos + p.vel * dt
            elif p.state == "wait":
                p.vel[:] = 0
                p.wait_t += dt
                meta = city.sw_edge[(p.a, p.b)][1]
                p0, p1 = city.sw_nodes[p.a], city.sw_nodes[p.b]
                if self._cross_allowed(meta, t, cars_state, p0, p1) and p.wait_t > 0.5:
                    p.state = "cross"
                    p.crossing = (p0.copy(), p1.copy())
                elif p.wait_t > 25:     # give up, turn around
                    p.a, p.b = p.b, p.a
                    p.pos = city.sw_nodes[p.a].copy()
                    opts = [q for q in city.sw_adj[p.a] if city.sw_edge[(p.a, q)][0] == "walk"]
                    if opts:
                        p.b = int(rng.choice(opts))
                    p.state = "walk"
            elif p.state == "cross":
                tgt = city.sw_nodes[p.b]
                vec = tgt - p.pos
                dist = math.hypot(vec[0], vec[1])
                if dist < 0.3:
                    p.state = "walk"
                    p.crossing = None
                    opts = [q for q in city.sw_adj[p.b] if q != p.a] or [p.a]
                    nb = int(rng.choice(opts))
                    p.a, p.b = p.b, nb
                    if city.sw_edge[(p.a, p.b)][0] == "cross":
                        p.state = "wait"
                        p.wait_t = 0
                    return
                p.vel = vec / dist * p.speed * 1.15
                p.pos = p.pos + p.vel * dt
            elif p.state in ("jay", "run"):
                vec = p.target - p.pos
                dist = math.hypot(vec[0], vec[1])
                if dist < 0.3:
                    # back on a sidewalk: become a normal walker again
                    p.state = "walk"
                    p.kind = "walker" if p.kind == "emerger" else p.kind
                    k = int(np.argmin(np.linalg.norm(city.sw_nodes - p.pos, axis=1)))
                    opts = [q for q in city.sw_adj[k] if city.sw_edge[(k, q)][0] == "walk"]
                    p.a, p.b = k, int(rng.choice(opts)) if opts else k
                    # walk towards whichever end is ahead
                    p.a, p.b = (p.b, p.a) if rng.random() < 0.5 else (p.a, p.b)
                    return
                p.vel = vec / dist * p.speed
                p.pos = p.pos + p.vel * dt
            if math.hypot(p.vel[0], p.vel[1]) > 0.05:
                p.heading = float(np.arctan2(p.vel[1], p.vel[0]))

    def _gap_ok(self, pos, cars):
        d = np.hypot(cars[:, 0] - pos[0], cars[:, 1] - pos[1])
        return not bool(((d < 25) & (cars[:, 2] > 1.0)).any())

    def recycle(self, agent_pos, far=130.0):
        self.peds = [p for p in self.peds
                     if p.kind == "emerger" and p.state == "hidden"
                     or np.linalg.norm(p.pos - agent_pos) < far or p.in_road]
        self.populate(agent_pos)

    def circles(self):
        """Visible pedestrians as circles (hidden kids are still physically there!)."""
        if not self.peds:
            return np.zeros((0, 3))
        return np.array([[p.pos[0], p.pos[1], PED_R] for p in self.peds])
