"""
worlds.py - simple worlds for the first game levels.

RoadWorld: a single empty two-lane road across open grass, ending in a row
of parking bays. Level 2 uses a straight road, level 3 a road with bends.
There are no intersections, buildings, traffic or pedestrians.

It offers the same interface the environment and renderer use from City
(drivable(), static_boxes(), slots, build_route(), ...), so the rest of the
simulator works unchanged. Anything a city has but a road does not (nodes,
lights, sidewalks, ...) is simply empty.
"""
from __future__ import annotations

import numpy as np

from .city import (BAY_STRAIGHT, CURVE_LEN, END_BAYS, LANE_OFF, LANE_W, SLOT_D, SLOT_W,
                   TARGET_BAYS, Path, right_of, unit)

ROAD_HALF = LANE_W + 0.6          # two lanes + a small shoulder each side
MARGIN_WORLD = 30.0               # free space around the road (metres)
NO_LIMIT = 30.0                   # "no speed limit" (m/s): faster than the car can go
PAD_LIMIT = 15 / 3.6              # creeping speed on the parking pad


class RoadWorld:
    kind = "road"

    def __init__(self, rng, length=(160, 240), bends=0, lot_occupancy=0.3):
        self.rng = rng
        self.bends = bends
        for _ in range(50):
            center = self._centerline(rng, length, bends)
            if self._ok(center):
                break
        # shift so everything is inside [MARGIN, ...] like the city
        lo = center.min(0) - (ROAD_HALF + 30)
        center = center - lo
        self.center = center
        d = np.diff(center, axis=0)
        self.seg_a, self.seg_b = center[:-1], center[1:]
        self.seg_len = np.linalg.norm(d, axis=1)
        # end of the road and its direction -> parking pad
        f = unit(center[-1] - center[-6])
        r = right_of(f)
        self.end, self.f, self.r = center[-1], f, r
        self._build_pad(lot_occupancy)
        pts = np.concatenate([center, [self.pad_far + r * (END_BAYS + 1) * SLOT_W,
                                       self.pad_far - r * (END_BAYS + 1) * SLOT_W]])
        self.width = float(pts[:, 0].max() + 30)
        self.height = float(pts[:, 1].max() + 30)
        # decoration: trees well away from the road
        self.trees = []
        for _ in range(int(len(center) / 6)):
            p = np.array([rng.uniform(0, self.width), rng.uniform(0, self.height)])
            if self._dist_to_center(p[None])[0] > ROAD_HALF + 6 and not self.point_in_pad(p, 4):
                self.trees.append((p[0], p[1], rng.uniform(0.7, 1.2)))
        # empty city features
        self.nodes = np.zeros((0, 2))
        self.adj, self.control, self.lights = {}, [], {}
        self.buildings = np.zeros((0, 5))
        self.building_colors = []
        self.parked = np.zeros((0, 5))
        self.parked_meta, self.parked_gaps, self.parks = [], [], []
        self.sw_nodes = np.zeros((0, 2))
        self.sw_adj, self.sw_edge = {}, {}
        self.xs = self.ys = np.zeros(0)
        self.nx = self.ny = 0
        self._static_boxes = None

    # ------------------------------------------------------------ geometry
    @staticmethod
    def _centerline(rng, length, bends, step=0.5):
        """Polyline: straight pieces joined by circular bends."""
        total = rng.uniform(*length)
        pts = [np.zeros(2)]
        heading = rng.uniform(0, 2 * np.pi)
        n_pieces = 2 * bends + 1
        straight = total / (bends + 1) if bends else total
        for k in range(n_pieces):
            if k % 2 == 0:                            # straight piece
                L = straight * rng.uniform(0.6, 1.0) if bends else total
                for _ in range(int(L / step)):
                    pts.append(pts[-1] + step * np.array([np.cos(heading), np.sin(heading)]))
            else:                                     # bend
                R = rng.uniform(18, 35)
                ang = rng.choice([-1, 1]) * np.deg2rad(rng.uniform(35, 90))
                n = int(abs(ang) * R / step)
                for _ in range(n):
                    heading += ang / n
                    pts.append(pts[-1] + step * np.array([np.cos(heading), np.sin(heading)]))
        # a final straight run-in so the car arrives straight at the pad
        for _ in range(int(25 / step)):
            pts.append(pts[-1] + step * np.array([np.cos(heading), np.sin(heading)]))
        return np.array(pts)

    @staticmethod
    def _ok(c):
        """Reject roads that come close to themselves."""
        sub = c[::8]
        d = np.linalg.norm(sub[:, None] - sub[None], axis=2)
        idx = np.arange(len(sub))
        far = np.abs(idx[:, None] - idx[None]) > 12
        return not (far & (d < 4 * ROAD_HALF + 10)).any()

    def _dist_to_center(self, pts):
        a, b = self.seg_a[None], self.seg_b[None]
        p = np.atleast_2d(pts)[:, None]
        ab = b - a
        t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-9), 0, 1)
        proj = a + t[..., None] * ab
        return np.sqrt(((p - proj) ** 2).sum(-1)).min(1)

    # ------------------------------------------------------------ parking pad
    def _build_pad(self, occupancy):
        f, r, e = self.f, self.r, self.end
        self.pad_bay_along = CURVE_LEN + BAY_STRAIGHT        # bay centre, measured from road end
        self.pad_far = e + f * (self.pad_bay_along + SLOT_D / 2)
        heading = float(np.arctan2(f[1], f[0]))
        self.slots = []
        for j in range(-END_BAYS, END_BAYS + 1):
            c = e + f * self.pad_bay_along + r * (j * SLOT_W)
            self.slots.append({"center": c, "side": 0, "r": self.pad_bay_along, "axis": f.copy(),
                               "heading": heading, "end": True, "j": j})
        self.slot_occupied = np.array([self.rng.random() < occupancy for _ in self.slots])
        self.target_slot = None

    def point_in_pad(self, p, margin=0.0):
        rel = np.asarray(p) - self.end
        a, b = rel @ self.f, rel @ self.r
        return (-margin <= a <= self.pad_bay_along + SLOT_D / 2 + margin
                and abs(b) <= (END_BAYS + 0.5) * SLOT_W + margin)

    def lot_polys(self):
        f = self.f
        ang = float(np.arctan2(f[1], f[0]))
        L = self.pad_bay_along + SLOT_D / 2
        c = self.end + f * (L / 2)
        return np.array([[c[0], c[1], L / 2, (END_BAYS + 0.5) * SLOT_W, ang]])

    def choose_target_slot(self, rng, between_cars=False):
        ends = {sl["j"]: i for i, sl in enumerate(self.slots)}
        j = int(rng.integers(-TARGET_BAYS, TARGET_BAYS + 1))
        i = ends[j]
        self.target_slot = i
        self.slot_occupied[i] = False
        if between_cars:
            for jj in (j - 1, j + 1):
                if jj in ends:
                    self.slot_occupied[ends[jj]] = True
        self._static_boxes = None
        return i

    def lot_parked_boxes(self):
        out = [[s["center"][0], s["center"][1], 2.3, 0.92, s["heading"]]
               for i, s in enumerate(self.slots) if self.slot_occupied[i]]
        return np.array(out) if out else np.zeros((0, 5))

    def static_boxes(self):
        if self._static_boxes is None:
            self._static_boxes = self.lot_parked_boxes()
        return self._static_boxes

    # ------------------------------------------------------------ queries
    def drivable(self, pts):
        pts = np.atleast_2d(pts)
        on = self._dist_to_center(pts) <= ROAD_HALF
        for k in np.where(~on)[0]:
            on[k] = self.point_in_pad(pts[k])
        return on

    def on_sidewalk(self, pts):
        return np.zeros(len(np.atleast_2d(pts)), dtype=bool)

    def light_state(self, node, axis, t):
        return None

    def is_school(self, a, b):
        return False

    # ------------------------------------------------------------ route
    def build_route(self, rng, *_hops, start_edge=None, start_s=None, step=0.5):
        """Right lane along the whole road, S-curve onto the pad, into the bay."""
        c = self.center
        tang = np.gradient(c, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        normal = np.stack([-tang[:, 1], tang[:, 0]], 1)          # right of travel
        lane = c + normal * LANE_OFF
        sl = self.slots[self.target_slot]
        y_t = sl["j"] * SLOT_W                                    # lateral offset of the bay
        f, r, e = self.f, self.r, self.end
        tail = []
        for a in np.arange(step, CURVE_LEN, step):
            u = a / CURVE_LEN
            y = LANE_OFF + (y_t - LANE_OFF) * (u * u * (3 - 2 * u))
            tail.append(e + f * a + r * y)
        for a in np.arange(CURVE_LEN, self.pad_bay_along, step):
            tail.append(e + f * a + r * y_t)
        tail.append(sl["center"])
        pts = np.concatenate([lane, np.array(tail)])
        n_road = len(lane)
        limits = np.full(len(pts), NO_LIMIT)
        limits[n_road:] = PAD_LIMIT
        path = Path(pts, limits, np.zeros(len(pts), dtype=bool), nodes=[])
        path.in_lot_s = float(path.s[n_road])
        return path, None
