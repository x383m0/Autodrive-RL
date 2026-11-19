"""
city.py - procedural city generation + lane-level paths.

THE CITY
========
A Manhattan-style grid of streets. Column/row spacings are random (55-85 m),
so every city looks different but the topology stays simple and robust:

      x0        x1           x2
  y0  +---------+------------+        + = intersection ("node")
      | block   |   block    |        - | = two-way street with
  y1  +---------+------------+            1 driving lane + 1 parking strip
      | block   |  PARKING   |            per direction, sidewalks outside
  y2  +---------+------------+

Street cross-section (distances from the street centre line):

   0.0 .. 3.5  driving lane   (right-hand traffic: you drive at +1.75 m)
   3.5 .. 6.0  parking strip  (parked cars, kids hiding between them)
   6.0 .. 9.0  sidewalk       (pedestrians walk here)
   9.0 ..      block interior (buildings, trees, or the parking lot)

Each intersection is controlled by a TRAFFIC LIGHT, an ALL-WAY STOP, or
nothing (grid corners / curriculum stage 0). Crosswalks sit at 6..9 m from
the intersection centre, and the stop line is just behind them at 9.5 m.

PATHS
=====
Cars (NPCs and the agent) never steer "freely" in the simulator logic: an NPC
moves along a `Path` (a dense polyline following the right lane, with circular
arcs through intersections). The agent DOES steer freely, but its route is a
Path too and is used for GPS waypoints, progress measurement and lane rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---- street geometry constants (metres) -----------------------------------
LANE_W = 3.5                 # driving lane width
PARK_W = 2.5                 # parking strip width
ROAD_HW = LANE_W + PARK_W    # half width of the asphalt (6.0)
SIDEWALK_W = 3.0
CROSSWALK_W = 3.0
STOPLINE_D = ROAD_HW + CROSSWALK_W + 0.5   # stop line distance from node centre
LANE_OFF = LANE_W / 2.0      # lane centre offset from street centre line
PARKED_OFF = LANE_W + PARK_W / 2.0
CORNER_OFF = ROAD_HW + SIDEWALK_W / 2.0    # sidewalk centre offset

# ---- speed limits (m/s). 1 m/s = 3.6 km/h --------------------------------
KMH = 1 / 3.6
LIMIT_CHOICES = [30 * KMH, 40 * KMH, 50 * KMH]
SCHOOL_LIMIT = 20 * KMH
LOT_LIMIT = 15 * KMH

# ---- parking lot ---------------------------------------------------------
SLOT_W = 3.0      # width of one parking slot (along the aisle)
SLOT_D = 5.5      # depth of one slot
AISLE_HW = 3.5    # half width of the lot aisle

# ---- traffic light timing -------------------------------------------------
YELLOW_T = 3.0
ALLRED_T = 1.0

DIRS = {"E": np.array([1.0, 0.0]), "S": np.array([0.0, 1.0]),
        "W": np.array([-1.0, 0.0]), "N": np.array([0.0, -1.0])}


def right_of(v):
    """Vector rotated 90 deg clockwise on screen (= the 'right' side)."""
    return np.array([-v[1], v[0]])


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ==========================================================================
# Traffic lights
# ==========================================================================
@dataclass
class TrafficLight:
    """Two-phase light. Axis 0 = traffic moving along x (E/W), 1 = along y."""
    green0: float
    green1: float
    offset: float

    @property
    def cycle(self):
        return self.green0 + self.green1 + 2 * (YELLOW_T + ALLRED_T)

    def state(self, t: float, axis: int) -> str:
        """'G', 'Y' or 'R' for traffic travelling along `axis` at time t."""
        tc = (t + self.offset) % self.cycle
        g0, g1 = self.green0, self.green1
        # timeline: G0 | Y0 | allred | G1 | Y1 | allred
        b = [g0, g0 + YELLOW_T, g0 + YELLOW_T + ALLRED_T,
             g0 + YELLOW_T + ALLRED_T + g1, g0 + 2 * YELLOW_T + ALLRED_T + g1]
        if tc < b[0]:
            s0, s1 = "G", "R"
        elif tc < b[1]:
            s0, s1 = "Y", "R"
        elif tc < b[2]:
            s0, s1 = "R", "R"
        elif tc < b[3]:
            s0, s1 = "R", "G"
        elif tc < b[4]:
            s0, s1 = "R", "Y"
        else:
            s0, s1 = "R", "R"
        return s0 if axis == 0 else s1

    def time_to_change(self, t, axis):
        """Seconds until the state for `axis` changes (used by scenarios/HUD)."""
        s = self.state(t, axis)
        dt = 0.0
        while dt < self.cycle and self.state(t + dt, axis) == s:
            dt += 0.1
        return dt


# ==========================================================================
# Path: a dense polyline with metadata
# ==========================================================================
@dataclass
class Path:
    pts: np.ndarray                 # (M,2) points, ~0.5 m apart
    limits: np.ndarray              # (M,) speed limit at each point
    school: np.ndarray              # (M,) bool, inside a school zone
    events: list = field(default_factory=list)   # intersection controls
    crosswalks: list = field(default_factory=list)  # (s0, s1) spans
    in_lot_s: float = 1e9           # s where the path enters the parking lot
    nodes: list = field(default_factory=list)

    def __post_init__(self):
        d = np.diff(self.pts, axis=0)
        seg = np.linalg.norm(d, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        hd = np.arctan2(d[:, 1], d[:, 0])
        self.headings = np.concatenate([hd, hd[-1:]])
        self.length = float(self.s[-1])

    # --- sampling ---------------------------------------------------------
    def idx(self, s):
        """Index of the first sample at or after arc-length s."""
        if np.ndim(s) == 0:                       # fast scalar path (hot loop)
            i = int(np.searchsorted(self.s, s))
            return min(max(i, 0), len(self.s) - 1)
        return np.clip(np.searchsorted(self.s, s), 0, len(self.s) - 1)

    def pos(self, s):
        """Interpolated point(s) at arc-length s."""
        if np.ndim(s) == 0:                       # fast scalar path (hot loop)
            s = min(max(float(s), 0.0), self.length)
            i = min(max(int(np.searchsorted(self.s, s)) - 1, 0), len(self.s) - 2)
            ds = self.s[i + 1] - self.s[i]
            t = (s - self.s[i]) / ds if ds > 1e-9 else 0.0
            return self.pts[i] * (1 - t) + self.pts[i + 1] * t
        s = np.clip(s, 0.0, self.length)
        i = np.clip(np.searchsorted(self.s, s) - 1, 0, len(self.s) - 2)
        t = (s - self.s[i]) / np.maximum(self.s[i + 1] - self.s[i], 1e-9)
        t = np.asarray(t)[..., None]
        return self.pts[i] * (1 - t) + self.pts[i + 1] * t

    def heading(self, s):
        return float(self.headings[self.idx(s)]) if np.ndim(s) == 0 else self.headings[self.idx(s)]

    def limit(self, s):
        return float(self.limits[self.idx(s)])

    def project(self, p, hint: int = 0, window: int = 120):
        """Closest point on the path to p, searching near index `hint`.

        Returns (s, signed lateral offset [+ = right of path], index).
        Searching a window around the last index keeps this O(window) and
        avoids snapping to a far-away part of the route that crosses itself.
        """
        lo = max(0, hint - window // 4)
        hi = min(len(self.pts) - 1, hint + window)
        a = self.pts[lo:hi]
        b = self.pts[lo + 1:hi + 1]
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(1), 1e-12)
        t = np.clip(((p - a) * ab).sum(1) / L2, 0, 1)
        proj = a + t[:, None] * ab
        d2 = ((p - proj) ** 2).sum(1)
        k = int(np.argmin(d2))
        i = lo + k
        s = self.s[i] + t[k] * np.sqrt(L2[k])
        tang = ab[k] / np.sqrt(L2[k])
        rel = p - proj[k]
        lat = -tang[1] * rel[0] + tang[0] * rel[1]     # dot(rel, right_of(tang))
        return float(s), float(lat), i


# ==========================================================================
# The city
# ==========================================================================
class City:
    def __init__(self, rng: np.random.Generator, nx=4, ny=4, controls=True,
                 parked_prob=0.4, school_prob=0.15, lot_occupancy=0.5,
                 light_prob=0.5):
        self.rng = rng
        self.nx, self.ny = nx, ny
        # random but tidy street spacing
        self.xs = np.concatenate([[0.0], np.cumsum(rng.uniform(58, 82, nx - 1))])
        self.ys = np.concatenate([[0.0], np.cumsum(rng.uniform(58, 82, ny - 1))])
        self.nodes = np.array([[x, y] for y in self.ys for x in self.xs])
        self.width = self.xs[-1]
        self.height = self.ys[-1]

        # adjacency: full grid
        self.adj = {n: [] for n in range(nx * ny)}
        for j in range(ny):
            for i in range(nx):
                n = self.nid(i, j)
                if i + 1 < nx:
                    self._link(n, self.nid(i + 1, j))
                if j + 1 < ny:
                    self._link(n, self.nid(i, j + 1))

        # --- intersection controls -----------------------------------------
        self.control = []
        self.lights = {}
        for n in range(nx * ny):
            deg = len(self.adj[n])
            if not controls or deg <= 2:
                self.control.append("none")
            elif rng.random() < light_prob:
                self.control.append("light")
                self.lights[n] = TrafficLight(rng.uniform(9, 16), rng.uniform(9, 16),
                                              rng.uniform(0, 40))
            else:
                self.control.append("stop")

        # --- speed limits: per street line, plus school-zone segments --------
        self.row_limit = rng.choice(LIMIT_CHOICES, size=ny, p=[0.35, 0.4, 0.25])
        self.col_limit = rng.choice(LIMIT_CHOICES, size=nx, p=[0.35, 0.4, 0.25])
        self.school_edges = set()
        for a in self.adj:
            for b in self.adj[a]:
                if a < b and rng.random() < school_prob:
                    self.school_edges.add((a, b))

        # --- the parking lot (one block) -------------------------------------
        self._build_lot(lot_occupancy)
        # --- buildings, trees ---------------------------------------------
        self._build_blocks()
        # --- parked cars along streets ---------------------------------------
        self._build_parked(parked_prob)
        # --- sidewalk graph for pedestrians -----------------------------------
        self._build_sidewalks()
        self._static_boxes = None

    # ------------------------------------------------------------------ util
    def nid(self, i, j):
        return j * self.nx + i

    def ij(self, n):
        return n % self.nx, n // self.nx

    def _link(self, a, b):
        self.adj[a].append(b)
        self.adj[b].append(a)

    def edge_dir(self, a, b):
        return unit(self.nodes[b] - self.nodes[a])

    def edge_axis(self, a, b):
        """0 if the street a-b runs along x, 1 if along y."""
        d = self.nodes[b] - self.nodes[a]
        return 0 if abs(d[0]) > abs(d[1]) else 1

    def edge_limit(self, a, b):
        key = (min(a, b), max(a, b))
        if key in self.school_edges:
            return SCHOOL_LIMIT
        i, j = self.ij(a)
        return float(self.row_limit[j] if self.edge_axis(a, b) == 0 else self.col_limit[i])

    def is_school(self, a, b):
        return (min(a, b), max(a, b)) in self.school_edges

    def light_state(self, node, axis, t):
        if self.control[node] != "light":
            return None
        return self.lights[node].state(t, axis)

    # ------------------------------------------------------------- lot
    def _build_lot(self, occupancy):
        """Pick a block for the parking lot and lay out a single-aisle lot.

        The lot entrance is at the middle of one street edge (u -> v), on the
        RIGHT side of travel u->v, so a car driving u->v turns right into it.
        """
        rng = self.rng
        ci = rng.integers(0, self.nx - 1)
        cj = rng.integers(0, self.ny - 1)
        corners = [self.nid(ci, cj), self.nid(ci + 1, cj), self.nid(ci + 1, cj + 1),
                   self.nid(ci, cj + 1)]
        self.lot_cell = (ci, cj)
        # edges of the cell in clockwise order: top (0->1), right (1->2), ...
        # travelling clockwise around a cell on screen keeps the cell on the RIGHT
        k = rng.integers(0, 4)
        u, v = corners[k], corners[(k + 1) % 4]
        self.lot_edge = (u, v)
        d = self.edge_dir(u, v)
        r = right_of(d)
        m = (self.nodes[u] + self.nodes[v]) / 2.0
        depth = np.linalg.norm(self.nodes[corners[(k + 2) % 4]] - self.nodes[v])
        self.lot_d, self.lot_r, self.lot_m = d, r, m
        self.lot_depth = depth
        a0 = ROAD_HW + SIDEWALK_W            # where lot surface starts (along r)
        a1 = depth - ROAD_HW - SIDEWALK_W    # where it ends
        self.lot_aisle = (a0, a1 - 1.0)
        # slots on both sides of the aisle
        slots = []
        rr = a0 + 4.0 + SLOT_W / 2
        while rr + SLOT_W / 2 < a1 - 2.0:
            for side in (-1, 1):   # -1 => on the right when driving along r
                c = m + r * rr + d * side * (AISLE_HW + SLOT_D / 2)
                slots.append({"center": c, "side": side, "r": rr,
                              "heading": float(np.arctan2(*(d * side)[::-1]))})
            rr += SLOT_W
        self.slots = slots
        self.slot_occupied = np.array([rng.random() < occupancy for _ in slots])
        self.target_slot = None

    def choose_target_slot(self, rng, between_cars=False):
        """Pick an empty slot (optionally one with both neighbours occupied)."""
        n = len(self.slots)
        cands = []
        for i in range(n):
            if between_cars:
                same = [j for j in range(n) if self.slots[j]["side"] == self.slots[i]["side"]
                        and abs(self.slots[j]["r"] - self.slots[i]["r"]) < SLOT_W * 1.5 and j != i]
                if len(same) == 2:
                    cands.append(i)
            else:
                cands.append(i)
        # avoid the first two rows near the entrance so there is a bit of aisle
        cands = [i for i in cands if self.slots[i]["r"] > self.lot_aisle[0] + 6] or cands
        i = int(rng.choice(cands))
        self.target_slot = i
        self.slot_occupied[i] = False
        if between_cars:
            for j in range(n):
                if (self.slots[j]["side"] == self.slots[i]["side"]
                        and abs(self.slots[j]["r"] - self.slots[i]["r"]) < SLOT_W * 1.5 and j != i):
                    self.slot_occupied[j] = True
        self._static_boxes = None
        return i

    def lot_polys(self):
        """Rectangles (as OBB rows) of drivable lot surface: driveway+aisle+slots."""
        d, r, m = self.lot_d, self.lot_r, self.lot_m
        ang = float(np.arctan2(r[1], r[0]))
        a0, a1 = self.lot_aisle
        rects = []
        # driveway through the sidewalk + aisle
        c = m + r * ((ROAD_HW + a1) / 2)
        rects.append([c[0], c[1], (a1 - ROAD_HW) / 2, AISLE_HW, ang])
        # slot rows
        r_lo = min(s["r"] for s in self.slots) - SLOT_W / 2
        r_hi = max(s["r"] for s in self.slots) + SLOT_W / 2
        for side in (-1, 1):
            c = m + r * ((r_lo + r_hi) / 2) + d * side * (AISLE_HW + SLOT_D / 2)
            rects.append([c[0], c[1], (r_hi - r_lo) / 2, SLOT_D / 2, ang])
        return np.array(rects)

    # ------------------------------------------------------------- blocks
    def _build_blocks(self):
        rng = self.rng
        boxes, colors, trees, parks = [], [], [], []
        inset = ROAD_HW + SIDEWALK_W + 0.8
        for j in range(self.ny - 1):
            for i in range(self.nx - 1):
                x0, x1 = self.xs[i] + inset, self.xs[i + 1] - inset
                y0, y1 = self.ys[j] + inset, self.ys[j + 1] - inset
                if (i, j) == self.lot_cell:
                    # trees in the lot's landscaped margins
                    for _ in range(10):
                        p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
                        if not self.point_in_lot_surface(p, margin=3.5):
                            trees.append((p[0], p[1], rng.uniform(0.7, 1.1)))
                    continue
                if rng.random() < 0.12:            # a little park
                    parks.append((x0, y0, x1, y1))
                    for _ in range(rng.integers(6, 12)):
                        trees.append((rng.uniform(x0 + 3, x1 - 3), rng.uniform(y0 + 3, y1 - 3),
                                      rng.uniform(0.8, 1.3)))
                    continue
                # split block into 2x2-ish buildings with random gaps
                nxs = rng.integers(1, 3)
                nys = rng.integers(1, 3)
                xe = np.linspace(x0, x1, nxs + 1)
                ye = np.linspace(y0, y1, nys + 1)
                for a in range(nxs):
                    for b in range(nys):
                        bx0, bx1 = xe[a] + (1.5 if a > 0 else 0), xe[a + 1] - (1.5 if a < nxs - 1 else 0)
                        by0, by1 = ye[b] + (1.5 if b > 0 else 0), ye[b + 1] - (1.5 if b < nys - 1 else 0)
                        # random setback from the sidewalk (front yards)
                        sb = rng.uniform(0, 4, size=4)
                        bx0 += sb[0]; bx1 -= sb[1]; by0 += sb[2]; by1 -= sb[3]
                        if bx1 - bx0 < 6 or by1 - by0 < 6:
                            continue
                        boxes.append([(bx0 + bx1) / 2, (by0 + by1) / 2, (bx1 - bx0) / 2,
                                      (by1 - by0) / 2, 0.0])
                        colors.append(int(rng.integers(0, 6)))
                        if rng.random() < 0.5:
                            trees.append((bx0 - 1.2 + rng.uniform(-0.5, 0.5),
                                          rng.uniform(by0, by1), rng.uniform(0.5, 0.8)))
        self.buildings = np.array(boxes) if boxes else np.zeros((0, 5))
        self.building_colors = colors
        self.trees = trees
        self.parks = parks

    def point_in_lot_surface(self, p, margin=0.0):
        rel = p - self.lot_m
        a = rel @ self.lot_r
        b = rel @ self.lot_d
        a0, a1 = self.lot_aisle
        if ROAD_HW - margin <= a <= a1 + margin and abs(b) <= AISLE_HW + margin:
            return True
        r_lo = min(s["r"] for s in self.slots) - SLOT_W / 2
        r_hi = max(s["r"] for s in self.slots) + SLOT_W / 2
        return (r_lo - margin <= a <= r_hi + margin) and abs(b) <= AISLE_HW + SLOT_D + margin

    # ------------------------------------------------------------- parked
    def _build_parked(self, prob):
        """Parked cars in the parking strips; remember the gaps between them
        (kids hide there and step out in front of you)."""
        rng = self.rng
        boxes, meta, gaps = [], [], []
        for a in self.adj:
            for b in self.adj[a]:
                # each directed edge a->b owns the parking strip on its right
                if rng.random() > prob:
                    continue
                d = self.edge_dir(a, b)
                r = right_of(d)
                L = np.linalg.norm(self.nodes[b] - self.nodes[a])
                ang = float(np.arctan2(d[1], d[0]))
                s = STOPLINE_D + 4.0 + rng.uniform(0, 6)
                prev_end = None
                while s < L - STOPLINE_D - 6.0:
                    c = self.nodes[a] + d * s + r * PARKED_OFF
                    if (a, b) == self.lot_edge and abs(s - L / 2) < 10:
                        s += 6.0
                        continue
                    if rng.random() < 0.75:
                        clen = rng.uniform(4.2, 4.8)
                        boxes.append([c[0], c[1], clen / 2, 0.92, ang])
                        meta.append({"edge": (a, b), "s": s, "color": int(rng.integers(0, 11))})
                        if prev_end is not None and s - clen / 2 - prev_end > 0.9:
                            gs = (prev_end + s - clen / 2) / 2
                            gaps.append({"edge": (a, b), "s": gs,
                                         "pos": self.nodes[a] + d * gs + r * PARKED_OFF})
                        prev_end = s + clen / 2
                        s += clen + rng.uniform(1.0, 3.5)
                    else:
                        prev_end = None
                        s += rng.uniform(5, 9)
        self.parked = np.array(boxes) if boxes else np.zeros((0, 5))
        self.parked_meta = meta
        self.parked_gaps = gaps

    def lot_parked_boxes(self):
        out = []
        for i, sl in enumerate(self.slots):
            if self.slot_occupied[i]:
                c = sl["center"]
                out.append([c[0], c[1], 2.3, 0.92, sl["heading"]])
        return np.array(out) if out else np.zeros((0, 5))

    def static_boxes(self):
        """All static obstacles as OBBs: buildings + parked cars + lot cars."""
        if self._static_boxes is None:
            lp = self.lot_parked_boxes()
            self._static_boxes = np.concatenate([self.buildings, self.parked, lp], 0)
            self._n_buildings = len(self.buildings)
        return self._static_boxes

    def remove_parked_near(self, p, radius):
        """Used by scenarios to clear space."""
        if len(self.parked):
            keep = np.linalg.norm(self.parked[:, :2] - p, axis=1) > radius
            self.parked = self.parked[keep]
            self.parked_meta = [m for m, k in zip(self.parked_meta, keep) if k]
        self._static_boxes = None

    # ------------------------------------------------------------- sidewalks
    def _build_sidewalks(self):
        """Graph of sidewalk corner points; edges are sidewalk segments or crosswalks."""
        self.sw_nodes = []        # positions
        self.sw_index = {}        # (node, qx, qy) -> id
        for n in range(len(self.nodes)):
            for qx in (-1, 1):
                for qy in (-1, 1):
                    self.sw_index[(n, qx, qy)] = len(self.sw_nodes)
                    self.sw_nodes.append(self.nodes[n] + CORNER_OFF * np.array([qx, qy]))
        self.sw_nodes = np.array(self.sw_nodes)
        self.sw_adj = {k: [] for k in range(len(self.sw_nodes))}
        # edge type: ('walk', None) or ('cross', (node, axis_of_crossed_street))
        self.sw_edge = {}

        def add(p, q, kind):
            self.sw_adj[p].append(q)
            self.sw_adj[q].append(p)
            self.sw_edge[(p, q)] = kind
            self.sw_edge[(q, p)] = kind

        for n in range(len(self.nodes)):
            i, j = self.ij(n)
            S = self.sw_index
            # walk along the street to the next intersection
            if i + 1 < self.nx:
                m = self.nid(i + 1, j)
                add(S[(n, 1, -1)], S[(m, -1, -1)], ("walk", None))
                add(S[(n, 1, 1)], S[(m, -1, 1)], ("walk", None))
            if j + 1 < self.ny:
                m = self.nid(i, j + 1)
                add(S[(n, -1, 1)], S[(m, -1, -1)], ("walk", None))
                add(S[(n, 1, 1)], S[(m, 1, -1)], ("walk", None))
            # crosswalks across each existing arm
            if i + 1 < self.nx:       # east arm: street along x, crossing moves in y
                add(S[(n, 1, -1)], S[(n, 1, 1)], ("cross", (n, 0)))
            if i > 0:
                add(S[(n, -1, -1)], S[(n, -1, 1)], ("cross", (n, 0)))
            if j + 1 < self.ny:       # south arm: street along y
                add(S[(n, -1, 1)], S[(n, 1, 1)], ("cross", (n, 1)))
            if j > 0:
                add(S[(n, -1, -1)], S[(n, 1, -1)], ("cross", (n, 1)))
            # outer sides of the grid have no arm: the two corners on that side
            # are joined by a plain sidewalk running past the street end.
            if i == 0:
                add(S[(n, -1, -1)], S[(n, -1, 1)], ("walk", None))
            if i == self.nx - 1:
                add(S[(n, 1, -1)], S[(n, 1, 1)], ("walk", None))
            if j == 0:
                add(S[(n, -1, -1)], S[(n, 1, -1)], ("walk", None))
            if j == self.ny - 1:
                add(S[(n, -1, 1)], S[(n, 1, 1)], ("walk", None))

    # ------------------------------------------------------------- queries
    def drivable(self, pts: np.ndarray) -> np.ndarray:
        """Bool per point: on asphalt (street, intersection) or lot surface."""
        pts = np.atleast_2d(pts)
        dx = np.min(np.abs(pts[:, 0:1] - self.xs[None, :]), axis=1)
        dy = np.min(np.abs(pts[:, 1:2] - self.ys[None, :]), axis=1)
        inside_x = (pts[:, 0] >= -ROAD_HW) & (pts[:, 0] <= self.width + ROAD_HW)
        inside_y = (pts[:, 1] >= -ROAD_HW) & (pts[:, 1] <= self.height + ROAD_HW)
        on = ((dx <= ROAD_HW) & inside_y) | ((dy <= ROAD_HW) & inside_x)
        out = on.copy()
        for k in np.where(~on)[0]:
            out[k] = self.point_in_lot_surface(pts[k])
        return out

    def on_sidewalk(self, pts):
        pts = np.atleast_2d(pts)
        dx = np.min(np.abs(pts[:, 0:1] - self.xs[None, :]), axis=1)
        dy = np.min(np.abs(pts[:, 1:2] - self.ys[None, :]), axis=1)
        return (np.minimum(dx, dy) > ROAD_HW) & (np.minimum(dx, dy) <= ROAD_HW + SIDEWALK_W)

    def random_directed_edge(self, rng):
        a = int(rng.integers(0, len(self.nodes)))
        b = int(rng.choice(self.adj[a]))
        return a, b

    # ==================================================================
    # Lane path construction
    # ==================================================================
    def lane_point(self, a, b, s):
        """Point at distance s along directed edge a->b, in the right lane."""
        d = self.edge_dir(a, b)
        return self.nodes[a] + d * s + right_of(d) * LANE_OFF

    def build_path(self, nodes_seq, start_s=None, end_s=None, step=0.5, extra_tail=None):
        """Dense right-lane polyline for a node sequence.

        nodes_seq: [n0, n1, ..., nk]  (consecutive nodes are adjacent)
        start_s:   distance along the first edge where the path starts
                   (default: right after the first intersection box)
        end_s:     distance along the last edge where the path stops
        extra_tail: optional list of (pts, limit) to append (the parking lot)
        """
        pts, lims, sch = [], [], []
        events, cws = [], []

        def add_seg(p0, p1, lim, school):
            L = np.linalg.norm(p1 - p0)
            n = max(int(L / step), 1)
            t = (np.arange(n) / n)[:, None]
            pts.extend(list(p0 * (1 - t) + p1 * t))
            lims.extend([lim] * n)
            sch.extend([school] * n)

        def add_arc(center, radius, a0, a1, lim):
            # sweep from angle a0 to a1 (shortest way)
            da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
            n = max(int(abs(da) * radius / step), 2)
            for k in range(n):
                ang = a0 + da * k / n
                pts.append(center + radius * np.array([np.cos(ang), np.sin(ang)]))
                lims.append(lim)
                sch.append(False)

        k_last = len(nodes_seq) - 2
        for k in range(len(nodes_seq) - 1):
            a, b = nodes_seq[k], nodes_seq[k + 1]
            d = self.edge_dir(a, b)
            L = float(np.linalg.norm(self.nodes[b] - self.nodes[a]))
            lim = self.edge_limit(a, b)
            school = self.is_school(a, b)
            s0 = ROAD_HW if k > 0 else (start_s if start_s is not None else ROAD_HW)
            s1 = L - ROAD_HW if k < k_last else (end_s if end_s is not None else L - ROAD_HW)
            # straight part of this edge
            seg_start_len = len(pts)
            add_seg(self.lane_point(a, b, s0), self.lane_point(a, b, s1), lim, school)
            # remember the control at the end of this edge (node b)
            if k < k_last or (end_s is None):
                ctrl = self.control[b]
                # path length so far is approx (len(pts))*step but compute exactly later
                events.append({"node": b, "kind": ctrl, "axis": self.edge_axis(a, b),
                               "_stop_local": L - STOPLINE_D - s0, "_seg_start": seg_start_len,
                               "edge": (a, b)})
            if k < k_last:
                c = nodes_seq[k + 2]
                d2 = self.edge_dir(b, c)
                cross = d[0] * d2[1] - d[1] * d2[0]   # >0 => right turn (screen coords)
                p_in = self.lane_point(a, b, L - ROAD_HW)
                if abs(cross) < 1e-6:
                    add_seg(p_in, self.lane_point(b, c, ROAD_HW), lim, False)
                    events[-1]["turn"] = "S"
                else:
                    if cross > 0:      # right turn: small radius
                        R = ROAD_HW - LANE_OFF
                        center = p_in + right_of(d) * R
                        events[-1]["turn"] = "R"
                    else:              # left turn: big radius
                        R = ROAD_HW + LANE_OFF
                        center = p_in - right_of(d) * R
                        events[-1]["turn"] = "L"
                    p_out = self.lane_point(b, c, ROAD_HW)
                    a0 = np.arctan2(*(p_in - center)[::-1])
                    a1 = np.arctan2(*(p_out - center)[::-1])
                    add_arc(center, R, a0, a1, min(lim, 25 * KMH))
        if extra_tail is not None:
            for seg_pts, lim in extra_tail:
                for p in seg_pts:
                    pts.append(p)
                    lims.append(lim)
                    sch.append(False)
        else:
            # final point
            a, b = nodes_seq[-2], nodes_seq[-1]
            L = float(np.linalg.norm(self.nodes[b] - self.nodes[a]))
            pts.append(self.lane_point(a, b, end_s if end_s is not None else L - ROAD_HW))
            lims.append(lims[-1])
            sch.append(sch[-1])

        P = np.array(pts)
        # remove duplicate consecutive points
        keep = np.concatenate([[True], np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6])
        idx_map = np.cumsum(keep) - 1
        P = P[keep]
        path = Path(P, np.array(lims)[keep], np.array(sch)[keep], nodes=list(nodes_seq))
        # resolve event arc-lengths now that s is known
        for ev in events:
            i0 = idx_map[min(ev["_seg_start"], len(idx_map) - 1)]
            s_line = path.s[i0] + ev.pop("_stop_local")
            ev.pop("_seg_start")
            ev["s"] = float(s_line)                         # stop line
            ev["s_cross0"] = float(s_line + 0.5)            # crosswalk start
            ev["s_box"] = float(s_line + 0.5 + CROSSWALK_W)  # intersection box entry
            ev["s_exit"] = float(ev["s_box"] + 2 * ROAD_HW + 4)  # roughly past the box
            path.crosswalks.append((ev["s_cross0"], ev["s_box"]))
            path.crosswalks.append((ev["s_exit"] - 4, ev["s_exit"] - 4 + CROSSWALK_W))
        path.events = [e for e in events if e["s"] <= path.length + 1]
        return path

    # ------------------------------------------------------------- routes
    def hops_to(self, target):
        """Hop count from every directed edge to `target` (no U-turns)."""
        from collections import deque
        dist = {target: 0}
        q = deque([target])
        while q:
            e = q.popleft()
            a, b = e
            # predecessors: (p, a) with p != b
            for p in self.adj[a]:
                if p == b:
                    continue
                pe = (p, a)
                if pe not in dist:
                    dist[pe] = dist[e] + 1
                    q.append(pe)
        return dist

    def lot_tail(self):
        """Points from the lot driveway to the target slot centre (+limits)."""
        d, r, m = self.lot_d, self.lot_r, self.lot_m
        sl = self.slots[self.target_slot]
        R1 = 6.0
        # arc 1: from the lane (offset LANE_OFF) turning right into the driveway
        p0 = m - d * R1 + r * LANE_OFF
        c1 = p0 + r * R1
        pts1 = []
        n = 30
        for k in range(n + 1):
            ang = -np.pi / 2 * (1 - k / n)       # in the (d, r) frame
            # point = c1 + R1*(cos(phi)*d + sin(phi)*r) with phi from -90 (p0) to 0
            phi = -np.pi / 2 + (np.pi / 2) * k / n
            pts1.append(c1 + R1 * (np.cos(phi) * d + np.sin(phi) * r) * 1.0)
        # c1 + R1*(cos(-90) d + sin(-90) r) = c1 - R1 r = p0  OK; end = c1 + R1 d
        p_end1 = pts1[-1]
        # straight along the aisle to the start of arc 2. Like a human driver we
        # first swing wide (away from the slot side) so the turn ends early and
        # there is a few metres of straight line to settle into the slot.
        R2 = 4.5
        side = sl["side"]
        swing = 1.8
        a_end = sl["r"] - R2
        aisle_pts = []
        a_start = float((p_end1 - m) @ r)
        for a in np.arange(a_start + 0.5, a_end, 0.5):
            u = np.clip((a - (a_end - 9.0)) / 9.0, 0, 1)
            u = u * u * (3 - 2 * u)                      # smoothstep 0 -> 1
            aisle_pts.append(m + r * a - d * side * swing * u)
        p2 = m + r * a_end - d * side * swing
        # turning towards d*side; center offset towards that side
        c2 = p2 + d * side * R2
        arc2 = []
        for k in range(n + 1):
            phi = np.pi * k / (2 * n)
            # start at p2 (= c2 - side*d*R2) heading r, end at c2 + r*R2 heading side*d
            arc2.append(c2 - d * side * R2 * np.cos(phi) + r * R2 * np.sin(phi))
        p_end2 = arc2[-1]
        tail_pts = []
        end = sl["center"]
        L = np.linalg.norm(end - p_end2)
        for t in np.arange(0.5, L + 1e-6, 0.5):
            tail_pts.append(p_end2 + (end - p_end2) * min(t / L, 1.0))
        tail_pts.append(end)
        return [(pts1, LOT_LIMIT), (aisle_pts, LOT_LIMIT), (arc2, LOT_LIMIT * 0.8),
                (tail_pts, LOT_LIMIT * 0.6)]

    def build_route(self, rng, min_hops=1, max_hops=4, start_edge=None, start_s=None):
        """Agent route: start on some lane, drive to the lot and into the slot.

        Returns (Path, start_edge).
        """
        u, v = self.lot_edge
        target = (u, v)
        hops = self.hops_to(target)
        if start_edge is None:
            cands = [e for e, h in hops.items() if min_hops <= h <= max_hops]
            if not cands:
                cands = [e for e in hops if e != target]
            start_edge = cands[int(rng.integers(0, len(cands)))]
        # greedy descent with random tie breaking
        seq = [start_edge[0], start_edge[1]]
        e = start_edge
        while e != target:
            a, b = e
            nxt = [(b, c) for c in self.adj[b] if c != a and hops.get((b, c), 1e9) == hops[e] - 1]
            e = nxt[int(rng.integers(0, len(nxt)))]
            seq.append(e[1])
        L0 = float(np.linalg.norm(self.nodes[start_edge[1]] - self.nodes[start_edge[0]]))
        if start_s is None:
            hi = L0 - STOPLINE_D - 12 if len(seq) > 2 else L0 / 2 - 22
            start_s = float(rng.uniform(STOPLINE_D + 2, max(hi, STOPLINE_D + 3)))
        Ld = float(np.linalg.norm(self.nodes[v] - self.nodes[u]))
        end_s = Ld / 2 - 6.0          # where arc 1 into the driveway begins
        tail = self.lot_tail()
        path = self.build_path(seq, start_s=start_s, end_s=end_s, extra_tail=tail)
        # s where the lot begins = total length minus the tail length
        tp = np.concatenate([np.array(t[0]) for t in tail if len(t[0])], 0)
        tail_len = float(np.sum(np.linalg.norm(np.diff(tp, axis=0), axis=1)))
        path.in_lot_s = path.length - tail_len
        return path, start_edge

    def random_walk_nodes(self, rng, start_edge, n_edges=8):
        seq = [start_edge[0], start_edge[1]]
        for _ in range(n_edges):
            a, b = seq[-2], seq[-1]
            opts = [c for c in self.adj[b] if c != a] or [a]
            seq.append(int(rng.choice(opts)))
        return seq
