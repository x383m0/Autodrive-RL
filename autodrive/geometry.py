"""
geometry.py - small, fast, vectorised 2D geometry helpers.

Everything in the simulator lives in a flat 2D world measured in METRES.
Coordinate convention (chosen to match pygame's screen so rendering is trivial):

    +x  -> right / "east"
    +y  -> DOWN  / "south"
    heading angle theta is measured from +x, and because +y points down,
    increasing theta rotates CLOCKWISE on screen (i.e. a right turn).

    forward unit vector  f = ( cos theta,  sin theta)
    right   unit vector  r = (-sin theta,  cos theta)   (90 deg clockwise of f)

Objects are approximated by two primitives:
    * Oriented boxes (OBB)  -> cars, buildings, parked cars
      stored as rows [cx, cy, half_length, half_width, angle]
    * Circles               -> pedestrians
      stored as rows [cx, cy, radius]

The ray caster is the heart of the "camera" sensors, so it is written with
numpy broadcasting: R rays x B boxes are intersected in one shot, no Python
loops. With ~70 rays and ~100 nearby boxes this takes a few tens of
microseconds.
"""
from __future__ import annotations

import numpy as np

INF = 1e9


def wrap_angle(a):
    """Wrap an angle (or array of angles) into [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def rot(theta: float) -> np.ndarray:
    """2x2 rotation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def to_local(points: np.ndarray, origin, theta: float) -> np.ndarray:
    """Express world points in a frame located at `origin` with heading `theta`.

    Returned columns are (forward, right): x>0 means "in front", y>0 "to the right".
    """
    d = np.asarray(points, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    c, s = np.cos(theta), np.sin(theta)
    fwd = d[..., 0] * c + d[..., 1] * s
    rgt = -d[..., 0] * s + d[..., 1] * c
    return np.stack([fwd, rgt], axis=-1)


def obb_corners(box) -> np.ndarray:
    """Return the 4 corners (4,2) of a single oriented box [cx,cy,hl,hw,ang]."""
    cx, cy, hl, hw, a = box
    c, s = np.cos(a), np.sin(a)
    f = np.array([c, s]) * hl
    r = np.array([-s, c]) * hw
    ctr = np.array([cx, cy])
    return np.array([ctr + f + r, ctr + f - r, ctr - f - r, ctr - f + r])


# --------------------------------------------------------------------------
# Ray casting
# --------------------------------------------------------------------------
def ray_obb(origins: np.ndarray, dirs: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Distance along each ray to each oriented box (slab method).

    origins: (R,2) ray start points
    dirs:    (R,2) unit direction vectors
    boxes:   (B,5) [cx, cy, half_len, half_wid, angle]
    returns: (R,B) hit distance, INF where the ray misses (or box is behind).

    Idea: rotate every ray into each box's local frame; there the box is an
    axis-aligned rectangle [-hl,hl] x [-hw,hw] and the classic slab test
    applies: intersect the ray's parameter intervals for the x-slab and the
    y-slab, a hit exists if the intervals overlap at positive t.
    """
    if len(boxes) == 0:
        return np.full((len(origins), 0), INF)
    c = np.cos(boxes[:, 4])[None, :]
    s = np.sin(boxes[:, 4])[None, :]
    dx = origins[:, 0:1] - boxes[None, :, 0]
    dy = origins[:, 1:2] - boxes[None, :, 1]
    # ray origin in box frame
    ox = dx * c + dy * s
    oy = -dx * s + dy * c
    # ray direction in box frame
    vx = dirs[:, 0:1] * c + dirs[:, 1:2] * s
    vy = -dirs[:, 0:1] * s + dirs[:, 1:2] * c
    hl = boxes[None, :, 2]
    hw = boxes[None, :, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_x = 1.0 / np.where(np.abs(vx) < 1e-12, 1e-12, vx)
        inv_y = 1.0 / np.where(np.abs(vy) < 1e-12, 1e-12, vy)
        t1x, t2x = (-hl - ox) * inv_x, (hl - ox) * inv_x
        t1y, t2y = (-hw - oy) * inv_y, (hw - oy) * inv_y
    tmin = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
    tmax = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
    hit = (tmax >= np.maximum(tmin, 0.0))
    # if the origin is inside the box tmin<0: report 0 distance
    t = np.where(tmin > 0, tmin, 0.0)
    return np.where(hit, t, INF)


def ray_circle(origins: np.ndarray, dirs: np.ndarray, circles: np.ndarray) -> np.ndarray:
    """Distance along each ray to each circle. circles: (C,3) [cx,cy,r]."""
    if len(circles) == 0:
        return np.full((len(origins), 0), INF)
    mx = origins[:, 0:1] - circles[None, :, 0]
    my = origins[:, 1:2] - circles[None, :, 1]
    b = mx * dirs[:, 0:1] + my * dirs[:, 1:2]
    cc = mx * mx + my * my - circles[None, :, 2] ** 2
    disc = b * b - cc
    sq = np.sqrt(np.maximum(disc, 0.0))
    t = -b - sq
    t = np.where(t < 0, -b + sq, t)          # origin inside the circle
    ok = (disc >= 0) & (t >= 0)
    return np.where(ok, np.maximum(-b - sq, 0.0), INF)


# --------------------------------------------------------------------------
# Overlap tests (collision detection)
# --------------------------------------------------------------------------
def obb_overlap_many(box, boxes: np.ndarray) -> np.ndarray:
    """Separating-axis test of ONE box against MANY boxes. Returns bool (B,).

    Two convex shapes do not overlap iff there is an axis on which their
    projections are disjoint. For rectangles only 4 candidate axes exist
    (the 2 edge normals of each rectangle).
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=bool)
    cx, cy, hl, hw, a = box
    ax = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])     # (2,2)
    bc, bs = np.cos(boxes[:, 4]), np.sin(boxes[:, 4])
    bx = np.stack([np.stack([bc, bs], -1), np.stack([-bs, bc], -1)], 1)  # (B,2,2)
    d = boxes[:, :2] - np.array([cx, cy])                                # (B,2)
    sep = np.zeros(len(boxes), dtype=bool)
    # axes of box A
    for k in range(2):
        n = ax[k]
        ra = hl if k == 0 else hw
        rb = boxes[:, 2] * np.abs(bx[:, 0] @ n) + boxes[:, 3] * np.abs(bx[:, 1] @ n)
        sep |= np.abs(d @ n) > ra + rb
    # axes of each box B
    for k in range(2):
        n = bx[:, k]                                                     # (B,2)
        ra = hl * np.abs(n @ ax[0]) + hw * np.abs(n @ ax[1])
        rb = boxes[:, 2] if k == 0 else boxes[:, 3]
        sep |= np.abs((d * n).sum(1)) > ra + rb
    return ~sep


def obb_circle_overlap(box, circles: np.ndarray) -> np.ndarray:
    """One oriented box vs many circles -> bool (C,)."""
    if len(circles) == 0:
        return np.zeros(0, dtype=bool)
    cx, cy, hl, hw, a = box
    loc = to_local(circles[:, :2], (cx, cy), a)
    qx = np.clip(loc[:, 0], -hl, hl)
    qy = np.clip(loc[:, 1], -hw, hw)
    return (loc[:, 0] - qx) ** 2 + (loc[:, 1] - qy) ** 2 < circles[:, 2] ** 2


def dist_to_box_boundary(hl: float, hw: float, local_angles: np.ndarray) -> np.ndarray:
    """Distance from a box centre to its boundary along given local directions.

    Used by the ultrasonic sensors: rays start at the car centre, so we subtract
    the part of the ray that is still inside our own bodywork.
    """
    c = np.abs(np.cos(local_angles)) + 1e-9
    s = np.abs(np.sin(local_angles)) + 1e-9
    return np.minimum(hl / c, hw / s)


def point_segment_distance(p, a, b):
    """Distance from points p (N,2) to segment a-b."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=1)
