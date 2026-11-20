"""
sensors.py - what the agent can SEE. It is not omniscient.

The agent's perception is a set of "semantic depth cameras": each camera
casts a fan of rays inside its field of view; every ray stops at the FIRST
object it hits (so things behind a parked car or around a building corner
are invisible = real occlusion) and reports
    * how far the hit is (normalised closeness, 0 = nothing in range)
    * what class it hit: static (building / parked car), vehicle, pedestrian
    * whether a hit vehicle is signalling a turn: -1 left, +1 right, 0 none
      (the car's own left/right, like reading its blinker on camera)

This mimics what a vision stack (camera + object detector + depth estimate)
hands to a planner, while staying cheap enough to train on a CPU.

Camera layout (yaw relative to the car's nose, + = clockwise/right):

    name          yaw   FOV    rays  range
    front_wide      0   120 deg  21   45 m    general awareness
    front_narrow    0    30 deg   9  100 m    far look-ahead (fast roads)
    left          -90   100 deg   9   30 m    cross traffic, kids
    right         +90   100 deg   9   30 m
    rear          180   100 deg   9   40 m
    ultrasonic   ring   360 deg  12    5 m    parking sensors (distance from bodywork)

Plus "sign/lane reading" from the front camera (lane offset, the state of
the next traffic light / stop sign, the posted speed limit) which the
environment computes and only reveals when it is within camera range.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import INF, dist_to_box_boundary, ray_circle, ray_obb
from .vehicle import LENGTH, WIDTH


@dataclass
class Camera:
    name: str
    yaw_deg: float
    fov_deg: float
    n_rays: int
    range: float
    mount: tuple  # (forward, right) offset on the car body

    def local_angles(self):
        if self.n_rays == 1:
            return np.array([np.deg2rad(self.yaw_deg)])
        half = np.deg2rad(self.fov_deg) / 2
        return np.deg2rad(self.yaw_deg) + np.linspace(-half, half, self.n_rays)


CAMERAS = [
    Camera("front_wide", 0, 120, 21, 45.0, (2.0, 0.0)),
    Camera("front_narrow", 0, 30, 9, 100.0, (2.0, 0.0)),
    Camera("left", -90, 100, 9, 30.0, (0.3, -0.9)),
    Camera("right", 90, 100, 9, 30.0, (0.3, 0.9)),
    Camera("rear", 180, 100, 9, 40.0, (-2.2, 0.0)),
]
ULTRA_RAYS = 12
ULTRA_RANGE = 5.0

N_CAM_RAYS = sum(c.n_rays for c in CAMERAS)
N_RAYS = N_CAM_RAYS + ULTRA_RAYS
CLS_NONE, CLS_STATIC, CLS_VEHICLE, CLS_PED = 0, 1, 2, 3
RAY_FEATURES = 5          # closeness, static, vehicle, pedestrian, turn signal


class SensorSuite:
    def __init__(self):
        loc_o, loc_a, rng, cam_id = [], [], [], []
        for ci, c in enumerate(CAMERAS):
            a = c.local_angles()
            loc_a.append(a)
            loc_o.append(np.tile(np.array(c.mount, dtype=float), (len(a), 1)))
            rng.append(np.full(len(a), c.range))
            cam_id.append(np.full(len(a), ci))
        ua = np.linspace(0, 2 * np.pi, ULTRA_RAYS, endpoint=False)
        loc_a.append(ua)
        loc_o.append(np.zeros((ULTRA_RAYS, 2)))
        rng.append(np.full(ULTRA_RAYS, ULTRA_RANGE))
        cam_id.append(np.full(ULTRA_RAYS, len(CAMERAS)))
        self.local_ang = np.concatenate(loc_a)
        self.local_org = np.concatenate(loc_o)
        self.ranges = np.concatenate(rng)
        self.cam_id = np.concatenate(cam_id)
        # ultrasonic rays start at the car centre: subtract the bodywork part
        self.body_off = np.zeros(N_RAYS)
        self.body_off[N_CAM_RAYS:] = dist_to_box_boundary(LENGTH / 2, WIDTH / 2, ua)
        self.max_range = float(self.ranges.max())

    def scan(self, x, y, theta, static_boxes, vehicle_boxes, ped_circles, vehicle_signals=None):
        """Cast all rays. Returns (dist (R,), cls (R,), origins (R,2), dirs (R,2), sig (R,)).

        vehicle_signals: per vehicle box, -1 (left blinker), +1 (right) or 0."""
        c, s = np.cos(theta), np.sin(theta)
        # origins in world frame
        fo, ro = self.local_org[:, 0], self.local_org[:, 1]
        ox = x + fo * c - ro * s
        oy = y + fo * s + ro * c
        origins = np.stack([ox, oy], 1)
        ang = theta + self.local_ang
        dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
        R = self.max_range + 6.0
        me = np.array([x, y])

        def keep(arr, extra):
            if len(arr) == 0:
                return np.zeros(0, dtype=bool)
            return np.linalg.norm(arr[:, :2] - me, axis=1) < R + extra

        sb = static_boxes[keep(static_boxes, 30.0)] if len(static_boxes) else static_boxes   # buildings can be big
        kv = keep(vehicle_boxes, 3.0)
        vb = vehicle_boxes[kv] if len(vehicle_boxes) else vehicle_boxes
        vs = (np.asarray(vehicle_signals, dtype=np.float32)[kv] if vehicle_signals is not None and len(vb)
              else np.zeros(len(vb), dtype=np.float32))
        sig = np.zeros(N_RAYS, dtype=np.float32)
        pc = ped_circles[keep(ped_circles, 1.0)] if len(ped_circles) else ped_circles
        best = np.full(N_RAYS, INF)
        cls = np.zeros(N_RAYS, dtype=np.int64)
        # static + vehicle boxes in ONE vectorised call, then split by class
        boxes = np.concatenate([sb.reshape(-1, 5), vb.reshape(-1, 5)], 0)
        if len(boxes):
            d = ray_obb(origins, dirs, boxes)
            j = d.argmin(axis=1)
            best = d[np.arange(len(d)), j]
            cls = np.where(j < len(sb), CLS_STATIC, CLS_VEHICLE)
            if len(vb):
                sig = np.where(j >= len(sb), vs[np.clip(j - len(sb), 0, len(vb) - 1)], 0.0).astype(np.float32)
        if len(pc):
            d = ray_circle(origins, dirs, pc).min(axis=1)
            upd = d < best
            best = np.where(upd, d, best)
            cls = np.where(upd, CLS_PED, cls)
            sig = np.where(upd, 0.0, sig).astype(np.float32)
        dist = best - self.body_off
        miss = dist > self.ranges
        dist = np.where(miss, self.ranges, np.maximum(dist, 0.0))
        cls = np.where(miss, CLS_NONE, cls)
        sig = np.where(miss, 0.0, sig).astype(np.float32)
        return dist, cls, origins, dirs, sig

    def encode(self, dist, cls, sig=None):
        """(R,5): closeness in [0,1] (1 = touching, 0 = nothing), 3 class flags, turn signal."""
        close = 1.0 - dist / self.ranges
        out = np.zeros((N_RAYS, RAY_FEATURES), dtype=np.float32)
        out[:, 0] = close
        out[:, 1] = cls == CLS_STATIC
        out[:, 2] = cls == CLS_VEHICLE
        out[:, 3] = cls == CLS_PED
        if sig is not None:
            out[:, 4] = sig
        return out.reshape(-1)
