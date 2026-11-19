"""
sensors.py - what the agent can SEE. It is not omniscient.

The agent's perception is a set of "semantic depth cameras": each camera
casts a fan of rays inside its field of view; every ray stops at the FIRST
object it hits (so things behind a parked car or around a building corner
are invisible = real occlusion) and reports
    * how far the hit is (normalised closeness, 0 = nothing in range)
    * what class it hit: static (building / parked car), vehicle, pedestrian

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

    def scan(self, x, y, theta, static_boxes, vehicle_boxes, ped_circles):
        """Cast all rays. Returns (dist (R,), cls (R,), origins (R,2), dirs (R,2))."""
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

        def cull(arr, extra):
            if len(arr) == 0:
                return arr
            dd = np.linalg.norm(arr[:, :2] - me, axis=1)
            return arr[dd < R + extra]

        sb = cull(static_boxes, 30.0)   # buildings can be big
        vb = cull(vehicle_boxes, 3.0)
        pc = cull(ped_circles, 1.0)
        best = np.full(N_RAYS, INF)
        cls = np.zeros(N_RAYS, dtype=np.int64)
        # static + vehicle boxes in ONE vectorised call, then split by class
        boxes = np.concatenate([sb.reshape(-1, 5), vb.reshape(-1, 5)], 0)
        if len(boxes):
            d = ray_obb(origins, dirs, boxes)
            j = d.argmin(axis=1)
            best = d[np.arange(len(d)), j]
            cls = np.where(j < len(sb), CLS_STATIC, CLS_VEHICLE)
        if len(pc):
            d = ray_circle(origins, dirs, pc).min(axis=1)
            upd = d < best
            best = np.where(upd, d, best)
            cls = np.where(upd, CLS_PED, cls)
        dist = best - self.body_off
        miss = dist > self.ranges
        dist = np.where(miss, self.ranges, np.maximum(dist, 0.0))
        cls = np.where(miss, CLS_NONE, cls)
        return dist, cls, origins, dirs

    def encode(self, dist, cls):
        """(R,4): closeness in [0,1] (1 = touching, 0 = nothing) + 3 class flags."""
        close = 1.0 - dist / self.ranges
        out = np.zeros((N_RAYS, 4), dtype=np.float32)
        out[:, 0] = close
        out[:, 1] = cls == CLS_STATIC
        out[:, 2] = cls == CLS_VEHICLE
        out[:, 3] = cls == CLS_PED
        return out.reshape(-1)
