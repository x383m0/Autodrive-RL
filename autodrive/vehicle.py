"""
vehicle.py - the agent car's physics: a kinematic bicycle model.

The bicycle model collapses the 4 wheels into one front and one rear wheel
on the car's centre line. It ignores tyre slip (fine at city speeds) but
captures the important things: a car cannot move sideways, it turns
around a point on the rear axle line, and the tighter you steer the
smaller the circle.

            lf        lr
   front o-----x-----------o rear       x = centre of mass (our reference point)
         ^ steering angle delta

   beta     = atan( lr / (lf+lr) * tan(delta) )    slip angle of the CoM velocity
   x'       = v * cos(theta + beta)
   y'       = v * sin(theta + beta)
   theta'   = v / lr * sin(beta)
   v'       = a

Controls coming from the policy are NORMALISED to [-1, 1]:
   steer_cmd  ->  target steering angle = sign(c) * c^2 * MAX_STEER
                  (squared: fine control for small lane corrections, full lock
                  still available; the wheel moves at a limited rate)
   speed_cmd  ->  TARGET SPEED = ((speed_cmd + 1) / 2)^2 * CMD_MAX_SPEED
                  -1 = "stop", 0 = 4 m/s, 0.5 = 9 m/s, +1 = 16 m/s (58 km/h).
                  Squared so the slow speeds needed for parking get half of the
                  command range (precise creeping), not a tiny sliver of it.
                  A cruise controller turns it into acceleration:
                      a = clip(K_SPEED * (target - v), -MAX_BRAKE, MAX_ACCEL)
                  (no reverse gear: v >= 0)

Why a target speed and not a raw pedal? This is how real self-driving stacks
split the work (the planner picks a speed, a low-level controller tracks it),
and it matters a lot for learning: with a raw pedal, the random exploration
noise of a fresh policy averages out to BRAKING (the brake is stronger than
the engine), so the car mostly stands still and learns nothing. With a speed
target, a fresh policy rolls along at a moderate speed and keeping a steady
speed is simply a steady action.
"""
from __future__ import annotations

import numpy as np

from .geometry import wrap_angle

LENGTH = 4.6          # metres
WIDTH = 1.9
WHEELBASE = 2.7
LF = 1.2              # CoM -> front axle
LR = WHEELBASE - LF   # CoM -> rear axle
MAX_STEER = np.deg2rad(38)
STEER_RATE = np.deg2rad(120)      # rad/s the steering wheel can turn
MAX_ACCEL = 3.5                   # m/s^2
MAX_BRAKE = 8.0                   # m/s^2 (emergency stop)
MAX_SPEED = 20.0                  # m/s (72 km/h)
CMD_MAX_SPEED = 16.0              # m/s: speed_cmd = +1 asks for this
K_SPEED = 2.0                     # cruise controller gain (1/s)
DRAG = 0.02                       # gentle rolling resistance when coasting


class BicycleCar:
    def __init__(self, x, y, theta, v=0.0):
        self.x, self.y, self.theta, self.v = float(x), float(y), float(theta), float(v)
        self.delta = 0.0           # actual steering angle
        self.a = 0.0               # actual longitudinal acceleration (for HUD / comfort)
        self.prev_a = 0.0
        self.yaw_rate = 0.0
        self.lat_acc = 0.0
        self.braking = False
        self.target_speed = v
        self.odometer = 0.0

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def box(self):
        """OBB row [cx, cy, half_len, half_wid, angle] for collision/rays."""
        return np.array([self.x, self.y, LENGTH / 2, WIDTH / 2, self.theta])

    def step(self, steer_cmd: float, speed_cmd: float, dt: float, substeps: int = 2):
        steer_cmd = float(np.clip(steer_cmd, -1, 1))
        speed_cmd = float(np.clip(speed_cmd, -1, 1))
        target = np.sign(steer_cmd) * steer_cmd ** 2 * MAX_STEER
        self.target_speed = ((speed_cmd + 1.0) / 2.0) ** 2 * CMD_MAX_SPEED
        acc = K_SPEED * (self.target_speed - self.v)
        if speed_cmd <= -0.98:
            acc = -MAX_BRAKE                      # "stop now": full brakes
        acc = float(np.clip(acc, -MAX_BRAKE, MAX_ACCEL * (1.0 - self.v / (MAX_SPEED * 1.05))))
        self.braking = acc < -1.0
        acc -= DRAG * self.v
        self.prev_a = self.a
        h = dt / substeps
        theta0 = self.theta
        for _ in range(substeps):
            # steering actuator with rate limit
            dd = np.clip(target - self.delta, -STEER_RATE * h, STEER_RATE * h)
            self.delta += dd
            beta = np.arctan(LR / WHEELBASE * np.tan(self.delta))
            v_new = max(0.0, min(MAX_SPEED, self.v + acc * h))
            v_mid = 0.5 * (self.v + v_new)
            self.x += v_mid * np.cos(self.theta + beta) * h
            self.y += v_mid * np.sin(self.theta + beta) * h
            self.theta = float(wrap_angle(self.theta + v_mid / LR * np.sin(beta) * h))
            self.odometer += v_mid * h
            self.v = v_new
        self.a = acc if self.v > 0 or acc > 0 else 0.0
        self.yaw_rate = float(wrap_angle(self.theta - theta0)) / dt
        self.lat_acc = self.v * self.yaw_rate
