"""A hand-written (privileged) driver used only to sanity-check that tasks are solvable.

Pure pursuit on the GPS route + speed control that respects limits, lights,
stop signs and stops in the parking slot. It reads ground truth, so it is NOT
a fair baseline for the RL agent - it is a test fixture.
"""
import numpy as np
from autodrive.geometry import to_local, wrap_angle
from autodrive.vehicle import WHEELBASE, MAX_STEER, LENGTH


def expert_action(env):
    car, route = env.car, env.route
    look = max(4.0, car.v * 0.9)
    tgt = route.pos(env.s + look)
    loc = to_local(tgt[None], car.pos, car.theta)[0]
    curv = 2 * loc[1] / max(loc @ loc, 1e-6)
    steer = np.clip(np.arctan(WHEELBASE * curv) / MAX_STEER, -1, 1)
    # target speed
    v_t = route.limit(env.s + 3) * 0.95
    # curvature ahead
    for o in (5, 10, 15):
        dh = abs(wrap_angle(route.heading(env.s + o) - car.theta))
        if dh > 0.3:
            v_t = min(v_t, 5.0)
    togo = route.length - env.s
    v_t = min(v_t, np.sqrt(2 * 1.5 * max(togo - 0.2, 0)))
    s_front = env.s + LENGTH / 2
    i, ev = env._next_event(s_front)
    if ev is not None:
        d = ev["s"] - s_front - 0.5
        stop = False
        if ev["kind"] == "light":
            stt = env.city.light_state(ev["node"], ev["axis"], env.t)
            stop = stt == "R" or (stt == "Y" and d > car.v ** 2 / 6)
        elif ev["kind"] == "stop":
            stop = env.stop_min_speed.get(i, 99) > 0.3
        if stop and d > -0.5:
            v_t = min(v_t, np.sqrt(2 * 2.5 * max(d, 0)))
    # obstacles straight ahead from the front cameras
    fr = env.sensors.cam_id <= 1
    ang = np.abs(env.sensors.local_ang) < 0.25
    m = fr & ang & (env.ray_cls >= 2)
    if m.any():
        d = env.ray_dist[m].min() - 3.0
        v_t = min(v_t, np.sqrt(2 * 3.0 * max(d, 0)))
    # the action is a target speed: -1 = stop ... +1 = 16 m/s
    cmd = np.clip(v_t / 16.0 * 2 - 1, -1, 1)
    if v_t < 0.1:
        cmd = -1.0
    return np.array([steer, cmd])
