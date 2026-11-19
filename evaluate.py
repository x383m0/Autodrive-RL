#!/usr/bin/env python
"""
evaluate.py - test a trained car on cities it has NEVER seen.

    python evaluate.py experiments/balanced
    python evaluate.py experiments/balanced --n 100 --failure-videos 3   # also save 3 failure mp4s

Training draws city seeds from [0, 10,000,000). Evaluation uses seeds starting
at 50,000,000, so every evaluation city is new. All cities use the hardest
curriculum stage ("S4 full city": traffic, pedestrians, kids behind parked
cars, school zones, erratic drivers). The policy acts deterministically
(no exploration noise).

Writes experiments/<name>/eval/:
    episodes.csv     one row per city
    summary.json     all metrics (machine readable, used by compare.py)
    summary.md       the same, human readable
    failures/*.mp4   videos of failed episodes, only with --failure-videos N
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
EVAL_SEED0 = 50_000_000

_policy = None
_env = None


def _init(exp_dir, stage):
    global _policy, _env
    import torch
    torch.set_num_threads(1)
    from autodrive.env import AutoDriveEnv
    from autodrive.video import load_policy
    _policy, cfg = load_policy(exp_dir)
    _env = AutoDriveEnv(reward_config=cfg, stage=stage, replay_prob=0.0)


def _run(seed):
    obs, _ = _env.reset(seed=seed, options={"stage": _env.stage})
    while True:
        obs, r, term, trunc, info = _env.step(_policy(obs))
        if term or trunc:
            st = info["episode_stats"]
            st["seed"] = seed
            st["km"] = _env.car.odometer / 1000.0
            return st


def wilson(k, n, z=1.96):
    """95% confidence interval for a success rate (Wilson score interval)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(eps):
    n = len(eps)
    km = sum(e["km"] for e in eps) or 1e-9
    succ = sum(e["success"] for e in eps)
    status = Counter(e["status"] for e in eps)
    clean = sum(e["success"] and e["red_violations"] + e["stop_violations"] + e["crosswalk_violations"] == 0
                for e in eps)
    parked = [e for e in eps if e["success"]]
    lo, hi = wilson(succ, n)
    return {
        "episodes": n,
        "success_rate": succ / n,
        "success_ci95": [lo, hi],
        "clean_success_rate": clean / n,
        "collision_rate": status["collision"] / n,
        "pedestrian_hit_rate": status["pedestrian_hit"] / n,
        "off_route_rate": status["off_route"] / n,
        "stuck_rate": status["stuck"] / n,
        "timeout_rate": status["timeout"] / n,
        "km_driven": km,
        "red_lights_per_km": sum(e["red_violations"] for e in eps) / km,
        "stop_signs_per_km": sum(e["stop_violations"] for e in eps) / km,
        "crosswalk_violations_per_km": sum(e["crosswalk_violations"] for e in eps) / km,
        "speeding_time_frac": float(np.mean([e["speeding_frac"] for e in eps])),
        "school_zone_max_over_kmh": float(np.max([e["school_over_limit"] for e in eps]) * 3.6),
        "mean_speed_kmh": float(np.mean([e["mean_speed"] for e in eps]) * 3.6),
        "lane_offset_mean_m": float(np.mean([e["lane_dev_mean"] for e in eps])),
        "offroad_time_frac": float(np.mean([e["offroad_frac"] for e in eps])),
        "route_completion_mean": float(np.mean([e["progress_frac"] for e in eps])),
        "parking_error_mean": float(np.mean([e["park_err"] for e in parked])) if parked else None,
        "time_to_park_s_mean": float(np.mean([e["time_s"] for e in parked])) if parked else None,
        "mean_return": float(np.mean([e["return"] for e in eps])),
        "outcomes": dict(status),
    }


def to_markdown(name, s, stage_name):
    lo, hi = s["success_ci95"]
    pct = lambda v: f"{100 * v:.1f}%"
    rows = [
        ("Parked successfully", f"{pct(s['success_rate'])}  (95% CI {pct(lo)}-{pct(hi)})"),
        ("Parked with zero rule violations", pct(s["clean_success_rate"])),
        ("Collision (car/building)", pct(s["collision_rate"])),
        ("Pedestrian hit", pct(s["pedestrian_hit_rate"])),
        ("Left the route", pct(s["off_route_rate"])),
        ("Got stuck / timed out", f"{pct(s['stuck_rate'])} / {pct(s['timeout_rate'])}"),
        ("Average route completed", pct(s["route_completion_mean"])),
        ("Red lights run per km", f"{s['red_lights_per_km']:.2f}"),
        ("Stop signs rolled per km", f"{s['stop_signs_per_km']:.2f}"),
        ("Crosswalk violations per km", f"{s['crosswalk_violations_per_km']:.2f}"),
        ("Time spent speeding", pct(s["speeding_time_frac"])),
        ("Worst school-zone overspeed", f"{s['school_zone_max_over_kmh']:.1f} km/h"),
        ("Average speed", f"{s['mean_speed_kmh']:.1f} km/h"),
        ("Average distance from lane centre", f"{s['lane_offset_mean_m']:.2f} m"),
        ("Time with wheels off the road", pct(s["offroad_time_frac"])),
        ("Parking error (0 = perfect, 1 = at tolerance)",
         f"{s['parking_error_mean']:.2f}" if s["parking_error_mean"] is not None else "-"),
        ("Time to park (successful trips)",
         f"{s['time_to_park_s_mean']:.1f} s" if s["time_to_park_s_mean"] is not None else "-"),
        ("Total distance driven", f"{s['km_driven']:.1f} km"),
    ]
    out = [f"# Evaluation: {name}", "",
           f"{s['episodes']} unseen random cities, stage **{stage_name}**, deterministic policy.", "",
           "| Metric | Value |", "|---|---|"]
    out += [f"| {a} | {b} |" for a, b in rows]
    out += ["", "Outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(s["outcomes"].items()))]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", help="experiment folder, e.g. experiments/balanced")
    ap.add_argument("--n", type=int, default=100, help="number of unseen cities")
    ap.add_argument("--workers", type=int, default=min(4, mp.cpu_count()))
    ap.add_argument("--stage", type=int, default=None, help="curriculum stage (default: hardest)")
    ap.add_argument("--failure-videos", type=int, default=0,
                    help="save mp4s of this many failed episodes (default 0 = no files)")
    args = ap.parse_args()

    from autodrive.curriculum import EVAL_STAGE, STAGES
    exp = Path(args.exp)
    if not (exp / "model.zip").exists():
        sys.exit(f"No model.zip in {exp} - train first (python train.py --name {exp.name}).")
    stage = EVAL_STAGE if args.stage is None else args.stage
    out = exp / "eval"
    out.mkdir(exist_ok=True)
    seeds = [EVAL_SEED0 + i for i in range(args.n)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(str(exp), stage)) as pool:
        eps = []
        for i, st in enumerate(pool.imap(_run, seeds)):
            eps.append(st)
            if (i + 1) % 10 == 0:
                print(f"[eval] {i + 1}/{args.n} cities, success so far "
                      f"{np.mean([e['success'] for e in eps]):.0%}", flush=True)
    s = summarize(eps)
    s["stage"] = stage
    s["stage_name"] = STAGES[stage]["name"]
    s["eval_seconds"] = round(time.time() - t0, 1)
    fields = ["seed", "status", "success", "return", "time_s", "route_len", "progress_frac", "km",
              "mean_speed", "red_violations", "stop_violations", "crosswalk_violations", "speeding_frac",
              "school_over_limit", "lane_dev_mean", "offroad_frac", "park_err"]
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in eps:
            w.writerow({k: (round(e[k], 4) if isinstance(e[k], float) else e[k]) for k in fields})
    json.dump(s, open(out / "summary.json", "w"), indent=1)
    md = to_markdown(exp.name, s, s["stage_name"])
    (out / "summary.md").write_text(md)
    print(md)

    # ---- failure videos: one per failure type first, then any ---------------------
    if args.failure_videos > 0:
        fails = [e for e in eps if not e["success"]]
        picked, seen = [], set()
        for e in fails:
            if e["status"] not in seen:
                picked.append(e)
                seen.add(e["status"])
        for e in fails:
            if len(picked) >= args.failure_videos:
                break
            if e not in picked:
                picked.append(e)
        picked = picked[: args.failure_videos]
        if picked:
            from autodrive.audio import SoundBank
            from autodrive.env import AutoDriveEnv
            from autodrive.video import load_policy, record_episode
            policy, cfg = load_policy(exp)
            env = AutoDriveEnv(reward_config=cfg, stage=stage, replay_prob=0.0)
            bank = SoundBank()
            (out / "failures").mkdir(exist_ok=True)
            for e in picked:
                p = out / "failures" / f"{e['status']}_seed{e['seed']}.mp4"
                st = record_episode(env, policy, p, seed=e["seed"], options={"stage": stage}, bank=bank)
                same = "reproduced" if st["status"] == e["status"] else f"re-run ended as {st['status']}"
                print(f"[eval] failure video {p.name} ({same})")


if __name__ == "__main__":
    main()
