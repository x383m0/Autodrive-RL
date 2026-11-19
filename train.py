#!/usr/bin/env python
"""
train.py - train a self-driving policy with PPO and an automatic curriculum.

Examples
--------
    python train.py --reward rewards.yaml --name my_first_car
    python train.py --reward rewards/cautious.yaml --name careful --minutes 20
    python train.py --reward rewards.yaml --name smoke --quick        # 2-minute smoke test
    python train.py --name my_first_car --resume                      # continue a run

Everything for an experiment goes to experiments/<name>/:
    config.yaml        the exact reward config + training arguments used
    model.zip          final policy (Stable-Baselines3 PPO)
    vecnormalize.pkl   observation/reward normalisation statistics (needed to run the model!)
    episodes.csv       one row per finished episode (outcome, violations, per-term rewards)
    progress.csv       one row per PPO update (timesteps, stage, success rate, ...)
    curriculum.json    when each curriculum stage was reached
    plots/             training curves
    videos/            demo mp4s (with sound), only with --video
    checkpoints/       periodic checkpoints for --resume (git-ignored)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
MIN_STD = 0.2           # action noise restored when entering a stage with new actors
sys.path.insert(0, str(ROOT))


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reward", default="rewards.yaml", help="reward config YAML")
    ap.add_argument("--name", default=None, help="experiment name (folder under experiments/)")
    ap.add_argument("--minutes", type=float, default=45.0, help="wall-clock training budget")
    ap.add_argument("--steps", type=int, default=20_000_000, help="max environment steps")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="2-minute smoke test, tiny eval")
    ap.add_argument("--resume", action="store_true", help="continue from experiments/<name>/checkpoints")
    ap.add_argument("--start-stage", type=int, default=0)
    ap.add_argument("--video", action="store_true",
                    help="record demo mp4s after training (off by default; use watch.py to view live)")
    ap.add_argument("--no-video", action="store_true", help=argparse.SUPPRESS)   # old flag, now the default
    ap.add_argument("--video-episodes", type=int, default=2)
    return ap.parse_args()


def make_env_fn(reward_cfg, stage, rank, seed):
    def _f():
        from autodrive.env import AutoDriveEnv
        env = AutoDriveEnv(reward_config=reward_cfg, stage=stage, seed_range=(0, 10_000_000))
        env.reset(seed=seed * 1000 + rank)
        return env
    return _f


def main():
    args = parse()
    import torch
    torch.set_num_threads(2)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from autodrive.curriculum import STAGES, CurriculumTracker
    from autodrive.rewards import TERMS, RewardConfigError, describe_config, load_reward_config

    if args.quick:
        args.minutes = min(args.minutes, 2.0)
        args.video_episodes = 1
    name = args.name or Path(args.reward).stem
    out = ROOT / "experiments" / name
    if not args.resume:
        # check the reward file BEFORE creating anything, and name the user's file
        try:
            load_reward_config(args.reward)
        except RewardConfigError as e:
            msg = str(e)
            if "No reward terms are switched on" in msg:
                msg += ("\n\nTip: don't train the template itself. Copy it and switch terms on:\n"
                        "    cp rewards/template.yaml rewards/mine.yaml\n"
                        "    (edit rewards/mine.yaml: enabled: false -> true, weight: 0 -> a number)\n"
                        "    python train.py --reward rewards/mine.yaml --name mine")
            print(f"\n[reward config problem] {msg}\n")
            sys.exit(1)
    ckpt_dir = out / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    # ---- reward config (on resume, reuse the saved one) -------------------------
    if args.resume and (out / "config.yaml").exists():
        saved = yaml.safe_load(open(out / "config.yaml"))
        reward_path = out / "reward_config.yaml"
        print(f"[resume] using saved reward config {reward_path}")
    else:
        reward_path = out / "reward_config.yaml"
        shutil.copy(args.reward, reward_path)
    try:
        reward_cfg = load_reward_config(reward_path)
    except RewardConfigError as e:
        print(f"\n[reward config problem] {e}\n")
        sys.exit(1)
    print(f"[train] rewards switched on in '{reward_cfg['name']}':\n{describe_config(reward_cfg)}")
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump({"reward_file": str(args.reward), "reward": reward_cfg,
                        "train_args": vars(args)}, f, sort_keys=False)

    # ---- curriculum state ----------------------------------------------------
    tracker = CurriculumTracker(start_stage=args.start_stage)
    t_offset = 0
    cur_file = out / "curriculum.json"
    if args.resume and cur_file.exists():
        cj = json.load(open(cur_file))
        tracker.stage = cj["stage"]
        tracker.history = [tuple(h) for h in cj["history"]]
        t_offset = cj.get("timesteps", 0)
        print(f"[resume] curriculum stage {tracker.stage} ({STAGES[tracker.stage]['name']}), "
              f"{t_offset} steps done")

    venv = SubprocVecEnv([make_env_fn(reward_cfg, tracker.stage, i, args.seed + int(time.time()) % 1000 * args.resume)
                          for i in range(args.n_envs)])
    if args.resume and (ckpt_dir / "vecnormalize.pkl").exists():
        venv = VecNormalize.load(str(ckpt_dir / "vecnormalize.pkl"), venv)
        venv.training = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    if args.resume and (ckpt_dir / "model.zip").exists():
        model = PPO.load(str(ckpt_dir / "model.zip"), env=venv, device="cpu")
        print("[resume] loaded checkpoint model")
    else:
        model = PPO(
            "MlpPolicy", venv, device="cpu", seed=args.seed,
            n_steps=512, batch_size=512, n_epochs=6, learning_rate=3e-4,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.002,
            vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0),
            verbose=0)

    # ---- logging callback -------------------------------------------------------
    ep_fields = ["timesteps", "wall_s", "stage", "success", "status", "return", "steps", "route_len",
                 "progress_frac", "mean_speed", "red_violations", "stop_violations",
                 "crosswalk_violations", "speeding_frac", "offroad_frac", "stall_frac",
                 "lane_dev_mean", "turns_per_100m", "park_err", "city_seed"] + [f"r_{k}" for k in TERMS]
    ep_mode = "a" if args.resume and (out / "episodes.csv").exists() else "w"
    ep_file = open(out / "episodes.csv", ep_mode, newline="")
    ep_writer = csv.DictWriter(ep_file, fieldnames=ep_fields)
    if ep_mode == "w":
        ep_writer.writeheader()
    pr_fields = ["timesteps", "wall_s", "stage", "stage_success_rate", "success_rate_100",
                 "mean_return_100", "collision_rate_100", "ped_hit_rate_100", "fps"]
    pr_mode = "a" if args.resume and (out / "progress.csv").exists() else "w"
    pr_file = open(out / "progress.csv", pr_mode, newline="")
    pr_writer = csv.DictWriter(pr_file, fieldnames=pr_fields)
    if pr_mode == "w":
        pr_writer.writeheader()

    t_start = time.time()
    budget = args.minutes * 60.0

    class Cb(BaseCallback):
        def __init__(self):
            super().__init__()
            self.recent = deque(maxlen=100)
            self.last_ckpt = 0
            self.last_print = 0

        def _on_step(self):
            ts = self.num_timesteps          # restored from the checkpoint on --resume
            for info in self.locals["infos"]:
                st = info.get("episode_stats")
                if st is None:
                    continue
                tracker.record(st["success"], st["stage"])
                self.recent.append(st)
                row = {k: st.get(k) for k in ep_fields if k in st}
                row.update({"timesteps": ts, "wall_s": round(time.time() - t_start, 1)})
                for k, v in st["terms"].items():
                    row[f"r_{k}"] = round(v, 3)
                for k in ("return", "progress_frac", "mean_speed", "speeding_frac", "offroad_frac",
                          "stall_frac", "lane_dev_mean", "turns_per_100m"):
                    row[k] = round(float(row[k]), 4)
                ep_writer.writerow(row)
            if tracker.should_advance():
                # snapshot the model that just PASSED this stage (small files),
                # so a later collapse never loses a good model
                snap = out / "stages" / STAGES[tracker.stage]["name"].split()[0]
                snap.mkdir(parents=True, exist_ok=True)
                self.model.save(str(snap / "model.zip"))
                self.training_env.save(str(snap / "vecnormalize.pkl"))
                k = tracker.advance(ts)
                self.training_env.env_method("set_stage", k)
                # Re-inject exploration: on easy stages PPO's action noise
                # shrinks towards zero (we measured 0.05), and a nearly
                # deterministic policy cannot discover the new skills a harder
                # stage needs (it got WORSE in the traffic stage). So each new
                # stage with NEW ACTORS (traffic, pedestrians) starts with at least
                # MIN_STD of exploration noise. Not for the parking/rules stages:
                # precise parking needs LOW noise, and those pass quickly anyway.
                if STAGES[k]["n_cars"] > STAGES[k - 1]["n_cars"] or STAGES[k]["n_peds"] > STAGES[k - 1]["n_peds"] \
                        or k == len(STAGES) - 1:
                    with torch.no_grad():
                        self.model.policy.log_std.clamp_(min=float(np.log(MIN_STD)))
                    print(f"[train] exploration noise raised to >= {MIN_STD} for the new stage", flush=True)
                print(f"\n=== CURRICULUM: advancing to stage {k}: {STAGES[k]['name']} at {ts} steps ===\n",
                      flush=True)
            if ts - self.last_ckpt >= 250_000:
                self.last_ckpt = ts
                self.save_ckpt(ts)
            if time.time() - t_start > budget:
                print("[train] time budget reached")
                return False
            return True

        def save_ckpt(self, ts):
            self.model.save(str(ckpt_dir / "model.zip"))
            self.training_env.save(str(ckpt_dir / "vecnormalize.pkl"))
            json.dump({"stage": tracker.stage, "history": tracker.history, "timesteps": ts},
                      open(cur_file, "w"), indent=1)

        def _on_rollout_end(self):
            ts = self.num_timesteps
            r = list(self.recent)
            wall = time.time() - t_start
            row = {"timesteps": ts, "wall_s": round(wall, 1), "stage": tracker.stage,
                   "stage_success_rate": round(tracker.rate, 3),
                   "success_rate_100": round(np.mean([e["success"] for e in r]), 3) if r else 0,
                   "mean_return_100": round(np.mean([e["return"] for e in r]), 2) if r else 0,
                   "collision_rate_100": round(np.mean([e["collision"] for e in r]), 3) if r else 0,
                   "ped_hit_rate_100": round(np.mean([e["ped_hit"] for e in r]), 3) if r else 0,
                   "fps": int((self.num_timesteps - start_steps) / max(wall, 1e-6))}
            pr_writer.writerow(row)
            pr_file.flush()
            ep_file.flush()
            if wall - self.last_print > 30:
                self.last_print = wall
                print(f"[{wall/60:5.1f} min] steps={ts:>9,} stage={tracker.stage} "
                      f"stage_succ={tracker.rate:.2f} succ100={row['success_rate_100']:.2f} "
                      f"ret={row['mean_return_100']:7.2f} coll={row['collision_rate_100']:.2f} "
                      f"ped={row['ped_hit_rate_100']:.2f} fps={row['fps']}", flush=True)

    start_steps = model.num_timesteps        # > 0 when resuming
    cb = Cb()
    print(f"[train] experiment '{name}' reward='{reward_cfg['name']}' envs={args.n_envs} "
          f"budget={args.minutes} min -> {out}")
    model.learn(total_timesteps=args.steps, callback=cb, reset_num_timesteps=not args.resume,
                progress_bar=False)
    total = model.num_timesteps
    cb.save_ckpt(total)
    model.save(str(out / "model.zip"))
    venv.save(str(out / "vecnormalize.pkl"))
    ep_file.close()
    pr_file.close()
    json.dump({"stage": tracker.stage, "stage_name": STAGES[tracker.stage]["name"],
               "history": tracker.history, "timesteps": int(total),
               "wall_minutes": round((time.time() - t_start) / 60, 2)},
              open(cur_file, "w"), indent=1)
    venv.close()
    print(f"[train] done: {total:,} steps, final stage {tracker.stage} "
          f"({STAGES[tracker.stage]['name']}), {(time.time()-t_start)/60:.1f} min")

    # ---- plots + demo videos ------------------------------------------------------
    try:
        from autodrive.plots import training_plots
        training_plots(out)
    except Exception as e:  # plotting must never kill a finished training run
        print("[train] plotting failed:", e)
    if args.video:
        try:
            from autodrive.video import record_episodes
            record_episodes(out, n=args.video_episodes, stage=tracker.stage, prefix="demo")
        except Exception as e:
            print("[train] video export failed:", e)


if __name__ == "__main__":
    main()
