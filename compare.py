#!/usr/bin/env python
"""
compare.py - side-by-side report of two experiments.

    python compare.py balanced cautious
    python compare.py experiments/balanced experiments/mine --n 50 --no-video

Produces experiments/compare/<a>_vs_<b>/:
    report.md          reward settings diff, evaluation metrics side by side,
                       plain-English summary of who does what better
    curves.png         training curves of both runs
    metrics.png        key evaluation metrics as paired bars
    side_by_side.mp4   both cars driving the SAME unseen city at the same time
                       (left = a, right = b; stereo sound: left ear = a, right = b)

If an experiment has no evaluation yet, it is evaluated first (100 unseen
cities by default, --n to change).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# metric key, label, format, higher_is_better
METRICS = [
    ("success_rate", "Parked successfully", "pct", True),
    ("clean_success_rate", "Parked with zero violations", "pct", True),
    ("collision_rate", "Collisions", "pct", False),
    ("pedestrian_hit_rate", "Pedestrian hits", "pct", False),
    ("off_route_rate", "Left the route", "pct", False),
    ("route_completion_mean", "Route completed (avg)", "pct", True),
    ("red_lights_per_km", "Red lights run / km", "f2", False),
    ("stop_signs_per_km", "Stop signs rolled / km", "f2", False),
    ("crosswalk_violations_per_km", "Crosswalk violations / km", "f2", False),
    ("speeding_time_frac", "Time spent speeding", "pct", False),
    ("mean_speed_kmh", "Average speed (km/h)", "f1", None),
    ("lane_offset_mean_m", "Distance from lane centre (m)", "f2", False),
    ("parking_error_mean", "Parking error (0 = perfect)", "f2", False),
    ("time_to_park_s_mean", "Time to park (s)", "f1", False),
]


def resolve(x):
    p = Path(x)
    if not p.exists():
        p = ROOT / "experiments" / x
    if not (p / "model.zip").exists():
        sys.exit(f"'{x}': no trained model found (looked in {p}).")
    return p


def fmt(v, kind):
    if v is None:
        return "-"
    return {"pct": f"{100 * v:.1f}%", "f2": f"{v:.2f}", "f1": f"{v:.1f}"}[kind]


def ensure_eval(exp, n):
    f = exp / "eval" / "summary.json"
    if f.exists():
        s = json.load(open(f))
        if s.get("episodes", 0) >= n:
            return s
    print(f"[compare] evaluating {exp.name} on {n} unseen cities ...")
    subprocess.run([sys.executable, str(ROOT / "evaluate.py"), str(exp), "--n", str(n),
                    "--failure-videos", "0"], check=True)
    return json.load(open(f))


def reward_diff(a, b):
    from autodrive.rewards import TERMS, load_reward_config
    ca = load_reward_config(a / "reward_config.yaml")
    cb = load_reward_config(b / "reward_config.yaml")
    rows = []
    for k in TERMS:
        ta, tb = ca["terms"][k], cb["terms"][k]
        va = f"{ta['weight']:g}" if ta["enabled"] and ta["weight"] > 0 else "off"
        vb = f"{tb['weight']:g}" if tb["enabled"] and tb["weight"] > 0 else "off"
        rows.append((k, va, vb, va != vb))
    srows = []
    for k in ca["settings"]:
        va, vb = ca["settings"][k], cb["settings"][k]
        if va != vb:
            srows.append((k, va, vb))
    return rows, srows


def verdict(sa, sb, na, nb):
    lines = []
    for key, label, kind, hib in METRICS:
        va, vb = sa.get(key), sb.get(key)
        if va is None or vb is None or hib is None:
            continue
        if abs(va - vb) < (0.03 if kind == "pct" else 0.05 * max(abs(va), abs(vb), 1e-6)):
            continue
        better = na if (va > vb) == hib else nb
        lines.append(f"- **{better}** is better at *{label.lower()}* ({fmt(va, kind)} vs {fmt(vb, kind)}).")
    if not lines:
        lines.append("- The two cars perform about the same on every metric.")
    # honesty about sample size
    lo_a, hi_a = sa["success_ci95"]
    lo_b, hi_b = sb["success_ci95"]
    if not (hi_a < lo_b or hi_b < lo_a):
        lines.append(f"- Note: the success-rate confidence intervals overlap "
                     f"({100*lo_a:.0f}-{100*hi_a:.0f}% vs {100*lo_b:.0f}-{100*hi_b:.0f}%), so that "
                     f"difference may be luck. Evaluate on more cities (--n) to be sure.")
    return lines


def plots(a, b, sa, sb, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from autodrive.plots import PALETTE, _fmt_steps, _style, read_csv, rolling
    _style(plt)
    # training curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for exp, col in ((a, PALETTE[0]), (b, PALETTE[1])):
        eps = read_csv(exp / "episodes.csv")
        steps = np.array([float(e["timesteps"]) for e in eps])
        succ = np.array([e["success"] == "True" for e in eps], float)
        ret = np.array([float(e["return"]) for e in eps])
        axes[0].plot(steps, rolling(succ, 200) * 100, color=col, label=exp.name)
        axes[1].plot(steps, rolling(ret, 200), color=col, label=exp.name)
    axes[0].set_title("Parked successfully during training (%, rolling 200)", pad=10)
    axes[0].set_ylim(0, 100)
    axes[1].set_title("Episode return during training (rolling 200)", pad=10)
    for ax in axes:
        _fmt_steps(ax)
        ax.set_xlabel("environment steps")
        ax.legend()
    fig.text(0.01, 0.01, "Each run's own reward scale - returns of different reward configs are not "
             "directly comparable; success rate is.", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "curves.png", dpi=110)
    plt.close(fig)
    # paired bars, small multiples (each metric on its own axis = no misleading shared scale)
    keys = [m for m in METRICS if sa.get(m[0]) is not None and sb.get(m[0]) is not None]
    cols_n = 4
    rows_n = int(np.ceil(len(keys) / cols_n))
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(14, 2.5 * rows_n), squeeze=False)
    for i, (key, label, kind, hib) in enumerate(keys):
        ax = axes[i // cols_n][i % cols_n]
        va, vb = sa[key], sb[key]
        scale = 100 if kind == "pct" else 1
        bars = ax.bar([0, 1], [va * scale, vb * scale], color=[PALETTE[0], PALETTE[1]], width=0.6)
        ax.set_xticks([0, 1], [a.name, b.name], fontsize=8)
        ax.set_title(label + ("" if hib is None else ("  (higher = better)" if hib else "  (lower = better)")),
                     fontsize=9)
        for bar, v in zip(bars, (va, vb)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt(v, kind), ha="center",
                    va="bottom", fontsize=8, color="#0b0b0b")
        ax.margins(y=0.25)
        ax.grid(axis="x", visible=False)
    for j in range(len(keys), rows_n * cols_n):
        axes[j // cols_n][j % cols_n].axis("off")
    fig.suptitle(f"Evaluation on unseen cities: {a.name} (blue) vs {b.name} (orange)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "metrics.png", dpi=105)
    plt.close(fig)


def side_by_side_video(a, b, out, seed, stage):
    """Both cars in the same unseen city; frames joined horizontally, stereo sound."""
    import os

    from autodrive.audio import SoundBank, mix_soundtrack
    from autodrive.env import AutoDriveEnv
    from autodrive.render import Renderer
    from autodrive.video import FPS, SUBFRAMES, FrameWriter, _audio_frame, load_policy, mux
    import wave
    bank = SoundBank()
    sides = []
    for exp in (a, b):
        pol, cfg = load_policy(exp)
        env = AutoDriveEnv(reward_config=cfg, stage=stage, replay_prob=0.0)
        obs, _ = env.reset(seed=seed, options={"stage": stage})
        env.renderer = Renderer()
        sides.append(dict(env=env, pol=pol, obs=obs, done=False, audio=[], name=exp.name))
    W, H = 960, 540
    writer = FrameWriter(out / "sbs.tmp.mp4", 2 * W, H)
    import pygame
    hold = 0
    while True:
        frames = [[], []]
        for i, sd in enumerate(sides):
            env, r = sd["env"], sd["env"].renderer
            if not sd["done"]:
                sd["obs"], _, te, tr, _ = env.step(sd["pol"](sd["obs"]))
                sd["done"] = te or tr
                events = list(env.events)
            else:
                events = []
            r.begin_step()
            for k in range(SUBFRAMES):
                f = r.draw(env, (k + 1) / SUBFRAMES, 1.0 / FPS)
                frames[i].append(f)
                if sd["done"] and not events:
                    sd["audio"].append(dict(speed=0.0, throttle=0.0, brake=0.0, blink=False, events=[]))
                else:
                    sd["audio"].append(_audio_frame(env, events if k == 0 else []))
        for k in range(SUBFRAMES):
            tiles = []
            for i, sd in enumerate(sides):
                surf = pygame.surfarray.make_surface(frames[i][k].swapaxes(0, 1))
                surf = pygame.transform.smoothscale(surf, (W, H))
                lab = sd["env"].renderer.fonts["medb"].render(f"{'AB'[i]}: {sd['name']}", True, (255, 255, 255))
                bg = pygame.Surface((lab.get_width() + 20, lab.get_height() + 10), pygame.SRCALPHA)
                bg.fill((0, 0, 0, 170))
                surf.blit(bg, (W // 2 - bg.get_width() // 2, H - 44))
                surf.blit(lab, (W // 2 - lab.get_width() // 2, H - 39))
                tiles.append(pygame.surfarray.array3d(surf).swapaxes(0, 1))
            joined = np.concatenate(tiles, axis=1)
            joined[:, W - 2:W + 2] = 20
            writer.write(joined)
        if all(sd["done"] for sd in sides):
            hold += 1
            if hold > 15:
                break
        if len(sides[0]["audio"]) > 20 * 150:
            break
    writer.close()
    la = mix_soundtrack(sides[0]["audio"], FPS, bank)
    lb = mix_soundtrack(sides[1]["audio"], FPS, bank)
    n = max(len(la), len(lb))
    st = np.zeros((n, 2), np.float32)
    st[:len(la), 0] = la
    st[:len(lb), 1] = lb
    pcm = (np.clip(st, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(out / "sbs.tmp.wav"), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(pcm.tobytes())
    mux(out / "sbs.tmp.mp4", out / "sbs.tmp.wav", out / "side_by_side.mp4")
    os.remove(out / "sbs.tmp.mp4")
    os.remove(out / "sbs.tmp.wav")
    return [sd["env"].status for sd in sides]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n", type=int, default=100, help="unseen cities to evaluate on (if not done yet)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--video-seed", type=int, default=50_000_007)
    args = ap.parse_args()
    a, b = resolve(args.a), resolve(args.b)
    out = ROOT / "experiments" / "compare" / f"{a.name}_vs_{b.name}"
    out.mkdir(parents=True, exist_ok=True)
    sa, sb = ensure_eval(a, args.n), ensure_eval(b, args.n)
    rows, srows = reward_diff(a, b)
    plots(a, b, sa, sb, out)
    vid_line = ""
    if not args.no_video:
        print("[compare] rendering side-by-side video ...")
        sts = side_by_side_video(a, b, out, args.video_seed, sa.get("stage", 4))
        vid_line = (f"\n## Side-by-side drive\n\n`side_by_side.mp4`: both cars in the same unseen city "
                    f"(seed {args.video_seed}). Left = **{a.name}** ({sts[0]}), right = **{b.name}** "
                    f"({sts[1]}). Stereo: left ear = {a.name}, right ear = {b.name}.\n")

    def curj(exp):
        f = exp / "curriculum.json"
        return json.load(open(f)) if f.exists() else {}
    ja, jb = curj(a), curj(b)
    md = [f"# {a.name} vs {b.name}", "",
          "## Summary", ""] + verdict(sa, sb, a.name, b.name) + [
          "", "## Reward settings", "",
          f"| Term | {a.name} | {b.name} |", "|---|---|---|"]
    md += [f"| {'**' + k + '**' if d else k} | {va} | {vb} |" for k, va, vb, d in rows]
    if srows:
        md += ["", f"| Setting | {a.name} | {b.name} |", "|---|---|---|"]
        md += [f"| {k} | {va} | {vb} |" for k, va, vb in srows]
    md += ["", "Rows in **bold** differ.", "",
           "## Training", "",
           f"| | {a.name} | {b.name} |", "|---|---|---|",
           f"| Steps trained | {ja.get('timesteps', '?'):,} | {jb.get('timesteps', '?'):,} |"
           if isinstance(ja.get('timesteps'), int) and isinstance(jb.get('timesteps'), int)
           else f"| Steps trained | {ja.get('timesteps', '?')} | {jb.get('timesteps', '?')} |",
           f"| Final curriculum stage | {ja.get('stage_name', '?')} | {jb.get('stage_name', '?')} |",
           f"| Wall-clock minutes | {ja.get('wall_minutes', '?')} | {jb.get('wall_minutes', '?')} |",
           "", "![training curves](curves.png)", "",
           f"## Evaluation ({sa['episodes']} / {sb['episodes']} unseen cities, {sa.get('stage_name', '')})", "",
           f"| Metric | {a.name} | {b.name} |", "|---|---|---|"]
    for key, label, kind, hib in METRICS:
        va, vb = sa.get(key), sb.get(key)
        mark_a = mark_b = ""
        if va is not None and vb is not None and hib is not None and va != vb:
            if (va > vb) == hib:
                mark_a = " ✔"
            else:
                mark_b = " ✔"
        md.append(f"| {label} | {fmt(va, kind)}{mark_a} | {fmt(vb, kind)}{mark_b} |")
    md += ["", "✔ = better of the two.", "", "![metrics](metrics.png)", vid_line]
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[compare] report written to {out / 'report.md'}")


if __name__ == "__main__":
    main()
