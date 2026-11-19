"""
video.py - render episodes to mp4 (with sound).

    load_policy(exp_dir)        -> a function obs -> action using the saved model
    record_episode(env, policy, path, ...)  -> run one episode, write an mp4
    record_episodes(exp_dir, n) -> demo videos into experiments/<name>/videos/

The simulation runs at 10 Hz; we draw 2 frames per step with interpolated
positions, so videos play smoothly at 20 fps. While recording we log what
the sound system needs per frame, mix the soundtrack (audio.py) and mux it
into the mp4 with ffmpeg.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .audio import SR, SoundBank, mix_soundtrack, write_wav

SUBFRAMES = 3                  # frames drawn per 0.1 s sim step -> 30 fps videos
FPS = 10 * SUBFRAMES


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def load_policy(exp_dir, deterministic=True):
    """Return (policy_fn, reward_cfg) for a trained experiment folder."""
    from stable_baselines3 import PPO
    import pickle

    exp_dir = Path(exp_dir)
    model = PPO.load(str(exp_dir / "model.zip"), device="cpu")
    with open(exp_dir / "vecnormalize.pkl", "rb") as f:
        vn = pickle.load(f)
    mean, var, eps, clip = vn.obs_rms.mean, vn.obs_rms.var, vn.epsilon, vn.clip_obs

    def policy(obs):
        o = np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
        a, _ = model.predict(o, deterministic=deterministic)
        return a

    from .rewards import load_reward_config
    cfg_path = exp_dir / "reward_config.yaml"
    reward_cfg = load_reward_config(cfg_path) if cfg_path.exists() else None
    return policy, reward_cfg


class FrameWriter:
    """Streams RGB frames into ffmpeg (no huge in-memory frame lists)."""

    def __init__(self, path, w, h, fps=FPS):
        self.path = str(path)
        self.proc = subprocess.Popen(
            [ffmpeg_exe(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "26", "-pix_fmt", "yuv420p", self.path],
            stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def mux(video, wav, out):
    subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-i", str(video), "-i", str(wav),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", str(out)], check=True)


def _audio_frame(env, events):
    car = env.car
    a = env.last_action
    blink = env.blinker() is not None and (env.renderer.blink_t % 0.7) < 0.35
    evs = []
    for e in events:
        d = float(np.linalg.norm(np.asarray(e[1]) - car.pos)) if len(e) > 1 else 0.0
        evs.append((e[0], d))
    return dict(speed=car.v, throttle=float(np.clip(car.a / 3.5, 0, 1)), brake=float(np.clip(-car.a / 8.0, 0, 1)),
                blink=bool(blink), events=evs)


def record_episode(env, policy, path, seed=None, options=None, max_steps=None, hold_end=1.5,
                   bank=None, frame_hook=None):
    """Run one episode with rendering. Returns the episode_stats dict."""
    from .render import Renderer

    obs, _ = env.reset(seed=seed, options=options)
    if env.renderer is None:
        env.renderer = Renderer()
    r = env.renderer
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_v = path.with_suffix(".tmp.mp4")
    tmp_a = path.with_suffix(".tmp.wav")
    writer = FrameWriter(tmp_v, r.W, r.H)
    audio_frames = []
    stats = None
    steps = 0
    r.begin_step()
    frame = r.draw(env, 1.0, 1.0 / FPS)
    while True:
        action = policy(obs)
        obs, rew, term, trunc, info = env.step(action)
        steps += 1
        r.begin_step()
        events = list(env.events)
        for k in range(SUBFRAMES):
            frame = r.draw(env, (k + 1) / SUBFRAMES, 1.0 / FPS)
            if frame_hook:
                frame = frame_hook(frame, env)
            writer.write(frame)
            audio_frames.append(_audio_frame(env, events if k == 0 else []))
        if term or trunc or (max_steps and steps >= max_steps):
            stats = info.get("episode_stats") or env.episode_stats()
            break
    # hold the last frame (with the banner) for a moment
    for _ in range(int(hold_end * FPS)):
        frame = r.draw(env, 1.0, 1.0 / FPS)
        writer.write(frame)
        audio_frames.append(dict(speed=0.0, throttle=0.0, brake=0.0, blink=False, events=[]))
    writer.close()
    audio = mix_soundtrack(audio_frames, FPS, bank)
    write_wav(tmp_a, audio)
    mux(tmp_v, tmp_a, path)
    os.remove(tmp_v)
    os.remove(tmp_a)
    return stats


def record_episodes(exp_dir, n=2, stage=4, prefix="demo", seeds=None):
    """Record n demo episodes of a trained experiment."""
    from .env import AutoDriveEnv

    exp_dir = Path(exp_dir)
    policy, cfg = load_policy(exp_dir)
    env = AutoDriveEnv(reward_config=cfg, stage=stage, replay_prob=0.0)
    bank = SoundBank()
    seeds = seeds or [20_000_000 + i for i in range(n)]
    out = []
    for i, sd in enumerate(seeds[:n]):
        p = exp_dir / "videos" / f"{prefix}_{i + 1}_seed{sd}.mp4"
        st = record_episode(env, policy, p, seed=sd, options={"stage": stage}, bank=bank)
        print(f"[video] {p.name}: {st['status']} return={st['return']:.1f}")
        out.append((p, st))
    return out
