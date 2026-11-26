#!/usr/bin/env python
"""
watch.py - spectate a trained car LIVE in your browser.

    python watch.py                                  # shipped model, http://localhost:8000
    python watch.py --exp experiments/mine           # your own experiment
    python watch.py --exp experiments/mine --follow  # watch WHILE it trains: reloads
                                                     # the newest checkpoint automatically
    python watch.py --stage 5 --port 8080
    python watch.py --fps 60                         # smoother (needs a bit more CPU)

Then open the printed address. The page shows a live video stream, buttons
(pause, speed, next city, stage, sensors/HUD on/off), running statistics,
and plays the sound effects in the browser (click "Sound on" - browsers only
allow audio after a click).

How it works (no extra dependencies, standard library only):
  * a simulation thread runs the policy in the environment and renders
    frames with the same renderer as the videos (autodrive/render.py)
  * GET /stream.mjpg   an MJPEG stream (a never-ending sequence of JPEGs;
                       every browser shows it as a live <img>)
  * GET /events        Server-Sent Events: per-frame state (speed, braking,
                       blinker, sound events, episode results) as JSON
  * GET /control?...   buttons on the page call this
  * GET /sounds/<f>    the sound files from assets/sounds, played with WebAudio

By default the server only listens on this computer (127.0.0.1). Use
--host 0.0.0.0 to watch from another device on your network.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SOUNDS = ROOT / "assets" / "sounds"


# =====================================================================
# Simulation thread
# =====================================================================
class Spectator:
    def __init__(self, exp: Path, stage: int | None, follow: bool, quality: int, seed: int, fps: int = 30,
                 stages=None, allow_untrained=False):
        """stages: list of world settings to drive in (default: the main curriculum).
        allow_untrained: if no model exists yet, drive an untrained car (game mode)."""
        import torch
        torch.set_num_threads(1)
        from autodrive.curriculum import EVAL_STAGE, STAGES
        from autodrive.env import AutoDriveEnv
        from autodrive.render import Renderer

        self.exp = exp
        self.follow = follow
        self.quality = quality
        self.fps = int(np.clip(fps, 10, 60))
        self.stages = stages if stages is not None else STAGES
        self.stage = (len(self.stages) - 1 if stages is not None else EVAL_STAGE) if stage is None else stage
        self.allow_untrained = allow_untrained
        self.policy_mtime = None
        self.policy_src = None
        self.reward_cfg = None
        self.stopped = False
        self._load_policy(force=True)
        self.env = AutoDriveEnv(reward_config=self.reward_cfg, stage=self.stage, replay_prob=0.0,
                                stages=self.stages)
        self.env.renderer = Renderer()
        self.rng = np.random.default_rng(seed)

        # shared state (guarded by self.cond)
        self.cond = threading.Condition()
        self.jpeg = b""
        self.frame_id = 0
        self.events = deque(maxlen=200)       # (id, dict) for /events
        self.event_id = 0
        self.paused = False
        self.zoom = 1.0
        self.size = (1280, 720)
        self.watchers = 0                     # open video streams; nobody watching = no rendering
        self.speed = 1.0
        self.request_new = False
        self.results = deque(maxlen=200)
        self.episodes = 0
        self.ep_start = time.time()

    # ---------------------------------------------------------------
    def _policy_dir(self):
        """--follow: newest of experiments/<name>/checkpoints and the final model."""
        cands = [self.exp, self.exp / "checkpoints"] if self.follow else [self.exp]
        best = None
        for d in cands:
            f = d / "model.zip"
            if f.exists() and (d / "vecnormalize.pkl").exists():
                if best is None or f.stat().st_mtime > best[1]:
                    best = (d, f.stat().st_mtime)
        return best

    def _load_policy(self, force=False):
        from autodrive.rewards import load_reward_config
        from autodrive.video import load_policy
        found = self._policy_dir()
        if found is None:
            if force and self.allow_untrained:
                # nothing trained yet: an untrained car just wobbles along,
                # scored with the player's own reward (never the defaults)
                for f in ("reward_config.yaml", "my_reward.yaml"):
                    if (self.exp / f).exists():
                        self.reward_cfg = load_reward_config(self.exp / f, allow_empty=True)
                        break
                else:
                    self.reward_cfg = load_reward_config(None)
                    self.reward_cfg["terms"] = {k: dict(t, enabled=False) for k, t in self.reward_cfg["terms"].items()}
                rng = np.random.default_rng(0)
                self.policy = lambda obs: np.array([rng.normal(0, 0.3), rng.normal(-0.2, 0.4)])
                self.policy_src = "untrained car (waiting for the first checkpoint)"
                return False
            if force:
                sys.exit(f"No model.zip + vecnormalize.pkl in {self.exp}"
                         f"{' or its checkpoints/' if self.follow else ''}. Train first.")
            return False
        d, mtime = found
        if not force and mtime == self.policy_mtime:
            return False
        time.sleep(0.5 if not force else 0)      # let the trainer finish writing
        try:
            policy, cfg = load_policy(d)
        except Exception as e:                      # half-written file: try again later
            print("[watch] could not load model yet:", e)
            return False
        self.policy = policy
        cfg_path = self.exp / "reward_config.yaml"
        self.reward_cfg = cfg or (load_reward_config(cfg_path, allow_empty=True) if cfg_path.exists() else None)
        self.policy_mtime = mtime
        self.policy_src = f"{d.relative_to(ROOT) if d.is_relative_to(ROOT) else d}/model.zip"
        if not force:
            print(f"[watch] reloaded newer model from {self.policy_src}")
        return True

    def _new_episode(self):
        seed = int(self.rng.integers(60_000_000, 70_000_000))   # never-seen cities
        self.obs, _ = self.env.reset(seed=seed, options={"stage": self.stage})
        self.city_seed = seed
        self.ep_start = time.time()
        self._push({"type": "episode", "stage": self.stages[self.stage]["name"], "seed": seed,
                    "model": self.policy_src})

    def _push(self, d):
        with self.cond:
            self.event_id += 1
            self.events.append((self.event_id, d))
            self.cond.notify_all()

    def _publish(self, frame):
        from PIL import Image
        buf = io.BytesIO()
        # subsampling=0: full-resolution colour (4:4:4). The default halves colour
        # resolution, which blurs coloured edges and text the most.
        Image.fromarray(frame).save(buf, "JPEG", quality=self.quality, subsampling=0)
        with self.cond:
            self.jpeg = buf.getvalue()
            self.frame_id += 1
            self.cond.notify_all()

    def _show(self, frame, until):
        """Publish a frame, then wait until its display time is over."""
        self._publish(frame)
        dt = until - time.time()
        if dt > 0:
            time.sleep(dt)

    def run(self):
        """Simulation + rendering loop.

        The simulation advances in 0.1 s steps. Between two steps we draw
        several frames with interpolated positions (the renderer's `alpha`),
        each shown at its own moment, so motion looks smooth at --fps frames
        per second. At high playback speeds, or on a slow computer, fewer
        in-between frames are drawn so the simulation keeps up.
        """
        env, r = self.env, self.env.renderer
        self._new_episode()
        r.begin_step()
        self._publish(r.draw(env, 1.0, 1.0 / self.fps))
        last_check = time.time()
        render_ms = 10.0                     # running estimate of draw + encode time
        hold_until = None
        while not self.stopped:
            if self.follow and time.time() - last_check > 3:
                last_check = time.time()
                if self._load_policy():
                    self._push({"type": "model", "model": self.policy_src})
                    # show the curriculum stage the trainer is currently practising
                    try:
                        k = int(json.load(open(self.exp / "curriculum.json"))["stage"])
                        if k != self.stage and 0 <= k < len(self.stages):
                            self.stage = k
                            self.request_new = True
                    except Exception:
                        pass
            if self.request_new:
                self.request_new = False
                hold_until = None
                self._new_episode()
            frame_dt = 1.0 / self.fps
            if self.watchers <= 0:
                # nobody is looking: don't burn CPU (training may be running)
                time.sleep(0.2)
                continue
            if self.paused:
                self._show(r.draw(env, 1.0, 0.0), time.time() + frame_dt)
                continue
            if hold_until is not None:
                # show the result banner (particles still animate) for ~2 s
                self._show(r.draw(env, 1.0, frame_dt), time.time() + frame_dt)
                if time.time() > hold_until:
                    hold_until = None
                    self._new_episode()
                continue

            step_wall = 0.1 / self.speed     # wall-clock time one sim step should take
            t_step = time.time()
            self.obs, rew, term, trunc, info = env.step(self.policy(self.obs))
            r.begin_step()
            self.stream_fps = 0.9 * getattr(self, "stream_fps", 0.0) + 0.1 * (
                (1.0 / max(step_wall, 1e-3)) * max(1, min(int(round(step_wall * self.fps)),
                                                         int(step_wall / (render_ms / 1000.0 * 1.15)))))
            events = [{"name": e[0], "dist": float(np.linalg.norm(np.asarray(e[1]) - env.car.pos))}
                      for e in env.events if e[0] in ("crash", "ped_hit", "horn", "light_change", "chime")]
            car = env.car
            blink = env.blinker() is not None and (r.blink_t % 0.7) < 0.35
            self._push({"type": "state", "speed": round(car.v, 2),
                        "throttle": round(float(np.clip(car.a / 3.5, 0, 1)), 2),
                        "brake": round(float(np.clip(-car.a / 8.0, 0, 1)), 2),
                        "blink": bool(blink), "events": events,
                        "t": round(env.t, 1), "ret": round(env.ep_return, 2),
                        "fps": round(getattr(self, "stream_fps", 0.0))})
            # how many frames fit into this step: target fps, limited by render speed
            n_sub = max(1, min(int(round(step_wall * self.fps)),
                               int(step_wall / (render_ms / 1000.0 * 1.15))))
            for k in range(n_sub):
                t_r = time.time()
                frame = r.draw(env, (k + 1) / n_sub, 0.1 / n_sub)
                self._publish(frame)
                render_ms = 0.9 * render_ms + 0.1 * (time.time() - t_r) * 1000.0
                wait = t_step + step_wall * (k + 1) / n_sub - time.time()
                if wait > 0:
                    time.sleep(wait)
            if term or trunc:
                st = info.get("episode_stats") or env.episode_stats()
                self.episodes += 1
                self.results.append(st["status"])
                ok = sum(s == "parked" for s in self.results)
                self._push({"type": "result", "status": st["status"], "return": round(st["return"], 1),
                            "time": round(st["time_s"], 1), "episodes": self.episodes,
                            "success_rate": round(ok / len(self.results), 3),
                            "recent": list(self.results)[-12:]})
                hold_until = time.time() + 2.0

    # --------------------------------------------------------------- controls
    def control(self, q):
        cmd = q.get("cmd", [""])[0]
        if cmd == "pause":
            self.paused = not self.paused
        elif cmd == "speed":
            self.speed = float(np.clip(float(q.get("v", ["1"])[0]), 0.25, 16))
        elif cmd == "next":
            self.request_new = True
        elif cmd == "stage":
            self.stage = int(np.clip(int(q.get("v", ["0"])[0]), 0, len(self.stages) - 1))
            self.results.clear()
            self.request_new = True
        elif cmd == "sensors":
            self.env.renderer.show_sensors = not self.env.renderer.show_sensors
        elif cmd == "hud":
            self.env.renderer.show_hud = not self.env.renderer.show_hud
        elif cmd == "zoom":
            self.zoom = self.env.renderer.set_zoom(float(q.get("v", ["1"])[0]))
        elif cmd == "size":                    # the viewer's real pixel size: render at it (sharp)
            self.size = self.env.renderer.set_output_size(int(float(q.get("w", ["1280"])[0])),
                                                          int(float(q.get("h", ["720"])[0])))
        return {"paused": self.paused, "speed": self.speed, "stage": self.stage, "zoom": self.zoom,
                "sensors": self.env.renderer.show_sensors, "hud": self.env.renderer.show_hud}


# =====================================================================
# Web page
# =====================================================================
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Autodrive Live</title>
<style>
:root { --bg:#0f1115; --panel:#181b22; --line:#2a2f3a; --fg:#e8eaee; --dim:#9aa1ad;
        --ok:#5cc878; --bad:#e8584f; --warn:#f0a43c; --accent:#4aa3ff; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; }
header { display:flex; align-items:center; gap:12px; padding:10px 16px; border-bottom:1px solid var(--line); flex-wrap:wrap; }
header h1 { font-size:16px; margin:0 8px 0 0; letter-spacing:.04em; }
button, select { background:var(--panel); color:var(--fg); border:1px solid var(--line); border-radius:8px;
                 padding:6px 10px; font:inherit; cursor:pointer; -webkit-tap-highlight-color:transparent;
                 transition-property:border-color, color, background-color; transition-duration:150ms; transition-timing-function:ease-out; }
@media (hover: hover) { button:hover:not(:disabled), select:hover { border-color:var(--accent); } }
@media (prefers-reduced-motion: no-preference) {
  button { transition-property:border-color, color, background-color, scale; }
  button:not(:disabled):active { scale:0.96; }
  .zoom button:not(:disabled):active { scale:1; }       /* used rapidly: instant feedback only */
}
/* zoom: -/+ joined like a stepper; nested radius 8 = 7 + 1 border */
.zoom { display:inline-flex; align-items:stretch; border:1px solid var(--line); border-radius:8px; background:var(--panel); overflow:hidden; }
.zoom button { border:none; border-radius:0; background:transparent; padding:0 9px; display:grid; place-items:center; color:var(--dim); }
.zoom button svg { width:12px; height:12px; display:block; }
.zoom #zl { min-width:52px; padding:6px 4px; color:var(--fg); font-size:12px; font-variant-numeric:tabular-nums; }
@media (hover: hover) { .zoom button:hover { background:#232834; color:var(--fg); } }
button.on { border-color:var(--accent); color:var(--accent); }
main { display:grid; grid-template-columns: minmax(0,1fr) 280px; gap:16px; padding:16px; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } }
#view { width:100%; border-radius:10px; background:#000; display:block; outline:1px solid oklch(1 0 0 / 0.1); outline-offset:-1px; cursor:zoom-in; }
aside { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; align-self:start; }
aside h2 { font-size:12px; color:var(--dim); text-transform:uppercase; letter-spacing:.08em; margin:14px 0 6px; }
aside h2:first-child { margin-top:0; }
.big { font-size:28px; font-weight:700; }
.row { display:flex; justify-content:space-between; gap:8px; padding:2px 0; }
.dim { color:var(--dim); }
.chips { display:flex; flex-wrap:wrap; gap:4px; }
.chip { font-size:11px; padding:2px 6px; border-radius:6px; background:#222733; }
.chip.parked { color:var(--ok); } .chip.collision, .chip.pedestrian_hit { color:var(--bad); }
.chip.off_route, .chip.stuck, .chip.timeout { color:var(--warn); }
code { color:var(--dim); font-size:12px; word-break:break-all; }
</style></head>
<body>
<header>
  <h1>AUTODRIVE LIVE</h1>
  <button id="pause">Pause</button>
  <span class="dim">Speed</span>
  <select id="speed"><option value="0.5">0.5</option><option value="1" selected>1</option><option value="2">2</option><option value="4">4</option><option value="8">8</option></select>
  <button id="next">Next city</button>
  <span class="dim">Stage</span><select id="stage"></select>
  <button id="sensors" class="on">Sensors</button>
  <button id="hud" class="on">HUD</button>
  <button id="sound">Sound on</button>
  <span class="dim">Zoom</span><span class="zoom"><button id="zout" title="zoom out" aria-label="zoom out"><svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6h8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg></button><button id="zl" title="reset zoom">100%</button><button id="zin" title="zoom in" aria-label="zoom in"><svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6h8M6 2v8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg></button></span>
</header>
<main>
  <img id="view" src="/stream.mjpg" alt="live view of the self-driving car">
  <aside>
    <h2>This session</h2>
    <div class="big" id="rate">-</div>
    <div class="dim" id="eps">0 episodes parked</div>
    <h2>Last outcomes</h2>
    <div class="chips" id="recent"></div>
    <h2>Now</h2>
    <div class="row"><span class="dim">City seed</span><span id="seed">-</span></div>
    <div class="row"><span class="dim">Time</span><span id="t">-</span></div>
    <div class="row"><span class="dim">Speed</span><span id="v">-</span></div>
    <div class="row"><span class="dim">Return</span><span id="ret">-</span></div>
    <div class="row"><span class="dim">Stream</span><span id="fps">-</span></div>
    <h2>Model</h2>
    <code id="model">-</code>
  </aside>
</main>
<script>
const $ = id => document.getElementById(id);
const STAGES = __STAGES__;
STAGES.forEach((n, i) => { const o = document.createElement('option'); o.value = i; o.textContent = n; $('stage').appendChild(o); });
$('stage').value = __STAGE__;
const ctl = (q) => fetch('/control?' + q).then(r => r.json()).then(s => {
  $('pause').textContent = s.paused ? 'Resume' : 'Pause';
  $('speed').value = String(s.speed); $('stage').value = s.stage;
  $('sensors').classList.toggle('on', s.sensors); $('hud').classList.toggle('on', s.hud);
});
ctl('');   // read the current settings when the page opens
$('pause').onclick = () => ctl('cmd=pause');
$('next').onclick = () => ctl('cmd=next');
// render at the real on-screen pixel size (sharp on big / high-DPI screens)
function sendSize() {
  const v = $('view'), d = window.devicePixelRatio || 1;
  const w = Math.round(v.clientWidth * d), h = Math.round(w * 9 / 16);
  if (w > 0) ctl('cmd=size&w=' + w + '&h=' + h);
}
let sizeT = null;
window.addEventListener('resize', () => { clearTimeout(sizeT); sizeT = setTimeout(sendSize, 300); });
setTimeout(sendSize, 500);
let Z = 1;
function zoom(z) { Z = Math.min(3, Math.max(0.5, Math.round(z * 100) / 100)); $('zl').textContent = Math.round(Z * 100) + '%'; ctl('cmd=zoom&v=' + Z); }
$('zin').onclick = () => zoom(Z * 1.25); $('zout').onclick = () => zoom(Z / 1.25); $('zl').onclick = () => zoom(1);
$('view').addEventListener('wheel', e => { e.preventDefault(); zoom(Z * Math.exp(-e.deltaY * 0.0015)); }, {passive:false});
$('speed').onchange = e => ctl('cmd=speed&v=' + e.target.value);
$('stage').onchange = e => ctl('cmd=stage&v=' + e.target.value);
$('sensors').onclick = () => ctl('cmd=sensors');
$('hud').onclick = () => ctl('cmd=hud');

// ---------------- sound (WebAudio, same files as the videos) ----------------
let ac = null, buf = {}, engine = null, engGain = null, squeal = null, sqGain = null, lastBlink = false;
const FILES = {engine:'engine_loop.ogg', brake:'brake_squeal.ogg', horn:'horn.ogg', crash1:'crash_metal.ogg',
  crash2:'crash_glass.ogg', thud:'ped_hit.ogg', tick:'turn_tick.ogg', ping:'light_change.ogg',
  chime:'park_chime.ogg', ambient:'ambient_city.ogg', wind:'wind_loop.ogg'};
async function load(name) {
  try { const r = await fetch('/sounds/' + FILES[name]); buf[name] = await ac.decodeAudioData(await r.arrayBuffer()); }
  catch (e) { console.warn('sound missing', name); }
}
function loop(name, gain) {
  if (!buf[name]) return [null, null];
  const s = ac.createBufferSource(); s.buffer = buf[name]; s.loop = true;
  const g = ac.createGain(); g.gain.value = gain; s.connect(g).connect(ac.destination); s.start(); return [s, g];
}
function play(name, gain) {
  if (!ac || !buf[name]) return;
  const s = ac.createBufferSource(); s.buffer = buf[name];
  const g = ac.createGain(); g.gain.value = gain; s.connect(g).connect(ac.destination); s.start();
}
$('sound').onclick = async () => {
  if (ac) { ac.close(); ac = null; $('sound').textContent = 'Sound on'; $('sound').classList.remove('on'); return; }
  ac = new AudioContext();
  await Promise.all(Object.keys(FILES).map(load));
  [engine, engGain] = loop('engine', 0.15); [squeal, sqGain] = loop('brake', 0.0);
  loop('ambient', 0.10); loop('wind', 0.06);
  $('sound').textContent = 'Sound off'; $('sound').classList.add('on');
};

// ---------------- live state ----------------
const es = new EventSource('/events');
es.onmessage = (m) => {
  const d = JSON.parse(m.data);
  if (d.type === 'state') {
    $('t').textContent = d.t.toFixed(1) + ' s';
    $('v').textContent = (d.speed * 3.6).toFixed(0) + ' km/h';
    $('ret').textContent = d.ret.toFixed(1);
    $('fps').textContent = d.fps + ' fps';
    if (ac && engine) {
      engine.playbackRate.value = 0.55 + d.speed / 20 + 0.12 * d.throttle;
      engGain.gain.value = 0.16 + 0.10 * d.throttle + 0.04 * d.speed / 20;
      const sq = Math.max(0, (d.brake - 0.45) / 0.55) * Math.min(1, Math.max(0, (d.speed - 3) / 5));
      if (sqGain) sqGain.gain.value = 0.35 * sq;
      if (d.blink !== lastBlink) play('tick', 0.25);
      lastBlink = d.blink;
      for (const e of d.events) {
        const att = 1 / (1 + Math.max(e.dist, 0) / 15);
        if (e.name === 'crash') { play('crash1', 0.9); play('crash2', 0.6); }
        else if (e.name === 'ped_hit') play('thud', 0.9);
        else if (e.name === 'horn') play('horn', 0.7 * att);
        else if (e.name === 'light_change') play('ping', 0.35);
        else if (e.name === 'chime') play('chime', 0.6);
      }
    }
  } else if (d.type === 'result') {
    $('rate').textContent = Math.round(d.success_rate * 100) + '% parked';
    $('eps').textContent = d.episodes + ' episodes this session';
    $('recent').innerHTML = d.recent.slice().reverse().map(s => `<span class="chip ${s}">${s.replace('_', ' ')}</span>`).join('');
  } else if (d.type === 'episode') {
    $('seed').textContent = d.seed; $('model').textContent = d.model;
    $('stage').value = STAGES.indexOf(d.stage);
  } else if (d.type === 'model') {
    $('model').textContent = d.model + ' (reloaded)';
  }
};
</script>
</body></html>
"""


def make_handler(sp: Spectator):
    stage_names = [s["name"] for s in sp.stages]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):          # keep the console quiet
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path == "/":
                    page = PAGE.replace("__STAGES__", json.dumps(stage_names)).replace("__STAGE__", str(sp.stage))
                    self._send(200, page.encode(), "text/html; charset=utf-8")
                elif u.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    last = -1
                    sp.watchers += 1
                    try:
                        while True:
                            with sp.cond:
                                sp.cond.wait_for(lambda: sp.frame_id != last, timeout=5)
                                jpg, last = sp.jpeg, sp.frame_id
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    finally:
                        sp.watchers -= 1
                elif u.path == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    with sp.cond:
                        last = sp.event_id
                        # replay the latest episode/result so a new page is filled in at once
                        backlog = [d for _, d in sp.events if d["type"] in ("episode", "result")][-2:]
                    for d in backlog:
                        self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
                    while True:
                        with sp.cond:
                            sp.cond.wait_for(lambda: sp.event_id != last, timeout=15)
                            new = [(i, d) for i, d in sp.events if i > last]
                            last = sp.event_id
                        if not new:
                            self.wfile.write(b": keep-alive\n\n")
                        for _, d in new:
                            self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
                        self.wfile.flush()
                elif u.path == "/control":
                    self._send(200, json.dumps(sp.control(parse_qs(u.query))).encode(), "application/json")
                elif u.path.startswith("/sounds/"):
                    f = (SOUNDS / Path(u.path).name).resolve()
                    if f.parent == SOUNDS.resolve() and f.exists():
                        self._send(200, f.read_bytes(), "audio/ogg")
                    else:
                        self._send(404, b"not found", "text/plain")
                else:
                    self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass                        # browser tab closed

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="experiments/balanced_45min", help="experiment folder")
    ap.add_argument("--stage", type=int, default=None, help="curriculum stage (default: full city)")
    ap.add_argument("--follow", action="store_true",
                    help="reload the newest checkpoint while that experiment is training")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = reachable from your network")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality of the stream")
    ap.add_argument("--fps", type=int, default=30, help="frames per second of the stream (10-60)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    exp = Path(args.exp)
    if not exp.exists():
        exp = ROOT / "experiments" / args.exp
    sp = Spectator(exp.resolve(), args.stage, args.follow, args.quality, args.seed, args.fps)
    threading.Thread(target=sp.run, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(sp))
    srv.daemon_threads = True
    host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"[watch] model: {sp.policy_src}   stage: {sp.stages[sp.stage]['name']}")
    print(f"[watch] open  http://{host}:{args.port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[watch] bye")


if __name__ == "__main__":
    main()
