#!/usr/bin/env python
"""
game.py - "Reward Designer": the training game, in your browser.

    python game.py            # then open http://localhost:8000

You never drive the car. In each level you DESIGN ITS REWARD (switch reward
terms on, choose weights), choose how long to train, watch it learn live, and
a hidden test on roads it has never seen gives you a score and stars.
One star unlocks the next level. See GAME_DESIGN.md.

Everything runs locally:
  * training is `train.py --level N` in a background process
  * the live view is the same streaming viewer as watch.py
  * your progress is saved in games/progress.json, your cars in
    experiments/game/ (both ignored by git)
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
GAMES = ROOT / "games"
PROGRESS = GAMES / "progress.json"
RUNS = ROOT / "experiments" / "game"
SOUNDS = ROOT / "assets" / "sounds"

from autodrive.levels import BY_ID, LEVELS, run_test, stars_for          # noqa: E402
from autodrive.rewards import DEFAULT_SETTINGS, DEFAULT_TERMS, TERMS      # noqa: E402

# plain-English help for each reward term (shown in the workshop)
HELP = {
    "progress": "Points for every metre driven along the GPS route. Without it the car has no reason to go anywhere.",
    "goal_reached": "One-off bonus for ending inside the parking spot.",
    "parking_precision": "Extra one-off bonus for parking neatly (centred and straight).",
    "collision": "Crashing into a car or building. Ends the drive.",
    "pedestrian_hit": "Hitting a pedestrian. Ends the drive.",
    "pedestrian_proximity": "Driving fast close to a pedestrian (every step it happens).",
    "crosswalk_yield": "Driving over a crosswalk while someone is walking on it.",
    "speeding": "Going over the speed limit (only school zones have one in this game).",
    "red_light": "Running a red light.",
    "stop_sign": "Not stopping at a stop sign.",
    "lane_deviation": "Drifting away from the middle of the lane / the planned path (every step).",
    "off_road": "Wheels on the grass or sidewalk (every step).",
    "comfort": "Jerky driving: sudden braking, fast steering. Too big and the car becomes timid.",
    "stall": "Standing still for no reason (every step).",
    "time": "A small cost for every step, so the car hurries.",
    "off_route": "Wandering more than 6 m away from the route. Ends the drive.",
    "stuck": "Standing still for 12 seconds for no reason. Ends the drive.",
    "speed_bonus": "Points for raw speed, no matter where the car goes. A classic trap...",
}


# =====================================================================
# Progress (saved between sessions)
# =====================================================================
def load_progress():
    try:
        return json.load(open(PROGRESS))
    except Exception:
        return {}


def save_progress(p):
    GAMES.mkdir(exist_ok=True)
    json.dump(p, open(PROGRESS, "w"), indent=1)


def unlocked(progress, lid):
    return lid == 1 or progress.get(str(lid - 1), {}).get("stars", 0) >= 1


# =====================================================================
# Hints: look at what happened and point at a reward that might help
# =====================================================================
def hints(level, weights, result, train_rows):
    """Hints only ever point at reward terms this level actually offers."""
    on = {k for k, (en, w) in weights.items() if en and w > 0}
    have = set(level["terms"])

    def ask(terms, question):
        """The question, if one of these terms is available but switched off."""
        return question if (set(terms) & have) and not (set(terms) & on) else ""
    d = result["drives"]
    n = len(d)
    share = lambda st: sum(x["status"] == st for x in d) / n
    out = []
    parked = share("parked")
    if "goal_reached" in have and "goal_reached" not in on and parked < 0.5:
        out.append("Nothing rewards actually ending in the spot. Is there a reward for parking?")
    if "progress" in have and "progress" not in on and np.mean([x["progress"] for x in d]) < 0.5:
        out.append("The car barely moves along the route. What would make driving forward worth it?")
    if share("stuck") > 0.25 or np.mean([x["stall_frac"] for x in d]) > 0.3:
        if ({"stall", "stuck"} & have) and not ({"stall", "stuck"} & on):
            out.append("Your car often just stops and waits. Nothing punishes that yet.")
        else:
            out.append("Your car still freezes a lot. Maybe your penalties make trying look too risky "
                       "(e.g. a huge collision or comfort weight), or standing still is too cheap.")
    if share("collision") > 0.25:
        out.append("Lots of crashes. " + ask(["collision"], "Does crashing cost anything?"))
    if share("off_route") > 0.25:
        out.append("The car leaves its route often. " + (
            ask(["off_route", "lane_deviation"], "Does anything keep it on the path?")
            or "Maybe it drives too fast to follow the turns, or needs more training time."))
    if share("timeout") > 0.25:
        out.append("Drives run out of time: the car is too slow or wanders.")
    if level["score"] == "speed" and parked > 0.5:
        speeds = [x["avg_speed_kmh"] for x in d if x["status"] == "parked"]
        if speeds and np.mean(speeds) < 30:
            out.append(f"It parks, but slowly (avg {np.mean(speeds):.0f} km/h). What would make it hurry?"
                       + (" (Is a penalty making it timid?)" if {"comfort", "collision"} & on else ""))
    if level["score"] == "precision" and parked > 0.5 and "parking_precision" not in on:
        out.append("It parks, but nothing rewards parking neatly.")
    if "speed_bonus" in on and np.mean([x["turns_per_100m"] for x in d]) > 0.6:
        out.append("speed_bonus is on and the car turns a lot... is it driving in circles to farm speed?")
    if level.get("rules"):
        if sum(x["red"] for x in d) and "red_light" not in on:
            out.append("It runs red lights and nothing tells it not to.")
        if sum(x["stop"] for x in d) and "stop_sign" not in on:
            out.append("It rolls through stop signs and nothing tells it not to.")
        if sum(x["school_over_kmh"] > 3 for x in d) and "speeding" in have and "speeding" not in on:
            out.append("It speeds through school zones. Is there a speed penalty?")
    if train_rows and len(train_rows) > 5:
        last = [float(r["success_rate_100"]) for r in train_rows[-5:]]
        if max(last) > parked + 0.3:
            out.append("It parked more often during training than in the test - maybe train a bit longer.")
    if not out:
        out.append("Nice! To get more stars: faster, cleaner, fewer mistakes - or train a bit longer.")
    return [h.strip() for h in out]


# =====================================================================
# The game server state
# =====================================================================
class Game:
    def __init__(self, fps, quality):
        self.fps, self.quality = fps, quality
        self.progress = load_progress()
        self.lock = threading.Lock()
        self.phase = "idle"          # idle | training | testing | done
        self.level_id = 1
        self.run_dir = None
        self.proc = None
        self.t_start = 0.0
        self.minutes = 0.0
        self.weights = {}
        self.test_done = 0
        self.test_total = 0
        self.result = None
        self.log_tail = []
        self.sp = None
        self.view_level(1)

    # ----------------------------------------------------- live viewer
    def _set_spectator(self, exp, stages, follow, stage=None):
        from watch import Spectator
        old = self.sp
        sp = Spectator(Path(exp), stage, follow, self.quality, int(time.time()) % 10_000, self.fps,
                       stages=stages, allow_untrained=True)
        threading.Thread(target=sp.run, daemon=True).start()
        self.sp = sp
        if old is not None:
            old.stopped = True

    def view_level(self, lid):
        """Idle view: your best car for this level (or an untrained one) on test-like roads."""
        lv = BY_ID[lid]
        self.level_id = lid
        best = self.progress.get(str(lid), {}).get("best_run")
        exp = Path(best) if best and (Path(best) / "model.zip").exists() else RUNS / "_none"
        self._set_spectator(exp, [lv["test"]], follow=False, stage=0)

    # ----------------------------------------------------- training
    def start(self, lid, weights, minutes, init_prev):
        with self.lock:
            if self.phase in ("training", "testing"):
                return "already busy"
            lv = BY_ID[lid]
            if not lv["available"] or not unlocked(self.progress, lid):
                return "level locked"
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run = RUNS / f"L{lid}_{stamp}"
            run.mkdir(parents=True, exist_ok=True)
            terms = {k: dict(enabled=bool(en), weight=float(w)) for k, (en, w) in weights.items()
                     if k in lv["terms"]}
            cfg = {"name": f"level{lid}_{stamp}", "description": "designed in game.py", "terms": terms}
            (run / "my_reward.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            cmd = [sys.executable, str(ROOT / "train.py"), "--reward", str(run / "my_reward.yaml"),
                   "--name", f"game/{run.name}", "--level", str(lid), "--minutes", str(minutes),
                   "--ckpt-every", "40000"]
            prev = self.progress.get(str(lid - 1), {}).get("best_run")
            if init_prev and prev and (Path(prev) / "model.zip").exists():
                cmd += ["--init-from", prev]
            self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, bufsize=1)
            self.phase, self.level_id, self.run_dir = "training", lid, run
            self.t_start, self.minutes, self.weights = time.time(), float(minutes), weights
            self.result, self.log_tail = None, []
            self.test_done = self.test_total = 0
        self._set_spectator(run, lv["curriculum"], follow=True, stage=0)
        threading.Thread(target=self._watch_training, args=(lv, run), daemon=True).start()
        return "ok"

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()          # train.py saves the car and exits

    def _watch_training(self, lv, run):
        for line in self.proc.stdout:
            line = line.rstrip()
            if line.startswith(("[", "===")) and "rewards switched on" not in line:
                self.log_tail = (self.log_tail + [line.strip("= ")])[-12:]
        self.proc.wait()
        if not (run / "model.zip").exists():
            self.phase = "done"
            self.result = {"error": "Training failed - see the log.", "log": self.log_tail}
            return
        # ---- hidden test ----------------------------------------------------
        self.phase = "testing"
        self.test_total = 12
        self._set_spectator(run, [lv["test"]], follow=False, stage=0)

        def tick(i, d):
            self.test_done = i + 1
        res = run_test(run, lv, n=self.test_total, on_drive=tick)
        rows = read_progress(run)
        res["hints"] = hints(lv, self.weights, res, rows)
        res["minutes"] = round((time.time() - self.t_start) / 60, 1)
        p = self.progress.setdefault(str(lv["id"]), {})
        p["attempts"] = p.get("attempts", 0) + 1
        p["last_weights"] = {k: v for k, v in self.weights.items() if k in lv["terms"]}
        res["new_best"] = res["score"] > p.get("best", -1)
        if res["new_best"]:
            p.update(best=res["score"], stars=max(p.get("stars", 0), res["stars"]), best_run=str(run),
                     best_weights={k: v for k, v in self.weights.items() if k in lv["terms"]})
        else:
            p["stars"] = max(p.get("stars", 0), res["stars"])
        save_progress(self.progress)
        json.dump(res, open(run / "test_result.json", "w"), indent=1)
        self.result = res
        self.phase = "done"

    # ----------------------------------------------------- state for the page
    def state(self):
        lv = BY_ID[self.level_id]
        rows = read_progress(self.run_dir) if self.run_dir and self.phase != "idle" else []
        stage_name = None
        if self.run_dir and (self.run_dir / "curriculum.json").exists():
            try:
                k = json.load(open(self.run_dir / "curriculum.json"))["stage"]
                stage_name = lv["curriculum"][k]["name"]
            except Exception:
                pass
        if rows and stage_name is None:
            stage_name = lv["curriculum"][int(rows[-1]["stage"])]["name"]
        return {
            "phase": self.phase, "level": self.level_id,
            "elapsed": round(time.time() - self.t_start, 1) if self.phase == "training" else 0,
            "minutes": self.minutes, "stage": stage_name,
            "curve": [[float(r["wall_s"]), float(r["success_rate_100"])] for r in rows][-300:],
            "steps": int(rows[-1]["timesteps"]) if rows else 0,
            "test": [self.test_done, self.test_total], "result": self.result, "log": self.log_tail[-6:],
            "viewer": self.sp.policy_src if self.sp else None,
            "progress": self.progress,
        }


def read_progress(run):
    f = Path(run) / "progress.csv" if run else None
    if not f or not f.exists():
        return []
    try:
        return list(csv.DictReader(open(f)))
    except Exception:
        return []


def levels_payload(progress):
    out = []
    for lv in LEVELS:
        prev_terms = set(BY_ID[lv["id"] - 1]["terms"]) if lv["id"] > 1 else set()
        out.append({
            "id": lv["id"], "name": lv["name"], "brief": lv["brief"], "available": lv["available"],
            "unlocked": unlocked(progress, lv["id"]), "score": lv["score"], "rules": bool(lv.get("rules")),
            "minutes": lv["minutes"], "stars": list(lv["stars"]),
            "terms": [{"name": t, "sign": TERMS[t][0], "kind": TERMS[t][1], "help": HELP.get(t, TERMS[t][2]),
                       "suggested": DEFAULT_TERMS[t]["weight"], "new": t not in prev_terms and lv["id"] > 1}
                      for t in lv["terms"]],
        })
    return out


# =====================================================================
# Web page
# =====================================================================
PAGE = (ROOT / "assets" / "game.html")


def make_handler(game: Game):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if u.path == "/api/train":
                w = {k: (bool(v.get("enabled")), float(v.get("weight", 0))) for k, v in body["terms"].items()}
                msg = game.start(int(body["level"]), w, float(np.clip(float(body["minutes"]), 0.5, 120)),
                                 bool(body.get("init_prev")))
                self._json({"status": msg})
            elif u.path == "/api/stop":
                game.stop()
                self._json({"status": "ok"})
            elif u.path == "/api/view":
                if game.phase in ("idle", "done"):
                    game.phase = "idle"
                    game.view_level(int(body["level"]))
                self._json({"status": "ok"})
            else:
                self._json({"error": "unknown"}, 404)

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path == "/":
                    self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif u.path == "/api/levels":
                    self._json({"levels": levels_payload(game.progress), "settings": DEFAULT_SETTINGS})
                elif u.path == "/api/state":
                    self._json(game.state())
                elif u.path == "/control":
                    self._json(game.sp.control(parse_qs(u.query)))
                elif u.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    last, sp = -1, None
                    while True:
                        if game.sp is not sp:              # viewer switched (new run / level)
                            sp, last = game.sp, -1
                        with sp.cond:
                            sp.cond.wait_for(lambda: sp.frame_id != last or game.sp is not sp, timeout=5)
                            jpg, last = sp.jpeg, sp.frame_id
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                elif u.path == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    sp, last = None, 0
                    while True:
                        if game.sp is not sp:
                            sp = game.sp
                            with sp.cond:
                                last = sp.event_id
                        with sp.cond:
                            sp.cond.wait_for(lambda: sp.event_id != last or game.sp is not sp, timeout=15)
                            new = [(i, d) for i, d in sp.events if i > last]
                            last = sp.event_id
                        if not new:
                            self.wfile.write(b": keep-alive\n\n")
                        for _, d in new:
                            self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
                        self.wfile.flush()
                elif u.path.startswith("/sounds/"):
                    f = (SOUNDS / Path(u.path).name).resolve()
                    if f.parent == SOUNDS.resolve() and f.exists():
                        self._send(200, f.read_bytes(), "audio/ogg")
                    else:
                        self._send(404, b"not found", "text/plain")
                else:
                    self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=80)
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    game = Game(args.fps, args.quality)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(game))
    srv.daemon_threads = True
    print(f"[game] Reward Designer running: open http://localhost:{args.port}   (Ctrl+C to quit)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        game.stop()
        print("\n[game] bye")


if __name__ == "__main__":
    main()
