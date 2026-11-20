#!/usr/bin/env python
"""
game.py - "Reward Designer": the training game, in your browser.

    python game.py            # then open http://localhost:8000

You never drive the car. In each level you DESIGN ITS REWARD: start from an
empty list, add rewards one by one (built-in ones, or your own rules made
with the rule builder), give each a number of points, choose how long to
train, watch it learn live, and a hidden test on roads it has never seen
gives you a score and stars. One star unlocks the next level.
See GAME_DESIGN.md and .plans/reward-workshop-redesign-spec.md.

Everything runs locally:
  * training is `train.py --level N` in a background process
  * the live view is the same streaming viewer as watch.py, and it only
    renders while a browser is actually watching it
  * your progress:        games/progress.json
    your custom rewards:  games/custom_rewards.json
    your trained cars:    experiments/game/      (all ignored by git)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import threading
import time
import uuid
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
LIBRARY = GAMES / "custom_rewards.json"
RUNS = ROOT / "experiments" / "game"
SOUNDS = ROOT / "assets" / "sounds"
PAGE = ROOT / "assets" / "game.html"

from autodrive.levels import BY_ID, LEVELS, run_test                         # noqa: E402
from autodrive.rewards import (MEASUREMENTS, OPERATORS, TERMS, RewardConfigError,  # noqa: E402
                               parse_config, to_yaml_dict)

# plain-English help for each built-in reward (no numbers on purpose)
HELP = {
    "progress": "Points for every metre the car drives along its GPS route (driving backwards gives the opposite).",
    "goal_reached": "Points once, when the car ends the drive parked inside the spot.",
    "parking_precision": "Points once when parked, scaled by how centred and straight the car is (perfect = full points).",
    "collision": "Points once when the car hits a car, a parked car or a building. The drive ends.",
    "pedestrian_hit": "Points once when the car hits a pedestrian. The drive ends.",
    "pedestrian_proximity": "Points every step, scaled by how close (0-1) a pedestrian is while the car is moving.",
    "crosswalk_yield": "Points once each time the car drives over a crosswalk someone is using.",
    "speeding": "Points every step for each m/s above the speed limit (only school zones have one in this game).",
    "red_light": "Points once each time the car crosses a stop line on red.",
    "stop_sign": "Points once each time the car passes a stop sign without stopping.",
    "lane_deviation": "Points every step for each metre the car is away from its lane / planned path.",
    "off_road": "Points every step while wheels are on the grass or sidewalk (scaled by how much of the car).",
    "comfort": "Points every step for jerky driving: sudden braking/accelerating and fast steering.",
    "stall": "Points every step while the car stands still for no reason.",
    "time": "Points every single step, whatever happens.",
    "off_route": "Points once when the car wanders more than 6 m from its route. The drive ends.",
    "stuck": "Points once when the car has stood still for 12 s for no reason. The drive ends.",
    "speed_bonus": "Points every step for each m/s of speed, wherever the car is going.",
}


# =====================================================================
# Saved data
# =====================================================================
def _load(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default


def _save(path, data):
    GAMES.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    json.dump(data, open(tmp, "w"), indent=1)
    tmp.replace(path)


DEBUG = False                # --debug: every built level is open, no stars needed


def unlocked(progress, lid):
    return DEBUG or lid == 1 or progress.get(str(lid - 1), {}).get("stars", 0) >= 1


# =====================================================================
# Hints: look at what happened and point at what might help
# =====================================================================
def hints(level, cfg, result, train_rows):
    t = cfg["terms"]
    pos = lambda k: t[k]["enabled"] and t[k]["weight"] > 0
    neg = lambda k: t[k]["enabled"] and t[k]["weight"] < 0
    custom_on = [c for c in cfg.get("custom", []) if c["enabled"] and c["give"] != 0]
    mentions = lambda m, sign: any(sign * c["give"] > 0 and any(x["m"] == m for g in c["when"] for x in g)
                                   for c in custom_on)
    d = result["drives"]
    n = len(d)
    share = lambda st: sum(x["status"] == st for x in d) / n
    parked = share("parked")
    out = []
    if not (pos("goal_reached") or pos("parking_precision") or mentions("parked", +1)) and parked < 0.5:
        out.append("Nothing in your design gives points for actually parking.")
    if not (pos("progress") or pos("speed_bonus")) and np.mean([x["progress"] for x in d]) < 0.5:
        out.append("The car barely moves along the route. What would make driving forward worth points?")
    if share("stuck") > 0.25 or np.mean([x["stall_frac"] for x in d]) > 0.3:
        if not (neg("stall") or neg("stuck") or mentions("got_stuck", -1)):
            out.append("Your car often just stops and waits - and that costs it nothing.")
        else:
            out.append("Your car still freezes a lot. Maybe trying looks too risky to it "
                       "(very big crash or comfort penalties), or standing still is too cheap.")
    if share("collision") > 0.25:
        if pos("collision"):
            out.append("Lots of crashes... your design GIVES points for crashing.")
        elif not (neg("collision") or mentions("crashed", -1)):
            out.append("Lots of crashes, and crashing costs nothing in your design.")
        else:
            out.append("Lots of crashes. The car may be going too fast for what it can see, or needs more training.")
    if share("off_route") > 0.25:
        out.append("The car leaves its route often. " + (
            "Maybe it drives too fast for the turns, or needs more training."
            if (neg("off_route") or neg("lane_deviation")) else "Does anything make leaving the path costly?"))
    if share("timeout") > 0.25:
        out.append("Drives run out of time: the car is too slow or wanders.")
    if level["score"] == "speed" and parked > 0.5:
        speeds = [x["avg_speed_kmh"] for x in d if x["status"] == "parked"]
        if speeds and np.mean(speeds) < 30:
            out.append(f"It parks, but slowly (avg {np.mean(speeds):.0f} km/h). What would make it hurry?")
    if level["score"] == "precision" and parked > 0.5 and not pos("parking_precision"):
        out.append("It parks, but nothing gives extra points for parking neatly.")
    if pos("speed_bonus") and np.mean([x["turns_per_100m"] for x in d]) > 0.6:
        out.append("speed_bonus gives points and the car turns a lot... is it driving in circles to farm speed?")
    if level.get("rules"):
        if sum(x["red"] for x in d) and not neg("red_light"):
            out.append("It runs red lights, and that costs it nothing.")
        if sum(x["stop"] for x in d) and not neg("stop_sign"):
            out.append("It rolls through stop signs, and that costs it nothing.")
        if sum(x["school_over_kmh"] > 3 for x in d) and not neg("speeding"):
            out.append("It speeds through school zones, and that costs it nothing.")
    if train_rows and len(train_rows) > 5:
        last = [float(r["success_rate_100"]) for r in train_rows[-5:]]
        if max(last) > parked + 0.3:
            out.append("It parked more often during training than in the test - maybe train a bit longer.")
    if not out:
        out.append("Nice! To get more stars: faster, cleaner, fewer mistakes - or train a bit longer.")
    return out


# =====================================================================
# The game server state
# =====================================================================
class Game:
    def __init__(self, fps, quality):
        self.view_opts = {}                  # last zoom / screen size sent by the page
        self.fps, self.quality = fps, quality
        self.progress = _load(PROGRESS, {})
        self.library = _load(LIBRARY, [])
        self.lock = threading.Lock()
        self.phase = "idle"          # idle | training | testing | done
        self.level_id = 1
        self.run_dir = None
        self.proc = None
        self.t_start = 0.0
        self.minutes = 0.0
        self.recipe = None
        self.cfg = None
        self.test_done = self.test_total = 0
        self.result = None
        self.log_tail = []
        self.sp = None
        self.previews = {}
        self.view_level(1)

    # ----------------------------------------------------- live viewer
    def _set_spectator(self, exp, stages, follow, stage=None):
        from watch import Spectator
        old = self.sp
        sp = Spectator(Path(exp), stage, follow, self.quality, int(time.time()) % 10_000, self.fps,
                       stages=stages, allow_untrained=True)
        for q in self.view_opts.values():    # keep the player's zoom and screen size across runs
            sp.control(q)
        threading.Thread(target=sp.run, daemon=True).start()
        self.sp = sp
        if old is not None:
            old.stopped = True

    def view_level(self, lid):
        """Idle view of a level: your best car, or NO live view if never trained here."""
        lv = BY_ID[lid]
        self.level_id = lid
        p = self.progress.get(str(lid), {})
        best = p.get("best_run")
        if self.sp is not None:
            self.sp.stopped = True
            self.sp = None
        if p.get("attempts") and best and (Path(best) / "model.zip").exists():
            self._set_spectator(Path(best), [lv["test"]], follow=False, stage=0)

    def preview_png(self, lid):
        """A still map of the level (cached)."""
        if lid not in self.previews:
            from PIL import Image

            from autodrive.env import AutoDriveEnv
            from autodrive.render import Renderer
            env = AutoDriveEnv(stages=[BY_ID[lid]["test"]], replay_prob=0.0)
            env.reset(seed=4242 + lid, options={"stage": 0})
            buf = io.BytesIO()
            Image.fromarray(Renderer().overview(env)).save(buf, "PNG")
            self.previews[lid] = buf.getvalue()
        return self.previews[lid]

    # ----------------------------------------------------- custom reward library
    def library_save(self, item):
        """Add or replace a custom reward (live link: levels use the latest version)."""
        item = {k: item.get(k) for k in ("id", "name", "when", "give", "times", "paid")}
        item["id"] = item["id"] or uuid.uuid4().hex[:10]
        parse_config({"format": 2, "custom": [item]}, "your custom reward", allow_empty=True)  # validate
        self.library = [x for x in self.library if x["id"] != item["id"]] + [item]
        _save(LIBRARY, self.library)
        return item

    def library_delete(self, rid):
        self.library = [x for x in self.library if x["id"] != rid]
        _save(LIBRARY, self.library)

    # ----------------------------------------------------- training
    def build_config(self, lid, recipe):
        """recipe = {terms: {name: points}, custom_ids: [...]} -> validated config."""
        lib = {x["id"]: x for x in self.library}
        user = {"format": 2, "name": f"level{lid}",
                "terms": {k: {"enabled": True, "weight": float(v)} for k, v in recipe.get("terms", {}).items()
                          if k in TERMS and k in BY_ID[lid]["terms"]},          # only this level's rewards
                "custom": [{k: lib[i][k] for k in ("name", "when", "give", "times", "paid")}
                           for i in recipe.get("custom_ids", []) if i in lib]}
        return parse_config(user, "your design", allow_empty=True)

    def start(self, lid, recipe, minutes, init_prev):
        with self.lock:
            if self.phase in ("training", "testing"):
                return "Already training."
            lv = BY_ID[lid]
            if not lv["available"] or not unlocked(self.progress, lid):
                return "This level is locked."
            try:
                cfg = self.build_config(lid, recipe)
            except (RewardConfigError, ValueError, TypeError) as e:
                return str(e)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run = RUNS / f"L{lid}_{stamp}"
            run.mkdir(parents=True, exist_ok=True)
            # full snapshot of the rewards used (custom rewards included)
            (run / "my_reward.yaml").write_text(yaml.safe_dump(to_yaml_dict(cfg), sort_keys=False))
            cmd = [sys.executable, str(ROOT / "train.py"), "--reward", str(run / "my_reward.yaml"),
                   "--name", f"game/{run.name}", "--level", str(lid), "--minutes", str(minutes),
                   "--ckpt-every", "40000"]
            prev = self.progress.get(str(lid - 1), {}).get("best_run")
            if init_prev and prev and (Path(prev) / "model.zip").exists():
                cmd += ["--init-from", prev]
            self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, bufsize=1)
            self.phase, self.level_id, self.run_dir = "training", lid, run
            self.t_start, self.minutes, self.recipe, self.cfg = time.time(), float(minutes), recipe, cfg
            self.result, self.log_tail = None, []
            self.test_done = self.test_total = 0
            p = self.progress.setdefault(str(lid), {})
            p["attempts"] = p.get("attempts", 0) + 1          # the live view is now "unlocked"
            _save(PROGRESS, self.progress)
        self._set_spectator(run, lv["curriculum"], follow=True, stage=0)
        threading.Thread(target=self._watch_training, args=(lv, run), daemon=True).start()
        return "ok"

    def stop(self):
        proc, run = self.proc, self.run_dir
        if not (proc and proc.poll() is None and run):
            return
        # ask train.py to stop through a file: it finishes the current update,
        # saves the car and exits (proc.terminate() kills it at once on Windows)
        (run / "STOP").write_text("stop")

        def hard_stop():                   # last resort if it does not exit in time
            try:
                proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                proc.terminate()
        threading.Thread(target=hard_stop, daemon=True).start()

    def _watch_training(self, lv, run):
        for line in self.proc.stdout:
            line = line.rstrip()
            if line.startswith(("[", "===")) and "rewards switched on" not in line:
                self.log_tail = (self.log_tail + [line.strip("= ")])[-12:]
        self.proc.wait()
        (run / "STOP").unlink(missing_ok=True)
        ck = run / "checkpoints"
        if not (run / "model.zip").exists() and (ck / "model.zip").exists() and (ck / "vecnormalize.pkl").exists():
            # killed before the final save: test the last checkpoint instead
            shutil.copy(ck / "model.zip", run / "model.zip")
            shutil.copy(ck / "vecnormalize.pkl", run / "vecnormalize.pkl")
            self.log_tail = (self.log_tail + ["stopped early - using the last saved checkpoint"])[-12:]
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
        res["hints"] = hints(lv, self.cfg, res, rows)
        res["minutes"] = round((time.time() - self.t_start) / 60, 1)
        p = self.progress.setdefault(str(lv["id"]), {})
        res["new_best"] = res["score"] > p.get("best", -1)
        if res["new_best"]:
            p.update(best=res["score"], best_run=str(run), best_recipe=self.recipe)
        p["stars"] = max(p.get("stars", 0), res["stars"])
        _save(PROGRESS, self.progress)
        json.dump(res, open(run / "test_result.json", "w"), indent=1)
        self.result = res
        self.phase = "done"

    # ----------------------------------------------------- state for the page
    def state(self):
        lv = BY_ID[self.level_id]
        busy = self.phase in ("training", "testing")
        rows = read_progress(self.run_dir) if self.run_dir and self.phase != "idle" else []
        stage_name = None
        if busy and self.run_dir and (self.run_dir / "curriculum.json").exists():
            try:
                stage_name = lv["curriculum"][json.load(open(self.run_dir / "curriculum.json"))["stage"]]["name"]
            except Exception:
                pass
        if busy and rows and stage_name is None:
            stage_name = lv["curriculum"][int(rows[-1]["stage"])]["name"]
        return {
            "phase": self.phase, "level": self.level_id, "live": self.sp is not None,
            "elapsed": round(time.time() - self.t_start, 1) if self.phase == "training" else 0,
            "minutes": self.minutes, "stage": stage_name,
            "curve": [[float(r["wall_s"]), float(r["success_rate_100"])] for r in rows][-300:],
            "steps": int(rows[-1]["timesteps"]) if rows else 0,
            "test": [self.test_done, self.test_total], "result": self.result, "log": self.log_tail[-6:],
            "viewer": self.sp.policy_src if self.sp else None,
            "progress": self.progress, "library": self.library,
        }


def read_progress(run):
    f = Path(run) / "progress.csv" if run else None
    if not f or not f.exists():
        return []
    try:
        return list(csv.DictReader(open(f)))
    except Exception:
        return []


def catalog():
    """Everything the page needs to build the workshop (no default numbers)."""
    return {
        "debug": DEBUG,
        "levels": [{"id": lv["id"], "name": lv["name"], "brief": lv["brief"], "available": lv["available"],
                    "terms": lv["terms"],
                    "new_terms": [t for t in lv["terms"] if lv["id"] > 1 and t not in BY_ID[lv["id"] - 1]["terms"]],
                    "score": lv["score"], "rules": bool(lv.get("rules")), "minutes": lv["minutes"],
                    "stars": list(lv["stars"])} for lv in LEVELS],
        "terms": [{"name": k, "paid": "once" if TERMS[k][1] == "event" else "every step",
                   "help": HELP[k]} for k in TERMS],
        "measurements": [{"name": k, "label": v[0], "unit": v[1], "type": v[2], "group": v[3]}
                         for k, v in MEASUREMENTS.items()],
        "operators": list(OPERATORS),
    }


# =====================================================================
# HTTP
# =====================================================================
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
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"status": "bad request"}, 400)
            try:
                if u.path == "/api/train":
                    msg = game.start(int(body["level"]), body.get("recipe", {}),
                                     float(np.clip(float(body["minutes"]), 0.5, 120)), bool(body.get("init_prev")))
                    self._json({"status": msg})
                elif u.path == "/api/stop":
                    game.stop()
                    self._json({"status": "ok"})
                elif u.path == "/api/view":
                    if game.phase in ("idle", "done"):
                        game.phase = "idle"
                        game.result = None
                        game.view_level(int(body["level"]))
                    self._json({"status": "ok"})
                elif u.path == "/api/library/save":
                    self._json({"status": "ok", "item": game.library_save(body["item"])})
                elif u.path == "/api/library/delete":
                    game.library_delete(body["id"])
                    self._json({"status": "ok"})
                else:
                    self._json({"error": "unknown"}, 404)
            except (RewardConfigError, KeyError, ValueError, TypeError) as e:
                self._json({"status": str(e)})

        def _mjpeg(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sp, last = None, -1
            try:
                while True:
                    if game.sp is not sp:              # viewer switched (new run / level / none)
                        if sp is not None:
                            sp.watchers -= 1
                        sp, last = game.sp, -1
                        if sp is not None:
                            sp.watchers += 1
                    if sp is None:
                        time.sleep(0.3)
                        continue
                    with sp.cond:
                        sp.cond.wait_for(lambda: sp.frame_id != last or game.sp is not sp, timeout=5)
                        jpg, last = sp.jpeg, sp.frame_id
                    if jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            finally:
                if sp is not None:
                    sp.watchers -= 1

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sp, last = None, 0
            while True:
                if game.sp is not sp:
                    sp = game.sp
                    last = sp.event_id if sp is not None else 0
                if sp is None:
                    time.sleep(1)
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                with sp.cond:
                    sp.cond.wait_for(lambda: sp.event_id != last or game.sp is not sp, timeout=15)
                    new = [(i, d) for i, d in sp.events if i > last]
                    last = sp.event_id
                if not new:
                    self.wfile.write(b": keep-alive\n\n")
                for _, d in new:
                    self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
                self.wfile.flush()

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/":
                    self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif u.path == "/api/catalog":
                    self._json(catalog())
                elif u.path == "/api/state":
                    self._json(game.state())
                elif u.path == "/api/preview.png":
                    self._send(200, game.preview_png(int(q.get("level", ["1"])[0])), "image/png")
                elif u.path == "/control":
                    cmd = q.get("cmd", [""])[0]
                    if cmd in ("zoom", "size"):          # remembered for every later live view
                        game.view_opts[cmd] = q
                    self._json(game.sp.control(q) if game.sp else {})
                elif u.path == "/stream.mjpg":
                    self._mjpeg()
                elif u.path == "/events":
                    self._events()
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
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--debug", action="store_true",
                    help="debug mode: unlock every level to try it (progress is still saved)")
    args = ap.parse_args()
    global DEBUG
    DEBUG = args.debug
    RUNS.mkdir(parents=True, exist_ok=True)
    game = Game(args.fps, args.quality)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(game))
    srv.daemon_threads = True
    print(f"[game] Reward Designer running: open http://localhost:{args.port}   (Ctrl+C to quit)"
          + ("\n[game] DEBUG MODE: all levels unlocked" if DEBUG else ""))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        game.stop()
        print("\n[game] bye")


if __name__ == "__main__":
    main()
