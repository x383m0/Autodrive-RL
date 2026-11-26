"""
play.py - the entry point of the packaged game (AutodriveGame.exe and friends).

Double-click it (or run `python play.py`): it starts the Reward Designer and
opens it in the web browser. The same program also runs a training job when
the game asks for one (`--train ...`), because a packaged app has no separate
Python to run train.py with.

    play.py                 start the game, open the browser
    play.py --debug         ... with every level unlocked
    play.py --train ARGS    run train.py with ARGS (used by the game itself)
    play.py --selftest      quick check that the packaged app works (CI)
"""
import multiprocessing
import sys


def _selftest():
    """Start the game, ask it for its level list, train for a few seconds, test the car."""
    import json
    import shutil
    import subprocess
    import tempfile
    import threading
    import time
    import urllib.request
    from pathlib import Path

    import os
    tmp = Path(tempfile.mkdtemp(prefix="autodrive_selftest_"))
    os.environ["AUTODRIVE_DATA"] = str(tmp)
    import game
    ready = threading.Event()
    box = {}

    def on_ready(url, srv):
        box["url"], box["srv"] = url, srv
        ready.set()
    threading.Thread(target=game.main, args=(["--port", "8765"], on_ready), daemon=True).start()
    assert ready.wait(60), "game did not start"
    cat = json.loads(urllib.request.urlopen(box["url"] + "/api/catalog", timeout=30).read())
    print("[selftest] game is up:", len(cat["levels"]), "levels")
    from autodrive.paths import train_command
    cmd = train_command() + ["--reward", str(Path(game.ROOT) / "rewards.yaml"), "--name", "selftest",
                             "--level", "1", "--minutes", "0.3", "--n-envs", "2"]
    t = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(out.stdout[-800:], out.stderr[-1500:])
    run = tmp / "experiments" / "selftest"
    assert (run / "model.zip").exists(), "training produced no model"
    print(f"[selftest] training ok ({time.time() - t:.0f} s)")
    from autodrive.levels import BY_ID, run_test
    r = run_test(run, BY_ID[1], n=2)
    print("[selftest] test drives ok, score", round(r["score"]))
    box["srv"].shutdown()
    shutil.rmtree(tmp, ignore_errors=True)
    print("[selftest] PASSED")


def main():
    multiprocessing.freeze_support()          # training workers re-run this program
    args = sys.argv[1:]
    if args[:1] == ["--train"]:
        sys.argv = [sys.argv[0]] + args[1:]
        import train
        train.main()
        return
    if args[:1] == ["--selftest"]:
        _selftest()
        return
    import webbrowser
    import game

    def on_ready(url, srv):
        print(f"\n  The game is running. If your browser did not open, go to {url}\n"
              "  Close this window to quit.\n")
        webbrowser.open(url)
    game.main(args, on_ready)


if __name__ == "__main__":
    main()
