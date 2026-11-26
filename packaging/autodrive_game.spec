# PyInstaller recipe for the one-folder game: dist/AutodriveGame/AutodriveGame(.exe)
#   pyinstaller packaging/autodrive_game.spec --noconfirm
# Build it on each OS you want to ship (Windows / macOS / Linux); see
# .github/workflows/build-game.yml, which does exactly that on GitHub.
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = [
    (os.path.join(ROOT, "assets"), "assets"),
    (os.path.join(ROOT, "rewards"), "rewards"),
    (os.path.join(ROOT, "rewards.yaml"), "."),
] + collect_data_files("stable_baselines3")       # version.txt
hidden = (collect_submodules("autodrive") + ["train", "game", "watch"]
          + collect_submodules("stable_baselines3") + collect_submodules("gymnasium.envs.registration"))

a = Analysis(
    [os.path.join(ROOT, "play.py")],
    pathex=[ROOT],
    datas=datas,
    hiddenimports=hidden,
    # not needed by the game, keeps the download smaller
    excludes=["tensorboard", "torch.utils.tensorboard", "IPython", "jupyter", "notebook", "pytest",
              "tkinter", "imageio_ffmpeg"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AutodriveGame",
          console=True,            # the window shows the game address and can be closed to quit
          icon=None)
coll = COLLECT(exe, a.binaries, a.datas, name="AutodriveGame")
