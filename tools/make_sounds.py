#!/usr/bin/env python
"""Write the synthesised fallback sounds (horn, brake squeal, city ambience)
into assets/sounds/ so they are visible and easy to replace with your own.

    python tools/make_sounds.py          # only writes files that are missing
    python tools/make_sounds.py --force  # regenerate them
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autodrive.audio import SOUNDS, synth_ambient, synth_brake, synth_horn, write_wav  # noqa: E402
from autodrive.video import ffmpeg_exe  # noqa: E402

force = "--force" in sys.argv
for name, fn in (("horn", synth_horn), ("brake_squeal", synth_brake), ("ambient_city", synth_ambient)):
    ogg = SOUNDS / f"{name}.ogg"
    if ogg.exists() and not force:
        print("exists:", ogg.name)
        continue
    wav = SOUNDS / f"{name}.tmp.wav"
    write_wav(wav, fn())
    subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-i", str(wav), "-c:a", "libvorbis", "-q:a", "4",
                    str(ogg)], check=True)
    wav.unlink()
    print("wrote:", ogg.name)
