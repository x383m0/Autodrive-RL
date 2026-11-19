"""
audio.py - sound effects for exported videos (offline mixing, no sound card).

There is no audio device in a headless container (and sound is off during
training anyway), so instead of playing sounds live we RECORD what happened
in every video frame (speed, throttle, braking, blinker, events) and mix a
soundtrack afterwards with numpy:

    engine      looping sample, played faster (= higher pitch) with speed
    brakes      squeal loop, louder the harder we brake at speed
    horn        NPCs honk when the agent blocks them for no reason
    crash       metal + glass impact
    ped_hit     a dull thud
    turn_tick   the agent's blinker relay
    light ping  when the next traffic light ahead changes colour
    chime       parked successfully
    ambient     city hum + wind, looping quietly underneath

Real recordings live in assets/sounds/ (Kenney CC0 packs). Anything missing
is SYNTHESISED here (see `synth_*`), so you can delete or replace any file.
`sound_report()` says which sounds are real files and which are synthetic.
"""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

SR = 44100
SOUNDS = Path(__file__).resolve().parent.parent / "assets" / "sounds"

# role -> list of candidate files (first one that loads wins)
FILES = {
    "engine": ["engine_loop.ogg", "engine_loop.wav"],
    "brake": ["brake_squeal.ogg", "brake_squeal.wav"],
    "horn": ["horn.ogg", "horn.wav"],
    "crash_metal": ["crash_metal.ogg", "crash_metal.wav"],
    "crash_glass": ["crash_glass.ogg", "crash_glass.wav"],
    "ped_hit": ["ped_hit.ogg", "ped_hit.wav"],
    "tick": ["turn_tick.ogg", "turn_tick.wav"],
    "light": ["light_change.ogg", "light_change.wav"],
    "chime": ["park_chime.ogg", "park_chime.wav"],
    "ambient": ["ambient_city.ogg", "ambient_city.wav"],
    "wind": ["wind_loop.ogg", "wind_loop.wav"],
}


def decode(path: Path) -> np.ndarray | None:
    """Decode any audio file to mono float32 at SR using ffmpeg."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = "ffmpeg"
    try:
        out = subprocess.run([exe, "-v", "quiet", "-i", str(path), "-f", "f32le", "-ac", "1",
                              "-ar", str(SR), "-"], capture_output=True, timeout=30).stdout
    except Exception:
        return None
    x = np.frombuffer(out, np.float32).copy()
    if len(x) < 100 or not np.isfinite(x).all():
        return None
    return x


# ---------------------------------------------------------------- synthesis
def _env(n, attack=0.01, release=0.2):
    t = np.arange(n) / SR
    e = np.minimum(1.0, t / max(attack, 1e-4))
    tail = np.clip((n / SR - t) / max(release, 1e-4), 0, 1)
    return e * tail


def synth_horn(dur=0.55):
    """Classic two-tone car horn (F4 + A4) with a buzzy, slightly detuned timbre."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for f in (349.2, 440.0):
        for h, a in ((1, 1.0), (2, 0.5), (3, 0.35), (4, 0.2), (5, 0.12)):
            x += a * np.sin(2 * np.pi * f * h * t * (1 + 0.002 * np.sin(2 * np.pi * 5 * t)))
    x = np.tanh(1.8 * x / 3)
    return (0.6 * x * _env(n, 0.015, 0.08)).astype(np.float32)


def synth_brake(dur=2.0, seed=1):
    """Tyre/brake squeal: narrow band noise around ~2.6 kHz with wobble (loopable)."""
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 2600 + 180 * np.sin(2 * np.pi * 3.1 * t) + 90 * np.sin(2 * np.pi * 7.3 * t)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) + 0.4 * np.sin(2 * ph)
    x *= 0.6 + 0.4 * np.abs(np.convolve(rng.standard_normal(n), np.ones(400) / 400, "same")) * 3
    x += 0.15 * rng.standard_normal(n)
    # crossfade the ends so it loops cleanly
    k = int(0.05 * SR)
    x[:k] = x[:k] * np.linspace(0, 1, k) + x[-k:] * np.linspace(1, 0, k)
    return (0.25 * x / np.abs(x).max()).astype(np.float32)


def synth_ambient(dur=20.0, seed=2):
    """City ambience: low traffic rumble (brown noise), distant passing cars,
    and the odd far-away horn. Loopable."""
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    brown = np.cumsum(rng.standard_normal(n))
    brown -= np.convolve(brown, np.ones(4410) / 4410, "same")    # remove drift
    brown /= np.abs(brown).max() + 1e-9
    x = 0.5 * brown
    t = np.arange(n) / SR
    for _ in range(int(dur / 3)):                                  # passing cars (whoosh)
        c = rng.uniform(0, dur)
        w = rng.uniform(1.0, 2.5)
        g = np.exp(-((t - c) / w) ** 2)
        noise = np.convolve(rng.standard_normal(n), np.ones(30) / 30, "same")
        x += 0.25 * g * noise / (np.abs(noise).max() + 1e-9)
    for _ in range(2):                                             # far horns
        c = int(rng.uniform(1, dur - 2) * SR)
        h = synth_horn(rng.uniform(0.2, 0.4)) * 0.08
        x[c:c + len(h)] += h[: n - c]
    k = int(0.5 * SR)
    x[:k] = x[:k] * np.linspace(0, 1, k) + x[-k:] * np.linspace(1, 0, k)
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def synth_engine(dur=2.0):
    """Fallback engine: a low pulsed buzz (4-cylinder firing at ~30 Hz)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f0 = 30.0
    x = sum(a * np.sin(2 * np.pi * f0 * h * t) for h, a in ((1, 1), (2, 0.6), (3, 0.4), (4, 0.3), (6, 0.15)))
    x = np.tanh(x)
    return (0.5 * x).astype(np.float32)


def synth_tick():
    n = int(0.03 * SR)
    t = np.arange(n) / SR
    return (0.5 * np.sin(2 * np.pi * 2200 * t) * np.exp(-t * 300)).astype(np.float32)


def synth_ping():
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    return (0.4 * (np.sin(2 * np.pi * 1760 * t) + 0.3 * np.sin(2 * np.pi * 3520 * t)) * np.exp(-t * 9)).astype(np.float32)


def synth_chime():
    out = np.zeros(int(0.9 * SR), np.float32)
    for i, f in enumerate((784.0, 988.0, 1175.0)):
        n = int(0.5 * SR)
        t = np.arange(n) / SR
        s = 0.35 * np.sin(2 * np.pi * f * t) * np.exp(-t * 6)
        o = int(i * 0.12 * SR)
        out[o:o + n] += s
    return out


def synth_thud():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(3)
    x = np.sin(2 * np.pi * 70 * t * (1 - t)) * np.exp(-t * 14) + 0.2 * rng.standard_normal(n) * np.exp(-t * 40)
    return (0.8 * x).astype(np.float32)


def synth_crash():
    n = int(0.8 * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(4)
    x = rng.standard_normal(n) * np.exp(-t * 6) + 0.6 * np.sin(2 * np.pi * 90 * t) * np.exp(-t * 10)
    return (0.6 * x / np.abs(x).max()).astype(np.float32)


SYNTH = {"engine": synth_engine, "brake": synth_brake, "horn": synth_horn, "crash_metal": synth_crash,
         "crash_glass": synth_crash, "ped_hit": synth_thud, "tick": synth_tick, "light": synth_ping,
         "chime": synth_chime, "ambient": synth_ambient, "wind": lambda: np.zeros(SR, np.float32)}


# files written by tools/make_sounds.py (synthesised, not recordings)
GENERATED_FILES = {"horn.ogg", "brake_squeal.ogg", "ambient_city.ogg"}


class SoundBank:
    """Loads every role from assets/sounds, synthesising whatever is missing."""

    def __init__(self):
        self.snd, self.source = {}, {}
        for role, names in FILES.items():
            x = None
            for nm in names:
                p = SOUNDS / nm
                if p.exists():
                    x = decode(p)
                    if x is not None:
                        kind = ("synthesised file (tools/make_sounds.py)" if nm in GENERATED_FILES
                                else "recorded file (CC0, see assets/CREDITS.md)")
                        self.source[role] = f"{kind}: assets/sounds/{nm}"
                        break
            if x is None:
                x = SYNTH[role]()
                self.source[role] = "synthesised on the fly (no file found)"
            peak = np.abs(x).max()
            self.snd[role] = (x / peak * 0.9).astype(np.float32) if peak > 0 else x


def sound_report():
    return SoundBank().source


# ---------------------------------------------------------------- mixing
def _place(buf, x, start, gain):
    s = int(start)
    if s >= len(buf) or s + len(x) <= 0:
        return
    a, b = max(0, s), min(len(buf), s + len(x))
    buf[a:b] += gain * x[a - s:b - s]


def _loop_resample(src, rates, out_len):
    """Play `src` in a loop with a time-varying playback rate (pitch)."""
    phase = np.cumsum(rates) % len(src)
    i0 = np.floor(phase).astype(np.int64)
    frac = (phase - i0).astype(np.float32)
    i1 = (i0 + 1) % len(src)
    return src[i0] * (1 - frac) + src[i1] * frac


def mix_soundtrack(frames: list, fps: float, bank: SoundBank | None = None) -> np.ndarray:
    """frames: list of dicts with keys
         speed (m/s), throttle (0..1), brake (0..1), blink (bool, lamp on),
         events (list of (name, distance_m)).
    Returns mono float32 audio at SR covering len(frames)/fps seconds."""
    bank = bank or SoundBank()
    n = int(len(frames) / fps * SR) + SR // 2
    spf = SR / fps
    out = np.zeros(n, np.float32)
    # per-sample control curves (interpolated from per-frame values)
    ft = np.arange(len(frames)) * spf
    st = np.arange(n)

    def curve(key):
        v = np.array([f[key] for f in frames], np.float32)
        return np.interp(st, ft, v).astype(np.float32)

    speed = curve("speed")
    thr = curve("throttle")
    brk = curve("brake")
    # engine: rate 0.55 at idle up to ~1.6 at 20 m/s, a bit higher under load
    rate = 0.55 + speed / 20.0 * 1.0 + 0.12 * thr
    eng = _loop_resample(bank.snd["engine"], rate, n)
    out += eng * (0.16 + 0.10 * thr + 0.04 * speed / 20)
    # brake squeal only when braking hard at speed
    sq = np.clip((brk - 0.45) / 0.55, 0, 1) * np.clip((speed - 3) / 5, 0, 1)
    if sq.max() > 0:
        b = bank.snd["brake"]
        out += np.resize(b, n) * sq * 0.35
    # ambience
    out += np.resize(bank.snd["ambient"], n) * 0.10
    out += np.resize(bank.snd["wind"], n) * 0.06
    # one-shots
    prev_blink = False
    for k, f in enumerate(frames):
        t0 = k * spf
        if f["blink"] != prev_blink:
            _place(out, bank.snd["tick"], t0, 0.25)
        prev_blink = f["blink"]
        for name, dist in f["events"]:
            att = 1.0 / (1.0 + max(dist, 0) / 15.0)         # quieter when far away
            if name == "crash":
                _place(out, bank.snd["crash_metal"], t0, 0.9)
                _place(out, bank.snd["crash_glass"], t0 + 0.03 * SR, 0.6)
            elif name == "ped_hit":
                _place(out, bank.snd["ped_hit"], t0, 0.9)
            elif name == "horn":
                _place(out, bank.snd["horn"], t0, 0.7 * att)
            elif name == "light_change":
                _place(out, bank.snd["light"], t0, 0.35)
            elif name == "chime":
                _place(out, bank.snd["chime"], t0, 0.6)
    # gentle limiter
    out = np.tanh(out * 1.2) * 0.9
    return out.astype(np.float32)


def write_wav(path, x, sr=SR):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
