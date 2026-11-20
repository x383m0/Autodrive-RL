"""Top-down pedestrians drawn procedurally, in the same style as the cars.

A person seen from above: shoulders (shirt), head (hair, a sliver of face at
the front), arms and legs that swing while walking. Kids (the ones that run
out from behind parked cars) are smaller and wear a bright school backpack.
Each sprite faces UP (the renderer rotates it to the walking direction) and
is drawn at SS x the final size, then smooth-scaled down.
"""
import math

import numpy as np
import pygame

SS = 3
OUT = 72                 # sprite size in pixels (square); covers CANVAS_M metres
CANVAS_M = 1.0
N_FRAMES = 8             # walk cycle; frame -1 = standing

SHIRTS = [(70, 110, 170), (178, 64, 58), (64, 132, 92), (222, 190, 70), (120, 84, 150), (226, 226, 220),
          (54, 58, 66), (210, 120, 60), (60, 150, 160), (150, 150, 150), (190, 90, 130), (90, 120, 70)]
HAIR = [(30, 26, 24), (88, 58, 36), (196, 160, 96), (40, 36, 34), (130, 70, 40), (170, 170, 170)]
SKIN = [(236, 200, 170), (200, 150, 110), (150, 100, 70), (100, 66, 46), (222, 180, 140)]
PANTS = [(46, 52, 70), (60, 60, 64), (90, 80, 64), (40, 70, 110)]
BACKPACKS = [(255, 120, 40), (60, 180, 255), (250, 210, 40), (240, 70, 120)]
_CACHE = {}


def _mul(c, k):
    return tuple(int(np.clip(v * k, 0, 255)) for v in c)


def make_person(variant, frame=-1, kid=False, out=OUT):
    """RGBA sprite of size out x out. frame: 0..N_FRAMES-1 (walking) or -1 (standing)."""
    rng = np.random.default_rng(variant)
    shirt = SHIRTS[variant % len(SHIRTS)]
    hair = HAIR[int(rng.integers(len(HAIR)))]
    skin = SKIN[int(rng.integers(len(SKIN)))]
    pants = PANTS[int(rng.integers(len(PANTS)))]
    N = out * SS
    S = pygame.Surface((N, N), pygame.SRCALPHA)
    m = N / CANVAS_M * (0.78 if kid else 1.0)       # pixels per metre (kids are smaller)
    cx, cy = N / 2, N / 2

    def E(col, x, y, w, h):                         # ellipse centred at (x, y) metres from the centre
        pygame.draw.ellipse(S, col, (cx + (x - w / 2) * m, cy + (y - h / 2) * m, w * m, h * m))

    swing = 0.0 if frame < 0 else math.sin(2 * math.pi * frame / N_FRAMES)
    # legs (under everything): one forward, one back
    for s in (-1, 1):
        dy = -s * swing * 0.09
        E(_mul(pants, 0.8), s * 0.08, 0.02 + dy, 0.11, 0.20)
        E((34, 32, 30), s * 0.08, -0.05 + dy * 1.4, 0.09, 0.08)        # shoe
    # arms swing opposite to the legs
    for s in (-1, 1):
        dy = s * swing * 0.06
        E(_mul(shirt, 0.72), s * 0.225, 0.01 + dy, 0.10, 0.20)
        E(skin, s * 0.225, -0.075 + dy * 1.4, 0.075, 0.075)            # hand
    # backpack (kids) sits behind the shoulders
    if kid:
        bp = BACKPACKS[variant % len(BACKPACKS)]
        pygame.draw.rect(S, _mul(bp, 0.7), (cx - 0.17 * m, cy + 0.04 * m, 0.34 * m, 0.20 * m), border_radius=int(0.06 * m))
        pygame.draw.rect(S, bp, (cx - 0.15 * m, cy + 0.05 * m, 0.30 * m, 0.16 * m), border_radius=int(0.05 * m))
    # shoulders / torso
    E(_mul(shirt, 0.62), 0, 0, 0.50, 0.28)
    E(shirt, 0, -0.005, 0.46, 0.24)
    E(_mul(shirt, 1.15), -0.06, -0.03, 0.22, 0.10)                     # highlight
    # head: face sliver at the front, hair on top
    E(_mul(skin, 0.85), 0, -0.03, 0.24, 0.25)                          # face / neck edge
    E(hair, 0, 0.0, 0.225, 0.215)
    E(_mul(hair, 1.35), -0.03, -0.03, 0.07, 0.06)                      # hair shine
    return pygame.transform.smoothscale(S, (out, out))


def person(variant, frame=-1, kid=False):
    """Cached sprite."""
    k = (variant % len(SHIRTS), frame, kid)
    if k not in _CACHE:
        _CACHE[k] = make_person(*k)
    return _CACHE[k]
