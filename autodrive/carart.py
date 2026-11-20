"""High-resolution top-down car sprites, drawn procedurally.

Every car is drawn nose-up at 4x the final size, shaded (rounded body,
glass reflections, panel seams, lights, mirrors) and smooth-scaled down, so
edges are anti-aliased and details stay crisp when the view is zoomed in.
No image files are needed.
"""
import numpy as np
import pygame

SS = 4                   # supersampling factor
OUT_H = 256              # final sprite height in pixels (car length)
ASPECT = 0.48            # sprite width / length (matches the renderer)

GLASS = (28, 36, 48)
GLASS_HI = (92, 118, 142)
TRIM = (24, 25, 28)
HEAD = (246, 244, 226)
TAIL = (196, 28, 34)
TAIL_HI = (255, 96, 90)

# body styles: proportions along the car (0 = nose, 1 = tail) and width
STYLES = {
    #            body_w  nose_r tail_r  wind0  wind1  roof1  rear1  cabin_w
    "sedan":    (0.86,   0.30,  0.24,   0.30,  0.42,  0.70,  0.80,  0.78),
    "hatch":    (0.86,   0.30,  0.20,   0.28,  0.40,  0.76,  0.86,  0.80),
    "suv":      (0.90,   0.22,  0.14,   0.26,  0.36,  0.80,  0.90,  0.84),
    "van":      (0.90,   0.18,  0.10,   0.12,  0.22,  0.92,  0.96,  0.86),
    "pickup":   (0.90,   0.22,  0.10,   0.24,  0.34,  0.52,  0.56,  0.84),
    "sport":    (0.88,   0.36,  0.26,   0.34,  0.48,  0.66,  0.78,  0.72),
}


def _rrect(S, col, rect, rt, rb):
    x, y, w, h = [int(v) for v in rect]
    pygame.draw.rect(S, col, (x, y, w, h), border_top_left_radius=int(rt), border_top_right_radius=int(rt),
                     border_bottom_left_radius=int(rb), border_bottom_right_radius=int(rb))


def _shade(S, mask, body_x0, body_w, strength=1.0):
    """Cylinder shading across the width + a soft highlight, only on `mask` pixels."""
    arr = pygame.surfarray.pixels3d(S)
    W, H = arr.shape[:2]
    x = (np.arange(W) - (body_x0 + body_w / 2)) / (body_w / 2)
    x = np.clip(np.abs(x), 0, 1)
    prof = 1.10 - 0.42 * x ** 2.2                   # bright crown, darker flanks
    hi = 0.10 * np.exp(-((np.arange(W) - (body_x0 + body_w * 0.36)) / (body_w * 0.07)) ** 2)
    y = np.arange(H) / H
    yprof = 1.0 - 0.10 * (np.abs(y - 0.5) * 2) ** 3   # ends slightly darker
    k = (prof[:, None] * yprof[None, :] + hi[:, None]) ** strength
    a = arr.astype(np.float32)
    a = np.where(mask[..., None], np.clip(a * k[..., None], 0, 255), a)
    arr[:] = a.astype(np.uint8)
    del arr


def _glass(S, poly, w):
    """Dark glass with a diagonal reflection band."""
    pygame.draw.polygon(S, GLASS, poly)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    band = pygame.Surface(S.get_size(), pygame.SRCALPHA)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    hgt = y1 - y0
    refl = [(x0 + (x1 - x0) * 0.18, y1), (x0 + (x1 - x0) * 0.42, y1),
            (x0 + (x1 - x0) * 0.62, y0), (x0 + (x1 - x0) * 0.38, y0)]
    pygame.draw.polygon(band, (*GLASS_HI, 70), refl)
    pygame.draw.polygon(band, (*GLASS_HI, 34), [(refl[1][0] + w * 0.02, y1), (refl[1][0] + w * 0.07, y1),
                                                (refl[2][0] + w * 0.07, y0), (refl[2][0] + w * 0.02, y0)])
    clip = pygame.Surface(S.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(clip, (255, 255, 255, 255), poly)
    band.blit(clip, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    S.blit(band, (0, 0))
    pygame.draw.polygon(S, TRIM, poly, max(2, int(w * 0.012)))
    return hgt


def make_car(color, style="sedan", glass_roof=False, taxi=False, out_h=OUT_H):
    """Return a nose-up RGBA car sprite of height `out_h` (and width out_h*ASPECT)."""
    bw, nose_r, tail_r, w0, w1, r1, rr1, cw = STYLES[style]
    H = out_h * SS
    W = int(round(out_h * ASPECT)) * SS
    S = pygame.Surface((W, H), pygame.SRCALPHA)
    col = tuple(int(c) for c in color)
    dark = tuple(int(c * 0.55) for c in col)
    body_w = W * bw
    bx = (W - body_w) / 2
    # ---- mirrors (behind the body) ----
    my = H * (w0 + 0.03)
    for side in (-1, 1):
        cx = W / 2 + side * (body_w / 2 + W * 0.03)
        pygame.draw.ellipse(S, dark, (cx - W * 0.055, my - H * 0.018, W * 0.11, H * 0.04))
    # ---- body ----
    _rrect(S, dark, (bx, 0, body_w, H), body_w * nose_r * 1.4, body_w * tail_r * 1.4)
    inset = W * 0.014
    _rrect(S, col, (bx + inset, inset, body_w - 2 * inset, H - 2 * inset),
           body_w * nose_r * 1.35, body_w * tail_r * 1.35)
    arr_a = pygame.surfarray.pixels_alpha(S)
    body_mask = arr_a > 0
    del arr_a
    _shade(S, body_mask, bx, body_w)
    # ---- cabin ----
    cab_w = body_w * cw
    cx0 = (W - cab_w) / 2
    # windshield: trapezoid, wider at the bottom
    ws = [(cx0 + cab_w * 0.08, H * w0), (cx0 + cab_w * 0.92, H * w0),
          (cx0 + cab_w * 0.99, H * w1), (cx0 + cab_w * 0.01, H * w1)]
    _glass(S, ws, W)
    # roof
    roof = (cx0, H * w1, cab_w, H * (r1 - w1))
    if glass_roof:
        _glass(S, [(roof[0] + cab_w * 0.04, roof[1] + H * 0.006), (roof[0] + cab_w * 0.96, roof[1] + H * 0.006),
                   (roof[0] + cab_w * 0.96, roof[1] + roof[3] - H * 0.006),
                   (roof[0] + cab_w * 0.04, roof[1] + roof[3] - H * 0.006)], W)
    else:
        _rrect(S, tuple(min(255, int(c * 1.04)) for c in col), roof, cab_w * 0.08, cab_w * 0.08)
        pygame.draw.rect(S, dark, roof, max(2, int(W * 0.01)), border_radius=int(cab_w * 0.08))
    if style == "suv":                      # roof rails
        for side in (0.10, 0.86):
            pygame.draw.rect(S, TRIM, (roof[0] + cab_w * side, roof[1] + H * 0.02, cab_w * 0.04, roof[3] - H * 0.04),
                             border_radius=int(W * 0.01))
    # side windows (thin dark strips along the cabin)
    for side in (-1, 1):
        x = W / 2 + side * cab_w / 2
        strip = [(x, H * w1), (x - side * cab_w * 0.10, H * w1), (x - side * cab_w * 0.10, H * r1),
                 (x, H * r1)]
        pygame.draw.polygon(S, GLASS, strip)
    # rear window
    if style == "pickup":
        rw = [(cx0 + cab_w * 0.04, H * r1), (cx0 + cab_w * 0.96, H * r1),
              (cx0 + cab_w * 0.92, H * rr1), (cx0 + cab_w * 0.08, H * rr1)]
        _glass(S, rw, W)
        # cargo bed
        bed = (bx + body_w * 0.08, H * 0.60, body_w * 0.84, H * 0.36)
        pygame.draw.rect(S, tuple(int(c * 0.42) for c in col), bed, border_radius=int(W * 0.02))
        pygame.draw.rect(S, dark, bed, max(2, int(W * 0.014)), border_radius=int(W * 0.02))
        for i in range(1, 5):
            yy = bed[1] + bed[3] * i / 5
            pygame.draw.line(S, tuple(int(c * 0.34) for c in col), (bed[0] + W * 0.02, yy),
                             (bed[0] + bed[2] - W * 0.02, yy), max(1, int(W * 0.008)))
    else:
        rw = [(cx0 + cab_w * 0.02, H * r1), (cx0 + cab_w * 0.98, H * r1),
              (cx0 + cab_w * 0.90, H * rr1), (cx0 + cab_w * 0.10, H * rr1)]
        _glass(S, rw, W)
    # ---- panel seams: hood line, door lines, trunk line ----
    seam = tuple(int(c * 0.62) for c in col)
    lw = max(2, int(W * 0.008))
    pygame.draw.line(S, seam, (bx + body_w * 0.16, H * (w0 - 0.012)), (bx + body_w * 0.84, H * (w0 - 0.012)), lw)
    for side in (-1, 1):
        x = W / 2 + side * body_w * 0.47
        for yy in (w1 + (r1 - w1) * 0.45,):
            pygame.draw.line(S, seam, (x, H * yy), (x - side * body_w * 0.05, H * yy), lw)
        # hood creases
        hx = W / 2 + side * body_w * 0.22
        pygame.draw.line(S, tuple(min(255, int(c * 1.08)) for c in col), (hx, H * 0.05), (hx, H * (w0 - 0.03)), lw)
    if style not in ("van", "pickup"):
        pygame.draw.line(S, seam, (bx + body_w * 0.18, H * (rr1 + 0.015)), (bx + body_w * 0.82, H * (rr1 + 0.015)), lw)
    # ---- lights ----
    for side in (-1, 1):
        hx = W / 2 + side * body_w * 0.25
        hy = H * (0.016 + 0.03 * nose_r)
        pygame.draw.ellipse(S, TRIM, (hx - body_w * 0.10, hy, body_w * 0.20, H * 0.030))
        pygame.draw.ellipse(S, HEAD, (hx - body_w * 0.085, hy + H * 0.005, body_w * 0.17, H * 0.020))
        tx = W / 2 + side * body_w * 0.27
        ty = H * (0.962 - 0.03 * tail_r)
        pygame.draw.rect(S, TAIL, (tx - body_w * 0.11, ty, body_w * 0.22, H * 0.022),
                         border_radius=int(W * 0.02))
        pygame.draw.rect(S, TAIL_HI, (tx - body_w * 0.06, ty + H * 0.005, body_w * 0.10, H * 0.009),
                         border_radius=int(W * 0.01))
    if taxi:                                 # roof sign
        sx, sy = W / 2, H * (w1 + (r1 - w1) * 0.35)
        pygame.draw.rect(S, TRIM, (sx - cab_w * 0.30, sy - H * 0.026, cab_w * 0.60, H * 0.052), border_radius=int(W * 0.02))
        pygame.draw.rect(S, (250, 236, 170), (sx - cab_w * 0.27, sy - H * 0.019, cab_w * 0.54, H * 0.038),
                         border_radius=int(W * 0.015))
    # final soft shading over everything (roof, glass) so the car reads as rounded
    a = pygame.surfarray.pixels_alpha(S)
    m = a > 0
    del a
    _shade(S, m, bx, body_w, strength=0.55)
    return pygame.transform.smoothscale(S, (W // SS, H // SS))


# NPC fleet: realistic, slightly muted paints
NPC_FLEET = [
    ((232, 232, 228), "sedan"), ((168, 172, 178), "suv"), ((34, 36, 40), "sedan"),
    ((44, 70, 120), "hatch"), ((150, 34, 36), "sedan"), ((48, 84, 64), "suv"),
    ((96, 104, 116), "pickup"), ((210, 200, 176), "van"), ((226, 176, 30), "sedan"),
    ((70, 120, 170), "sport"), ((120, 34, 40), "hatch"), ((236, 236, 232), "van"),
]


_CACHE = {}


def npc_fleet(out_h=OUT_H):
    if ("npc", out_h) not in _CACHE:          # drawn once per process
        _CACHE["npc", out_h] = [make_car(c, s, taxi=(i == 8), out_h=out_h) for i, (c, s) in enumerate(NPC_FLEET)]
    return _CACHE["npc", out_h]


def agent_car(out_h=OUT_H):
    """The learning car: pearl white sedan with a glass roof."""
    if ("agent", out_h) not in _CACHE:
        _CACHE["agent", out_h] = make_car((240, 242, 246), "sedan", glass_roof=True, out_h=out_h)
    return _CACHE["agent", out_h]
