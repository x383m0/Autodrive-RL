"""High-resolution top-down car models, drawn procedurally (no image files).

Shapes follow real top-down car art: a rounded-rectangle body with fender
bulges and a slim waist, a big curved windshield, thin side windows along
the cabin, a curved rear window, swept headlights and wrap-around tail
lights tucked into the corners, small black mirrors, hood creases and door
seams. Body styles: sedan, hatchback, compact, SUV, van, pickup, sports car
(+ taxi). Every model is drawn at SS x the final size and smooth-scaled down.

Each model is a dict of layers (all nose-up, same size):
  body        RGBA car with lights off
  brake       RGBA overlay: lit brake lights
  blink_L/R   RGBA overlay: lit turn signals (left = -x side, right = +x side)
  glass       mask (alpha) of the glass, weighted (windshield brightest)
  dmg_<c>     damaged body for corner c in FL/FR/RL/RR (made on demand)
  layout      wheel positions etc. in fractions of the sprite size
The renderer composes wheels + body + lit lights + sun reflection per frame.
"""
import math

import numpy as np
import pygame

SS = 3                   # supersampling factor
OUT_H = 256              # sprite height in pixels (car length); 2x the largest zoom need
ASPECT = 0.48            # sprite width / length (the renderer uses the same)
BODY_W = 0.84            # body width / sprite width (the rest is room for mirrors / wheels)

GLASS_FRONT = (46, 58, 72)
GLASS_SIDE = (24, 30, 38)
GLASS_REAR = (34, 42, 54)
TRIM = (20, 21, 24)
MIRROR = (14, 14, 16)
HEAD_OFF = (206, 210, 214)
TAIL_OFF = (128, 22, 28)
TAIL_ON = (255, 40, 40)
TAIL_HOT = (255, 170, 160)
AMBER_OFF = (150, 100, 30)
AMBER_ON = (255, 170, 30)

# Style geometry: v runs along the car (0 = nose, 1 = tail), u across it
# (0 = centre line, 1 = body side). Glass edges are curved: (centre v, corner v, half-span u).
STYLES = {
    "sedan": dict(prof=(.93, .97, 1, .985, .965, .985, 1, .98, .95), rx=.42, ryf=.07, ryr=.06,
                  wf=(.300, .335, .86), wr=(.430, .450, .78), rf=(.705, .685, .78), rr=(.805, .785, .84),
                  hood_lines=2, trunk=True),
    "hatch": dict(prof=(.93, .97, 1, .985, .97, .985, 1, .99, .96), rx=.45, ryf=.075, ryr=.05,
                  wf=(.280, .315, .87), wr=(.415, .435, .80), rf=(.865, .850, .80), rr=(.925, .905, .82),
                  hood_lines=2, spoiler=True),
    "compact": dict(prof=(.92, .97, 1, .98, .965, .98, 1, .985, .96), rx=.48, ryf=.08, ryr=.07,
                    wf=(.240, .285, .88), wr=(.400, .420, .80), rf=(.800, .780, .80), rr=(.910, .885, .84),
                    hood_lines=0, spoiler=True),
    "suv": dict(prof=(.95, .985, 1, .995, .985, .995, 1, .995, .97), rx=.34, ryf=.06, ryr=.045,
                wf=(.250, .280, .88), wr=(.370, .385, .84), rf=(.880, .870, .84), rr=(.940, .930, .86),
                hood_lines=2, rails=True),
    "van": dict(prof=(.95, .99, 1, 1, 1, 1, 1, .995, .98), rx=.30, ryf=.06, ryr=.035,
                wf=(.150, .180, .88), wr=(.275, .290, .86), rf=(.955, .950, .86), rr=(.975, .970, .86),
                hood_lines=0, ribs=True),
    "pickup": dict(prof=(.95, .985, 1, .995, .985, .99, .995, .99, .98), rx=.34, ryf=.06, ryr=.03,
                   wf=(.245, .275, .88), wr=(.345, .360, .84), rf=(.505, .500, .84), rr=(.545, .540, .84),
                   hood_lines=2, bed=True),
    "sport": dict(prof=(.90, .95, 1, .97, .925, .965, 1, .985, .95), rx=.46, ryf=.085, ryr=.07,
                  wf=(.375, .410, .80), wr=(.500, .515, .70), rf=(.715, .700, .66), rr=(.830, .815, .66),
                  hood_lines=2, hood_bulge=True, rear_window_rect=True),
}
PROF_V = (0, .06, .2, .36, .5, .66, .8, .94, 1)


class _Geo:
    """Maps car coordinates (u across, v along) to supersampled pixels."""

    def __init__(self, W, H, st):
        self.W, self.H, self.st = W, H, st
        self.hw = W * BODY_W / 2
        self.cx = W / 2
        self._outline = self._outline_uv()

    def prof(self, v):
        return float(np.interp(v, PROF_V, self.st["prof"]))

    def P(self, u, v, inset=0.0):
        """(u, v) -> pixel; inset (pixels) shrinks the body evenly."""
        y = inset + v * (self.H - 2 * inset)
        x = self.cx + u * (self.hw - inset) * self.prof(v)
        return (x, y)

    def _outline_uv(self, n=14):
        st = self.st
        rx, ryf, ryr = st["rx"], st["ryf"], st["ryr"]
        fa, ra = 0.012, 0.008               # front / rear bumper arcs
        right = []
        for i in range(n + 1):                                  # front edge, centre -> corner
            u = (1 - rx) * i / n
            right.append((u, fa * (u / (1 - rx)) ** 2))
        for i in range(1, n + 1):                               # front corner
            t = math.pi / 2 * i / n
            right.append((1 - rx + rx * math.sin(t), fa + ryf * (1 - math.cos(t))))
        self.k_side_f = len(right) - 1
        for i in range(0, n + 1):                               # rear corner
            t = math.pi / 2 * i / n
            right.append((1 - rx + rx * math.cos(t), 1 - ra - ryr + ryr * math.sin(t)))
        self.k_side_r = len(right) - n - 1
        for i in range(1, n + 1):                               # rear edge, corner -> centre
            u = (1 - rx) * (1 - i / n)
            right.append((u, 1 - ra * (u / (1 - rx)) ** 2))
        self.k_end = len(right) - 1
        left = [(-u, v) for (u, v) in reversed(right[1:-1])]
        return right + left

    def outline(self, inset=0.0):
        return [self.P(u, v, inset) for (u, v) in self._outline]

    def front_path(self):
        return self._outline[:self.k_side_f + 1]

    def rear_path(self):
        return self._outline[self.k_side_r:self.k_end + 1]

    @staticmethod
    def edge(spec, n=16, flip=False):
        """A curved glass edge: list of (u, v) from -span to +span."""
        vc, vk, span = spec
        out = [(-span + 2 * span * i / n, 0.0) for i in range(n + 1)]
        out = [(u, vk + (vc - vk) * (1 - (u / span) ** 2)) for u, _ in out]
        return list(reversed(out)) if flip else out


def _poly(S, col, pts, width=0):
    if len(pts) >= 3:
        pygame.draw.polygon(S, col, [(int(round(x)), int(round(y))) for x, y in pts], width)


def _clip(layer, mask):
    a = pygame.surfarray.pixels_alpha(layer)
    a[~mask] = 0
    del a


def _shade(S, mask, cx, hw, strength=1.0):
    """Rounded-body shading: bright crown, darker flanks and ends, soft highlight."""
    arr = pygame.surfarray.pixels3d(S)
    W, H = arr.shape[:2]
    x = np.clip(np.abs((np.arange(W) - cx) / hw), 0, 1)
    prof = 1.10 - 0.45 * x ** 2.4
    hi = 0.09 * np.exp(-((np.arange(W) - (cx - hw * 0.30)) / (hw * 0.16)) ** 2)
    y = np.arange(H) / H
    yprof = 1.0 - 0.12 * (np.abs(y - 0.5) * 2) ** 4
    k = (prof[:, None] * yprof[None, :] + hi[:, None]) ** strength
    a = arr.astype(np.float32)
    arr[:] = np.where(mask[..., None], np.clip(a * k[..., None], 0, 255), a).astype(np.uint8)
    del arr


def _mul(c, k):
    return tuple(int(np.clip(v * k, 0, 255)) for v in c)


def make_car(color, style="sedan", sunroof=False, glass_roof=False, rack=False, stripes=False,
             black_roof=False, taxi=False, out_h=OUT_H):
    st = STYLES[style]
    H = out_h * SS
    W = int(round(out_h * ASPECT)) * SS
    g = _Geo(W, H, st)
    P, edge = g.P, g.edge
    col = tuple(int(c) for c in color)
    dark, seam = _mul(col, 0.50), _mul(col, 0.66)
    lw = max(2, int(W * 0.010))

    S = pygame.Surface((W, H), pygame.SRCALPHA)
    shape = pygame.Surface((W, H), pygame.SRCALPHA)
    _poly(shape, (255, 255, 255), g.outline())
    body_mask = pygame.surfarray.array_alpha(shape) > 0

    # ---- body: dark rim + paint, then shading ---------------------------------
    _poly(S, dark, g.outline())
    _poly(S, col, g.outline(inset=W * 0.016))
    _shade(S, body_mask, g.cx, g.hw)

    # ---- hood creases, hood bulge, cowl ------------------------------------------
    wf, wr, rf, rr = st["wf"], st["wr"], st["rf"], st["rr"]
    if st.get("hood_lines"):
        for s in (-1, 1):
            a, b = P(s * 0.30, 0.04), P(s * 0.42, wf[1] - 0.02)
            pygame.draw.line(S, _mul(col, 1.12), a, b, lw)
            pygame.draw.line(S, seam, (a[0] + s * lw, a[1]), (b[0] + s * lw, b[1]), max(1, lw // 2))
    if st.get("hood_bulge"):
        hb = [P(-0.26, 0.05), P(0.26, 0.05), P(0.36, wf[1] - 0.03), P(-0.36, wf[1] - 0.03)]
        _poly(S, _mul(col, 1.06), hb)
        _poly(S, seam, hb, lw)
    pygame.draw.lines(S, seam, False, [P(u, v) for u, v in edge((wf[0] - 0.014, wf[1] - 0.014, wf[2] * 0.98))], lw)
    if stripes:
        sc = (34, 35, 38) if sum(col) > 360 else (232, 232, 228)
        for s in (-1, 1):
            _poly(S, sc, [P(s * 0.07, 0.008), P(s * 0.19, 0.008), P(s * 0.19, 0.992), P(s * 0.07, 0.992)])

    # ---- glass --------------------------------------------------------------------
    GL = pygame.Surface((W, H), pygame.SRCALPHA)           # glass weight mask
    ws = [P(u, v) for u, v in edge(wf) + edge(wr, flip=True)]
    _poly(S, GLASS_FRONT, ws)
    _poly(GL, (255, 255, 255, 255), ws)
    _poly(S, TRIM, ws, lw)
    # side windows: strips along the cabin, split by the B-pillar
    v0, v1 = wr[1] - 0.035, rf[1] + 0.02
    for s in (-1, 1):
        strip = [P(s * 0.86, wf[1] + 0.012), P(s * 0.955, v0 + 0.04), P(s * 0.965, v1 - 0.03),
                 P(s * 0.90, v1 + 0.02), P(s * 0.80, v1), P(s * 0.80, v0 + 0.02)]
        _poly(S, GLASS_SIDE, strip)
        _poly(GL, (255, 255, 255, 110), strip)
        bp = v0 + (v1 - v0) * (0.46 if style != "van" else 0.30)
        _poly(S, dark, [P(s * 0.79, bp - 0.011), P(s * 0.975, bp - 0.011), P(s * 0.975, bp + 0.011), P(s * 0.79, bp + 0.011)])
        for dv in (bp, v1 + 0.05):                       # door seams below the window line
            if dv < 0.95:
                pygame.draw.line(S, seam, P(s * 0.975, dv), P(s * 1.0, dv), lw)
    # rear window
    if st.get("rear_window_rect"):
        rw = [P(-0.52, rf[1]), P(0.52, rf[1]), P(0.56, rr[1]), P(-0.56, rr[1])]
    else:
        rw = [P(u, v) for u, v in edge(rf) + edge(rr, flip=True)]
    _poly(S, GLASS_REAR, rw)
    _poly(GL, (255, 255, 255, 190), rw)
    _poly(S, TRIM, rw, lw)

    # ---- roof ------------------------------------------------------------------------
    roof = [P(u, v) for u, v in edge(wr) + edge(rf, flip=True)]
    if glass_roof:
        _poly(S, GLASS_FRONT, roof)
        _poly(GL, (255, 255, 255, 230), roof)
    else:
        _poly(S, (36, 37, 40) if black_roof else _mul(col, 1.03), roof)
    _poly(S, TRIM if (black_roof or glass_roof) else seam, roof, lw)
    r0, r1 = wr[0], rf[0]
    if sunroof and not glass_roof:
        h = min(0.16, (r1 - r0) * 0.45)
        sr = [P(-0.42, r0 + 0.05), P(0.42, r0 + 0.05), P(0.42, r0 + 0.05 + h), P(-0.42, r0 + 0.05 + h)]
        _poly(S, GLASS_SIDE, sr)
        _poly(GL, (255, 255, 255, 200), sr)
        _poly(S, TRIM, sr, lw)
    if st.get("ribs"):
        for u in (-0.45, -0.15, 0.15, 0.45):
            pygame.draw.line(S, seam, P(u, r0 + 0.04), P(u, r1 - 0.03), lw)
            pygame.draw.line(S, _mul(col, 1.12), P(u + 0.03, r0 + 0.04), P(u + 0.03, r1 - 0.03), max(1, lw // 2))
    if st.get("rails") or rack:
        for s in (-1, 1):
            _poly(S, TRIM, [P(s * 0.64, r0 + 0.03), P(s * 0.72, r0 + 0.03), P(s * 0.72, r1 - 0.03), P(s * 0.64, r1 - 0.03)])
        if rack:
            for v in np.linspace(r0 + 0.07, r1 - 0.07, 3):
                pygame.draw.line(S, TRIM, P(-0.68, v), P(0.68, v), lw * 2)
    if taxi:
        tv = r0 + (r1 - r0) * 0.35
        _poly(S, TRIM, [P(-0.42, tv - 0.03), P(0.42, tv - 0.03), P(0.42, tv + 0.03), P(-0.42, tv + 0.03)])
        _poly(S, (250, 236, 170), [P(-0.37, tv - 0.02), P(0.37, tv - 0.02), P(0.37, tv + 0.02), P(-0.37, tv + 0.02)])

    # ---- tail: trunk lid / spoiler / pickup bed -----------------------------------------
    if st.get("trunk"):
        pygame.draw.lines(S, seam, False, [P(u, v + 0.03) for u, v in edge(rr)], lw)
    if st.get("spoiler"):
        sp = [P(u, v + 0.012) for u, v in edge(rr)] + [P(u, v + 0.04) for u, v in edge(rr, flip=True)]
        _poly(S, (30, 31, 34), sp)
    if st.get("bed"):
        bed = [P(-0.84, 0.585), P(0.84, 0.585), P(0.84, 0.965), P(-0.84, 0.965)]
        _poly(S, _mul(col, 0.40), bed)
        _poly(S, dark, bed, lw * 2)
        for v in np.linspace(0.62, 0.93, 6):
            pygame.draw.line(S, _mul(col, 0.30), P(-0.78, v), P(0.78, v), lw)

    # ---- mirrors: small black wedges at the base of the A-pillars --------------------------
    mv = wf[1] + 0.008
    for s in (-1, 1):
        _poly(S, MIRROR, [P(s * 0.93, mv), P(s * 1.13, mv + 0.012), P(s * 1.17, mv + 0.040), P(s * 0.93, mv + 0.050)])

    # ---- lights: swept shapes that follow the body corners ------------------------------
    fp = [p for p in g.front_path() if 0.40 <= p[0]]
    n = max(1, len(fp) - 1)
    head = [(u * 0.985, v + 0.010) for u, v in fp] + \
           [(u * 0.985 - 0.03, v + 0.016 + 0.026 * (1 - abs(2 * i / n - 1))) for i, (u, v) in reversed(list(enumerate(fp)))]
    rp = [p for p in g.rear_path() if 0.42 <= p[0]]
    tail = [(u * 0.985, v - 0.008) for u, v in rp] + [(u * 0.985 - 0.03, v - 0.030) for u, v in reversed(rp)]
    ind_f = [(u, v) for (u, v) in head if u > 0.84]
    ind_r = [(u, v) for (u, v) in tail if u > 0.84]

    def side(pts, s, k=1.0, dv=0.0):
        return [P(s * u * k, v + dv) for u, v in pts]

    L = pygame.Surface((W, H), pygame.SRCALPHA)
    for s in (-1, 1):
        _poly(L, TRIM, side(head, s))
        _poly(L, HEAD_OFF, side(head, s, 0.97, 0.003))
        _poly(L, AMBER_OFF, side(ind_f, s))
        _poly(L, TRIM, side(tail, s))
        _poly(L, TAIL_OFF, side(tail, s, 0.98, -0.003))
    _clip(L, body_mask)
    S.blit(L, (0, 0))
    _shade(S, body_mask, g.cx, g.hw, strength=0.35)

    # ---- overlays: brake and indicators lit -------------------------------------------------
    B = pygame.Surface((W, H), pygame.SRCALPHA)
    for s in (-1, 1):
        _poly(B, TAIL_ON, side(tail, s, 0.98, -0.003))
        tc = np.mean(side(tail, s), axis=0)
        pygame.draw.ellipse(B, TAIL_HOT, (tc[0] - W * 0.05, tc[1] - H * 0.005, W * 0.10, H * 0.010))
    _clip(B, body_mask)
    blinks = {}
    for name, s in (("blink_L", -1), ("blink_R", 1)):
        K = pygame.Surface((W, H), pygame.SRCALPHA)
        _poly(K, AMBER_ON, side(ind_f, s))
        _poly(K, AMBER_ON, side(ind_r, s))
        _clip(K, body_mask)
        blinks[name] = K

    out = (W // SS, H // SS)
    sm = pygame.transform.smoothscale
    return {"body": sm(S, out), "brake": sm(B, out), "glass": sm(GL, out),
            "blink_L": sm(blinks["blink_L"], out), "blink_R": sm(blinks["blink_R"], out),
            "_mask": sm(shape, out), "style": style, "color": col,
            "layout": {"axle_f": 0.20, "axle_r": 0.80 if style != "van" else 0.84,
                       "tire_u": 0.92, "tire_len": 0.14, "tire_w": 0.085, "body_w": BODY_W,
                       "ws_v": (wf[0] + wr[0]) / 2, "rw_v": (rf[0] + rr[0]) / 2}}


def damaged(model, corner, seed=0):
    """The body with a crumpled corner and cracked glass (corner: FL/FR/RL/RR). Cached."""
    key = "dmg_" + corner
    if key in model:
        return model[key]
    rng = np.random.default_rng(seed)
    S = model["body"].copy()
    W, H = S.get_size()
    front, side = corner[0] == "F", (1 if corner[1] == "R" else -1)
    cx = W / 2 + side * W * BODY_W / 2 * 0.62
    cy = H * (0.05 if front else 0.95)
    D = pygame.Surface((W, H), pygame.SRCALPHA)
    pts = []
    for i in range(14):
        a = 2 * math.pi * i / 14
        r = rng.uniform(0.6, 1.0)
        pts.append((cx + math.cos(a) * W * 0.30 * r, cy + math.sin(a) * H * 0.10 * r))
    pygame.draw.polygon(D, (52, 52, 56, 235), pts)
    for _ in range(7):                                   # creases in the crumpled metal
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.2, 0.8)
        p0 = (cx + math.cos(a) * W * 0.25 * r, cy + math.sin(a) * H * 0.08 * r)
        p1 = (p0[0] + rng.uniform(-1, 1) * W * 0.12, p0[1] + rng.uniform(-1, 1) * H * 0.04)
        pygame.draw.line(D, (128, 128, 134, 230), p0, p1, 2)
    for _ in range(5):                                   # scratches in the paint
        y0 = cy + (1 if front else -1) * rng.uniform(0.06, 0.16) * H
        x0 = cx + rng.uniform(-0.1, 0.1) * W
        pygame.draw.line(D, (205, 205, 210, 160), (x0, y0),
                         (x0 - side * rng.uniform(0.05, 0.2) * W, y0 + rng.uniform(-0.02, 0.02) * H), 1)
    _clip(D, pygame.surfarray.array_alpha(model["_mask"]) > 100)
    S.blit(D, (0, 0))
    # cracked glass: a spider web on the windshield (front hit) or the rear window
    gx, gy = W / 2 + side * W * 0.14, H * model["layout"]["ws_v" if front else "rw_v"]
    Cr = pygame.Surface((W, H), pygame.SRCALPHA)
    for i in range(9):
        a = 2 * math.pi * i / 9 + rng.uniform(-0.2, 0.2)
        r = rng.uniform(0.12, 0.30) * W
        pygame.draw.line(Cr, (230, 236, 240, 200), (gx, gy), (gx + math.cos(a) * r, gy + math.sin(a) * r), 1)
    for rr in (0.05, 0.11):
        ring = [(gx + math.cos(2 * math.pi * i / 9) * rr * W * rng.uniform(0.8, 1.2),
                 gy + math.sin(2 * math.pi * i / 9) * rr * W * rng.uniform(0.8, 1.2)) for i in range(9)]
        pygame.draw.lines(Cr, (230, 236, 240, 160), True, ring, 1)
    _clip(Cr, pygame.surfarray.array_alpha(model["glass"]) > 60)
    S.blit(Cr, (0, 0))
    model[key] = S
    return S


# ---------------------------------------------------------------- fleets
PAINTS = [(232, 232, 228), (170, 174, 180), (34, 36, 40), (44, 70, 120), (150, 34, 36), (48, 84, 64),
          (96, 104, 116), (210, 200, 176), (70, 120, 170), (120, 34, 40), (36, 120, 130), (196, 98, 40),
          (90, 60, 110), (140, 140, 130)]
STYLE_MIX = ["sedan", "sedan", "hatch", "compact", "suv", "suv", "van", "pickup", "sport", "hatch",
             "compact", "sedan"]
N_FLEET = 22
_CACHE = {}


def _variant(i, rng):
    style = STYLE_MIX[i % len(STYLE_MIX)]
    paint = np.array(PAINTS[rng.integers(len(PAINTS))], float)
    paint = np.clip(paint * rng.uniform(0.9, 1.08) + rng.normal(0, 5, 3), 0, 255)   # no two exactly alike
    return dict(color=tuple(int(c) for c in paint), style=style,
                sunroof=style in ("sedan", "hatch", "suv", "compact") and rng.random() < 0.35,
                rack=style in ("suv", "van") and rng.random() < 0.4,
                stripes=style == "sport" and rng.random() < 0.6,
                black_roof=style in ("hatch", "compact") and rng.random() < 0.35)


def npc_fleet(out_h=OUT_H):
    if ("npc", out_h) not in _CACHE:          # drawn once per process
        rng = np.random.default_rng(7)
        cars = [make_car(**_variant(i, rng), out_h=out_h) for i in range(N_FLEET)]
        cars.append(make_car((226, 178, 30), "sedan", taxi=True, out_h=out_h))
        _CACHE["npc", out_h] = cars
    return _CACHE["npc", out_h]


def agent_car(out_h=OUT_H):
    """The learning car: pearl white sedan with a glass roof."""
    if ("agent", out_h) not in _CACHE:
        _CACHE["agent", out_h] = make_car((240, 242, 246), "sedan", glass_roof=True, out_h=out_h)
    return _CACHE["agent", out_h]
