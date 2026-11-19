"""
render.py - top-down pygame renderer (headless friendly).

Layers, bottom to top:
    1. STATIC CITY (pre-rendered once per episode onto one big surface):
       grass, sidewalks, asphalt, lane markings, crosswalks, stop lines,
       school-zone paint, parking lot, building roofs + shadows, trees,
       parked cars, stop signs, speed-limit signs
    2. GROUND DECALS drawn during the episode: skid marks
    3. DYNAMIC: GPS route line, target slot, camera FOV cones + ray hits,
       NPC cars (brake lights, blinkers), pedestrians, the agent car,
       traffic light heads, particles (tyre smoke, crash debris)
    4. HUD: speed + limit, next traffic control, action bars, the per-term
       reward breakdown, mini-map, status banner

The camera follows the agent with a small look-ahead and smoothing.
Positions are interpolated between simulation steps (`alpha`) so videos
can run at 20 fps although the simulation runs at 10 Hz.

Sprites come from assets/sprites (Kenney CC0 "Racing Pack"); if a file is
missing, a simple generated shape is used instead, so you can swap files
freely.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from .city import (AISLE_HW, CORNER_OFF, CROSSWALK_W, LANE_OFF, LANE_W, PARKED_OFF, ROAD_HW,
                   SIDEWALK_W, SLOT_D, SLOT_W, STOPLINE_D, right_of)
from .geometry import obb_corners, wrap_angle
from .rewards import TERMS
from .sensors import CAMERAS, CLS_NONE, CLS_PED, CLS_STATIC, CLS_VEHICLE
from .vehicle import LENGTH, WIDTH

ASSETS = Path(__file__).resolve().parent.parent / "assets"
W, H = 1280, 720
PPM = 9.0                      # pixels per metre
MARGIN = 30.0                  # metres of scenery around the city

# ---- palette (one cohesive, slightly desaturated look) ---------------------
C = dict(
    grass=(104, 150, 86), grass2=(96, 141, 80), sidewalk=(196, 194, 186), curb=(160, 158, 150),
    asphalt=(66, 69, 74), asphalt2=(60, 63, 68), lot=(78, 81, 86), line=(236, 236, 230),
    yellow=(242, 196, 56), school=(118, 104, 62), roof=[(170, 120, 96), (126, 138, 150),
    (186, 170, 140), (142, 110, 120), (110, 128, 120), (200, 196, 186)],
    shadow=(0, 0, 0, 70), hud_bg=(16, 18, 24, 200), hud_fg=(235, 238, 242), hud_dim=(150, 156, 168),
    pos=(92, 200, 120), neg=(232, 88, 80), accent=(80, 170, 255), route=(70, 200, 255),
    target=(80, 230, 140),
)
CAM_COLORS = [(80, 170, 255), (160, 120, 255), (255, 180, 80), (255, 180, 80), (255, 110, 160),
              (200, 200, 200)]
CLS_COLORS = {CLS_STATIC: (255, 255, 255), CLS_VEHICLE: (255, 210, 60), CLS_PED: (255, 70, 70)}


def _font(size, bold=False):
    f = ASSETS / "fonts" / ("Inter-Bold.otf" if bold else "Inter-Regular.otf")
    try:
        return pygame.font.Font(str(f), size)
    except Exception:
        return pygame.font.Font(None, int(size * 1.3))


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "r", "color", "grow")

    def __init__(self, x, y, vx, vy, life, r, color, grow=0.0):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life = self.max_life = life
        self.r, self.color, self.grow = r, color, grow


class Renderer:
    def __init__(self, width=W, height=H, show_hud=True, show_sensors=True):
        pygame.init()
        try:
            pygame.display.set_mode((1, 1))
        except pygame.error:
            pass
        self.W, self.H = width, height
        self.screen = pygame.Surface((width, height))
        self.show_hud = show_hud
        self.show_sensors = show_sensors
        self.fonts = {k: _font(*v) for k, v in dict(
            tiny=(12,), small=(14,), med=(17,), medb=(17, True), big=(40, True), huge=(54, True),
            title=(13, True)).items()}
        self._load_sprites()
        self.city_id = None
        self.cam = None
        self.particles = []
        self.prev_pose = {}
        self.cur_pose = {}
        self.blink_t = 0.0
        self.frame = 0
        # two reusable transparent layers (creating full-screen surfaces every
        # frame was a big part of the render time)
        self.ov_under = pygame.Surface((width, height), pygame.SRCALPHA)   # route, slot, sensors
        self.ov_over = pygame.Surface((width, height), pygame.SRCALPHA)    # particles
        self.text_cache = {}

    # ------------------------------------------------------------ sprites
    def _load_img(self, rel):
        p = ASSETS / "sprites" / rel
        try:
            return pygame.image.load(str(p)).convert_alpha()
        except Exception:
            return None

    def _fallback_car(self, color):
        s = pygame.Surface((70, 130), pygame.SRCALPHA)
        pygame.draw.rect(s, color, (4, 2, 62, 126), border_radius=18)
        pygame.draw.rect(s, (30, 34, 40), (12, 28, 46, 22), border_radius=6)
        pygame.draw.rect(s, (30, 34, 40), (12, 86, 46, 18), border_radius=6)
        return s

    def _load_sprites(self):
        colors = ["black", "blue", "green", "red", "yellow"]
        npc = []
        for c in colors:
            for v in (1, 5):
                img = self._load_img(f"cars/car_{c}_{v}.png")
                npc.append(img)
        fallback_cols = [(40, 40, 44), (40, 120, 210), (50, 170, 90), (220, 90, 40), (240, 200, 40)]
        self.npc_src = [img if img is not None else self._fallback_car(fallback_cols[i // 2])
                        for i, img in enumerate(npc)]
        self.npc_src.append(self._load_img("cars/car_red_3.png") or self._fallback_car((200, 60, 50)))
        # the agent: a clean white-ish car, tinted from the blue sprite
        agent = self._load_img("cars/car_blue_1.png")
        if agent is None:
            agent = self._fallback_car((235, 238, 242))
        self.agent_src = self._tint(agent, (245, 247, 250))
        self.ped_src = []
        for p in sorted((ASSETS / "sprites" / "peds").glob("*.png")):
            img = self._load_img(f"peds/{p.name}")
            if img is not None:
                self.ped_src.append(img)
        if not self.ped_src:
            s = pygame.Surface((36, 24), pygame.SRCALPHA)
            pygame.draw.ellipse(s, (60, 110, 200), (0, 0, 36, 24))
            pygame.draw.circle(s, (230, 190, 150), (18, 12), 8)
            self.ped_src = [s]
        self.tree_src = [self._load_img("objects/tree_large.png"), self._load_img("objects/tree_small.png")]
        self.tree_src = [t for t in self.tree_src if t is not None]
        self.skid_src = self._load_img("objects/skidmark_short_1.png")
        # scaled + rotation caches
        self.car_scaled = {}
        self.rot_cache = {}

    @staticmethod
    def _tint(img, rgb):
        """Recolour the car body while keeping windows/shading."""
        out = img.copy()
        arr = pygame.surfarray.pixels3d(out)
        a = arr.astype(np.float32)
        lum = a.mean(axis=2, keepdims=True) / 255.0
        sat = (a.max(axis=2, keepdims=True) - a.min(axis=2, keepdims=True)) / 255.0
        body = np.array(rgb, np.float32)[None, None, :] * (0.55 + 0.45 * lum)
        mix = np.clip(sat * 2.0, 0, 1)
        arr[:] = (a * (1 - mix) + body * mix).astype(np.uint8)
        del arr
        return out

    def _car_surface(self, key, src, length_m, width_m):
        """Scale a car sprite (nose pointing UP in the file) to world size."""
        if key not in self.car_scaled:
            w = max(int(width_m * PPM), 4)
            h = max(int(length_m * PPM), 6)
            self.car_scaled[key] = pygame.transform.smoothscale(src, (w, h))
        return self.car_scaled[key]

    def _rotated(self, key, surf, theta):
        """Rotate a nose-up sprite to heading theta (cached every 3 degrees)."""
        deg = int(round(math.degrees(theta) / 3.0)) * 3 % 360
        k = (key, deg)
        r = self.rot_cache.get(k)
        if r is None:
            # sprite nose is up (-y); heading 0 means +x -> rotate by -(deg+90)
            r = pygame.transform.rotozoom(surf, -(deg + 90), 1.0)
            if len(self.rot_cache) > 4000:
                self.rot_cache.clear()
            self.rot_cache[k] = r
        return r

    # ------------------------------------------------------------ static city
    def w2s_static(self, p):
        return (int((p[0] + MARGIN) * PPM), int((p[1] + MARGIN) * PPM))

    def _build_static(self, env):
        city = env.city
        if getattr(city, "kind", "city") == "road":
            return self._build_static_road(env)
        Wm = city.width + 2 * MARGIN
        Hm = city.height + 2 * MARGIN
        S = pygame.Surface((int(Wm * PPM), int(Hm * PPM)))
        rng = np.random.default_rng(city.nx * 1000 + int(city.width))
        S.fill(C["grass"])
        # grass texture
        for _ in range(int(Wm * Hm / 6)):
            x, y = rng.uniform(0, Wm * PPM), rng.uniform(0, Hm * PPM)
            pygame.draw.circle(S, C["grass2"], (int(x), int(y)), int(rng.integers(2, 6)))
        ps = self.w2s_static
        k = PPM
        # sidewalk band around every street
        sw = ROAD_HW + SIDEWALK_W
        for x in city.xs:
            r = pygame.Rect(ps((x - sw, -sw)), (int(2 * sw * k), int((city.height + 2 * sw) * k)))
            pygame.draw.rect(S, C["sidewalk"], r)
        for y in city.ys:
            r = pygame.Rect(ps((-sw, y - sw)), (int((city.width + 2 * sw) * k), int(2 * sw * k)))
            pygame.draw.rect(S, C["sidewalk"], r)
        # sidewalk joints
        for x in city.xs:
            for yy in np.arange(-sw, city.height + sw, 2.5):
                for side in (-1, 1):
                    x0 = x + side * ROAD_HW
                    x1 = x + side * sw
                    pygame.draw.line(S, C["curb"], ps((x0, yy)), ps((x1, yy)), 1)
        for y in city.ys:
            for xx in np.arange(-sw, city.width + sw, 2.5):
                for side in (-1, 1):
                    pygame.draw.line(S, C["curb"], ps((xx, y + side * ROAD_HW)), ps((xx, y + side * sw)), 1)
        # parking lot surface
        for rect in city.lot_polys():
            pygame.draw.polygon(S, C["lot"], [ps(c) for c in obb_corners(rect)])
        # asphalt
        for x in city.xs:
            r = pygame.Rect(ps((x - ROAD_HW, -ROAD_HW)), (int(2 * ROAD_HW * k), int((city.height + 2 * ROAD_HW) * k)))
            pygame.draw.rect(S, C["asphalt"], r)
        for y in city.ys:
            r = pygame.Rect(ps((-ROAD_HW, y - ROAD_HW)), (int((city.width + 2 * ROAD_HW) * k), int(2 * ROAD_HW * k)))
            pygame.draw.rect(S, C["asphalt"], r)
        # curbs
        for x in city.xs:
            for side in (-1, 1):
                pygame.draw.line(S, C["curb"], ps((x + side * ROAD_HW, -ROAD_HW)),
                                 ps((x + side * ROAD_HW, city.height + ROAD_HW)), 2)
        for y in city.ys:
            for side in (-1, 1):
                pygame.draw.line(S, C["curb"], ps((-ROAD_HW, y + side * ROAD_HW)),
                                 ps((city.width + ROAD_HW, y + side * ROAD_HW)), 2)
        # asphalt speckle
        for _ in range(int(Wm * Hm / 3)):
            p = (rng.uniform(-ROAD_HW, city.width + ROAD_HW), rng.uniform(-ROAD_HW, city.height + ROAD_HW))
            if city.drivable(np.array(p))[0]:
                S.set_at(ps(p), C["asphalt2"])
        # per-edge markings
        for a in city.adj:
            for b in city.adj[a]:
                if a > b:
                    continue
                self._edge_markings(S, city, a, b)
        # crosswalks & stop lines
        for n in range(len(city.nodes)):
            for b in city.adj[n]:
                d = city.edge_dir(n, b)
                r = right_of(d)
                c = city.nodes[n]
                # zebra stripes across the arm (from ROAD_HW to ROAD_HW+CROSSWALK_W)
                for o in np.arange(-ROAD_HW + 0.5, ROAD_HW - 0.2, 1.0):
                    p0 = c + d * (ROAD_HW + 0.3) + r * o
                    p1 = c + d * (ROAD_HW + CROSSWALK_W - 0.3) + r * o
                    pygame.draw.line(S, C["line"], ps(p0), ps(p1), max(int(0.5 * k), 2))
                # stop line on the incoming lane (traffic coming from b into n)
                if city.control[n] in ("light", "stop"):
                    din = -d               # direction of travel towards n
                    rin = right_of(din)
                    q0 = c + d * STOPLINE_D + rin * 0.1
                    q1 = c + d * STOPLINE_D + rin * LANE_W
                    pygame.draw.line(S, C["line"], ps(q0), ps(q1), max(int(0.45 * k), 3))
                    if city.control[n] == "stop":
                        # "STOP" painted on the road
                        mid = c + d * (STOPLINE_D + 3.0) + rin * LANE_OFF
                        self._road_text(S, "STOP", mid, din, C["line"])
        # lot markings
        self._lot_markings(S, city)
        # building shadows then roofs
        sh = pygame.Surface(S.get_size(), pygame.SRCALPHA)
        for b in city.buildings:
            pts = [ps(c + np.array([1.6, 2.2])) for c in obb_corners(b)]
            pygame.draw.polygon(sh, C["shadow"], pts)
        S.blit(sh, (0, 0))
        for i, b in enumerate(city.buildings):
            col = C["roof"][city.building_colors[i] % len(C["roof"])]
            pts = [ps(c) for c in obb_corners(b)]
            pygame.draw.polygon(S, col, pts)
            dark = tuple(int(v * 0.75) for v in col)
            pygame.draw.polygon(S, dark, pts, 3)
            # inner parapet + rooftop units
            inner = b.copy()
            inner[2] -= 1.0
            inner[3] -= 1.0
            if inner[2] > 1 and inner[3] > 1:
                pygame.draw.polygon(S, tuple(int(v * 0.9) for v in col),
                                    [ps(c) for c in obb_corners(inner)], 1)
                brng = np.random.default_rng(i + 7)
                for _ in range(int(brng.integers(1, 4))):
                    u = brng.uniform(-inner[2] + 1.5, inner[2] - 1.5)
                    v = brng.uniform(-inner[3] + 1.5, inner[3] - 1.5)
                    box = [b[0] + u, b[1] + v, 1.0, 0.8, 0.0]
                    pygame.draw.polygon(S, (150, 152, 156), [ps(c) for c in obb_corners(box)])
                    pygame.draw.polygon(S, (110, 112, 116), [ps(c) for c in obb_corners(box)], 1)
        # parked cars on streets + in the lot
        lot_boxes = city.lot_parked_boxes()
        for i, pb in enumerate(list(city.parked) + list(lot_boxes)):
            color = city.parked_meta[i]["color"] if i < len(city.parked_meta) else (i * 7) % len(self.npc_src)
            self._blit_car(S, (pb[0], pb[1]), pb[4], self.npc_src[color % len(self.npc_src)],
                           ("p", color % len(self.npc_src)), static=True, length=pb[2] * 2)
        # trees (with soft shadow)
        for (x, y, s) in city.trees:
            if not self.tree_src:
                pygame.draw.circle(S, (60, 120, 60), ps((x, y)), int(3 * s * k))
                continue
            src = self.tree_src[0 if s > 0.9 else 1]
            size = int(6.5 * s * k)
            img = pygame.transform.smoothscale(src, (size, size))
            shadow = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow, (0, 0, 0, 60), (0, 0, size, size))
            px, py = ps((x, y))
            S.blit(shadow, (px - size // 2 + 6, py - size // 2 + 8))
            S.blit(img, (px - size // 2, py - size // 2))
        # signs
        for n in range(len(city.nodes)):
            if city.control[n] == "stop":
                for b in city.adj[n]:
                    d = city.edge_dir(n, b)
                    rin = right_of(-d)
                    pos = city.nodes[n] + d * (STOPLINE_D + 0.5) + rin * (ROAD_HW + 1.0)
                    self._stop_sign(S, pos)
        for a in city.adj:
            for b in city.adj[a]:
                d = city.edge_dir(a, b)
                pos = city.nodes[a] + d * (STOPLINE_D + 6) + right_of(d) * (ROAD_HW + 1.2)
                self._limit_sign(S, pos, city.edge_limit(a, b) * 3.6, city.is_school(a, b))
        return S

    def _build_static_road(self, env):
        """Static layer for the single-road world (game levels 2-3)."""
        from .worlds import ROAD_HALF
        city = env.city
        Wm, Hm = city.width + 2 * MARGIN, city.height + 2 * MARGIN
        S = pygame.Surface((int(Wm * PPM), int(Hm * PPM)))
        rng = np.random.default_rng(int(city.width * 7))
        S.fill(C["grass"])
        for _ in range(int(Wm * Hm / 6)):
            pygame.draw.circle(S, C["grass2"], (int(rng.uniform(0, Wm * PPM)), int(rng.uniform(0, Hm * PPM))),
                               int(rng.integers(2, 6)))
        ps = self.w2s_static
        c = city.center
        tang = np.gradient(c, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)

        def strip(off0, off1, color, step=4):
            for i in range(0, len(c) - step, step):
                j = i + step
                quad = [c[i] + nrm[i] * off0, c[j] + nrm[j] * off0, c[j] + nrm[j] * off1, c[i] + nrm[i] * off1]
                pygame.draw.polygon(S, color, [ps(q) for q in quad])
        strip(-ROAD_HALF - 0.6, ROAD_HALF + 0.6, C["curb"])          # gravel shoulder
        for rect in city.lot_polys():                                 # parking pad
            pygame.draw.polygon(S, C["lot"], [ps(q) for q in obb_corners(rect)])
        strip(-ROAD_HALF, ROAD_HALF, C["asphalt"])
        # markings: solid white edges, dashed yellow centre line
        for off in (-ROAD_HALF + 0.3, ROAD_HALF - 0.3):
            pygame.draw.lines(S, (220, 220, 214), False, [ps(q) for q in c[::4] + nrm[::4] * off], 2)
        for i in range(0, len(c) - 6, 12):
            pygame.draw.line(S, C["yellow"], ps(c[i]), ps(c[i + 6]), 2)
        # bays
        for sl in city.slots:
            ax = sl["axis"]
            w = np.array([-ax[1], ax[0]])
            for e in (-1, 1):
                pygame.draw.line(S, C["line"], ps(sl["center"] + w * e * SLOT_W / 2 - ax * SLOT_D / 2),
                                 ps(sl["center"] + w * e * SLOT_W / 2 + ax * SLOT_D / 2), 2)
        for i, pb in enumerate(city.lot_parked_boxes()):
            k = (i * 7) % len(self.npc_src)
            self._blit_car(S, (pb[0], pb[1]), pb[4], self.npc_src[k], ("p", k), static=True, length=pb[2] * 2)
        for (x, y, sz) in city.trees:
            if not self.tree_src:
                continue
            src = self.tree_src[0 if sz > 0.9 else 1]
            size = int(6.5 * sz * PPM)
            img = pygame.transform.smoothscale(src, (size, size))
            sh = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.ellipse(sh, (0, 0, 0, 60), (0, 0, size, size))
            px, py = ps((x, y))
            S.blit(sh, (px - size // 2 + 6, py - size // 2 + 8))
            S.blit(img, (px - size // 2, py - size // 2))
        return S

    def _road_text(self, S, text, pos, direction, color):
        # Road paint reads from the driver's seat: the word runs ACROSS the lane
        # and its top points in the direction of travel. Scaled to fit the lane.
        f = self.fonts["medb"]
        img = f.render(text, True, color)
        scale = min(1.0, (LANE_W - 0.6) * PPM / img.get_width())
        ang = math.degrees(math.atan2(direction[1], direction[0]))
        img = pygame.transform.rotozoom(img, -(ang + 90), scale)
        img.set_alpha(190)
        px, py = self.w2s_static(pos)
        S.blit(img, (px - img.get_width() // 2, py - img.get_height() // 2))

    def _edge_markings(self, S, city, a, b):
        ps = self.w2s_static
        d = city.edge_dir(a, b)
        r = right_of(d)
        A, B = city.nodes[a], city.nodes[b]
        L = np.linalg.norm(B - A)
        s0, s1 = ROAD_HW + CROSSWALK_W, L - ROAD_HW - CROSSWALK_W
        # school zone tint
        if city.is_school(a, b):
            for sgn in (-1, 1):
                pts = [A + d * s0, A + d * s1, A + d * s1 + r * sgn * LANE_W, A + d * s0 + r * sgn * LANE_W]
                pygame.draw.polygon(S, C["school"], [ps(p) for p in pts])
            for sgn, dd in ((1, d), (-1, -d)):
                mid = A + d * (L / 2) + r * sgn * LANE_OFF
                self._road_text(S, "SCHOOL", mid, dd, C["yellow"])
        # double yellow centre line
        for o in (-0.15, 0.15):
            pygame.draw.line(S, C["yellow"], ps(A + d * s0 + r * o), ps(A + d * s1 + r * o), 2)
        # dashed white lane/parking separator (each side)
        for sgn in (-1, 1):
            for t in np.arange(s0, s1 - 1, 3.0):
                pygame.draw.line(S, (200, 200, 196), ps(A + d * t + r * sgn * LANE_W),
                                 ps(A + d * min(t + 1.5, s1) + r * sgn * LANE_W), 2)

    def _lot_markings(self, S, city):
        ps = self.w2s_static
        d, r, m = city.lot_d, city.lot_r, city.lot_m
        for i, sl in enumerate(city.slots):
            c = sl["center"]
            ax = sl["axis"]                       # slot axis (angled)
            w = np.array([-ax[1], ax[0]])         # across the slot
            for e in (-1, 1):
                p0 = c + w * e * SLOT_W / 2 - ax * SLOT_D / 2
                p1 = c + w * e * SLOT_W / 2 + ax * SLOT_D / 2
                pygame.draw.line(S, C["line"], ps(p0), ps(p1), 2)
        # entrance arrow-ish marking
        a0, a1 = city.lot_aisle
        for t in np.arange(a0 + 2, a1 - 2, 6):
            pygame.draw.line(S, (210, 210, 205), ps(m + r * t), ps(m + r * (t + 2)), 2)

    def _stop_sign(self, S, pos):
        px, py = self.w2s_static(pos)
        R = int(0.85 * PPM)
        pts = [(px + R * math.cos(math.pi / 8 + i * math.pi / 4), py + R * math.sin(math.pi / 8 + i * math.pi / 4))
               for i in range(8)]
        pygame.draw.polygon(S, (200, 30, 36), pts)
        pygame.draw.polygon(S, (255, 255, 255), pts, 2)
        t = self.fonts["tiny"].render("STOP", True, (255, 255, 255))
        t = pygame.transform.smoothscale(t, (int(R * 1.4), int(R * 0.6)))
        S.blit(t, (px - t.get_width() // 2, py - t.get_height() // 2))

    def _limit_sign(self, S, pos, kmh, school):
        px, py = self.w2s_static(pos)
        w, h = int(1.8 * PPM), int(2.2 * PPM)
        rect = pygame.Rect(px - w // 2, py - h // 2, w, h)
        pygame.draw.rect(S, (250, 250, 248) if not school else (250, 220, 60), rect, border_radius=3)
        pygame.draw.rect(S, (30, 30, 30), rect, 1, border_radius=3)
        t = self.fonts["medb"].render(f"{int(round(kmh))}", True, (20, 20, 20))
        t = pygame.transform.smoothscale(t, (int(w * 0.8), int(h * 0.55)))
        S.blit(t, (px - t.get_width() // 2, py - t.get_height() // 2 + 2))

    def _blit_car(self, S, pos, theta, src, key, static=False, length=None, offset=(0, 0), shadow=True):
        length = length or 4.5
        surf = self._car_surface((key, round(length, 1)), src, length, length * 0.48)
        rot = self._rotated((key, round(length, 1)), surf, theta)
        if static:
            px, py = self.w2s_static(pos)
        else:
            px, py = self.w2s(pos)
        if shadow:
            sh = self._shadow_for(rot)
            S.blit(sh, (px - rot.get_width() // 2 + 4, py - rot.get_height() // 2 + 5))
        S.blit(rot, (px - rot.get_width() // 2, py - rot.get_height() // 2))

    def _shadow_for(self, rot):
        k = ("sh", id(rot))
        s = self.rot_cache.get(k)
        if s is None:
            mask = pygame.mask.from_surface(rot)
            s = mask.to_surface(setcolor=(0, 0, 0, 80), unsetcolor=(0, 0, 0, 0))
            self.rot_cache[k] = s
        return s

    # ------------------------------------------------------------ camera
    def w2s(self, p):
        return (int((p[0] - self.cam[0]) * PPM + self.W / 2),
                int((p[1] - self.cam[1]) * PPM + self.H / 2))

    def _lerp_pose(self, key, x, y, th, alpha):
        prev = self.prev_pose.get(key)
        self.cur_pose[key] = (x, y, th)
        if prev is None or alpha >= 1.0:
            return x, y, th
        px, py, pth = prev
        if abs(px - x) + abs(py - y) > 10:      # teleported (respawn)
            return x, y, th
        return (px + (x - px) * alpha, py + (y - py) * alpha,
                pth + wrap_angle(th - pth) * alpha)

    def begin_step(self):
        """Call once per simulation step (before drawing its sub-frames)."""
        self.prev_pose = self.cur_pose
        self.cur_pose = {}

    # ------------------------------------------------------------ main draw
    def draw(self, env, alpha=1.0, dt=0.1):
        key = (env.city_seed, id(env.city))
        if key != self.city_id:
            self.city_id = key
            self.static = self._build_static(env)
            self.ground = self.static.copy()
            self.cam = None
            self.particles = []
            self.prev_pose, self.cur_pose = {}, {}
            self.last_skid = None
        car = env.car
        ax, ay, ath = self._lerp_pose("agent", car.x, car.y, car.theta, alpha)
        # smooth following camera with look-ahead
        look = np.array([math.cos(ath), math.sin(ath)]) * (6 + car.v * 1.2)
        target = np.array([ax, ay]) + look
        if self.cam is None:
            self.cam = target.copy()
        else:
            # exponential smoothing that feels the same at any frame rate
            # (0.12 per frame was tuned at 20 fps)
            k = 1.0 - (1.0 - 0.12) ** (max(dt, 1e-3) / 0.05)
            self.cam = self.cam + (target - self.cam) * k
        self.blink_t += dt
        self.frame += 1
        scr = self.screen
        # --- ground --------------------------------------------------------
        self._skids(env, ax, ay, ath)
        ox = int((self.cam[0] + MARGIN) * PPM - self.W / 2)
        oy = int((self.cam[1] + MARGIN) * PPM - self.H / 2)
        scr.fill(C["grass"])
        scr.blit(self.ground, (-ox, -oy))
        # --- route + target slot ------------------------------------------------
        self.ov_under.fill((0, 0, 0, 0))
        self._route(env)
        # --- sensors --------------------------------------------------------------
        if self.show_sensors:
            self._sensors(env, ax, ay, ath)
        scr.blit(self.ov_under, (0, 0))
        # --- actors ---------------------------------------------------------------
        for c in env.traffic.cars:
            x, y, th = self._lerp_pose(c.id, c.x, c.y, c.theta, alpha)
            if abs(x - self.cam[0]) * PPM > self.W / 2 + 60 or abs(y - self.cam[1]) * PPM > self.H / 2 + 60:
                continue
            src = self.npc_src[c.color % len(self.npc_src)]
            self._blit_car(scr, (x, y), th, src, ("n", c.color % len(self.npc_src)), length=4.5)
            self._lights(scr, (x, y), th, 4.5, 1.85, c.braking, c.blinker)
        for i, p in enumerate(env.peds.peds):
            if p.state == "hidden" and not self.show_sensors:
                pass
            x, y, th = self._lerp_pose(("p", id(p)), p.pos[0], p.pos[1], p.heading, alpha)
            if abs(x - self.cam[0]) * PPM > self.W / 2 + 40 or abs(y - self.cam[1]) * PPM > self.H / 2 + 40:
                continue
            self._ped(scr, p, x, y, th)
        # agent
        self._blit_car(scr, (ax, ay), ath, self.agent_src, ("agent",), length=LENGTH)
        self._lights(scr, (ax, ay), ath, LENGTH, WIDTH, car.braking or car.v < 0.2, env.blinker(), agent=True)
        # traffic light heads
        self._traffic_lights(env)
        # particles
        self._particles(env, dt)
        # --- HUD -----------------------------------------------------------------
        if self.show_hud:
            self._hud(env)
        # raw RGB bytes -> (H, W, 3) array; much faster than surfarray + copy
        return np.frombuffer(pygame.image.tobytes(scr, "RGB"), np.uint8).reshape(self.H, self.W, 3)

    # ------------------------------------------------------------ pieces
    def _skids(self, env, x, y, th):
        car = env.car
        hard = (car.braking and car.v > 3.0 and car.a < -4.0) or abs(car.lat_acc) > 6.0
        if hard:
            f = np.array([math.cos(th), math.sin(th)])
            r = np.array([-f[1], f[0]])
            pts = [np.array([x, y]) - f * (LENGTH * 0.35) + r * s * (WIDTH * 0.4) for s in (-1, 1)]
            if self.last_skid is not None:
                for a, b in zip(self.last_skid, pts):
                    pa = self.w2s_static(a)
                    pb = self.w2s_static(b)
                    pygame.draw.line(self.ground, (34, 34, 36), pa, pb, 4)
            self.last_skid = pts
            # tyre smoke
            for p in pts:
                if np.random.random() < 0.6:
                    self.particles.append(Particle(p[0], p[1], np.random.uniform(-0.5, 0.5),
                                                   np.random.uniform(-0.5, 0.5), 1.2, 0.5,
                                                   (225, 225, 225), grow=1.2))
        else:
            self.last_skid = None
        for e in env.events:
            if e[0] == "crash":
                for _ in range(40):
                    a = np.random.uniform(0, 2 * np.pi)
                    sp = np.random.uniform(2, 9)
                    col = [(255, 210, 80), (255, 140, 40), (200, 200, 210)][np.random.randint(3)]
                    self.particles.append(Particle(e[1][0], e[1][1], math.cos(a) * sp, math.sin(a) * sp,
                                                   np.random.uniform(0.4, 1.0), 0.18, col))
                for _ in range(12):
                    self.particles.append(Particle(e[1][0], e[1][1], np.random.uniform(-1, 1),
                                                   np.random.uniform(-1, 1), 2.5, 0.8, (120, 120, 120), grow=1.5))
            if e[0] == "ped_hit":
                for _ in range(16):
                    a = np.random.uniform(0, 2 * np.pi)
                    self.particles.append(Particle(e[1][0], e[1][1], math.cos(a) * 3, math.sin(a) * 3,
                                                   0.8, 0.25, (255, 80, 80)))
            if e[0] == "chime":
                for _ in range(30):
                    a = np.random.uniform(0, 2 * np.pi)
                    sp = np.random.uniform(1, 4)
                    self.particles.append(Particle(e[1][0], e[1][1], math.cos(a) * sp, math.sin(a) * sp,
                                                   1.4, 0.2, C["target"]))

    def _particles(self, env, dt):
        if not self.particles:
            return
        surf = self.ov_over
        surf.fill((0, 0, 0, 0))
        alive = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0:
                continue
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vx *= 0.92
            p.vy *= 0.92
            p.r += p.grow * dt
            a = int(200 * p.life / p.max_life)
            sx, sy = self.w2s((p.x, p.y))
            pygame.draw.circle(surf, (*p.color, a), (sx, sy), max(int(p.r * PPM), 1))
            alive.append(p)
        self.particles = alive[-600:]
        self.screen.blit(surf, (0, 0))

    def _route(self, env):
        route = env.route
        s = np.arange(env.s + 3, min(env.s + 90, route.length), 2.0)
        if len(s) > 1:
            pts = [self.w2s(p) for p in route.pos(s)]
            surf = self.ov_under
            for i in range(0, len(pts) - 1, 2):
                a = int(150 * (1 - i / len(pts)))
                pygame.draw.line(surf, (*C["route"], a), pts[i], pts[i + 1], 3)
        # target slot (pulsing)
        sl = env.slot
        box = [sl["center"][0], sl["center"][1], SLOT_D / 2 - 0.2, SLOT_W / 2 - 0.2, sl["heading"]]
        pulse = 0.5 + 0.5 * math.sin(self.blink_t * 4)
        surf = self.ov_under
        pygame.draw.polygon(surf, (*C["target"], int(50 + 50 * pulse)), [self.w2s(c) for c in obb_corners(box)])
        pygame.draw.polygon(surf, (*C["target"], 230), [self.w2s(c) for c in obb_corners(box)], 3)

    def _sensors(self, env, ax, ay, ath):
        surf = self.ov_under
        sens = env.sensors
        # shift ray data to the interpolated pose
        dx, dy, dth = ax - env.car.x, ay - env.car.y, wrap_angle(ath - env.car.theta)
        start = 0
        for ci, cam in enumerate(CAMERAS + [None]):
            n = cam.n_rays if cam else len(sens.local_ang) - start
            idx = slice(start, start + n)
            start += n
            col = CAM_COLORS[ci]
            org = env.ray_org[idx] + np.array([dx, dy])
            ang = env.car.theta + dth + sens.local_ang[idx]
            dist = env.ray_dist[idx]
            ends = org + np.stack([np.cos(ang), np.sin(ang)], 1) * dist[:, None]
            if cam is not None:
                poly = [self.w2s(org[0])] + [self.w2s(e) for e in ends] + [self.w2s(org[-1])]
                pygame.draw.polygon(surf, (*col, 22), poly)
                pygame.draw.lines(surf, (*col, 90), False, [self.w2s(e) for e in ends], 1)
            else:
                pygame.draw.polygon(surf, (*col, 18), [self.w2s(e) for e in ends])
            for e, c in zip(ends, env.ray_cls[idx]):
                if c != CLS_NONE:
                    pygame.draw.circle(surf, (*CLS_COLORS[c], 220), self.w2s(e), 3)

    def _lights(self, S, pos, th, L, Wd, braking, blinker, agent=False):
        f = np.array([math.cos(th), math.sin(th)])
        r = np.array([-f[1], f[0]])
        p = np.array(pos)
        if braking:
            for s in (-1, 1):
                q = self.w2s(p - f * (L / 2 - 0.15) + r * s * (Wd / 2 - 0.35))
                pygame.draw.circle(S, (255, 40, 40), q, 3)
                gs = self._glow((255, 30, 30), 9)
                S.blit(gs, (q[0] - 9, q[1] - 9), special_flags=pygame.BLEND_RGBA_ADD)
        if agent:
            for s in (-1, 1):   # headlights
                q = self.w2s(p + f * (L / 2 - 0.1) + r * s * (Wd / 2 - 0.35))
                gs = self._glow((90, 90, 70), 8)
                S.blit(gs, (q[0] - 8, q[1] - 8), special_flags=pygame.BLEND_RGBA_ADD)
        if blinker and (self.blink_t % 0.7) < 0.35:
            s = 1 if blinker == "R" else -1
            for fo in (L / 2 - 0.2, -(L / 2 - 0.2)):
                q = self.w2s(p + f * fo + r * s * (Wd / 2 - 0.1))
                pygame.draw.circle(S, (255, 170, 30), q, 3)
                gs = self._glow((255, 140, 0), 10)
                S.blit(gs, (q[0] - 10, q[1] - 10), special_flags=pygame.BLEND_RGBA_ADD)

    def _glow(self, col, r):
        k = ("glow", col, r)
        g = self.rot_cache.get(k)
        if g is None:
            g = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
            for i in range(r, 0, -1):
                a = int(200 * (1 - i / r) ** 1.5)
                pygame.draw.circle(g, (*col, a), (r, r), i)
            self.rot_cache[k] = g
        return g

    def _ped(self, S, p, x, y, th):
        src = self.ped_src[p.sprite % len(self.ped_src)]
        w = int(1.25 * PPM)          # drawn a bit larger than life so people are easy to spot
        h = int(w * src.get_height() / src.get_width())
        bob = 1.0 + 0.06 * math.sin(p.anim * 3) if p.state not in ("wait", "hidden") else 1.0
        key = ("ped", p.sprite % len(self.ped_src), int(bob * 20))
        surf = self.car_scaled.get(key)
        if surf is None:
            surf = pygame.transform.smoothscale(src, (int(w * bob), int(h * bob)))
            self.car_scaled[key] = surf
        # sprite faces up: rotate to heading
        rot = self._rotated(key, surf, th)
        sx, sy = self.w2s((x, y))
        sh = self._shadow_for(rot)
        S.blit(sh, (sx - rot.get_width() // 2 + 2, sy - rot.get_height() // 2 + 3))
        if p.state == "hidden":
            rot = rot.copy()
            rot.set_alpha(150)
        S.blit(rot, (sx - rot.get_width() // 2, sy - rot.get_height() // 2))

    def _traffic_lights(self, env):
        city = env.city
        for n, lt in city.lights.items():
            c = city.nodes[n]
            if abs(c[0] - self.cam[0]) * PPM > self.W / 2 + 120 or abs(c[1] - self.cam[1]) * PPM > self.H / 2 + 120:
                continue
            for b in city.adj[n]:
                d = city.edge_dir(n, b)         # arm direction
                rin = right_of(-d)              # right side of incoming traffic
                pos = c + d * (STOPLINE_D - 0.5) + rin * (ROAD_HW + 0.9)
                st = city.light_state(n, city.edge_axis(n, b), env.t)
                self._signal_head(pos, -d, st)

    def _signal_head(self, pos, facing, st):
        px, py = self.w2s(pos)
        horiz = abs(facing[0]) > abs(facing[1])
        w, h = (14, 34) if not horiz else (34, 14)
        rect = pygame.Rect(px - w // 2, py - h // 2, w, h)
        pygame.draw.rect(self.screen, (25, 27, 30), rect, border_radius=4)
        pygame.draw.rect(self.screen, (90, 92, 96), rect, 1, border_radius=4)
        cols = {"R": (255, 50, 40), "Y": (255, 200, 40), "G": (60, 230, 100)}
        order = ["R", "Y", "G"]
        # order lamps so red is "far" from approaching traffic
        for i, k in enumerate(order):
            t = (i - 1) * 10
            if horiz:
                q = (px - int(np.sign(facing[0])) * t, py)
            else:
                q = (px, py - int(np.sign(facing[1])) * t)
            on = st == k
            pygame.draw.circle(self.screen, cols[k] if on else (55, 55, 58), q, 4)
            if on:
                g = self._glow(cols[k], 12)
                self.screen.blit(g, (q[0] - 12, q[1] - 12), special_flags=pygame.BLEND_RGBA_ADD)

    # ------------------------------------------------------------ HUD
    def _panel(self, rect, alpha=200):
        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (16, 18, 24, alpha), s.get_rect(), border_radius=12)
        pygame.draw.rect(s, (255, 255, 255, 25), s.get_rect(), 1, border_radius=12)
        self.screen.blit(s, rect.topleft)

    def _text(self, txt, pos, font="small", color=None, anchor="topleft"):
        key = (txt, font, color)
        img = self.text_cache.get(key)
        if img is None:
            img = self.fonts[font].render(txt, True, color or C["hud_fg"])
            if len(self.text_cache) > 3000:
                self.text_cache.clear()
            self.text_cache[key] = img
        r = img.get_rect(**{anchor: pos})
        self.screen.blit(img, r)
        return r

    def _hud(self, env):
        car = env.car
        f = self.fonts
        # ---------- left: speed cluster ----------
        self._panel(pygame.Rect(14, 14, 250, 196))
        kmh = car.v * 3.6
        limit = env.route.limit(env.s) * 3.6
        over = kmh > limit + env.reward_cfg["settings"]["speed_tolerance_kmh"]
        self._text(f"{kmh:3.0f}", (30, 22), "huge", C["neg"] if over else C["hud_fg"])
        self._text("km/h", (34, 84), "small", C["hud_dim"])
        # limit sign
        cx, cy = 210, 58
        if limit <= 60:
            pygame.draw.circle(self.screen, (250, 250, 248), (cx, cy), 30)
            pygame.draw.circle(self.screen, (210, 40, 40), (cx, cy), 30, 5)
            self._text(f"{limit:.0f}", (cx, cy), "medb", (20, 20, 20), anchor="center")
        else:                                   # game levels: no speed limit here
            pygame.draw.circle(self.screen, (250, 250, 248), (cx, cy), 30)
            pygame.draw.line(self.screen, (40, 40, 40), (cx - 19, cy + 19), (cx + 19, cy - 19), 4)
            self._text("NO LIMIT", (cx, cy + 40), "title", C["hud_dim"], anchor="center")
        school = env.route.school[min(env.route_idx, len(env.route.school) - 1)]
        if school:
            self._text("SCHOOL ZONE", (cx, cy + 40), "title", C["yellow"], anchor="center")
        # next control
        s_front = env.s + LENGTH / 2
        i, ev = env._next_event(s_front)
        y = 112
        if ev is not None and ev["s"] - s_front < 60:
            dist = ev["s"] - s_front
            if ev["kind"] == "light":
                st = env.city.light_state(ev["node"], ev["axis"], env.t)
                col = {"R": (255, 60, 50), "Y": (255, 200, 40), "G": (70, 220, 110)}[st]
                pygame.draw.circle(self.screen, col, (40, y + 10), 9)
                self._text(f"Light {dist:4.0f} m", (58, y + 1), "med")
            else:
                pygame.draw.circle(self.screen, (200, 30, 36), (40, y + 10), 9)
                done = env.stop_min_speed.get(i, 99) <= 0.5
                self._text(f"Stop {dist:4.0f} m" + ("  done" if done else ""), (58, y + 1), "med")
        else:
            self._text("No control ahead", (30, y + 1), "med", C["hud_dim"])
        # action bars
        a = env.last_action
        self._text("STEER", (30, 142), "title", C["hud_dim"])
        self._bar(92, 146, 150, 10, a[0], C["accent"], centered=True)
        acc = car.a / (3.5 if car.a >= 0 else 8.0)
        self._text("ACCEL", (30, 164), "title", C["hud_dim"])
        self._bar(92, 168, 150, 10, acc, C["pos"] if acc >= 0 else C["neg"], centered=True)
        stg = env.ep_stage
        self._text(f"{env.stages[stg]['name']}   t={env.t:5.1f}s", (30, 186), "tiny", C["hud_dim"])

        # ---------- right: reward breakdown ----------
        enabled = [k for k in TERMS if env.reward_cfg["terms"][k]["enabled"]
                   and env.reward_cfg["terms"][k]["weight"] > 0]
        ph = 64 + 19 * len(enabled)
        x0 = self.W - 314
        self._panel(pygame.Rect(x0, 14, 300, ph))
        self._text("REWARD", (x0 + 16, 24), "title", C["hud_dim"])
        self._text(f"{env.ep_return:+8.2f}", (x0 + 284, 20), "medb",
                   C["pos"] if env.ep_return >= 0 else C["neg"], anchor="topright")
        self._text(f"step {env.last_reward:+.3f}", (x0 + 284, 42), "tiny", C["hud_dim"], anchor="topright")
        vals = [env.ep[k] for k in enabled]
        vmax = max([abs(v) for v in vals] + [1.0])
        for j, k in enumerate(enabled):
            yy = 62 + j * 19
            v = env.ep[k]
            active = abs(env.last_terms.get(k, 0.0)) > 1e-9
            self._text(k.replace("_", " "), (x0 + 16, yy), "tiny", C["hud_fg"] if active else C["hud_dim"])
            bw = int(70 * abs(v) / vmax)
            col = C["pos"] if v >= 0 else C["neg"]
            bx = x0 + 160
            pygame.draw.rect(self.screen, (50, 54, 62), (bx, yy + 4, 70, 8), border_radius=3)
            if bw > 0:
                pygame.draw.rect(self.screen, col, (bx, yy + 4, bw, 8), border_radius=3)
            self._text(f"{v:+.1f}", (x0 + 284, yy), "tiny", col if abs(v) > 1e-6 else C["hud_dim"],
                       anchor="topright")

        # ---------- bottom-right: minimap ----------
        self._minimap(env)
        # ---------- status banner ----------
        st = env.status
        if st != "driving":
            msg, col = {"parked": ("PARKED", C["target"]), "collision": ("COLLISION", C["neg"]),
                        "pedestrian_hit": ("PEDESTRIAN HIT", C["neg"]),
                        "off_route": ("OFF ROUTE", (255, 170, 60)), "stuck": ("STUCK", (255, 170, 60)),
                        "timeout": ("TIME OUT", (255, 170, 60))}.get(st, (st.upper(), C["hud_fg"]))
            img = f["big"].render(msg, True, col)
            r = img.get_rect(center=(self.W // 2, 80))
            self._panel(r.inflate(40, 20), 210)
            self.screen.blit(img, r)
        self._text("AUTODRIVE-RL", (16, self.H - 24), "title", (255, 255, 255, 120))

    def _bar(self, x, y, w, h, v, col, centered=False):
        pygame.draw.rect(self.screen, (50, 54, 62), (x, y, w, h), border_radius=4)
        v = float(np.clip(v, -1, 1))
        if centered:
            mid = x + w // 2
            bw = int(abs(v) * w / 2)
            rx = mid if v >= 0 else mid - bw
            pygame.draw.rect(self.screen, col, (rx, y, bw, h), border_radius=4)
            pygame.draw.line(self.screen, (200, 200, 200), (mid, y - 2), (mid, y + h + 1), 1)

    def _minimap(self, env):
        city = env.city
        size = 190
        sc = size / (max(city.width, city.height) + 40)
        x0, y0 = self.W - size - 16, self.H - size - 16
        self._panel(pygame.Rect(x0 - 6, y0 - 6, size + 12, size + 12), 190)

        def mp(p):
            return (int(x0 + (p[0] + 20) * sc), int(y0 + (p[1] + 20) * sc))
        for x in city.xs:
            pygame.draw.line(self.screen, (90, 95, 105), mp((x, 0)), mp((x, city.height)), 3)
        for y in city.ys:
            pygame.draw.line(self.screen, (90, 95, 105), mp((0, y)), mp((city.width, y)), 3)
        pts = env.route.pts[::6]
        if len(pts) > 1:
            pygame.draw.lines(self.screen, C["route"], False, [mp(p) for p in pts], 2)
        for c in env.traffic.cars:
            pygame.draw.circle(self.screen, (230, 200, 70), mp(c.pos), 2)
        pygame.draw.circle(self.screen, C["target"], mp(env.slot["center"]), 4)
        pygame.draw.circle(self.screen, (255, 255, 255), mp(env.car.pos), 4)
