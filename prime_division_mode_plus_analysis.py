# prime_division_mode_plus_analysis.py
#
# PYGAME-ONLY
# Combines your Prime Division Mode (towers) with the Manim IDEAS:
#   (2) Sans-space ramp (primes only): s(n) = n - 2^(bit_length(n)-1)
#   (3) Density decay by bit bucket + ~1/ln(n) guide
#   (4) End card: Thanks For Watching! - ONOJK123
#
# Adds: BIG realtime "mod check" text in Mode 1 (updates every step)
#
# FINAL SEMANTICS (TOWERS MODE):
#   GREEN  = n % d != 0 OR synthetic prime-fill (work completed)
#   RED    = n % d == 0 (composite witness)
#   YELLOW = spacer / structural marker only
#
# HEIGHT RULE:
#   Prime n      → exactly n green dots tall
#   Composite n  → < n dots, ending in one red dot
#
# Controls:
#   1     : Mode 1  (Prime Division Towers)
#   2     : Mode 2  (Sans-space ramp, primes only)
#   3     : Mode 3  (Density decay by bit bucket)
#   4     : Mode 4  (End card)
#
#   SPACE : tap = step (Mode 1 only), hold = scroll-step (Mode 1 only)
#   A     : toggle auto-advance (Mode 1 only)
#   UP    : faster auto (Mode 1 only)
#   DOWN  : slower auto (Mode 1 only)
#   LEFT  : zoom out (all modes)
#   RIGHT : zoom in (all modes)
#   R     : reset n back to 1 (keeps dots + same csv)   (Mode 1 only)
#   C     : clear dots AND start a new csv              (Mode 1 only)
#   ESC   : quit

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pygame


# ----------------------------
# Config
# ----------------------------
W, H = 1280, 720
FPS = 60

N_MIN = 1
N_MAX = 300

# For ramp + density
P_MAX_RAMP = 250
MIN_BITS_DENS = 2
MAX_BITS_DENS = 16

# View / camera (towers)
ORIGIN_SCREEN_Y_FRAC = 0.90

# Colors
BG = (8, 8, 10)

COL_DOT_GREEN = (80, 255, 120)
COL_DOT_RED = (255, 80, 80)
COL_DOT_YELLOW = (255, 230, 90)
COL_COLLINE = (55, 55, 65)

COL_TEXT = (235, 235, 245)
COL_SUB = (160, 160, 175)
COL_HI = (255, 230, 90)
COL_BLUE = (110, 170, 255)
COL_PURPLE = (190, 120, 255)
COL_PANEL = (20, 20, 26)
COL_PANEL_STROKE = (95, 95, 115)
COL_WHITE = (230, 230, 230)

DOT_RADIUS = 2

# Scaling (global zoom)
SCALE_DEFAULT = 3.2
SCALE_MIN = 0.7
SCALE_MAX = 28.0
ZOOM_STEP = 1.12

# Timing (mode 1)
AUTO_CHECKS_PER_SEC_DEFAULT = 6.0
AUTO_CHECKS_PER_SEC_MIN = 0.5
AUTO_CHECKS_PER_SEC_MAX = 80.0
SPACE_REPEAT_RATE = 18.0


# ----------------------------
# Data
# ----------------------------
@dataclass(frozen=True)
class DotRec:
    n: int
    k: int
    color: str   # "green" | "red" | "yellow"


# ----------------------------
# Primes / math
# ----------------------------
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if (n % 2) == 0:
        return False
    r = int(math.isqrt(n))
    f = 3
    while f <= r:
        if (n % f) == 0:
            return False
        f += 2
    return True


def primes_upto(nmax: int) -> List[int]:
    return [n for n in range(2, nmax + 1) if is_prime(n)]


def sans_space(n: int) -> int:
    b = n.bit_length()
    return n - (1 << (b - 1))


# ----------------------------
# Divisors (sqrt trial)
# ----------------------------
def trial_divisors_for_n(n: int) -> List[int]:
    if n < 2:
        return []
    limit = int(math.isqrt(n))
    return list(range(2, limit + 1))


# ----------------------------
# CSV
# ----------------------------
def new_csv_path():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"prime_division_{ts}.csv"


def write_csv_header(path: str):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(["n", "k", "color"])


def append_dot(path: str, dot: DotRec):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([dot.n, dot.k, dot.color])


# ----------------------------
# Simulation (your exact semantics + realtime modcheck)
# ----------------------------
class PrimeStepper:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.dots: List[DotRec] = []

        self.n = N_MIN
        self.divs: List[int] = []
        self.i = 0
        self.k = 0
        self.state = "INIT"

        self.last_witness: Optional[int] = None

        # Realtime "what just happened" (for big HUD)
        self.last_check_d: Optional[int] = None
        self.last_check_r: Optional[int] = None
        self.last_event: str = "INIT"   # "TRIVIAL"|"CHECK"|"WITNESS"|"PRIME_FILL"|"NEXT_N"|"DONE"|"SYNC"

        self._sync()

    def _sync(self):
        self.last_witness = None
        self.last_check_d = None
        self.last_check_r = None
        self.last_event = "DONE" if self.n > N_MAX else "SYNC"

        if self.n > N_MAX:
            self.state = "DONE"
            return

        self.divs = trial_divisors_for_n(self.n)
        self.i = 0
        self.k = 0

        if self.n < 2:
            self.state = "TRIVIAL"
        elif self.divs:
            self.state = "CHECKING"
        else:
            self.state = "PRIME"

    def _next_n(self):
        self.n += 1
        self._sync()

    def reset_keep(self):
        # Reset n only; keep dots + same csv
        self.n = N_MIN
        self._sync()

    def clear_new_csv(self, csv_path: str):
        self.csv_path = csv_path
        self.dots = []
        self.n = N_MIN
        self._sync()

    def step(self):
        if self.state == "DONE":
            self.last_event = "DONE"
            return

        # TRIVIAL spacer
        if self.state == "TRIVIAL":
            self.last_event = "TRIVIAL"
            self.last_check_d = None
            self.last_check_r = None

            d = DotRec(self.n, 0, "yellow")
            self.dots.append(d)
            append_dot(self.csv_path, d)
            self._next_n()
            return

        # CHECKING divisors
        if self.state == "CHECKING":
            d = self.divs[self.i]
            r = self.n % d
            self.last_check_d = d
            self.last_check_r = r
            self.last_event = "CHECK"

            if r == 0:
                self.last_witness = d
                self.last_event = "WITNESS"

                dot = DotRec(self.n, self.k, "red")
                self.dots.append(dot)
                append_dot(self.csv_path, dot)
                self._next_n()
                return
            else:
                dot = DotRec(self.n, self.k, "green")
                self.dots.append(dot)
                append_dot(self.csv_path, dot)
                self.k += 1
                self.i += 1
                if self.i >= len(self.divs):
                    self.state = "PRIME"
                return

        # PRIME → fill to height n with green
        if self.state == "PRIME":
            self.last_event = "PRIME_FILL"
            self.last_check_d = None
            self.last_check_r = None

            if self.k < self.n:
                dot = DotRec(self.n, self.k, "green")
                self.dots.append(dot)
                append_dot(self.csv_path, dot)
                self.k += 1
                return
            else:
                self.last_event = "NEXT_N"
                self._next_n()
                return


# ----------------------------
# Rendering helpers
# ----------------------------
def font_cached(name: str, size: int, bold: bool = False) -> pygame.font.Font:
    key = (name, size, bold)
    if key in font_cached._cache:
        return font_cached._cache[key]
    f = pygame.font.SysFont(name, size, bold=bold)
    font_cached._cache[key] = f
    return f
font_cached._cache = {}


def draw_text(surf, text: str, x: int, y: int, size: int = 24,
              color: Tuple[int, int, int] = COL_TEXT,
              name: str = "dejavuserif", bold: bool = False,
              anchor: str = "topleft") -> pygame.Rect:
    f = font_cached(name, size, bold=bold)
    img = f.render(text, True, color)
    r = img.get_rect()
    setattr(r, anchor, (x, y))
    surf.blit(img, r)
    return r


def draw_panel(surf, rect: pygame.Rect, title: str, lines: List[Tuple[str, Tuple[int, int, int]]]):
    pygame.draw.rect(surf, COL_PANEL, rect, border_radius=10)
    pygame.draw.rect(surf, COL_PANEL_STROKE, rect, width=2, border_radius=10)
    x = rect.x + 14
    y = rect.y + 10
    draw_text(surf, title, x, y, size=22, color=COL_HI, bold=True, anchor="topleft")
    y += 30
    for t, c in lines:
        draw_text(surf, t, x, y, size=20, color=c, anchor="topleft")
        y += 24


# ----------------------------
# Towers world mapping (your original approach)
# ----------------------------
def world_to_screen(wx: float, wy: float, cx: float, cy: float, scale: float):
    sx = W * 0.5 + (wx - cx) * scale
    sy = H * ORIGIN_SCREEN_Y_FRAC - (wy - cy) * scale
    return int(sx), int(sy)


def draw_tower_grid(surface, cx: float, cy: float, scale: float):
    surface.fill(BG)
    _, by = world_to_screen(0, 0, cx, cy, scale)
    pygame.draw.line(surface, (70, 70, 85), (0, by), (W, by), 2)


# ----------------------------
# Graph mapping (screen-space axes)
# ----------------------------
def draw_axes(surface, rect: pygame.Rect, title: str, xlabel: str, ylabel: str,
              x_min: float, x_max: float, y_min: float, y_max: float,
              x_ticks: List[float], y_ticks: List[float]):
    surface.fill(BG)

    draw_text(surface, title, W // 2, 18, size=34, bold=True, anchor="midtop")

    pygame.draw.rect(surface, (40, 40, 50), rect, width=2, border_radius=10)

    draw_text(surface, xlabel, rect.centerx, rect.bottom + 10, size=22, color=COL_SUB, anchor="midtop")
    draw_text(surface, ylabel, rect.left, rect.top - 28, size=22, color=COL_SUB, anchor="topleft")

    def x_to_px(x: float) -> int:
        if x_max == x_min:
            return rect.left
        t = (x - x_min) / (x_max - x_min)
        return int(rect.left + t * rect.width)

    def y_to_py(y: float) -> int:
        if y_max == y_min:
            return rect.bottom
        t = (y - y_min) / (y_max - y_min)
        return int(rect.bottom - t * rect.height)

    # Ticks
    for xv in x_ticks:
        px = x_to_px(xv)
        pygame.draw.line(surface, (60, 60, 75), (px, rect.bottom), (px, rect.bottom + 6), 2)
        draw_text(surface, f"{int(xv)}", px, rect.bottom + 10, size=18, color=COL_SUB, anchor="midtop")

    for yv in y_ticks:
        py = y_to_py(yv)
        pygame.draw.line(surface, (60, 60, 75), (rect.left - 6, py), (rect.left, py), 2)
        lab = f"{yv:.2f}" if (abs(yv) < 10 and (yv % 1) != 0) else f"{yv:g}"
        draw_text(surface, lab, rect.left - 10, py, size=18, color=COL_SUB, anchor="midright")

    return x_to_px, y_to_py


# ----------------------------
# Precompute: ramp + density
# ----------------------------
def compute_ramp(P_MAX: int):
    primes_list = primes_upto(P_MAX)
    xs = list(range(1, len(primes_list) + 1))  # prime index k
    ys = [sans_space(p) for p in primes_list]

    # boundary markers: first prime index where p >= 2^m
    boundaries: List[Tuple[int, int]] = []
    m = 1
    while (1 << m) <= P_MAX:
        B = 1 << m
        idx = None
        for i, p in enumerate(primes_list):
            if p >= B:
                idx = i + 1
                break
        if idx is not None:
            boundaries.append((m, idx))
        m += 1

    return primes_list, xs, ys, boundaries


def compute_density(bits_min: int, bits_max: int):
    stats = []
    for b in range(bits_min, bits_max + 1):
        start = 2 ** (b - 1)
        end = (2 ** b) - 1
        size = end - start + 1
        primes_count = sum(1 for n in range(start, end + 1) if is_prime(n))
        dens = primes_count / size
        stats.append((b, start, end, dens))
    return stats


# ----------------------------
# Big modcheck HUD (Mode 1)
# ----------------------------
def draw_big_modcheck(screen, stepper: PrimeStepper, scale: float):
    big = ""
    big_color = COL_TEXT

    if stepper.state == "DONE":
        big = "DONE"
        big_color = COL_HI
    elif stepper.last_event == "TRIVIAL":
        big = f"{stepper.n} < 2  (spacer)"
        big_color = COL_DOT_YELLOW
    elif stepper.last_event in ("CHECK", "WITNESS") and stepper.last_check_d is not None:
        d = stepper.last_check_d
        r = stepper.last_check_r if stepper.last_check_r is not None else -1
        if r == 0:
            big = f"{stepper.n} % {d} = 0   →   COMPOSITE"
            big_color = COL_DOT_RED
        else:
            big = f"{stepper.n} % {d} = {r}"
            big_color = COL_TEXT
    elif stepper.last_event == "PRIME_FILL":
        big = f"PRIME FILL   {stepper.k} / {stepper.n}"
        big_color = COL_DOT_GREEN
    else:
        big = f"n = {stepper.n}"
        big_color = COL_SUB

    # Backing panel + centered text
    big_font = font_cached("dejavusansmono", 54, bold=True)
    big_img = big_font.render(big, True, big_color)
    big_rect = big_img.get_rect(center=(W // 2, 165))

    pad_x, pad_y = 18, 12
    back = pygame.Rect(
        big_rect.x - pad_x,
        big_rect.y - pad_y,
        big_rect.width + 2 * pad_x,
        big_rect.height + 2 * pad_y,
    )
    pygame.draw.rect(screen, COL_PANEL, back, border_radius=12)
    pygame.draw.rect(screen, COL_PANEL_STROKE, back, width=2, border_radius=12)
    screen.blit(big_img, big_rect)

    # Small helper line under it
    draw_text(
        screen,
        "SPACE step/hold • A auto • UP/DOWN speed • LEFT/RIGHT zoom • 1/2/3/4 modes",
        W // 2,
        back.bottom + 8,
        size=18,
        color=COL_SUB,
        anchor="midtop",
    )


# ----------------------------
# Main
# ----------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Prime Division Mode + Analysis (pygame)")
    clock = pygame.time.Clock()

    # CSV + stepper
    csv_path = new_csv_path()
    write_csv_header(csv_path)
    stepper = PrimeStepper(csv_path)

    # View camera for towers
    cx = (N_MIN + N_MAX) / 2
    cy = 0.0

    # Global zoom
    scale = SCALE_DEFAULT

    # Stepping controls (mode 1 only)
    auto = False
    auto_acc = 0.0
    auto_rate = AUTO_CHECKS_PER_SEC_DEFAULT

    space_held = False
    space_acc = 0.0

    # Modes:
    # 1 towers, 2 ramp, 3 density, 4 end
    mode = 1

    # Precompute analysis views
    primes_list, ramp_xs, ramp_ys, ramp_boundaries = compute_ramp(P_MAX_RAMP)
    dens_stats = compute_density(MIN_BITS_DENS, MAX_BITS_DENS)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        # -------- events --------
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False

                # Mode switches
                elif e.key == pygame.K_1:
                    mode = 1
                elif e.key == pygame.K_2:
                    mode = 2
                    auto = False
                    space_held = False
                elif e.key == pygame.K_3:
                    mode = 3
                    auto = False
                    space_held = False
                elif e.key == pygame.K_4:
                    mode = 4
                    auto = False
                    space_held = False

                # Zoom (all modes)
                elif e.key == pygame.K_LEFT:
                    scale = max(SCALE_MIN, scale / ZOOM_STEP)
                elif e.key == pygame.K_RIGHT:
                    scale = min(SCALE_MAX, scale * ZOOM_STEP)

                # Towers-mode controls
                if mode == 1:
                    if e.key == pygame.K_SPACE:
                        stepper.step()
                        space_held = True
                    elif e.key == pygame.K_a:
                        auto = not auto
                    elif e.key == pygame.K_UP:
                        auto_rate = min(AUTO_CHECKS_PER_SEC_MAX, auto_rate * 1.25)
                    elif e.key == pygame.K_DOWN:
                        auto_rate = max(AUTO_CHECKS_PER_SEC_MIN, auto_rate / 1.25)
                    elif e.key == pygame.K_r:
                        stepper.reset_keep()
                    elif e.key == pygame.K_c:
                        csv_path = new_csv_path()
                        write_csv_header(csv_path)
                        stepper.clear_new_csv(csv_path)

            elif e.type == pygame.KEYUP:
                if e.key == pygame.K_SPACE:
                    space_held = False
                    space_acc = 0.0

        # -------- stepping logic (mode 1 only) --------
        if mode == 1:
            if auto:
                auto_acc += dt
                step_dt = 1.0 / max(0.0001, auto_rate)
                while auto_acc >= step_dt:
                    auto_acc -= step_dt
                    stepper.step()

            if space_held and not auto:
                space_acc += dt
                step_dt = 1.0 / SPACE_REPEAT_RATE
                while space_acc >= step_dt:
                    space_acc -= step_dt
                    stepper.step()

        # -------- render --------
        if mode == 1:
            # Towers view
            draw_tower_grid(screen, cx, cy, scale)

            # Column lines
            maxk: Dict[int, int] = {}
            for d in stepper.dots:
                mk = maxk.get(d.n, -10)
                if d.k > mk:
                    maxk[d.n] = d.k

            for n, mk in maxk.items():
                sx0, sy0 = world_to_screen(n, 0, cx, cy, scale)
                sx1, sy1 = world_to_screen(n, mk + 0.2, cx, cy, scale)
                pygame.draw.line(screen, COL_COLLINE, (sx0, sy0), (sx1, sy1), 1)

            # Draw dots
            for dot in stepper.dots:
                sx, sy = world_to_screen(dot.n, dot.k, cx, cy, scale)
                col = (
                    COL_DOT_GREEN if dot.color == "green"
                    else COL_DOT_RED if dot.color == "red"
                    else COL_DOT_YELLOW
                )
                pygame.draw.circle(screen, col, (sx, sy), DOT_RADIUS)

            # Top HUD
            draw_text(screen, "MODE 1: PRIME DIVISION TOWERS", 14, 10, size=22, bold=True)
            draw_text(screen, "1 Towers   2 Ramp   3 Density   4 End", 14, 36, size=18, color=COL_SUB)

            st = stepper.state
            n_now = stepper.n
            auto_txt = "ON" if auto else "OFF"
            draw_text(
                screen,
                f"n={n_now}   state={st}   auto={auto_txt}   auto_rate={auto_rate:.2f}/s   zoom={scale:.2f}",
                14, 58, size=18, color=COL_SUB
            )

            # Legend panel
            panel_rect = pygame.Rect(W - 410, 10, 395, 128)
            lines = [
                ("GREEN  = n % d ≠ 0  (or prime-fill)", COL_DOT_GREEN),
                ("RED    = n % d = 0  (witness)", COL_DOT_RED),
                ("YELLOW = spacer (n < 2)", COL_DOT_YELLOW),
            ]
            draw_panel(screen, panel_rect, "Legend", lines)

            # Big realtime modcheck text
            draw_big_modcheck(screen, stepper, scale)

        elif mode == 2:
            # Sans-space ramp
            plot_rect = pygame.Rect(85, 120, W - 170, H - 220)

            if ramp_xs:
                x_min, x_max = 1, max(ramp_xs)
                y_min, y_max = 0, max(ramp_ys) if ramp_ys else 1
            else:
                x_min, x_max, y_min, y_max = 0, 1, 0, 1

            x_ticks = [1, x_max] if x_max > 1 else [1]
            if x_max > 10:
                x_ticks = [1, x_max // 2, x_max]
            y_ticks = [0, y_max] if y_max > 0 else [0]

            x_to_px, y_to_py = draw_axes(
                screen,
                plot_rect,
                "Sans-Space Ramp (Primes Only)",
                "prime index k",
                "s(p) = p - 2^(bitlen(p)-1)",
                x_min, x_max, y_min, y_max,
                x_ticks=x_ticks,
                y_ticks=y_ticks,
            )

            draw_text(screen, "MODE 2: SANS-SPACE RAMP", 14, 10, size=22, bold=True)
            draw_text(screen, "Blue line = bucket structure   Green dots = primes sampling that structure", 14, 36, size=18, color=COL_SUB)
            draw_text(screen, "Vertical markers show where primes first reach each power of two (bucket boundaries).", 14, 58, size=18, color=COL_SUB)

            if ramp_xs:
                pts = [(x_to_px(x), y_to_py(y)) for x, y in zip(ramp_xs, ramp_ys)]
                if len(pts) >= 2:
                    pygame.draw.lines(screen, COL_BLUE, False, pts, 3)

                for px, py in pts:
                    pygame.draw.circle(screen, COL_DOT_GREEN, (px, py), 3)

            for m, idx in ramp_boundaries:
                px = x_to_px(idx)
                pygame.draw.line(screen, COL_WHITE, (px, plot_rect.top), (px, plot_rect.bottom), 1)
                draw_text(screen, f"2^{m}", px, plot_rect.top - 8, size=16, color=COL_WHITE, name="dejavusansmono", anchor="midbottom")

            draw_text(screen, "1 Towers   2 Ramp   3 Density   4 End", 14, H - 28, size=18, color=COL_SUB)

        elif mode == 3:
            # Density decay by bit bucket
            plot_rect = pygame.Rect(85, 120, W - 170, H - 220)

            xs = [b for (b, _, _, _) in dens_stats]
            ys = [d for (_, _, _, d) in dens_stats]

            x_min, x_max = min(xs), max(xs)
            y_min = max(0.0, min(ys) - 0.02)
            y_max = min(1.0, max(ys) + 0.02)

            x_ticks = list(range(MIN_BITS_DENS, MAX_BITS_DENS + 1, 2))
            y_ticks = [round(y_min + i * (y_max - y_min) / 5.0, 2) for i in range(6)]

            x_to_px, y_to_py = draw_axes(
                screen,
                plot_rect,
                "Prime Density Thins Out With Scale",
                "bits (bucket width)",
                "prime density in [2^(b-1), 2^b-1]",
                x_min, x_max, y_min, y_max,
                x_ticks=x_ticks,
                y_ticks=y_ticks,
            )

            draw_text(screen, "MODE 3: DENSITY DECAY", 14, 10, size=22, bold=True)
            draw_text(screen, "Blue = measured density by bit bucket", 14, 36, size=18, color=COL_SUB)
            draw_text(screen, "Purple = ~ 1/ln(n) guide (n = midpoint of bucket)", 14, 58, size=18, color=COL_SUB)

            pts = [(x_to_px(b), y_to_py(d)) for (b, _, _, d) in dens_stats]
            if len(pts) >= 2:
                pygame.draw.lines(screen, COL_BLUE, False, pts, 3)
            for px, py in pts:
                pygame.draw.circle(screen, COL_BLUE, (px, py), 4)

            guide_pts = []
            for (b, start, end, _) in dens_stats:
                mid = 0.5 * (start + end)
                g = 1.0 / math.log(mid)
                guide_pts.append((x_to_px(b), y_to_py(g)))
            if len(guide_pts) >= 2:
                pygame.draw.lines(screen, COL_PURPLE, False, guide_pts, 3)

            panel_rect = pygame.Rect(W - 420, H - 190, 405, 170)
            lines = []
            prev = None
            for (b, _, _, d) in dens_stats:
                if prev is None:
                    lines.append((f"b={b:>2}   Δ%=+0.00", COL_TEXT))
                else:
                    dp = 0.0 if prev == 0 else 100.0 * (d - prev) / prev
                    sign = "+" if dp >= 0 else "-"
                    lines.append((f"b={b:>2}   Δ%={sign}{abs(dp):5.2f}", COL_TEXT))
                prev = d
            lines_show = lines[-5:]
            draw_panel(screen, panel_rect, "Per-bucket thinning (last 5)", lines_show)

            draw_text(screen, "1 Towers   2 Ramp   3 Density   4 End", 14, H - 28, size=18, color=COL_SUB)

        else:
            # End card
            screen.fill(BG)
            draw_text(screen, "Thanks For Watching!", W // 2, H // 2 - 50, size=64, bold=True, anchor="center")
            draw_text(screen, "- ONOJK123", W // 2, H // 2 + 20, size=40, color=COL_HI, anchor="center")
            draw_text(screen, "1 Towers   2 Ramp   3 Density   4 End", W // 2, H - 40, size=18, color=COL_SUB, anchor="center")

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
