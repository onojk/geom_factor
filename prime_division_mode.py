# prime_division_mode.py
#
# Prime Division Laser Graph (n=1..300)
# - SPACE tap: one mod check (one dot)
# - SPACE hold: repeatedly steps (scroll-like) at SPACE_REPEAT_RATE
# - A toggles auto-advance (separate from SPACE-hold)
# - Dots persist (vertical "laser" columns)
# - Columns build UP from a baseline near the bottom
# - Shows dots for ALL checks (including n % d != 0)
# - Stops early on first divisor hit (composite), otherwise ends as prime
# - Logs every check to a CSV
#
# Controls:
#   SPACE : tap = step, hold = scroll-step
#   A     : toggle auto-advance
#   UP    : faster auto (more checks/sec)
#   DOWN  : slower auto
#   LEFT  : zoom out
#   RIGHT : zoom in
#   R     : reset n back to 1 (keeps dots + same csv)
#   C     : clear dots AND start a new csv
#   ESC   : quit
#
# Run:
#   source .venv/bin/activate
#   python3 prime_division_mode.py

import csv
import math
import time
from dataclasses import dataclass
from datetime import datetime

import pygame


# ----------------------------
# Config
# ----------------------------
W, H = 1280, 720
FPS = 60

N_MIN = 1
N_MAX = 300

# Baseline position for y=0 (fraction of screen height)
ORIGIN_SCREEN_Y_FRAC = 0.90

# Visuals
BG = (8, 8, 10)
GRID_MAJOR = (24, 24, 30)
GRID_MINOR = (16, 16, 20)

COL_TEXT = (205, 205, 210)

COL_DOT_PASS = (80, 255, 120)      # n % d != 0
COL_DOT_HIT = (255, 80, 80)        # n % d == 0
COL_DOT_TRIVIAL = (200, 200, 90)   # n<2 marker
COL_COLLINE = (55, 55, 65)

DOT_RADIUS = 2

# Scaling (dense by default)
SCALE_DEFAULT = 3.2
SCALE_MIN = 1.0
SCALE_MAX = 25.0
ZOOM_STEP = 1.12

# Auto pacing
AUTO_CHECKS_PER_SEC_DEFAULT = 6.0
AUTO_CHECKS_PER_SEC_MIN = 0.2
AUTO_CHECKS_PER_SEC_MAX = 120.0

# SPACE-hold pacing (manual "scroll")
SPACE_REPEAT_RATE = 18.0  # checks/sec while holding space (feel free to tweak)


# ----------------------------
# Data
# ----------------------------
@dataclass(frozen=True)
class Dot:
    n: int
    k: int
    d: int
    rem: int
    is_hit: bool
    note: str


# ----------------------------
# Trial divisors
# ----------------------------
def trial_divisors_for_n(n: int):
    """
    Divisors we test:
      - n < 2: none
      - n == 2: none
      - n > 2: test d = 2..floor(sqrt(n))
    """
    if n < 2:
        return []
    if n == 2:
        return []
    limit = int(math.isqrt(n))
    if limit < 2:
        return []
    return list(range(2, limit + 1))


# ----------------------------
# Coordinate mapping
# ----------------------------
def world_to_screen(wx: float, wy: float, cx: float, cy: float, scale: float):
    # y increases UP in world; pygame y increases DOWN
    sx = W * 0.5 + (wx - cx) * scale
    baseline_sy = H * ORIGIN_SCREEN_Y_FRAC
    sy = baseline_sy - (wy - cy) * scale
    return int(sx), int(sy)


def draw_grid(surface, cx, cy, scale):
    surface.fill(BG)

    # visible bounds (approx)
    left_wx = (0 - W * 0.5) / scale + cx
    right_wx = (W - W * 0.5) / scale + cx

    baseline_sy = H * ORIGIN_SCREEN_Y_FRAC
    top_wy = (baseline_sy - 0) / scale + cy
    bottom_wy = (baseline_sy - H) / scale + cy

    min_wx = math.floor(left_wx) - 1
    max_wx = math.ceil(right_wx) + 1
    min_wy = math.floor(min(bottom_wy, top_wy)) - 1
    max_wy = math.ceil(max(bottom_wy, top_wy)) + 1

    # Coarse grid for dense scale
    if scale < 2.0:
        minor_step = 25.0
        major_step = 50.0
    elif scale < 4.0:
        minor_step = 10.0
        major_step = 20.0
    elif scale < 8.0:
        minor_step = 5.0
        major_step = 10.0
    else:
        minor_step = 1.0
        major_step = 5.0

    # Vertical lines
    x = math.floor(min_wx / minor_step) * minor_step
    while x <= max_wx:
        is_major = abs((x / major_step) - round(x / major_step)) < 1e-9
        col = GRID_MAJOR if is_major else GRID_MINOR
        sx, _ = world_to_screen(x, 0, cx, cy, scale)
        pygame.draw.line(surface, col, (sx, 0), (sx, H), 1)
        x += minor_step

    # Horizontal lines
    y = math.floor(min_wy / minor_step) * minor_step
    while y <= max_wy:
        is_major = abs((y / major_step) - round(y / major_step)) < 1e-9
        col = GRID_MAJOR if is_major else GRID_MINOR
        _, sy = world_to_screen(0, y, cx, cy, scale)
        pygame.draw.line(surface, col, (0, sy), (W, sy), 1)
        y += minor_step

    # Baseline (y=0)
    _, by = world_to_screen(0, 0, cx, cy, scale)
    pygame.draw.line(surface, (70, 70, 85), (0, by), (W, by), 2)


# ----------------------------
# CSV logging
# ----------------------------
def new_csv_path():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"prime_div_checks_1to{N_MAX}_{ts}.csv"


def write_csv_header(path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_unix", "n", "k", "d", "rem", "is_hit", "note"])


def append_dot_to_csv(path, dot: Dot):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([time.time(), dot.n, dot.k, dot.d, dot.rem, int(dot.is_hit), dot.note])


# ----------------------------
# Simulation state
# ----------------------------
class PrimeDivisionStepper:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.dots: list[Dot] = []

        self.n = N_MIN
        self.divs = trial_divisors_for_n(self.n)
        self.i = 0
        self.k = 0
        self.state = "INIT"  # TRIVIAL, CHECKING, PRIME, COMPOSITE, DONE
        self._sync_state_for_n()

    def _sync_state_for_n(self):
        if self.n > N_MAX:
            self.state = "DONE"
            return

        self.divs = trial_divisors_for_n(self.n)
        self.i = 0
        self.k = 0

        if self.n < 2:
            self.state = "TRIVIAL"
        elif self.n == 2:
            self.state = "PRIME"
        else:
            if len(self.divs) == 0:
                self.state = "PRIME"
            else:
                self.state = "CHECKING"

    def reset_n_only(self):
        self.n = N_MIN
        self._sync_state_for_n()

    def clear_and_new_csv(self):
        self.dots.clear()
        self.csv_path = new_csv_path()
        write_csv_header(self.csv_path)
        self.n = N_MIN
        self._sync_state_for_n()

    def _advance_to_next_n(self):
        self.n += 1
        self._sync_state_for_n()

    def step_one_check(self):
        """One dot-step (or advance if this n has no checks)."""
        if self.state == "DONE":
            return

        # n with no checks: SPACE advances (drop a trivial marker for n<2)
        if self.state in ("TRIVIAL", "PRIME"):
            if self.state == "TRIVIAL":
                dot = Dot(n=self.n, k=0, d=0, rem=0, is_hit=False, note="n<2")
                self.dots.append(dot)
                append_dot_to_csv(self.csv_path, dot)
            self._advance_to_next_n()
            return

        # composite: next SPACE moves on
        if self.state == "COMPOSITE":
            self._advance_to_next_n()
            return

        if self.state != "CHECKING":
            self._advance_to_next_n()
            return

        # CHECKING: one divisor per step
        if self.i >= len(self.divs):
            self.state = "PRIME"
            return

        d = self.divs[self.i]
        rem = self.n % d
        is_hit = (rem == 0)

        dot = Dot(n=self.n, k=self.k, d=d, rem=rem, is_hit=is_hit, note="")
        self.dots.append(dot)
        append_dot_to_csv(self.csv_path, dot)

        self.k += 1
        self.i += 1

        if is_hit:
            self.state = "COMPOSITE"
        elif self.i >= len(self.divs):
            self.state = "PRIME"


# ----------------------------
# Rendering helpers
# ----------------------------
def draw_text(surface, font, x, y, text, color=COL_TEXT):
    surface.blit(font.render(text, True, color), (x, y))


# ----------------------------
# Main
# ----------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Prime Division Laser Graph (1..300) — hold SPACE to scroll")
    clock = pygame.time.Clock()

    # Smaller fonts for dense view
    font = pygame.font.SysFont("DejaVu Sans Mono", 14)
    font_title = pygame.font.SysFont("DejaVu Serif", 40)

    csv_path = new_csv_path()
    write_csv_header(csv_path)
    stepper = PrimeDivisionStepper(csv_path)

    # Camera: center over range
    cx = (N_MIN + N_MAX) / 2.0
    cy = 0.0
    scale = SCALE_DEFAULT

    # Auto stepping
    auto = False
    auto_checks_per_sec = AUTO_CHECKS_PER_SEC_DEFAULT
    auto_acc = 0.0

    # SPACE hold stepping
    space_held = False
    space_acc = 0.0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False

                elif e.key == pygame.K_SPACE:
                    # immediate step on press, then repeat while held
                    stepper.step_one_check()
                    space_held = True

                elif e.key == pygame.K_a:
                    auto = not auto

                elif e.key == pygame.K_UP:
                    auto_checks_per_sec = min(AUTO_CHECKS_PER_SEC_MAX, auto_checks_per_sec * 1.25)

                elif e.key == pygame.K_DOWN:
                    auto_checks_per_sec = max(AUTO_CHECKS_PER_SEC_MIN, auto_checks_per_sec / 1.25)

                elif e.key == pygame.K_RIGHT:
                    scale = min(SCALE_MAX, scale * ZOOM_STEP)

                elif e.key == pygame.K_LEFT:
                    scale = max(SCALE_MIN, scale / ZOOM_STEP)

                elif e.key == pygame.K_r:
                    stepper.reset_n_only()

                elif e.key == pygame.K_c:
                    stepper.clear_and_new_csv()

            elif e.type == pygame.KEYUP:
                if e.key == pygame.K_SPACE:
                    space_held = False
                    space_acc = 0.0

        # Auto mode (one dot per tick)
        if auto and stepper.state != "DONE":
            auto_acc += dt
            period = 1.0 / max(1e-9, auto_checks_per_sec)
            while auto_acc >= period:
                auto_acc -= period
                stepper.step_one_check()

        # SPACE-hold mode (manual scroll stepping)
        # If auto is ON, auto wins and we ignore hold-space repeating
        if space_held and (not auto) and stepper.state != "DONE":
            space_acc += dt
            period = 1.0 / max(1e-9, SPACE_REPEAT_RATE)
            while space_acc >= period:
                space_acc -= period
                stepper.step_one_check()

        # Draw
        draw_grid(screen, cx, cy, scale)

        # Optional faint column lines (only if zoomed in enough)
        maxk = {}
        for dot in stepper.dots:
            maxk[dot.n] = max(maxk.get(dot.n, 0), dot.k)

        if scale >= 2.0:
            for n, mk in maxk.items():
                sx0, sy0 = world_to_screen(float(n), 0.0, cx, cy, scale)
                sx1, sy1 = world_to_screen(float(n), float(mk + 0.2), cx, cy, scale)
                pygame.draw.line(screen, COL_COLLINE, (sx0, sy0), (sx1, sy1), 1)

        # Dots
        for dot in stepper.dots:
            sx, sy = world_to_screen(float(dot.n), float(dot.k), cx, cy, scale)
            if dot.note == "n<2":
                col = COL_DOT_TRIVIAL
            else:
                col = COL_DOT_HIT if dot.is_hit else COL_DOT_PASS
            pygame.draw.circle(screen, col, (sx, sy), DOT_RADIUS)

        # UI
        draw_text(screen, font_title, 18, 10, "Prime Buckets", (140, 140, 150))

        n = stepper.n
        state = stepper.state
        next_d = None
        if state == "CHECKING" and stepper.i < len(stepper.divs):
            next_d = stepper.divs[stepper.i]

        line1 = f"n={n}/{N_MAX}   state={state}" + (f"   next d={next_d}" if next_d else "")
        line2 = f"auto={'ON' if auto else 'OFF'}  auto_speed={auto_checks_per_sec:.1f}/s  hold_space_speed={SPACE_REPEAT_RATE:.1f}/s"
        line3 = f"dots={len(stepper.dots)}   csv: {stepper.csv_path}"

        draw_text(screen, font, 18, 66, "SPACE tap=step, hold=scroll (one dot = one mod check)", (210, 210, 215))
        draw_text(screen, font, 18, 86, line1, (190, 190, 198))
        draw_text(screen, font, 18, 104, line2, (190, 190, 198))
        draw_text(screen, font, 18, 122, line3, (160, 160, 170))

        controls = "SPACE tap/hold | A auto | UP/DOWN auto speed | LEFT/RIGHT zoom | R reset n | C clear+new csv | ESC quit"
        draw_text(screen, font, 18, 142, controls, (170, 170, 180))

        legend = "Green: n%d!=0 (pass)   Red: n%d==0 (hit => composite witness)"
        draw_text(screen, font, 18, H - 22, legend, (150, 150, 165))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
