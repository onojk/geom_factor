# prime_division_mode.py
#
# Prime Division Laser Graph (n=1..300)
#
# FINAL SEMANTICS:
#   GREEN  = n % d != 0 OR synthetic prime-fill (work completed)
#   RED    = n % d == 0 (composite witness)
#   YELLOW = spacer / structural marker only
#
# HEIGHT RULE:
#   Prime n      → exactly n green dots tall
#   Composite n  → < n dots, ending in one red dot
#
# One dot == one discrete unit of work or structure
#
# Controls:
#   SPACE : tap = step, hold = scroll-step
#   A     : toggle auto-advance
#   UP    : faster auto
#   DOWN  : slower auto
#   LEFT  : zoom out
#   RIGHT : zoom in
#   R     : reset n back to 1 (keeps dots + same csv)
#   C     : clear dots AND start a new csv
#   ESC   : quit

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
ORIGIN_SCREEN_Y_FRAC = 0.90

# Colors
BG = (8, 8, 10)
GRID_MAJOR = (24, 24, 30)
GRID_MINOR = (16, 16, 20)

COL_DOT_GREEN = (80, 255, 120)
COL_DOT_RED = (255, 80, 80)
COL_DOT_YELLOW = (255, 230, 90)
COL_COLLINE = (55, 55, 65)

DOT_RADIUS = 2

# Scaling
SCALE_DEFAULT = 3.2
SCALE_MIN = 1.0
SCALE_MAX = 25.0
ZOOM_STEP = 1.12

# Timing
AUTO_CHECKS_PER_SEC_DEFAULT = 6.0
SPACE_REPEAT_RATE = 18.0


# ----------------------------
# Data
# ----------------------------
@dataclass(frozen=True)
class Dot:
    n: int
    k: int
    color: str   # "green" | "red" | "yellow"


# ----------------------------
# Divisors (sqrt trial)
# ----------------------------
def trial_divisors_for_n(n: int):
    if n < 2:
        return []
    limit = int(math.isqrt(n))
    return list(range(2, limit + 1))


# ----------------------------
# Coordinate mapping
# ----------------------------
def world_to_screen(wx, wy, cx, cy, scale):
    sx = W * 0.5 + (wx - cx) * scale
    sy = H * ORIGIN_SCREEN_Y_FRAC - (wy - cy) * scale
    return int(sx), int(sy)


def draw_grid(surface, cx, cy, scale):
    surface.fill(BG)
    _, by = world_to_screen(0, 0, cx, cy, scale)
    pygame.draw.line(surface, (70, 70, 85), (0, by), (W, by), 2)


# ----------------------------
# CSV
# ----------------------------
def new_csv_path():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"prime_division_{ts}.csv"


def write_csv_header(path):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(["n", "k", "color"])


def append_dot(path, dot):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([dot.n, dot.k, dot.color])


# ----------------------------
# Simulation
# ----------------------------
class PrimeStepper:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.dots = []

        self.n = N_MIN
        self.divs = []
        self.i = 0
        self.k = 0
        self.state = "INIT"

        self._sync()

    def _sync(self):
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

    def step(self):
        if self.state == "DONE":
            return

        # TRIVIAL spacer
        if self.state == "TRIVIAL":
            d = Dot(self.n, 0, "yellow")
            self.dots.append(d)
            append_dot(self.csv_path, d)
            self._next_n()
            return

        # CHECKING divisors
        if self.state == "CHECKING":
            d = self.divs[self.i]
            if self.n % d == 0:
                dot = Dot(self.n, self.k, "red")
                self.dots.append(dot)
                append_dot(self.csv_path, dot)
                self._next_n()
                return
            else:
                dot = Dot(self.n, self.k, "green")
                self.dots.append(dot)
                append_dot(self.csv_path, dot)
                self.k += 1
                self.i += 1
                if self.i >= len(self.divs):
                    self.state = "PRIME"
                return

        # PRIME → fill to height n with green
        if self.state == "PRIME":
            if self.k < self.n:
                dot = Dot(self.n, self.k, "green")
                self.dots.append(dot)
                append_dot(self.csv_path, dot)
                self.k += 1
                return
            else:
                self._next_n()
                return


# ----------------------------
# Main
# ----------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Prime Towers — n-tall primes")
    clock = pygame.time.Clock()

    csv_path = new_csv_path()
    write_csv_header(csv_path)
    stepper = PrimeStepper(csv_path)

    cx = (N_MIN + N_MAX) / 2
    cy = 0.0
    scale = SCALE_DEFAULT

    auto = False
    auto_acc = 0.0
    space_held = False
    space_acc = 0.0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    stepper.step()
                    space_held = True
                elif e.key == pygame.K_a:
                    auto = not auto
                elif e.key == pygame.K_LEFT:
                    scale = max(SCALE_MIN, scale / ZOOM_STEP)
                elif e.key == pygame.K_RIGHT:
                    scale = min(SCALE_MAX, scale * ZOOM_STEP)
            elif e.type == pygame.KEYUP:
                if e.key == pygame.K_SPACE:
                    space_held = False
                    space_acc = 0.0

        if auto:
            auto_acc += dt
            while auto_acc >= 1 / AUTO_CHECKS_PER_SEC_DEFAULT:
                auto_acc -= 1 / AUTO_CHECKS_PER_SEC_DEFAULT
                stepper.step()

        if space_held and not auto:
            space_acc += dt
            while space_acc >= 1 / SPACE_REPEAT_RATE:
                space_acc -= 1 / SPACE_REPEAT_RATE
                stepper.step()

        draw_grid(screen, cx, cy, scale)

        # Column lines
        maxk = {}
        for d in stepper.dots:
            maxk[d.n] = max(maxk.get(d.n, 0), d.k)
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

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
