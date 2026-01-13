# bit_bucket_prime_towers_slow.py
#
# Smooth slow-build Prime Towers with:
# - tap/hold SPACE stepping
# - auto-run stepping
# - auto-zone (advance bits when done)
# - SMOOTHNESS FIX: cap how many visible steps happen per frame
#
# Controls:
#   SPACE  tap = step once, HOLD = step repeatedly (visible steps)
#   A      auto-run steps
#   Z      toggle AUTO-ZONE (advance bits when DONE)
#   UP     faster auto/hold
#   DOWN   slower auto/hold
#   [ ]    bits down/up
#   C      toggle cumulative vs bucket-only
#   R      rebuild current view
#   ← →    zoom X
#   ESC    quit

import math
import pygame
from dataclasses import dataclass
from typing import Dict, List, Optional

# ---------------- CONFIG ----------------
W, H = 1280, 720
FPS = 60

BG = (8, 8, 12)
WHITE = (235, 235, 245)
SUB = (160, 160, 180)

GREEN = (80, 255, 120)
RED = (255, 90, 90)
YELLOW = (255, 220, 90)

COLLINE = (55, 55, 70)
DOT_R = 2

BITS_MIN, BITS_MAX = 1, 18
BITS_DEFAULT = 1

X_SCALE_DEFAULT = 8.0
ZOOM_STEP = 1.12

HUD_H = 155
SIDE_PAD = 30
TOP_PAD = 18
PLOT_TOP = HUD_H + 20
PLOT_BOTTOM_PAD = 28

AUTO_RATE_DEFAULT = 4.0   # visible steps/sec
HOLD_RATE_DEFAULT = 14.0  # visible steps/sec

ZONE_PAUSE_SEC = 0.35

# ---- smoothness: cap visible steps per frame ----
MAX_VISIBLE_STEPS_PER_FRAME = 10
DT_CAP = 0.08  # cap dt to avoid huge catch-up after a hitch (80ms)

# ---------------- DATA ----------------
@dataclass
class Dot:
    n: int
    k: int
    color: str  # green/red/yellow

# ---------------- HELPERS ----------------
def font(size, bold=False):
    return pygame.font.SysFont("dejavusansmono", size, bold)

def trial_divs(n: int):
    if n < 2:
        return []
    return list(range(2, int(math.isqrt(n)) + 1))

# ---------------- STEPPER ----------------
class BucketStepper:
    def __init__(self, L: int, R: int):
        self.L = L
        self.R = R
        self.n = L
        self.divs: List[int] = []
        self.i = 0
        self.k = 0
        self.state = "INIT"
        self.done = False
        self.curr_d: Optional[int] = None

    def step(self) -> Optional[Dot]:
        if self.done:
            self.curr_d = None
            return None

        if self.state == "INIT":
            self.curr_d = None
            if self.n > self.R:
                self.done = True
                return None
            if self.n < 2:
                self.state = "TRIVIAL"
            else:
                self.divs = trial_divs(self.n)
                self.i = 0
                self.k = 0
                self.state = "CHECK" if self.divs else "PRIME"

        if self.state == "TRIVIAL":
            self.curr_d = None
            d = Dot(self.n, 0, "yellow")
            self.n += 1
            self.state = "INIT"
            return d

        if self.state == "CHECK":
            d0 = self.divs[self.i]
            self.curr_d = d0

            if self.n % d0 == 0:
                dot = Dot(self.n, self.k, "red")
                self.n += 1
                self.state = "INIT"
                self.curr_d = None
                return dot
            else:
                dot = Dot(self.n, self.k, "green")
                self.k += 1
                self.i += 1
                if self.i >= len(self.divs):
                    self.state = "PRIME"
                    self.curr_d = None
                return dot

        if self.state == "PRIME":
            self.curr_d = None
            if self.k < self.n:
                dot = Dot(self.n, self.k, "green")
                self.k += 1
                return dot
            else:
                self.n += 1
                self.state = "INIT"
                return None

        return None

# ---------------- MAIN ----------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Bit-Bucket Prime Towers — Smooth Slow Build + Auto Zones")
    clock = pygame.time.Clock()

    bits = BITS_DEFAULT
    cumulative = True

    x_scale = X_SCALE_DEFAULT

    auto = False
    auto_rate = AUTO_RATE_DEFAULT
    auto_acc = 0.0

    space_held = False
    hold_rate = HOLD_RATE_DEFAULT
    hold_acc = 0.0

    auto_zone = True
    zone_pause = 0.0

    def current_range():
        if cumulative:
            return 1, (1 << bits) - 1
        return (1 << (bits - 1)), (1 << bits) - 1

    def reset_view(clear_dots=True):
        nonlocal L, R, stepper, cx, zone_pause, auto_acc, hold_acc
        L, R = current_range()
        stepper = BucketStepper(L, R)
        if clear_dots:
            dots.clear()
            maxk.clear()
        cx = (L + R) / 2.0
        zone_pause = ZONE_PAUSE_SEC
        auto_acc = 0.0
        hold_acc = 0.0

    def step_visible() -> Optional[Dot]:
        # skip internal None transitions, but bounded
        for _ in range(16):
            if stepper.done:
                return None
            d = stepper.step()
            if d is not None:
                return d
        return None

    L, R = current_range()
    stepper = BucketStepper(L, R)
    dots: List[Dot] = []
    maxk: Dict[int, int] = {}
    cx = (L + R) / 2.0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        if dt > DT_CAP:
            dt = DT_CAP

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False

                elif e.key == pygame.K_SPACE:
                    d = step_visible()
                    if d is not None:
                        dots.append(d)
                        maxk[d.n] = max(maxk.get(d.n, 0), d.k)
                    space_held = True

                elif e.key == pygame.K_a:
                    auto = not auto

                elif e.key == pygame.K_z:
                    auto_zone = not auto_zone

                elif e.key == pygame.K_UP:
                    auto_rate = min(240.0, auto_rate * 1.25)
                    hold_rate = min(480.0, hold_rate * 1.25)

                elif e.key == pygame.K_DOWN:
                    auto_rate = max(0.5, auto_rate / 1.25)
                    hold_rate = max(1.0, hold_rate / 1.25)

                elif e.key == pygame.K_LEFT:
                    x_scale = max(0.8, x_scale / ZOOM_STEP)

                elif e.key == pygame.K_RIGHT:
                    x_scale = min(120.0, x_scale * ZOOM_STEP)

                elif e.key == pygame.K_c:
                    cumulative = not cumulative
                    reset_view(clear_dots=True)

                elif e.key == pygame.K_LEFTBRACKET:
                    bits = max(BITS_MIN, bits - 1)
                    reset_view(clear_dots=True)

                elif e.key == pygame.K_RIGHTBRACKET:
                    bits = min(BITS_MAX, bits + 1)
                    reset_view(clear_dots=True)

                elif e.key == pygame.K_r:
                    reset_view(clear_dots=True)

            elif e.type == pygame.KEYUP:
                if e.key == pygame.K_SPACE:
                    space_held = False
                    hold_acc = 0.0

        # pause between zones
        if zone_pause > 0.0:
            zone_pause = max(0.0, zone_pause - dt)

        # HOLD SPACE (bounded per frame)
        if zone_pause == 0.0 and space_held and (not auto) and (not stepper.done):
            hold_acc += dt
            step_dt = 1.0 / max(0.0001, hold_rate)
            steps_this_frame = 0
            while hold_acc >= step_dt and (not stepper.done) and steps_this_frame < MAX_VISIBLE_STEPS_PER_FRAME:
                hold_acc -= step_dt
                d = step_visible()
                if d is not None:
                    dots.append(d)
                    maxk[d.n] = max(maxk.get(d.n, 0), d.k)
                steps_this_frame += 1

        # AUTO (bounded per frame)
        if zone_pause == 0.0 and auto and (not stepper.done):
            auto_acc += dt
            step_dt = 1.0 / max(0.0001, auto_rate)
            steps_this_frame = 0
            while auto_acc >= step_dt and (not stepper.done) and steps_this_frame < MAX_VISIBLE_STEPS_PER_FRAME:
                auto_acc -= step_dt
                d = step_visible()
                if d is not None:
                    dots.append(d)
                    maxk[d.n] = max(maxk.get(d.n, 0), d.k)
                steps_this_frame += 1

        # AUTO-ZONE transition
        if auto_zone and stepper.done and bits < BITS_MAX:
            bits += 1
            reset_view(clear_dots=(not cumulative))

        # -------- DRAW --------
        screen.fill(BG)

        plot_rect = pygame.Rect(
            SIDE_PAD,
            PLOT_TOP,
            W - 2 * SIDE_PAD,
            H - PLOT_TOP - PLOT_BOTTOM_PAD,
        )
        baseline = plot_rect.bottom - 10
        pygame.draw.line(screen, (70, 70, 85), (plot_rect.left, baseline), (plot_rect.right, baseline), 2)

        max_height = max(maxk.values()) if maxk else 1
        max_height = max(20, max_height)
        y_scale = (plot_rect.height - 30) / float(max_height)

        for n, mk in maxk.items():
            sx = int(W * 0.5 + (n - cx) * x_scale)
            if plot_rect.left <= sx <= plot_rect.right:
                y1 = int(baseline - mk * y_scale)
                pygame.draw.line(screen, COLLINE, (sx, baseline), (sx, y1), 1)

        for d in dots:
            sx = int(W * 0.5 + (d.n - cx) * x_scale)
            if not (plot_rect.left <= sx <= plot_rect.right):
                continue
            sy = int(baseline - d.k * y_scale)
            if sy < plot_rect.top:
                continue
            col = GREEN if d.color == "green" else RED if d.color == "red" else YELLOW
            pygame.draw.circle(screen, col, (sx, sy), DOT_R)

        hud = pygame.Rect(SIDE_PAD, TOP_PAD, W - 2 * SIDE_PAD, HUD_H)
        pygame.draw.rect(screen, (20, 20, 28), hud, border_radius=12)
        pygame.draw.rect(screen, (90, 90, 110), hud, width=2, border_radius=12)

        mode = "CUMULATIVE" if cumulative else "BUCKET"
        status = "DONE" if stepper.done else stepper.state
        cd = f"d={stepper.curr_d}" if stepper.curr_d is not None else "d=—"

        screen.blit(font(22, True).render("PRIME TOWERS — SMOOTH SLOW BUILD + AUTO ZONES", True, WHITE), (hud.x + 16, hud.y + 12))
        screen.blit(font(18).render(f"bits={bits}  mode={mode}  range=[{L}..{R}]  x_zoom={x_scale:.2f}", True, SUB), (hud.x + 16, hud.y + 48))
        screen.blit(font(18, True).render(f"state={status}   current n={stepper.n}   k={stepper.k}   {cd}", True, WHITE), (hud.x + 16, hud.y + 76))
        screen.blit(font(16).render("SPACE hold • A auto • Z auto-zones • UP/DOWN speed • [ ] bits • C mode • R rebuild • ← → zoom • ESC quit", True, SUB), (hud.x + 16, hud.y + 112))
        screen.blit(font(16).render(f"auto={auto} rate={auto_rate:.2f}/s   hold_rate={hold_rate:.2f}   auto_zone={auto_zone}", True, SUB), (hud.x + 16, hud.y + 132))

        if zone_pause > 0.0:
            screen.blit(font(18, True).render("...next bit bucket...", True, YELLOW), (hud.right - 260, hud.y + 76))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
