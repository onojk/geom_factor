# bit_bucket_prime_towers.py
#
# PYGAME — Prime Towers with BIT-BUCKET selection + optional CUMULATIVE view
#          and AUTO Y-SCALING so towers don't get insanely tall.
#
# Tower semantics:
#   Prime n      → exactly n green dots tall
#   Composite n  → green dots for each failed divisor, then ONE red witness dot
#   n < 2        → one yellow spacer dot
#
# Controls:
#   [     : bits -= 1
#   ]     : bits += 1
#   R     : rebuild
#   C     : toggle cumulative view (1..2^b-1)
#   B     : bucket-only view (2^(b-1)..2^b-1)
#   LEFT  : zoom out X
#   RIGHT : zoom in X
#   ESC   : quit
#
# Hover:
#   Shows n, status, witness, checks, and binary(n) colored:
#     ORANGE = bucket MSB (leading 1)
#     PURPLE = payload bits
#     YELLOW = parity bit (LSB)

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pygame

# ----------------------------
# Config
# ----------------------------
W, H = 1280, 720
FPS = 60

BG = (8, 8, 12)
WHITE = (235, 235, 245)
SUB = (160, 160, 180)

GREEN = (80, 255, 120)
RED = (255, 90, 90)
YELLOW = (255, 220, 90)

ORANGE = (255, 165, 80)
PURPLE = (190, 120, 255)

GRID_MAJOR = (24, 24, 32)
GRID_MINOR = (16, 16, 22)
COLLINE = (55, 55, 70)

PANEL = (20, 20, 28)
PANEL_STROKE = (90, 90, 110)

DOT_R = 2

BITS_MIN = 2
BITS_MAX = 16
BITS_DEFAULT = 8

# X zoom only (Y is auto-scaled)
X_SCALE_DEFAULT = 5.0
X_SCALE_MIN = 0.8
X_SCALE_MAX = 40.0
ZOOM_STEP = 1.12

# Layout
HUD_H = 155
HOVER_H = 165
TOP_PAD = 18
SIDE_PAD = 30
PLOT_TOP = HUD_H + 20
PLOT_BOTTOM_PAD = 28

# ----------------------------
# Helpers
# ----------------------------
def font(size: int, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont("dejavusansmono", size, bold=bold)

def clamp(x: float, a: float, b: float) -> float:
    return a if x < a else b if x > b else x

def bin_str(n: int) -> str:
    return format(n, "b")

def trial_divisors_for_n(n: int) -> List[int]:
    if n < 2:
        return []
    limit = int(math.isqrt(n))
    return list(range(2, limit + 1))

def draw_panel(screen: pygame.Surface, rect: pygame.Rect):
    pygame.draw.rect(screen, PANEL, rect, border_radius=12)
    pygame.draw.rect(screen, PANEL_STROKE, rect, width=2, border_radius=12)

# ----------------------------
# Dot data
# ----------------------------
@dataclass(frozen=True)
class Dot:
    n: int
    k: int
    color: str  # "green" | "red" | "yellow"

@dataclass
class NInfo:
    n: int
    is_prime: bool
    witness_d: Optional[int]
    checks_done: int

# ----------------------------
# Build range (bucket-only OR cumulative)
# ----------------------------
def build_range(L: int, R: int) -> Tuple[List[Dot], Dict[int, NInfo], Dict[int, int]]:
    dots: List[Dot] = []
    info: Dict[int, NInfo] = {}
    maxk: Dict[int, int] = {}

    for n in range(L, R + 1):
        if n < 2:
            dots.append(Dot(n, 0, "yellow"))
            info[n] = NInfo(n=n, is_prime=False, witness_d=None, checks_done=0)
            maxk[n] = 0
            continue

        divs = trial_divisors_for_n(n)
        k = 0
        witness = None
        checks_done = 0

        for d in divs:
            checks_done += 1
            if n % d == 0:
                witness = d
                dots.append(Dot(n, k, "red"))
                maxk[n] = max(maxk.get(n, 0), k)
                k += 1
                break
            else:
                dots.append(Dot(n, k, "green"))
                maxk[n] = max(maxk.get(n, 0), k)
                k += 1

        if witness is None:
            # prime fill to height n
            while k < n:
                dots.append(Dot(n, k, "green"))
                maxk[n] = max(maxk.get(n, 0), k)
                k += 1

        info[n] = NInfo(
            n=n,
            is_prime=(witness is None),
            witness_d=witness,
            checks_done=checks_done,
        )

    return dots, info, maxk

# ----------------------------
# Plot mapping with auto Y scale
# ----------------------------
def compute_y_scale(plot_rect: pygame.Rect, max_height: int) -> float:
    # map k=0..max_height into plot height (top..baseline)
    usable_h = max(80, plot_rect.height - 20)
    return usable_h / max(1, max_height + 2)

def x_world_to_screen(n: float, cx: float, x_scale: float) -> int:
    return int(W * 0.5 + (n - cx) * x_scale)

def k_to_screen_y(k: float, plot_rect: pygame.Rect, y_scale: float) -> int:
    baseline = plot_rect.bottom - 10
    return int(baseline - k * y_scale)

def draw_grid(screen: pygame.Surface, plot_rect: pygame.Rect, cx: float, x_scale: float):
    # plot bg
    pygame.draw.rect(screen, (10, 10, 16), plot_rect)
    pygame.draw.rect(screen, PANEL_STROKE, plot_rect, width=2)

    # baseline
    baseline_y = plot_rect.bottom - 10
    pygame.draw.line(screen, (70, 70, 85), (plot_rect.left, baseline_y), (plot_rect.right, baseline_y), 2)

    # vertical grid lines in world units
    if x_scale <= 0:
        return
    minor_world = max(1, int(70 / x_scale))
    major_world = minor_world * 5

    left_world = cx - (W * 0.5) / x_scale
    right_world = cx + (W * 0.5) / x_scale

    start = int(math.floor(left_world / minor_world) * minor_world)
    x = start
    while x <= right_world:
        sx = x_world_to_screen(x, cx, x_scale)
        if plot_rect.left <= sx <= plot_rect.right:
            col = GRID_MAJOR if (x % major_world == 0) else GRID_MINOR
            pygame.draw.line(screen, col, (sx, plot_rect.top), (sx, plot_rect.bottom), 1)
        x += minor_world

def draw_bit_colored_binary(screen: pygame.Surface, x: int, y: int, n: int):
    bits = bin_str(n)
    f = font(20, True)
    screen.blit(font(16, True).render("binary(n) =", True, WHITE), (x, y))

    bx = x + 120
    for i, ch in enumerate(bits):
        is_msb = (i == 0)
        is_lsb = (i == len(bits) - 1)
        if is_lsb:
            col = YELLOW
        elif is_msb:
            col = ORANGE
        else:
            col = PURPLE
        img = f.render(ch, True, col)
        screen.blit(img, (bx, y - 6))
        bx += img.get_width() + 3

    legend = "MSB(bucket)=ORANGE  payload=PURPLE  LSB(parity)=YELLOW"
    screen.blit(font(14).render(legend, True, SUB), (x, y + 22))

def hover_n_from_mouse(mx: int, cx: float, x_scale: float) -> int:
    if x_scale <= 0:
        return 0
    wx = cx + (mx - W * 0.5) / x_scale
    return int(round(wx))

# ----------------------------
# Main
# ----------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Bit Bucket Prime Towers (auto y-scale + cumulative)")
    clock = pygame.time.Clock()

    bits = BITS_DEFAULT
    cumulative = True  # start with earlier buckets visible

    x_scale = X_SCALE_DEFAULT

    def current_range(b: int, cum: bool) -> Tuple[int, int]:
        if cum:
            return 1, (1 << b) - 1
        else:
            return 1 << (b - 1), (1 << b) - 1

    L, R = current_range(bits, cumulative)
    dots, info, maxk = build_range(L, R)

    def rebuild():
        nonlocal L, R, dots, info, maxk, cx
        L, R = current_range(bits, cumulative)
        dots, info, maxk = build_range(L, R)
        cx = (L + R) / 2.0

    cx = (L + R) / 2.0

    running = True
    while running:
        _dt = clock.tick(FPS) / 1000.0

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_LEFT:
                    x_scale = max(X_SCALE_MIN, x_scale / ZOOM_STEP)
                elif e.key == pygame.K_RIGHT:
                    x_scale = min(X_SCALE_MAX, x_scale * ZOOM_STEP)
                elif e.key == pygame.K_LEFTBRACKET:  # [
                    bits = max(BITS_MIN, bits - 1)
                    rebuild()
                elif e.key == pygame.K_RIGHTBRACKET:  # ]
                    bits = min(BITS_MAX, bits + 1)
                    rebuild()
                elif e.key == pygame.K_r:
                    rebuild()
                elif e.key == pygame.K_c:
                    cumulative = not cumulative
                    rebuild()
                elif e.key == pygame.K_b:
                    cumulative = False
                    rebuild()

        # screen clear
        screen.fill(BG)

        # HUD
        hud = pygame.Rect(SIDE_PAD, TOP_PAD, W - 2 * SIDE_PAD, HUD_H)
        draw_panel(screen, hud)

        title = "BIT-BUCKET PRIME TOWERS  (auto y-scale)  —  bucket at a time OR cumulative"
        screen.blit(font(22, True).render(title, True, WHITE), (hud.x + 16, hud.y + 12))

        mode = "CUMULATIVE [1 .. 2^b-1]" if cumulative else "BUCKET ONLY [2^(b-1) .. 2^b-1]"
        line1 = f"bits b={bits}   mode={mode}"
        screen.blit(font(18).render(line1, True, SUB), (hud.x + 16, hud.y + 44))

        line2 = f"range=[{L:,} .. {R:,}]   size={R-L+1:,}   x_zoom={x_scale:.2f}"
        screen.blit(font(18).render(line2, True, SUB), (hud.x + 16, hud.y + 70))

        primes_in_range = sum(1 for n in range(L, R + 1) if n in info and info[n].is_prime and n >= 2)
        density = primes_in_range / max(1, (R - L + 1))
        line3 = f"primes={primes_in_range}   density={density:.3f}"
        screen.blit(font(18).render(line3, True, SUB), (hud.x + 16, hud.y + 96))

        controls = "[ ] bits  •  C toggle cumulative  •  B bucket-only  •  R rebuild  •  ← → zoom  •  ESC quit"
        screen.blit(font(16).render(controls, True, SUB), (hud.x + 16, hud.y + 126))

        # Plot area
        plot_rect = pygame.Rect(SIDE_PAD, PLOT_TOP, W - 2 * SIDE_PAD, H - PLOT_TOP - PLOT_BOTTOM_PAD)

        # Determine max height for Y auto-scale:
        # use the maximum visible k among all columns in range
        max_height = 0
        if maxk:
            max_height = max(maxk.values())
        # clamp so extremely huge values still fit sensibly
        # (not strictly necessary since y_scale auto fits, but keeps dot sizes readable)
        max_height = max(20, max_height)

        y_scale = compute_y_scale(plot_rect, max_height)

        draw_grid(screen, plot_rect, cx, x_scale)

        # Column lines
        for n, mk in maxk.items():
            sx = x_world_to_screen(n, cx, x_scale)
            if plot_rect.left <= sx <= plot_rect.right:
                y0 = k_to_screen_y(0, plot_rect, y_scale)
                y1 = k_to_screen_y(mk + 0.2, plot_rect, y_scale)
                pygame.draw.line(screen, COLLINE, (sx, y0), (sx, y1), 1)

        # Dots
        for d in dots:
            sx = x_world_to_screen(d.n, cx, x_scale)
            if not (plot_rect.left <= sx <= plot_rect.right):
                continue
            sy = k_to_screen_y(d.k, plot_rect, y_scale)
            if sy < plot_rect.top or sy > plot_rect.bottom:
                continue
            col = GREEN if d.color == "green" else RED if d.color == "red" else YELLOW
            pygame.draw.circle(screen, col, (sx, sy), DOT_R)

        # Hover panel
        mx, my = pygame.mouse.get_pos()
        n_hover = hover_n_from_mouse(mx, cx, x_scale)
        if L <= n_hover <= R and n_hover in info:
            hrect = pygame.Rect(SIDE_PAD, hud.bottom + 10, 560, HOVER_H)
            draw_panel(screen, hrect)
            ni = info[n_hover]

            status = "PRIME" if ni.is_prime and n_hover >= 2 else ("COMPOSITE" if n_hover >= 2 else "TRIVIAL")
            scol = GREEN if status == "PRIME" else RED if status == "COMPOSITE" else YELLOW

            screen.blit(font(20, True).render(f"n = {n_hover}", True, WHITE), (hrect.x + 14, hrect.y + 10))
            screen.blit(font(18, True).render(status, True, scol), (hrect.x + 14, hrect.y + 40))

            if status == "COMPOSITE":
                screen.blit(
                    font(16).render(f"witness d = {ni.witness_d}   checks before hit = {ni.checks_done}", True, SUB),
                    (hrect.x + 14, hrect.y + 68),
                )
            elif status == "PRIME":
                screen.blit(
                    font(16).render(f"no witness for any d ≤ √n   checks performed = {ni.checks_done}", True, SUB),
                    (hrect.x + 14, hrect.y + 68),
                )

            draw_bit_colored_binary(screen, hrect.x + 14, hrect.y + 96, n_hover)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
