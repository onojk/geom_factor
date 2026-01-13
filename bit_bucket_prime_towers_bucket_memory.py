# bit_bucket_prime_towers_bucket_memory.py
#
# Bit-bucket prime towers with FULLY VISIBLE dot reveals:
# - ONE dot max per frame (no batching)
# - One bit bucket at a time
# - Old buckets remain as memory
# - Crumb trail at powers of two
#
# Controls:
#   SPACE  pause / resume
#   UP     faster (fewer frames per dot)
#   DOWN   slower (more frames per dot)
#   [ ]    bits down / up
#   M      toggle memory
#   X      clear memory
#   B      toggle bucket boundary labels
#   ← →    zoom X
#   ESC    quit

import math
import pygame
from dataclasses import dataclass
from typing import Dict, List, Tuple

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

CRUMB_LINE = (120, 120, 150)
CRUMB_GLOW = (70, 70, 95)

BITS_MIN, BITS_MAX = 1, 18
BITS_START = 1

X_SCALE_DEFAULT = 8.0
ZOOM_STEP = 1.12

HUD_H = 155
SIDE_PAD = 30
TOP_PAD = 18
PLOT_TOP = HUD_H + 20
PLOT_BOTTOM_PAD = 28

# --- visibility control ---
FRAMES_PER_DOT_DEFAULT = 2   # 1 = every frame, 2 = every other frame, etc.
FRAMES_PER_DOT_MIN = 1
FRAMES_PER_DOT_MAX = 30

# ---------------- DATA ----------------
@dataclass(frozen=True)
class Dot:
    n: int
    k: int
    color: str  # green / red / yellow

# ---------------- MATH ----------------
def trial_divs(n: int):
    if n < 2:
        return []
    return list(range(2, int(math.isqrt(n)) + 1))

def bucket_range(bits: int) -> Tuple[int, int]:
    return (1 << (bits - 1)), (1 << bits) - 1

def build_bucket_dots(bits: int) -> List[Dot]:
    L, R = bucket_range(bits)
    out: List[Dot] = []

    for n in range(L, R + 1):
        if n < 2:
            out.append(Dot(n, 0, "yellow"))
            continue

        divs = trial_divs(n)
        if not divs:
            for k in range(n):
                out.append(Dot(n, k, "green"))
            continue

        k = 0
        found = False
        for d in divs:
            if n % d == 0:
                out.append(Dot(n, k, "red"))
                found = True
                break
            else:
                out.append(Dot(n, k, "green"))
                k += 1

        if not found:
            while k < n:
                out.append(Dot(n, k, "green"))
                k += 1

    return out

# ---------------- UI HELPERS ----------------
def font(size, bold=False):
    return pygame.font.SysFont("dejavusansmono", size, bold)

def color_of(tag: str):
    if tag == "green":
        return GREEN
    if tag == "red":
        return RED
    return YELLOW

# ---------------- MAIN ----------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Bit Bucket Prime Towers — Fully Visible Dots")
    clock = pygame.time.Clock()

    bits = BITS_START
    memory_on = True
    show_labels = True

    x_scale = X_SCALE_DEFAULT

    paused = False
    frames_per_dot = FRAMES_PER_DOT_DEFAULT
    frame_counter = 0

    dots: List[Dot] = []
    maxk: Dict[int, int] = {}
    crumb_bits: List[int] = []

    bucket_dots: List[Dot] = []
    bucket_idx = 0
    L = R = 1
    cx = 1.0

    def reset_bucket(new_bits: int, clear_memory: bool):
        nonlocal bits, bucket_dots, bucket_idx, L, R, cx, frame_counter
        bits = max(BITS_MIN, min(BITS_MAX, new_bits))
        L, R = bucket_range(bits)

        if clear_memory:
            dots.clear()
            maxk.clear()
            crumb_bits.clear()

        if bits not in crumb_bits:
            crumb_bits.append(bits)

        bucket_dots = build_bucket_dots(bits)
        bucket_idx = 0
        cx = (L + R) / 2.0
        frame_counter = 0

    reset_bucket(bits, clear_memory=True)

    running = True
    while running:
        clock.tick(FPS)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_UP:
                    frames_per_dot = max(FRAMES_PER_DOT_MIN, frames_per_dot - 1)
                elif e.key == pygame.K_DOWN:
                    frames_per_dot = min(FRAMES_PER_DOT_MAX, frames_per_dot + 1)
                elif e.key == pygame.K_LEFT:
                    x_scale = max(0.8, x_scale / ZOOM_STEP)
                elif e.key == pygame.K_RIGHT:
                    x_scale = min(120.0, x_scale * ZOOM_STEP)
                elif e.key == pygame.K_m:
                    memory_on = not memory_on
                elif e.key == pygame.K_b:
                    show_labels = not show_labels
                elif e.key == pygame.K_x:
                    dots.clear()
                    maxk.clear()
                    crumb_bits.clear()
                elif e.key == pygame.K_LEFTBRACKET:
                    reset_bucket(bits - 1, clear_memory=(not memory_on))
                elif e.key == pygame.K_RIGHTBRACKET:
                    reset_bucket(bits + 1, clear_memory=(not memory_on))

        # ---- reveal EXACTLY one dot (or none) ----
        if not paused and bucket_idx < len(bucket_dots):
            frame_counter += 1
            if frame_counter >= frames_per_dot:
                frame_counter = 0
                d = bucket_dots[bucket_idx]
                dots.append(d)
                maxk[d.n] = max(maxk.get(d.n, 0), d.k)
                bucket_idx += 1

        # auto-advance bucket
        if bucket_idx >= len(bucket_dots) and bits < BITS_MAX:
            reset_bucket(bits + 1, clear_memory=(not memory_on))

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

        # crumbs
        for b in crumb_bits:
            edge = (1 << b)
            sx = int(W * 0.5 + (edge - cx) * x_scale)
            if plot_rect.left <= sx <= plot_rect.right:
                pygame.draw.line(screen, CRUMB_GLOW, (sx, plot_rect.top), (sx, baseline), 1)
                pygame.draw.line(screen, CRUMB_LINE, (sx, plot_rect.top), (sx, baseline), 2)
                if show_labels:
                    lab = font(14, True).render(f"2^{b}", True, (200, 200, 220))
                    screen.blit(lab, (sx + 4, plot_rect.top + 4))

        # column lines
        for n, mk in maxk.items():
            sx = int(W * 0.5 + (n - cx) * x_scale)
            if plot_rect.left <= sx <= plot_rect.right:
                y1 = int(baseline - mk * y_scale)
                pygame.draw.line(screen, COLLINE, (sx, baseline), (sx, y1), 1)

        # dots
        for d in dots:
            sx = int(W * 0.5 + (d.n - cx) * x_scale)
            if plot_rect.left <= sx <= plot_rect.right:
                sy = int(baseline - d.k * y_scale)
                if sy >= plot_rect.top:
                    pygame.draw.circle(screen, color_of(d.color), (sx, sy), DOT_R)

        # HUD
        hud = pygame.Rect(SIDE_PAD, TOP_PAD, W - 2 * SIDE_PAD, HUD_H)
        pygame.draw.rect(screen, (20, 20, 28), hud, border_radius=12)
        pygame.draw.rect(screen, (90, 90, 110), hud, width=2, border_radius=12)

        screen.blit(font(22, True).render("BIT BUCKET PRIME TOWERS — EVERY DOT VISIBLE", True, WHITE), (hud.x + 16, hud.y + 12))
        screen.blit(font(18).render(
            f"bits={bits}  bucket=[{L}..{R}]  paused={paused}  frames/dot={frames_per_dot}",
            True, SUB
        ), (hud.x + 16, hud.y + 48))
        screen.blit(font(18, True).render(
            f"dots: {bucket_idx}/{len(bucket_dots)} (auto-advance)",
            True, WHITE
        ), (hud.x + 16, hud.y + 78))
        screen.blit(font(16).render(
            "SPACE pause • UP/DOWN speed • [ ] bits • M memory • X clear • B labels • ← → zoom • ESC quit",
            True, SUB
        ), (hud.x + 16, hud.y + 112))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
