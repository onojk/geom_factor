# bit_bucket_prime_towers_4k_export.py
#
# OFFLINE 4K MP4 RENDER
# - 3840x2160 @ 60fps
# - One dot revealed per frame (every dot visible)
# - One bit bucket at a time
# - Previous buckets remain (memory)
# - Bit-bucket crumb boundaries preserved
#
# Output:
#   bit_bucket_prime_towers_4k.mp4

import math
import pygame
import numpy as np
import imageio.v2 as imageio
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------------- VIDEO CONFIG ----------------
W, H = 3840, 2160          # 4K UHD
FPS = 60
OUTPUT_MP4 = "bit_bucket_prime_towers_4k.mp4"

# ---------------- VISUAL CONFIG ----------------
BG = (8, 8, 12)
WHITE = (235, 235, 245)
SUB = (160, 160, 180)

GREEN = (80, 255, 120)
RED = (255, 90, 90)
YELLOW = (255, 220, 90)

COLLINE = (55, 55, 70)
DOT_R = 3                 # slightly larger for 4K

CRUMB_LINE = (140, 140, 180)
CRUMB_GLOW = (80, 80, 110)

BITS_START = 1
BITS_MAX = 8             # ⚠️ increase cautiously (video length explodes)

X_SCALE = 18.0             # tuned for 4K

HUD_H = 260
SIDE_PAD = 80
TOP_PAD = 40
PLOT_TOP = HUD_H + 40
PLOT_BOTTOM_PAD = 60

FRAMES_PER_DOT = 1        # EXACTLY one dot per frame

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

# ---------------- RENDER ----------------
def render_video():
    pygame.init()
    screen = pygame.Surface((W, H))

    font_big = pygame.font.SysFont("dejavusansmono", 64, True)
    font_med = pygame.font.SysFont("dejavusansmono", 44)
    font_small = pygame.font.SysFont("dejavusansmono", 36)

    writer = imageio.get_writer(
        OUTPUT_MP4,
        fps=FPS,
        codec="libx264",
        bitrate="40M",
        pixelformat="yuv420p"
    )

    dots: List[Dot] = []
    maxk: Dict[int, int] = {}
    crumb_bits: List[int] = []

    bits = BITS_START

    plot_rect = pygame.Rect(
        SIDE_PAD,
        PLOT_TOP,
        W - 2 * SIDE_PAD,
        H - PLOT_TOP - PLOT_BOTTOM_PAD,
    )

    baseline = plot_rect.bottom - 20

    def draw_frame(cx, current_bits, reveal_idx, total_dots):
        screen.fill(BG)

        pygame.draw.line(
            screen, (70, 70, 85),
            (plot_rect.left, baseline),
            (plot_rect.right, baseline), 3
        )

        max_height = max(maxk.values()) if maxk else 1
        max_height = max(40, max_height)
        y_scale = (plot_rect.height - 60) / float(max_height)

        # crumbs
        for b in crumb_bits:
            edge = (1 << b)
            sx = int(W * 0.5 + (edge - cx) * X_SCALE)
            if plot_rect.left <= sx <= plot_rect.right:
                pygame.draw.line(screen, CRUMB_GLOW, (sx, plot_rect.top), (sx, baseline), 2)
                pygame.draw.line(screen, CRUMB_LINE, (sx, plot_rect.top), (sx, baseline), 3)

                lab = font_small.render(f"2^{b}", True, (200, 200, 220))
                screen.blit(lab, (sx + 8, plot_rect.top + 8))

        # columns
        for n, mk in maxk.items():
            sx = int(W * 0.5 + (n - cx) * X_SCALE)
            if plot_rect.left <= sx <= plot_rect.right:
                y1 = int(baseline - mk * y_scale)
                pygame.draw.line(screen, COLLINE, (sx, baseline), (sx, y1), 2)

        # dots
        for d in dots:
            sx = int(W * 0.5 + (d.n - cx) * X_SCALE)
            if plot_rect.left <= sx <= plot_rect.right:
                sy = int(baseline - d.k * y_scale)
                if sy >= plot_rect.top:
                    col = GREEN if d.color == "green" else RED if d.color == "red" else YELLOW
                    pygame.draw.circle(screen, col, (sx, sy), DOT_R)

        # HUD
        pygame.draw.rect(
            screen, (20, 20, 28),
            (SIDE_PAD, TOP_PAD, W - 2 * SIDE_PAD, HUD_H),
            border_radius=18
        )
        pygame.draw.rect(
            screen, (100, 100, 130),
            (SIDE_PAD, TOP_PAD, W - 2 * SIDE_PAD, HUD_H),
            width=3, border_radius=18
        )

        screen.blit(
            font_big.render("BIT BUCKET PRIME TOWERS", True, WHITE),
            (SIDE_PAD + 24, TOP_PAD + 24)
        )

        screen.blit(
            font_med.render(
                f"bits={current_bits}   bucket=[{1<<(current_bits-1)}..{(1<<current_bits)-1}]",
                True, SUB
            ),
            (SIDE_PAD + 24, TOP_PAD + 120)
        )

        screen.blit(
            font_med.render(
                f"dots: {reveal_idx}/{total_dots}",
                True, WHITE
            ),
            (SIDE_PAD + 24, TOP_PAD + 170)
        )

        frame = pygame.surfarray.array3d(screen)
        frame = np.transpose(frame, (1, 0, 2))  # pygame → imageio
        writer.append_data(frame)

    # ---- MAIN RENDER LOOP ----
    while bits <= BITS_MAX:
        L, R = bucket_range(bits)
        bucket_dots = build_bucket_dots(bits)
        cx = (L + R) / 2.0

        if bits not in crumb_bits:
            crumb_bits.append(bits)

        for idx, d in enumerate(bucket_dots):
            dots.append(d)
            maxk[d.n] = max(maxk.get(d.n, 0), d.k)
            draw_frame(cx, bits, idx + 1, len(bucket_dots))

        bits += 1

    writer.close()
    pygame.quit()

    print(f"✅ Render complete: {OUTPUT_MP4}")

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    render_video()
