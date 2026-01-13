# bit_bucket_sieve_dots.py
#
# PYGAME — Bit-bucket SEGMENTED SIEVE visualizer that also draws DOTS.
#
# Idea:
#   Pick a bit bucket [L, R] = [2^(b-1), 2^b - 1].
#   Run a segmented sieve INSIDE the bucket:
#     - compute base primes up to sqrt(R)
#     - for each base prime p, mark its multiples in [L, R]
#   Visualization:
#     - Each time a number n is newly marked composite by some p,
#       we add a RED dot to that column (a "witness stamp").
#     - After sieve completes, remaining unmarked n >= 2 get GREEN dots
#       (a small "prime glow stack").
#
# Controls:
#   SPACE : advance one base prime "sweep" (mark multiples for next p)
#   A     : toggle auto-advance
#   UP    : auto faster
#   DOWN  : auto slower
#   [     : bits -= 1  (change bucket)
#   ]     : bits += 1
#   R     : reset current bucket sieve (clear marks + dots)
#   ESC   : quit
#
# Hover:
#   Move mouse over columns to see n, binary(n) with:
#     ORANGE = bucket structure bit (leading 1)
#     PURPLE = payload bits (change inside bucket)
#     YELLOW = parity bit (LSB)

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
PURPLE = (190, 120, 255)
ORANGE = (255, 165, 80)

GRID_MAJOR = (24, 24, 32)
GRID_MINOR = (16, 16, 22)
AXIS = (90, 90, 110)

PANEL = (20, 20, 28)
PANEL_STROKE = (90, 90, 110)

DOT_R = 3

# “dot stacking” vertical spacing in pixels (UI space, not world-space)
DOT_DY = 8
MAX_DOTS_VISIBLE = 48  # per column

# bucket defaults
BITS_MIN = 3
BITS_MAX = 16
BITS_DEFAULT = 8  # [128..255], nice size for screen

# auto
AUTO_MIN = 0.25
AUTO_MAX = 30.0

# layout
TOP_H = 230
BOTTOM_MARGIN = 30
LEFT_MARGIN = 40
RIGHT_MARGIN = 40
PLOT_H = H - TOP_H - BOTTOM_MARGIN

# ----------------------------
# Helpers
# ----------------------------
def font(size: int, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont("dejavusansmono", size, bold=bold)

def clamp(x: float, a: float, b: float) -> float:
    return a if x < a else b if x > b else x

def bin_str(n: int) -> str:
    return format(n, "b")

def is_prime_small(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True

def primes_upto(nmax: int) -> List[int]:
    return [n for n in range(2, nmax + 1) if is_prime_small(n)]

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class ColumnDots:
    # Each dot is just a color; y-position is implicit by index
    dots: List[Tuple[int, int, int]]  # list of RGB colors
    witness_ps: List[int]             # which base primes marked this number

# ----------------------------
# Bit-bucket segmented sieve (stepwise)
# ----------------------------
class BucketSieve:
    def __init__(self, bits: int):
        self.bits = bits
        self.reset(bits)

    def reset(self, bits: Optional[int] = None):
        if bits is not None:
            self.bits = bits
        self.L = 1 << (self.bits - 1)
        self.R = (1 << self.bits) - 1
        self.N = self.R - self.L + 1

        self.base_primes = primes_upto(int(math.isqrt(self.R)))
        self.pi = 0  # index into base_primes

        # marked[i] corresponds to n = L + i
        self.marked = [False] * self.N
        self.mark_p = [0] * self.N  # first witness prime p (0 means none yet)

        # dot stacks per n
        self.cols: List[ColumnDots] = [ColumnDots(dots=[], witness_ps=[]) for _ in range(self.N)]

        self.done = False
        self.last_p: Optional[int] = None
        self.last_marked_count = 0

        # after done: prime glow applied once
        self._prime_glow_applied = False

    def _mark_number(self, n: int, p: int):
        """Mark n composite (if newly marked) and add a red dot stamp."""
        if n < self.L or n > self.R:
            return
        idx = n - self.L
        if n == p:
            return  # don't mark the prime itself inside the segment
        if not self.marked[idx]:
            self.marked[idx] = True
            self.mark_p[idx] = p
            self.last_marked_count += 1

        # Always record the witness p in the column, but only draw a dot if room
        col = self.cols[idx]
        if (not col.witness_ps) or (col.witness_ps and col.witness_ps[-1] != p):
            col.witness_ps.append(p)
            if len(col.dots) < MAX_DOTS_VISIBLE:
                col.dots.append(RED)

    def step_prime(self) -> bool:
        """
        Process one base prime p:
          mark multiples of p in [L,R].
        Returns True if a step occurred, False if already done.
        """
        if self.done:
            return False

        if self.pi >= len(self.base_primes):
            self.done = True
            self.last_p = None
            return False

        p = self.base_primes[self.pi]
        self.pi += 1

        self.last_p = p
        self.last_marked_count = 0

        # first multiple of p >= L
        start = (self.L + p - 1) // p * p
        # ensure we start at 2p at least (avoid marking p itself when p in segment)
        if start == p:
            start += p

        for m in range(start, self.R + 1, p):
            self._mark_number(m, p)

        return True

    def apply_prime_glow(self):
        """After sieve is done, add a few GREEN dots to unmarked primes >= 2."""
        if self._prime_glow_applied:
            return
        for i in range(self.N):
            n = self.L + i
            if n >= 2 and not self.marked[i]:
                # prime glow stack: 4–8 dots depending on size
                glow = 4 + min(6, int(math.log2(n)))
                col = self.cols[i]
                for _ in range(glow):
                    if len(col.dots) < MAX_DOTS_VISIBLE:
                        col.dots.append(GREEN)
        self._prime_glow_applied = True

    def count_primes_remaining(self) -> int:
        # only meaningful after done, but okay anytime as "current survivors"
        c = 0
        for i in range(self.N):
            n = self.L + i
            if n >= 2 and not self.marked[i]:
                c += 1
        return c

# ----------------------------
# Drawing helpers
# ----------------------------
def draw_panel(screen: pygame.Surface, rect: pygame.Rect):
    pygame.draw.rect(screen, PANEL, rect, border_radius=12)
    pygame.draw.rect(screen, PANEL_STROKE, rect, width=2, border_radius=12)

def draw_bit_colored_binary(screen: pygame.Surface, x: int, y: int, n: int):
    bits = bin_str(n)
    f = font(22, True)

    # label
    screen.blit(font(18, True).render("binary(n) =", True, WHITE), (x, y))

    bx = x + 130
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

    # legend
    leg = "MSB(structure)=ORANGE  payload=PURPLE  LSB(parity)=YELLOW"
    screen.blit(font(14).render(leg, True, SUB), (x, y + 26))

def nearest_column_index(plot_rect: pygame.Rect, L: int, R: int, mouse_x: int) -> Optional[int]:
    N = R - L + 1
    if N <= 0:
        return None
    if mouse_x < plot_rect.left or mouse_x > plot_rect.right:
        return None
    col_w = plot_rect.width / N
    idx = int((mouse_x - plot_rect.left) / col_w)
    idx = max(0, min(N - 1, idx))
    return idx

# ----------------------------
# Main
# ----------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Bit-Bucket Segmented Sieve (Dots)")
    clock = pygame.time.Clock()

    bits = BITS_DEFAULT
    sieve = BucketSieve(bits)

    auto = False
    auto_rate = 6.0
    auto_acc = 0.0

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
                    sieve.step_prime()
                elif e.key == pygame.K_a:
                    auto = not auto
                elif e.key == pygame.K_UP:
                    auto_rate = clamp(auto_rate * 1.25, AUTO_MIN, AUTO_MAX)
                elif e.key == pygame.K_DOWN:
                    auto_rate = clamp(auto_rate / 1.25, AUTO_MIN, AUTO_MAX)
                elif e.key == pygame.K_LEFTBRACKET:  # [
                    bits = max(BITS_MIN, bits - 1)
                    sieve.reset(bits)
                elif e.key == pygame.K_RIGHTBRACKET:  # ]
                    bits = min(BITS_MAX, bits + 1)
                    sieve.reset(bits)
                elif e.key == pygame.K_r:
                    sieve.reset(bits)

        # auto-advance
        if auto and not sieve.done:
            auto_acc += dt
            step_dt = 1.0 / max(0.0001, auto_rate)
            while auto_acc >= step_dt and not sieve.done:
                auto_acc -= step_dt
                sieve.step_prime()

        # when done, apply prime glow once
        if sieve.done:
            sieve.apply_prime_glow()

        # ----------------------------
        # Draw UI
        # ----------------------------
        screen.fill(BG)

        top_rect = pygame.Rect(LEFT_MARGIN, 20, W - LEFT_MARGIN - RIGHT_MARGIN, TOP_H - 40)
        draw_panel(screen, top_rect)

        plot_rect = pygame.Rect(LEFT_MARGIN, TOP_H, W - LEFT_MARGIN - RIGHT_MARGIN, PLOT_H)

        # Title + bucket info
        title = "BIT-BUCKET SEGMENTED SIEVE (MARK MULTIPLES INSIDE [2^(b-1), 2^b-1])"
        screen.blit(font(22, True).render(title, True, WHITE), (top_rect.x + 16, top_rect.y + 12))

        L, R = sieve.L, sieve.R
        info1 = f"bits b={bits}   bucket=[{L:,} .. {R:,}]   size={sieve.N:,}"
        screen.blit(font(18).render(info1, True, SUB), (top_rect.x + 16, top_rect.y + 48))

        base_total = len(sieve.base_primes)
        base_done = sieve.pi
        info2 = f"base primes up to sqrt(R) = {base_total}   processed = {base_done}"
        screen.blit(font(18).render(info2, True, SUB), (top_rect.x + 16, top_rect.y + 74))

        if sieve.last_p is None:
            lp = "(none)"
        else:
            lp = str(sieve.last_p)
        info3 = f"last sweep prime p = {lp}   newly marked this sweep = {sieve.last_marked_count}"
        screen.blit(font(18, True).render(info3, True, WHITE), (top_rect.x + 16, top_rect.y + 104))

        survivors = sieve.count_primes_remaining()
        status = "DONE (prime glow applied)" if sieve.done else "RUNNING"
        info4 = f"status = {status}   current survivors (unmarked ≥2) = {survivors}"
        screen.blit(font(18).render(info4, True, SUB), (top_rect.x + 16, top_rect.y + 132))

        controls = "SPACE step  •  A auto  •  UP/DOWN speed  •  [ ] change bits  •  R reset  •  ESC quit"
        screen.blit(font(16).render(controls, True, SUB), (top_rect.x + 16, top_rect.bottom - 28))

        # Plot background + axis line
        pygame.draw.rect(screen, (10, 10, 16), plot_rect)
        pygame.draw.rect(screen, PANEL_STROKE, plot_rect, width=2)

        axis_y = plot_rect.bottom - 10
        pygame.draw.line(screen, AXIS, (plot_rect.left, axis_y), (plot_rect.right, axis_y), 2)

        # Light vertical grid
        N = sieve.N
        if N > 0:
            # every ~32 columns draw a major line
            stride = max(1, N // 40)
            for i in range(0, N, stride):
                x = plot_rect.left + int(i * plot_rect.width / N)
                col = GRID_MAJOR if (i % (stride * 5) == 0) else GRID_MINOR
                pygame.draw.line(screen, col, (x, plot_rect.top), (x, plot_rect.bottom), 1)

        # Draw dots per column
        if N > 0:
            col_w = plot_rect.width / N
            for i in range(N):
                x = plot_rect.left + int((i + 0.5) * col_w)

                dots = sieve.cols[i].dots
                # stack upward from axis_y
                for k, c in enumerate(dots):
                    y = axis_y - k * DOT_DY
                    if y < plot_rect.top + 10:
                        break
                    pygame.draw.circle(screen, c, (x, y), DOT_R)

        # Hover info
        mx, my = pygame.mouse.get_pos()
        idx = nearest_column_index(plot_rect, sieve.L, sieve.R, mx)
        if idx is not None:
            n_hover = sieve.L + idx
            col = sieve.cols[idx]
            marked = sieve.marked[idx]
            first_p = sieve.mark_p[idx]

            # highlight column
            x_col = plot_rect.left + int((idx + 0.5) * plot_rect.width / sieve.N)
            pygame.draw.line(screen, (60, 60, 90), (x_col, plot_rect.top), (x_col, plot_rect.bottom), 2)

            # hover panel
            hover_rect = pygame.Rect(W - 420, 20, 380, 185)
            draw_panel(screen, hover_rect)
            screen.blit(font(18, True).render("HOVER", True, WHITE), (hover_rect.x + 14, hover_rect.y + 10))

            line1 = f"n = {n_hover}"
            screen.blit(font(22, True).render(line1, True, WHITE), (hover_rect.x + 14, hover_rect.y + 40))

            status_txt = "COMPOSITE" if (n_hover >= 2 and marked) else ("PRIME (survivor)" if n_hover >= 2 else "TRIVIAL")
            status_col = RED if (n_hover >= 2 and marked) else (GREEN if n_hover >= 2 else YELLOW)
            screen.blit(font(18, True).render(status_txt, True, status_col), (hover_rect.x + 14, hover_rect.y + 70))

            if n_hover >= 2 and marked:
                screen.blit(font(16).render(f"first witness p = {first_p}", True, SUB), (hover_rect.x + 14, hover_rect.y + 96))
                if col.witness_ps:
                    wps = ", ".join(map(str, col.witness_ps[:10]))
                    if len(col.witness_ps) > 10:
                        wps += " ..."
                    screen.blit(font(16).render(f"witness stamps: {wps}", True, SUB), (hover_rect.x + 14, hover_rect.y + 118))

            # binary with bucket-coloring
            draw_bit_colored_binary(screen, hover_rect.x + 14, hover_rect.y + 146, n_hover)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
