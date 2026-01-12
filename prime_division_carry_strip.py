# prime_division_carry_strip.py
#
# PYGAME — Visualize "carry bit" passing across an even/odd strip,
# then "passing" into PRIME FILL when n is prime.
#
# Controls:
#   SPACE : step / hold to repeat
#   A     : toggle auto-step
#   UP    : faster auto
#   DOWN  : slower auto
#   ESC   : quit

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame

# ----------------------------
# Config
# ----------------------------
W, H = 1280, 720
FPS = 60

N_MIN = 1
N_MAX = 220

BG = (8, 8, 12)
WHITE = (235, 235, 245)
SUB = (160, 160, 180)

GREEN = (80, 255, 120)
RED = (255, 90, 90)
YELLOW = (255, 220, 90)
BLUE = (120, 170, 255)
PURPLE = (190, 120, 255)
GREY = (100, 100, 120)
PANEL = (20, 20, 28)
PANEL_STROKE = (90, 90, 110)

DOT_R = 7

SPACE_REPEAT_RATE = 16.0
AUTO_MIN = 0.5
AUTO_MAX = 80.0

# Carry strip layout
STRIP_SLOTS = 18   # number of parity slots shown
SLOT_W = 46
SLOT_H = 56
SLOT_GAP = 10

# Smooth animation for carry dot
CARRY_LERP = 0.22  # 0..1 per frame; larger = snappier

# ----------------------------
# Helpers
# ----------------------------
def font(size: int, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont("dejavusansmono", size, bold=bold)

def clamp(x: float, a: float, b: float) -> float:
    return a if x < a else b if x > b else x

def is_prime(n: int) -> bool:
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

def trial_divisors(n: int) -> List[int]:
    if n < 2:
        return []
    lim = int(math.isqrt(n))
    return list(range(2, lim + 1))

def bin_str(n: int) -> str:
    if n < 0:
        return "0"
    return format(n, "b")

# ----------------------------
# Stepper state
# ----------------------------
@dataclass
class StepEvent:
    kind: str  # "TRIVIAL"|"CHECK"|"WITNESS"|"ENTER_PRIME"|"PRIME_FILL"|"NEXT_N"|"DONE"
    n: int
    d: Optional[int] = None
    r: Optional[int] = None

class Stepper:
    """
    State machine:
      TRIVIAL (n<2) -> NEXT
      CHECK divisors d=2..sqrt(n)
        if n%d==0 -> WITNESS -> NEXT
        else -> CHECK continues
      if all pass -> ENTER_PRIME then PRIME_FILL steps until k==n then NEXT
    """
    def __init__(self):
        self.n = N_MIN
        self.ds: List[int] = []
        self.i = 0
        self.k = 0
        self.state = "INIT"

        self.last_event: StepEvent = StepEvent("INIT", self.n)
        self._sync()

    def _sync(self):
        if self.n > N_MAX:
            self.state = "DONE"
            self.last_event = StepEvent("DONE", self.n)
            return

        self.ds = trial_divisors(self.n)
        self.i = 0
        self.k = 0

        if self.n < 2:
            self.state = "TRIVIAL"
        elif len(self.ds) == 0:
            # n is 2 or 3 (or any prime with sqrt < 2) -> enter prime immediately
            self.state = "PRIME"
            self.last_event = StepEvent("ENTER_PRIME", self.n)
        else:
            self.state = "CHECK"
            self.last_event = StepEvent("CHECK", self.n, d=self.ds[0], r=self.n % self.ds[0])

    def _next_n(self):
        self.n += 1
        self._sync()

    def step(self) -> StepEvent:
        if self.state == "DONE":
            self.last_event = StepEvent("DONE", self.n)
            return self.last_event

        if self.state == "TRIVIAL":
            ev = StepEvent("TRIVIAL", self.n)
            self._next_n()
            self.last_event = ev
            return ev

        if self.state == "CHECK":
            d = self.ds[self.i]
            r = self.n % d
            if r == 0:
                ev = StepEvent("WITNESS", self.n, d=d, r=r)
                self._next_n()
                self.last_event = ev
                return ev
            else:
                ev = StepEvent("CHECK", self.n, d=d, r=r)
                self.i += 1
                if self.i >= len(self.ds):
                    # exhausted checks -> prime phase begins
                    self.state = "PRIME"
                    self.last_event = ev
                    return ev
                self.last_event = ev
                return ev

        if self.state == "PRIME":
            # First frame entering prime, return ENTER_PRIME once
            if self.last_event.kind != "ENTER_PRIME" and self.k == 0:
                ev = StepEvent("ENTER_PRIME", self.n)
                self.last_event = ev
                return ev

            if self.k < self.n:
                ev = StepEvent("PRIME_FILL", self.n, d=None, r=None)
                self.k += 1
                self.last_event = ev
                return ev
            else:
                ev = StepEvent("NEXT_N", self.n)
                self._next_n()
                self.last_event = ev
                return ev

        self.last_event = StepEvent("INIT", self.n)
        return self.last_event

# ----------------------------
# Carry strip visual
# ----------------------------
class CarryStrip:
    """
    Shows a row of parity slots E/O as checks happen.
    A single "carry dot" moves slot-by-slot across.
    On prime entry, the carry dot "passes" into PRIME GATE.
    On witness, the carry collapses (turns red) and stops.
    """
    def __init__(self):
        self.reset_for_new_n(N_MIN)

        # Animated position (float, for smooth motion)
        self.carry_x = 0.0
        self.carry_y = 0.0
        self.target_x = 0.0
        self.target_y = 0.0

    def reset_for_new_n(self, n: int):
        self.n = n
        self.slots: List[str] = []      # "E" or "O"
        self.carry_bit = 0              # 0/1
        self.carry_alive = True
        self.in_prime_gate = False
        self.collapsed = False
        self.last_d: Optional[int] = None

        # Carry dot starts just before first slot
        self.carry_index = -1           # logical index (before 0)
        self.target_index = -1

    def _slot_center(self, strip_rect: pygame.Rect, idx: int) -> Tuple[float, float]:
        # idx: 0..STRIP_SLOTS-1
        x0 = strip_rect.x + 16
        y0 = strip_rect.y + strip_rect.height // 2
        x = x0 + idx * (SLOT_W + SLOT_GAP) + SLOT_W / 2
        y = y0
        return x, y

    def _pre_slot_center(self, strip_rect: pygame.Rect) -> Tuple[float, float]:
        # before slot 0
        x0 = strip_rect.x + 16
        y0 = strip_rect.y + strip_rect.height // 2
        return x0 - 18, y0

    def _gate_center(self, gate_rect: pygame.Rect) -> Tuple[float, float]:
        return gate_rect.centerx, gate_rect.centery

    def on_event(self, ev: StepEvent):
        # If n changed (NEXT stepper sync), reset strip
        # We detect n change by comparing stored self.n to ev.n
        # But note: WITNESS returns old n, then stepper already advanced.
        # We'll handle reset externally when we notice stepper.n != self.n.
        if ev.kind == "CHECK" and ev.d is not None:
            d = ev.d
            self.last_d = d

            # Record parity slot
            parity = "E" if (d % 2 == 0) else "O"
            self.slots.append(parity)
            if len(self.slots) > STRIP_SLOTS:
                self.slots = self.slots[-STRIP_SLOTS:]

            # "Carry process": flip carry bit whenever we pass an O slot
            if parity == "O":
                self.carry_bit ^= 1

            # Advance carry target to the newest slot position
            self.target_index = min(len(self.slots) - 1, STRIP_SLOTS - 1)

        elif ev.kind == "WITNESS":
            # Collapse carry at witness
            self.carry_alive = False
            self.collapsed = True

        elif ev.kind == "ENTER_PRIME":
            # Carry survives checks -> pass into prime gate
            if self.carry_alive:
                self.in_prime_gate = True

        elif ev.kind == "TRIVIAL":
            # treat trivial as a reset-ish behavior (carry irrelevant)
            self.carry_bit = 0
            self.carry_alive = True
            self.in_prime_gate = False
            self.collapsed = False
            self.target_index = -1

    def update_anim(self, strip_rect: pygame.Rect, gate_rect: pygame.Rect):
        # Decide target position
        if self.in_prime_gate and self.carry_alive and not self.collapsed:
            tx, ty = self._gate_center(gate_rect)
        else:
            if self.target_index < 0:
                tx, ty = self._pre_slot_center(strip_rect)
            else:
                tx, ty = self._slot_center(strip_rect, self.target_index)

        self.target_x, self.target_y = tx, ty

        # Initialize carry position on first update
        if self.carry_x == 0.0 and self.carry_y == 0.0:
            self.carry_x, self.carry_y = self._pre_slot_center(strip_rect)

        # Smooth move
        self.carry_x += (self.target_x - self.carry_x) * CARRY_LERP
        self.carry_y += (self.target_y - self.carry_y) * CARRY_LERP

    def draw(self, screen: pygame.Surface, strip_rect: pygame.Rect, gate_rect: pygame.Rect):
        # strip panel
        pygame.draw.rect(screen, PANEL, strip_rect, border_radius=12)
        pygame.draw.rect(screen, PANEL_STROKE, strip_rect, width=2, border_radius=12)

        # title
        f_title = font(18, bold=True)
        screen.blit(f_title.render("EVEN/ODD DIVISOR STRIP", True, WHITE), (strip_rect.x + 14, strip_rect.y + 10))

        # draw slots (fixed positions)
        base_x = strip_rect.x + 16
        mid_y = strip_rect.y + strip_rect.height // 2
        top_y = mid_y - SLOT_H / 2

        f_slot = font(24, bold=True)
        for i in range(STRIP_SLOTS):
            x = base_x + i * (SLOT_W + SLOT_GAP)
            r = pygame.Rect(int(x), int(top_y), SLOT_W, SLOT_H)
            pygame.draw.rect(screen, (28, 28, 38), r, border_radius=10)
            pygame.draw.rect(screen, (70, 70, 90), r, width=2, border_radius=10)

            if i < len(self.slots):
                ch = self.slots[i]
                col = BLUE if ch == "E" else PURPLE
                img = f_slot.render(ch, True, col)
                img_r = img.get_rect(center=r.center)
                screen.blit(img, img_r)

        # prime gate
        pygame.draw.rect(screen, (18, 18, 26), gate_rect, border_radius=12)
        pygame.draw.rect(screen, PANEL_STROKE, gate_rect, width=2, border_radius=12)

        f_gate = font(18, bold=True)
        screen.blit(f_gate.render("PRIME GATE", True, GREEN), (gate_rect.x + 14, gate_rect.y + 10))
        screen.blit(font(16).render("carry passes here", True, SUB), (gate_rect.x + 14, gate_rect.y + 38))

        # carry bit status text
        msg = f"CARRY BIT = {self.carry_bit}"
        col = GREEN if (self.carry_alive and not self.collapsed) else RED
        screen.blit(font(18, bold=True).render(msg, True, col), (strip_rect.x + 14, strip_rect.bottom - 30))

        # carry dot itself
        if self.collapsed:
            dot_col = RED
        else:
            dot_col = GREEN if self.carry_alive else RED

        pygame.draw.circle(screen, dot_col, (int(self.carry_x), int(self.carry_y)), DOT_R)

        # small label on the dot (0/1)
        lab = font(16, bold=True).render(str(self.carry_bit), True, (0, 0, 0))
        lab_r = lab.get_rect(center=(int(self.carry_x), int(self.carry_y)))
        screen.blit(lab, lab_r)

# ----------------------------
# Main
# ----------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Carry Bit Strip: even/odd → prime gate")
    clock = pygame.time.Clock()

    stepper = Stepper()
    strip = CarryStrip()
    strip.reset_for_new_n(stepper.n)

    auto = False
    auto_rate = 6.0
    auto_acc = 0.0

    space_held = False
    space_acc = 0.0

    # Layout rects
    strip_rect = pygame.Rect(40, 250, W - 80, 170)
    gate_rect = pygame.Rect(W - 260, 445, 220, 95)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        # -------- Events --------
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    ev = stepper.step()
                    # If n changed (stepper advanced internally), reset strip
                    if stepper.n != strip.n:
                        strip.reset_for_new_n(stepper.n)
                    strip.on_event(ev)
                    space_held = True
                elif e.key == pygame.K_a:
                    auto = not auto
                elif e.key == pygame.K_UP:
                    auto_rate = clamp(auto_rate * 1.25, AUTO_MIN, AUTO_MAX)
                elif e.key == pygame.K_DOWN:
                    auto_rate = clamp(auto_rate / 1.25, AUTO_MIN, AUTO_MAX)
            elif e.type == pygame.KEYUP:
                if e.key == pygame.K_SPACE:
                    space_held = False
                    space_acc = 0.0

        # -------- Stepping logic --------
        if auto:
            auto_acc += dt
            step_dt = 1.0 / max(0.0001, auto_rate)
            while auto_acc >= step_dt:
                auto_acc -= step_dt
                ev = stepper.step()
                if stepper.n != strip.n:
                    strip.reset_for_new_n(stepper.n)
                strip.on_event(ev)

        if space_held and not auto:
            space_acc += dt
            step_dt = 1.0 / SPACE_REPEAT_RATE
            while space_acc >= step_dt:
                space_acc -= step_dt
                ev = stepper.step()
                if stepper.n != strip.n:
                    strip.reset_for_new_n(stepper.n)
                strip.on_event(ev)

        # -------- Animate carry dot --------
        strip.update_anim(strip_rect, gate_rect)

        # -------- Draw --------
        screen.fill(BG)

        # Header
        screen.blit(font(28, bold=True).render("MOD CHECKS → PARITY STRIP → CARRY PASSES INTO PRIME", True, WHITE), (40, 30))
        screen.blit(font(18).render("SPACE step/hold  •  A auto  •  UP/DOWN speed  •  ESC quit", True, SUB), (40, 70))

        # Big n + binary
        n = stepper.n
        b = bin_str(n)
        n_line = f"n = {n}"
        screen.blit(font(44, bold=True).render(n_line, True, WHITE), (40, 120))

        # show bits of n, emphasizing LSB parity (even/odd)
        parity = "EVEN" if (n % 2 == 0) else "ODD"
        pcol = BLUE if parity == "EVEN" else PURPLE
        screen.blit(font(22, bold=True).render(f"binary(n) = {b}", True, WHITE), (40, 175))
        screen.blit(font(22, bold=True).render(f"LSB parity = {parity}", True, pcol), (40, 205))

        # Big realtime mod check line (when in CHECK)
        ev = stepper.last_event
        big_msg = ""
        big_col = WHITE
        if ev.kind == "CHECK" and ev.d is not None and ev.r is not None:
            big_msg = f"{ev.n} % {ev.d} = {ev.r}"
            big_col = WHITE
        elif ev.kind == "WITNESS" and ev.d is not None:
            big_msg = f"{ev.n} % {ev.d} = 0   →   COMPOSITE (carry collapses)"
            big_col = RED
        elif ev.kind == "ENTER_PRIME":
            big_msg = f"{ev.n} has no witness ≤ sqrt(n)  →  PRIME (carry passes into gate)"
            big_col = GREEN
        elif ev.kind == "PRIME_FILL":
            big_msg = f"PRIME FILL: building up to height n  (carry already passed)"
            big_col = GREEN
        elif ev.kind == "TRIVIAL":
            big_msg = f"{ev.n} < 2 (spacer)"
            big_col = YELLOW
        elif ev.kind == "DONE":
            big_msg = "DONE"
            big_col = WHITE

        if big_msg:
            # draw a panel behind it
            msg_font = font(26, bold=True)
            img = msg_font.render(big_msg, True, big_col)
            r = img.get_rect(center=(W // 2, 470))
            back = pygame.Rect(r.x - 16, r.y - 10, r.width + 32, r.height + 20)
            pygame.draw.rect(screen, PANEL, back, border_radius=12)
            pygame.draw.rect(screen, PANEL_STROKE, back, width=2, border_radius=12)
            screen.blit(img, r)

        # Draw strip + prime gate + carry dot
        strip.draw(screen, strip_rect, gate_rect)

        # Footer status
        st = stepper.state
        screen.blit(font(16).render(f"state={st}   auto={'ON' if auto else 'OFF'}   rate={auto_rate:.2f}/s", True, SUB), (40, H - 28))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
