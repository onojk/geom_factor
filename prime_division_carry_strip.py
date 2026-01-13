# prime_division_carry_strip.py
#
# PYGAME — Prime towers (dots) + carry strip UI
#         + binary(n) with bucket-meaning colors:
#             ORANGE = "bit-bucket structure" (leading 1; does not change inside bucket)
#             PURPLE = "payload bits" (change inside bucket)
#             YELLOW = parity bit (LSB) (odd/even survivor)
#         + E/O slots show 0/1 underneath
#
# Controls:
#   SPACE : step / hold to repeat
#   A     : toggle auto-step
#   UP    : faster auto
#   DOWN  : slower auto
#   LEFT  : zoom out
#   RIGHT : zoom in
#   ESC   : quit

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
N_MAX = 260

ORIGIN_SCREEN_Y_FRAC = 0.90  # where y=0 sits on screen

BG = (8, 8, 12)
WHITE = (235, 235, 245)
SUB = (160, 160, 180)

GREEN = (80, 255, 120)
RED = (255, 90, 90)
YELLOW = (255, 220, 90)

BLUE = (120, 170, 255)
PURPLE = (190, 120, 255)
ORANGE = (255, 165, 80)

GREY = (90, 90, 110)
GRID_MAJOR = (24, 24, 32)
GRID_MINOR = (16, 16, 22)

PANEL = (20, 20, 28)
PANEL_STROKE = (90, 90, 110)

DOT_R = 3

# Zoom
SCALE_DEFAULT = 3.4
SCALE_MIN = 1.0
SCALE_MAX = 18.0
ZOOM_STEP = 1.12

# Stepping
SPACE_REPEAT_RATE = 16.0
AUTO_MIN = 0.5
AUTO_MAX = 80.0

# Carry strip layout
STRIP_SLOTS = 18
SLOT_W = 46
SLOT_H = 56
SLOT_GAP = 10
CARRY_LERP = 0.22  # smoothing for the carry dot

# ----------------------------
# Helpers
# ----------------------------
def font(size: int, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont("dejavusansmono", size, bold=bold)


def clamp(x: float, a: float, b: float) -> float:
    return a if x < a else b if x > b else x


def bin_str(n: int) -> str:
    return format(n, "b") if n >= 0 else "0"


def trial_divisors(n: int) -> List[int]:
    if n < 2:
        return []
    lim = int(math.isqrt(n))
    return list(range(2, lim + 1))


# ----------------------------
# World <-> screen mapping for towers
# ----------------------------
def world_to_screen(wx: float, wy: float, cx: float, cy: float, scale: float) -> Tuple[int, int]:
    sx = W * 0.5 + (wx - cx) * scale
    sy = H * ORIGIN_SCREEN_Y_FRAC - (wy - cy) * scale
    return int(sx), int(sy)


def draw_background(surface: pygame.Surface, cx: float, cy: float, scale: float):
    surface.fill(BG)

    # baseline y=0
    _, by = world_to_screen(0, 0, cx, cy, scale)
    pygame.draw.line(surface, GREY, (0, by), (W, by), 2)

    # light vertical grid (minor/major)
    if scale <= 0:
        return

    minor_world = max(1, int(60 / scale))
    major_world = minor_world * 5

    left_world = cx - (W * 0.5) / scale
    right_world = cx + (W * 0.5) / scale

    start = int(math.floor(left_world / minor_world) * minor_world)
    x = start
    while x <= right_world:
        sx, _ = world_to_screen(x, 0, cx, cy, scale)
        col = GRID_MAJOR if (x % major_world == 0) else GRID_MINOR
        pygame.draw.line(surface, col, (sx, 0), (sx, H), 1)
        x += minor_world


# ----------------------------
# Data
# ----------------------------
@dataclass(frozen=True)
class Dot:
    n: int
    k: int
    color: str  # "green" | "red" | "yellow"


@dataclass
class StepEvent:
    kind: str  # "TRIVIAL"|"CHECK"|"WITNESS"|"ENTER_PRIME"|"PRIME_FILL"|"NEXT_N"|"DONE"
    n: int
    d: Optional[int] = None
    r: Optional[int] = None


# ----------------------------
# Prime-stepper (towers)
# ----------------------------
class PrimeStepper:
    """
    Tower semantics:
      - TRIVIAL: one yellow dot at k=0 then advance
      - CHECK: for d=2..sqrt(n):
          green dot per failed divisor; if hits divisor => red dot and advance
      - PRIME: fill green dots until k reaches n, then advance
    Emits StepEvent for UI.
    """

    def __init__(self):
        self.dots: List[Dot] = []

        self.n = N_MIN
        self.ds: List[int] = []
        self.i = 0
        self.k = 0
        self.state = "INIT"

        self.last_event = StepEvent("INIT", self.n)
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
            self.last_event = StepEvent("TRIVIAL", self.n)
        elif len(self.ds) == 0:
            self.state = "PRIME"
            self.last_event = StepEvent("ENTER_PRIME", self.n)
        else:
            self.state = "CHECK"
            d0 = self.ds[0]
            self.last_event = StepEvent("CHECK", self.n, d=d0, r=self.n % d0)

    def _next_n(self):
        self.n += 1
        self._sync()

    def step(self) -> StepEvent:
        if self.state == "DONE":
            self.last_event = StepEvent("DONE", self.n)
            return self.last_event

        if self.state == "TRIVIAL":
            self.dots.append(Dot(self.n, 0, "yellow"))
            ev = StepEvent("TRIVIAL", self.n)
            self._next_n()
            self.last_event = ev
            return ev

        if self.state == "CHECK":
            d = self.ds[self.i]
            r = self.n % d
            if r == 0:
                self.dots.append(Dot(self.n, self.k, "red"))
                ev = StepEvent("WITNESS", self.n, d=d, r=0)
                self._next_n()
                self.last_event = ev
                return ev
            else:
                self.dots.append(Dot(self.n, self.k, "green"))
                ev = StepEvent("CHECK", self.n, d=d, r=r)
                self.k += 1
                self.i += 1
                if self.i >= len(self.ds):
                    self.state = "PRIME"
                    self.k = 0  # prime fill starts at baseline
                self.last_event = ev
                return ev

        if self.state == "PRIME":
            if self.k == 0 and self.last_event.kind != "ENTER_PRIME":
                ev = StepEvent("ENTER_PRIME", self.n)
                self.last_event = ev
                return ev

            if self.k < self.n:
                self.dots.append(Dot(self.n, self.k, "green"))
                self.k += 1
                ev = StepEvent("PRIME_FILL", self.n)
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
# Carry strip visual (UI only)
# ----------------------------
class CarryStrip:
    """
    Shows parity slots of divisor checks (E/O) and one carry dot moving across.
    Carry bit flips on O slots.
    On ENTER_PRIME: carry dot moves into PRIME GATE.
    On WITNESS: carry collapses.
    """

    def __init__(self):
        self.reset_for_new_n(N_MIN)

        self.carry_x = 0.0
        self.carry_y = 0.0
        self.target_x = 0.0
        self.target_y = 0.0

    def reset_for_new_n(self, n: int):
        self.n = n
        self.slots: List[str] = []
        self.carry_bit = 0
        self.carry_alive = True
        self.in_prime_gate = False
        self.collapsed = False
        self.target_index = -1

    def _slot_center(self, strip_rect: pygame.Rect, idx: int) -> Tuple[float, float]:
        x0 = strip_rect.x + 16
        y0 = strip_rect.y + strip_rect.height // 2
        x = x0 + idx * (SLOT_W + SLOT_GAP) + SLOT_W / 2
        y = y0
        return x, y

    def _pre_slot_center(self, strip_rect: pygame.Rect) -> Tuple[float, float]:
        x0 = strip_rect.x + 16
        y0 = strip_rect.y + strip_rect.height // 2
        return x0 - 18, y0

    def _gate_center(self, gate_rect: pygame.Rect) -> Tuple[float, float]:
        return gate_rect.centerx, gate_rect.centery

    def on_event(self, ev: StepEvent):
        if ev.kind == "CHECK" and ev.d is not None:
            parity = "E" if (ev.d % 2 == 0) else "O"
            self.slots.append(parity)
            if len(self.slots) > STRIP_SLOTS:
                self.slots = self.slots[-STRIP_SLOTS:]

            if parity == "O":
                self.carry_bit ^= 1

            self.target_index = min(len(self.slots) - 1, STRIP_SLOTS - 1)

        elif ev.kind == "WITNESS":
            self.carry_alive = False
            self.collapsed = True

        elif ev.kind == "ENTER_PRIME":
            if self.carry_alive and not self.collapsed:
                self.in_prime_gate = True

        elif ev.kind == "TRIVIAL":
            self.carry_bit = 0
            self.carry_alive = True
            self.in_prime_gate = False
            self.collapsed = False
            self.target_index = -1

    def update_anim(self, strip_rect: pygame.Rect, gate_rect: pygame.Rect):
        if self.carry_x == 0.0 and self.carry_y == 0.0:
            self.carry_x, self.carry_y = self._pre_slot_center(strip_rect)

        if self.in_prime_gate and self.carry_alive and not self.collapsed:
            tx, ty = self._gate_center(gate_rect)
        else:
            if self.target_index < 0:
                tx, ty = self._pre_slot_center(strip_rect)
            else:
                tx, ty = self._slot_center(strip_rect, self.target_index)

        self.target_x, self.target_y = tx, ty
        self.carry_x += (self.target_x - self.carry_x) * CARRY_LERP
        self.carry_y += (self.target_y - self.carry_y) * CARRY_LERP

    def draw(self, screen: pygame.Surface, strip_rect: pygame.Rect, gate_rect: pygame.Rect):
        pygame.draw.rect(screen, PANEL, strip_rect, border_radius=12)
        pygame.draw.rect(screen, PANEL_STROKE, strip_rect, width=2, border_radius=12)

        screen.blit(font(18, True).render("EVEN/ODD DIVISOR STRIP  (d mod 2)", True, WHITE),
                    (strip_rect.x + 14, strip_rect.y + 10))

        base_x = strip_rect.x + 16
        mid_y = strip_rect.y + strip_rect.height // 2
        top_y = mid_y - SLOT_H / 2

        f_slot = font(24, True)
        f_bit = font(16, True)

        for i in range(STRIP_SLOTS):
            x = base_x + i * (SLOT_W + SLOT_GAP)
            r = pygame.Rect(int(x), int(top_y), SLOT_W, SLOT_H)
            pygame.draw.rect(screen, (28, 28, 38), r, border_radius=10)
            pygame.draw.rect(screen, (70, 70, 90), r, width=2, border_radius=10)

            if i < len(self.slots):
                ch = self.slots[i]
                bit = "0" if ch == "E" else "1"
                col = BLUE if ch == "E" else PURPLE

                img1 = f_slot.render(ch, True, col)
                screen.blit(img1, img1.get_rect(center=(r.centerx, r.centery - 10)))

                img2 = f_bit.render(bit, True, col)
                screen.blit(img2, img2.get_rect(center=(r.centerx, r.centery + 14)))

        # gate
        pygame.draw.rect(screen, (18, 18, 26), gate_rect, border_radius=12)
        pygame.draw.rect(screen, PANEL_STROKE, gate_rect, width=2, border_radius=12)
        screen.blit(font(18, True).render("PRIME GATE", True, GREEN), (gate_rect.x + 14, gate_rect.y + 10))
        screen.blit(font(16).render("carry passes here", True, SUB), (gate_rect.x + 14, gate_rect.y + 38))

        # carry bit text
        msg = f"CARRY BIT = {self.carry_bit}"
        col = GREEN if (self.carry_alive and not self.collapsed) else RED
        screen.blit(font(18, True).render(msg, True, col), (strip_rect.x + 14, strip_rect.bottom - 30))

        # carry dot
        dot_col = RED if self.collapsed else (GREEN if self.carry_alive else RED)
        pygame.draw.circle(screen, dot_col, (int(self.carry_x), int(self.carry_y)), 7)
        lab = font(16, True).render(str(self.carry_bit), True, (0, 0, 0))
        screen.blit(lab, lab.get_rect(center=(int(self.carry_x), int(self.carry_y))))


# ----------------------------
# Main
# ----------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Prime Towers + Bit-Bucket Coloring")
    clock = pygame.time.Clock()

    stepper = PrimeStepper()
    strip = CarryStrip()
    strip.reset_for_new_n(stepper.n)

    # Camera for towers
    cx = (N_MIN + N_MAX) / 2
    cy = 0.0
    scale = SCALE_DEFAULT

    # stepping controls
    auto = False
    auto_rate = 6.0
    auto_acc = 0.0
    space_held = False
    space_acc = 0.0

    # UI panels
    strip_rect = pygame.Rect(40, 250, W - 80, 170)
    gate_rect = pygame.Rect(W - 260, 445, 220, 95)

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
                    ev = stepper.step()
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
                elif e.key == pygame.K_LEFT:
                    scale = max(SCALE_MIN, scale / ZOOM_STEP)
                elif e.key == pygame.K_RIGHT:
                    scale = min(SCALE_MAX, scale * ZOOM_STEP)
            elif e.type == pygame.KEYUP:
                if e.key == pygame.K_SPACE:
                    space_held = False
                    space_acc = 0.0

        # auto
        if auto:
            auto_acc += dt
            step_dt = 1.0 / max(0.0001, auto_rate)
            while auto_acc >= step_dt:
                auto_acc -= step_dt
                ev = stepper.step()
                if stepper.n != strip.n:
                    strip.reset_for_new_n(stepper.n)
                strip.on_event(ev)

        # hold-to-repeat
        if space_held and not auto:
            space_acc += dt
            step_dt = 1.0 / SPACE_REPEAT_RATE
            while space_acc >= step_dt:
                space_acc -= step_dt
                ev = stepper.step()
                if stepper.n != strip.n:
                    strip.reset_for_new_n(stepper.n)
                strip.on_event(ev)

        # background + towers
        draw_background(screen, cx, cy, scale)

        # column lines
        maxk = {}
        for d in stepper.dots:
            maxk[d.n] = max(maxk.get(d.n, 0), d.k)
        for n_, mk in maxk.items():
            sx0, sy0 = world_to_screen(n_, 0, cx, cy, scale)
            sx1, sy1 = world_to_screen(n_, mk + 0.2, cx, cy, scale)
            pygame.draw.line(screen, (55, 55, 70), (sx0, sy0), (sx1, sy1), 1)

        # dots
        for dot in stepper.dots:
            sx, sy = world_to_screen(dot.n, dot.k, cx, cy, scale)
            col = GREEN if dot.color == "green" else RED if dot.color == "red" else YELLOW
            pygame.draw.circle(screen, col, (sx, sy), DOT_R)

        # animate carry dot
        strip.update_anim(strip_rect, gate_rect)

        # header text
        screen.blit(font(28, True).render("MOD CHECKS → PARITY STRIP → CARRY PASSES INTO PRIME", True, WHITE), (40, 22))
        screen.blit(font(18).render(
            "SPACE step/hold  •  A auto  •  UP/DOWN speed  •  LEFT/RIGHT zoom  •  ESC quit",
            True, SUB
        ), (40, 60))

        # n + binary with your color rule
        n = stepper.n
        bits = bin_str(n)

        screen.blit(font(44, True).render(f"n = {n}", True, WHITE), (40, 105))

        x0, y0 = 40, 160
        screen.blit(font(22, True).render("binary(n) = ", True, WHITE), (x0, y0))

        bx = x0 + 155
        f_bit = font(22, True)

        # Color rule:
        #   ORANGE: bucket-structure bit(s) that don't change (leading 1)
        #   PURPLE: bits that do change (payload bits)
        #   YELLOW: parity bit (LSB)
        for i, ch in enumerate(bits):
            is_msb = (i == 0 and len(bits) > 0)      # leading 1 in this bucket
            is_lsb = (i == len(bits) - 1)

            if is_lsb:
                col = YELLOW
            elif is_msb:
                col = ORANGE
            else:
                col = PURPLE

            img = f_bit.render(ch, True, col)
            screen.blit(img, (bx, y0))
            bx += img.get_width() + 4

        parity_label = "EVEN" if (n % 2 == 0) else "ODD"
        screen.blit(font(16, True).render(f"{parity_label}  (LSB)", True, YELLOW), (bx + 10, y0 + 4))
        screen.blit(font(16).render("MSB=structure (bucket) • middle=payload • LSB=parity", True, SUB), (40, 190))

        # big realtime mod-check line
        ev = stepper.last_event
        big_msg, big_col = "", WHITE
        if ev.kind == "CHECK" and ev.d is not None and ev.r is not None:
            big_msg, big_col = f"{ev.n} % {ev.d} = {ev.r}", WHITE
        elif ev.kind == "WITNESS" and ev.d is not None:
            big_msg, big_col = f"{ev.n} % {ev.d} = 0  →  COMPOSITE (carry collapses)", RED
        elif ev.kind == "ENTER_PRIME":
            big_msg, big_col = f"{ev.n} has no witness ≤ √n  →  PRIME (carry passes into gate)", GREEN
        elif ev.kind == "PRIME_FILL":
            big_msg, big_col = "PRIME FILL: building up to height n (carry already passed)", GREEN
        elif ev.kind == "TRIVIAL":
            big_msg, big_col = f"{ev.n} < 2 (spacer)", YELLOW
        elif ev.kind == "DONE":
            big_msg, big_col = "DONE", WHITE

        if big_msg:
            msg_font = font(26, True)
            img = msg_font.render(big_msg, True, big_col)
            r = img.get_rect(center=(W // 2, 470))
            back = pygame.Rect(r.x - 16, r.y - 10, r.width + 32, r.height + 20)
            pygame.draw.rect(screen, PANEL, back, border_radius=12)
            pygame.draw.rect(screen, PANEL_STROKE, back, width=2, border_radius=12)
            screen.blit(img, r)

        # strip + gate
        strip.draw(screen, strip_rect, gate_rect)

        # footer
        screen.blit(font(16).render(
            f"state={stepper.state}  auto={'ON' if auto else 'OFF'}  rate={auto_rate:.2f}/s  scale={scale:.2f}",
            True, SUB
        ), (40, H - 28))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
