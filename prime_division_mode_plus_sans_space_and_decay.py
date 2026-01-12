# prime_division_mode_plus_sans_space_and_decay.py
#
# Manim Community v0.19.x  (tested style/idioms for 0.19.1)
# Run:
#   python -m manim -pqh prime_division_mode_plus_sans_space_and_decay.py PrimeDivisionModePlusSansSpaceAndDecay
#
# Combines:
# 1) Your pygame "Prime Division Laser Graph" semantics (towers of dots per n)
#    - GREEN  = n % d != 0  OR synthetic prime-fill (work completed)
#    - RED    = n % d == 0  (composite witness)
#    - YELLOW = spacer / structural marker only (n < 2)
#    - HEIGHT RULE:
#        Prime n     -> exactly n green dots tall
#        Composite n -> < n dots, ending in one red dot (at first witness divisor)
#
# 2) The Manim interlude + graph ideas:
#    - Sans-space ramp (primes only): s(n) = n - 2^(bit_length(n)-1)
#    - Prime density decay by bit bucket (with ~ 1/ln(n) guide)
#    - End card: "Thanks For Watching! - ONOJK123"

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from manim import *


# ==================== MATH ====================

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


def trial_divisors_for_n(n: int) -> List[int]:
    # matches your pygame: d in [2..isqrt(n)]
    if n < 2:
        return []
    return list(range(2, int(math.isqrt(n)) + 1))


def primes_upto(nmax: int) -> List[int]:
    return [n for n in range(2, nmax + 1) if is_prime(n)]


def sans_space(n: int) -> int:
    b = n.bit_length()
    return n - (1 << (b - 1))


# ==================== DATA ====================

@dataclass
class TowerDot:
    n: int
    k: int
    color: str  # "green" | "red" | "yellow"


@dataclass
class BucketStats:
    bits: int
    start: int
    end: int
    size: int
    primes: int
    density: float
    delta_density: float | None


# ==================== STYLE ====================

COL = {
    "bg": BLACK,
    "text": WHITE,
    "sub": GREY_B,

    "green": GREEN_C,
    "red": RED_C,
    "yellow": YELLOW,

    "accent": PURPLE_B,
    "line": GREY_B,
    "dot": BLUE_B,
    "grid": GREY_E,
}

FONT = {
    "serif": "DejaVu Serif",
    "mono": "DejaVu Sans Mono",
}

LAY = {
    "side": 0.65,
    "top": 0.55,
    "bottom": 0.85,
    "title_size": 48,
    "subtitle_size": 26,
}


# ==================== HELPERS ====================

def clamp_to_frame(mob: Mobject, margin: float = 0.25) -> None:
    left_lim = -config.frame_width / 2 + margin
    right_lim = config.frame_width / 2 - margin
    bot_lim = -config.frame_height / 2 + margin
    top_lim = config.frame_height / 2 - margin

    dx = 0.0
    dy = 0.0
    if mob.get_right()[0] > right_lim:
        dx -= (mob.get_right()[0] - right_lim)
    if mob.get_left()[0] < left_lim:
        dx += (left_lim - mob.get_left()[0])
    if mob.get_bottom()[1] < bot_lim:
        dy += (mob.get_bottom()[1] - bot_lim)
    if mob.get_top()[1] > top_lim:
        dy -= (mob.get_top()[1] - top_lim)

    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
        mob.shift(np.array([dx, dy, 0.0]))


def scale_down_to_fit(mob: Mobject, max_w: float, max_h: float) -> None:
    if mob.width <= 0 or mob.height <= 0:
        return
    s = min(max_w / mob.width, max_h / mob.height, 1.0)
    mob.scale(s)


# ==================== PRIME DIVISION (TOWERS) ====================

def build_tower_for_n(n: int) -> Tuple[List[TowerDot], dict]:
    """
    Returns:
      (dots, meta)
    meta contains:
      kind: "trivial" | "prime" | "composite"
      witness: divisor or None
      checks: number of divisor checks performed (greens from checking stage)
    """
    if n < 2:
        return ([TowerDot(n=n, k=0, color="yellow")], {"kind": "trivial", "witness": None, "checks": 0})

    divs = trial_divisors_for_n(n)
    k = 0
    # CHECKING stage: green per non-divisor; red on first divisor
    for d in divs:
        if (n % d) == 0:
            dots = [TowerDot(n=n, k=i, color="green") for i in range(k)]
            dots.append(TowerDot(n=n, k=k, color="red"))
            return (dots, {"kind": "composite", "witness": d, "checks": k + 1})
        k += 1

    # PRIME stage: fill up to height n with green (your semantic)
    dots = [TowerDot(n=n, k=i, color="green") for i in range(n)]
    return (dots, {"kind": "prime", "witness": None, "checks": len(divs)})


class PrimeDivisionModePlusSansSpaceAndDecay(Scene):
    def construct(self):
        self.camera.background_color = COL["bg"]

        # Tune these to taste
        SPEED = 1.0
        N_MAX = 300

        # --------- Part 0: Title + legend ----------
        title = Text(
            "PRIME DIVISION MODE",
            font=FONT["serif"],
            weight=BOLD,
            font_size=62,
            color=COL["text"],
        ).to_edge(UP, buff=0.55)

        subtitle = Text(
            "Each dot is one discrete unit of work or structure",
            font=FONT["serif"],
            font_size=32,
            color=COL["sub"],
        ).next_to(title, DOWN, buff=0.18)

        legend = VGroup(
            VGroup(Dot(radius=0.08, color=COL["green"]), Text("GREEN = n % d ≠ 0  (or prime-fill)", font=FONT["serif"], font_size=24, color=COL["text"])).arrange(RIGHT, buff=0.18),
            VGroup(Dot(radius=0.08, color=COL["red"]),   Text("RED   = n % d = 0  (composite witness)", font=FONT["serif"], font_size=24, color=COL["text"])).arrange(RIGHT, buff=0.18),
            VGroup(Dot(radius=0.08, color=COL["yellow"]),Text("YELLOW = spacer (n < 2)", font=FONT["serif"], font_size=24, color=COL["text"])).arrange(RIGHT, buff=0.18),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)

        legend_box = RoundedRectangle(
            corner_radius=0.25,
            width=legend.width + 0.55,
            height=legend.height + 0.45,
            stroke_width=1.5,
            stroke_color=GREY_D,
        ).set_fill(color=BLACK, opacity=0.55)

        legend_pack = VGroup(legend_box, legend).center().shift(DOWN * 0.65)
        legend.move_to(legend_box.get_center())

        self.play(FadeIn(title, shift=DOWN * 0.15), run_time=0.8 * SPEED)
        self.play(FadeIn(subtitle), run_time=0.6 * SPEED)
        self.play(FadeIn(legend_pack, shift=UP * 0.10), run_time=0.8 * SPEED)
        self.wait(0.6 * SPEED)

        self.play(FadeOut(legend_pack), FadeOut(subtitle), run_time=0.7 * SPEED)

        # --------- Part 1: Tower graph ----------
        # Axes: x = n (1..N_MAX), y = k (work index)
        # y_max is N_MAX because prime towers reach height n
        y_max = N_MAX

        axes = Axes(
            x_range=[1, N_MAX, 50],
            y_range=[0, y_max, 50],
            x_length=12.2,
            y_length=5.8,
            tips=False,
            axis_config={"include_numbers": False, "stroke_width": 2},
        ).to_edge(DOWN, buff=1.05)
        axes.shift(RIGHT * 0.35)

        axes.get_x_axis().add_numbers([1, 50, 100, 150, 200, 250, 300], font_size=22, num_decimal_places=0)
        axes.get_y_axis().add_numbers([0, 50, 100, 150, 200, 250, 300], font_size=22, num_decimal_places=0)

        xlab = Text("n", font=FONT["serif"], font_size=28, color=COL["sub"]).next_to(axes.get_x_axis(), RIGHT, buff=0.18)
        ylab = Text("tower height k", font=FONT["serif"], font_size=26, color=COL["sub"]).rotate(PI / 2)
        ylab.next_to(axes.get_y_axis(), LEFT, buff=0.28)
        ylab.move_to(np.array([ylab.get_center()[0], axes.get_y_axis().get_center()[1], 0]))

        # Keep title, but shrink/position so it never fights the plot
        t_small = title.copy().scale(0.70).to_corner(UL, buff=0.55)

        # Status panel (top-right)
        panel = RoundedRectangle(
            corner_radius=0.22,
            width=5.1,
            height=1.7,
            stroke_width=1.8,
            stroke_color=GREY_B,
        ).set_fill(color=BLACK, opacity=0.62).to_corner(UR, buff=0.45)

        status_line1 = Text("n = 1", font=FONT["mono"], font_size=34, color=COL["text"])
        status_line2 = Text("spacer", font=FONT["serif"], font_size=26, color=COL["sub"])
        status = VGroup(status_line1, status_line2).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        status.move_to(panel.get_center()).shift(LEFT * 0.35)

        panel_pack = VGroup(panel, status)
        clamp_to_frame(panel_pack, margin=0.30)

        # Thin baseline gridline at y=0
        baseline = Line(
            axes.c2p(1, 0),
            axes.c2p(N_MAX, 0),
            stroke_width=3,
            color=GREY_D,
        )

        self.play(Transform(title, t_small), run_time=0.65 * SPEED)
        self.play(Create(axes), FadeIn(xlab), FadeIn(ylab), run_time=0.9 * SPEED)
        self.play(Create(baseline), run_time=0.35 * SPEED)
        self.play(FadeIn(panel_pack), run_time=0.5 * SPEED)

        # Build towers
        all_dots = VGroup()
        col_lines = VGroup()

        # Dot sizing: small enough that primes up to 300 won't explode visually
        dot_radius = 0.030

        # How fast to animate:
        # - show first chunk slowly for clarity
        # - then accelerate
        SLOW_N = 25
        STEP_PAUSE = 0.06  # little breath per n in slow region

        def set_status(n: int, meta: dict) -> AnimationGroup:
            kind = meta["kind"]
            witness = meta.get("witness", None)
            checks = meta.get("checks", 0)

            if kind == "trivial":
                line2 = "spacer (n < 2)"
                col = COL["yellow"]
            elif kind == "prime":
                line2 = f"prime → fill to height {n}"
                col = COL["green"]
            else:
                line2 = f"composite: witness d={witness}  (after {checks} checks)"
                col = COL["red"]

            new1 = Text(f"n = {n}", font=FONT["mono"], font_size=34, color=COL["text"])
            new2 = Text(line2, font=FONT["serif"], font_size=24, color=col if kind != "trivial" else COL["sub"])
            new_status = VGroup(new1, new2).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
            new_status.move_to(status.get_center())

            return Transform(status, new_status)

        for n in range(1, N_MAX + 1):
            dots_n, meta = build_tower_for_n(n)

            # Column line (structure marker), ending a hair above max k in that tower
            max_k = max(td.k for td in dots_n)
            cl = Line(
                axes.c2p(n, 0),
                axes.c2p(n, max_k + 0.7),
                stroke_width=1.2,
                color=GREY_E,
            ).set_opacity(0.55)
            col_lines.add(cl)

            # Dots for this n
            vg = VGroup()
            for td in dots_n:
                c = COL["green"] if td.color == "green" else COL["red"] if td.color == "red" else COL["yellow"]
                vg.add(Dot(point=axes.c2p(td.n, td.k), radius=dot_radius, color=c))
            all_dots.add(vg)

            if n <= SLOW_N:
                # For the first chunk, animate each tower so the rule is obvious
                self.play(Create(cl), run_time=0.12 * SPEED)
                self.play(set_status(n, meta), run_time=0.10 * SPEED)
                # Show dots progressively for that tower
                self.play(ShowIncreasingSubsets(vg), run_time=0.28 * SPEED, rate_func=linear)
                self.wait(STEP_PAUSE * SPEED)
            elif n == SLOW_N + 1:
                # Transition to faster build
                note = Text(
                    "Now we accelerate…",
                    font=FONT["serif"],
                    font_size=30,
                    color=COL["sub"],
                ).to_edge(DOWN, buff=0.35)
                self.play(FadeIn(note), run_time=0.35 * SPEED)
                self.wait(0.25 * SPEED)
                self.play(FadeOut(note), run_time=0.35 * SPEED)

            if n > SLOW_N:
                # Fast batch: add without heavy per-dot animation
                self.add(cl, vg)
                if (n % 30) == 0:
                    # Update status occasionally so it still "tracks"
                    self.play(set_status(n, meta), run_time=0.08 * SPEED)

        # Make sure everything is on screen
        self.add(col_lines, all_dots)
        self.wait(0.8 * SPEED)

        tower_close = Text(
            "Primes produce full-height towers.\nComposites stop early when a divisor is found.",
            font=FONT["serif"],
            font_size=28,
            color=COL["text"],
        ).to_edge(DOWN, buff=0.30)
        clamp_to_frame(tower_close, margin=0.25)
        self.play(FadeIn(tower_close), run_time=0.6 * SPEED)
        self.wait(1.0 * SPEED)
        self.play(FadeOut(tower_close), run_time=0.6 * SPEED)

        # Fade out tower view
        self.play(
            FadeOut(VGroup(axes, xlab, ylab, baseline, panel_pack, col_lines, all_dots)),
            run_time=0.8 * SPEED,
        )

        # --------- Bridge ----------
        bridge = VGroup(
            Text("THE BIG PICTURE", font=FONT["serif"], weight=BOLD, font_size=54, color=COL["text"]),
            Text("The structure comes from binary buckets.", font=FONT["serif"], font_size=30, color=COL["hi"]),
            Text("Prime density thins as scale increases.", font=FONT["serif"], font_size=30, color=COL["sub"]),
            Text("Next: (1) sans-space ramp  •  (2) density decay", font=FONT["serif"], font_size=28, color=COL["sub"]),
        ).arrange(DOWN, buff=0.22).center()

        self.play(FadeIn(bridge, shift=UP * 0.12), run_time=0.9 * SPEED)
        self.wait(0.8 * SPEED)
        self.play(FadeOut(bridge, shift=DOWN * 0.12), run_time=0.8 * SPEED)

        # --------- Part 2: Sans-space ramp (primes only) ----------
        P_MAX_RAMP = 250
        primes_list = primes_upto(P_MAX_RAMP)
        kmax = len(primes_list)

        xs = list(range(1, kmax + 1))
        ys = [sans_space(p) for p in primes_list]
        y_r_max = max(ys) if ys else 1

        ramp_title = Text(
            "Sans-Space Ramp (Primes Only)",
            font=FONT["serif"],
            weight=BOLD,
            font_size=44,
            color=COL["text"],
        ).to_edge(UP, buff=0.45)

        eq = MathTex(r"s(n)=n-2^{\lfloor \log_2 n \rfloor}").scale(0.95)
        eq.set_color(COL["hi"])
        eq.next_to(ramp_title, DOWN, buff=0.25)

        sub = Text(
            "Blue = bucket structure   •   Green = primes sampling that structure",
            font=FONT["serif"],
            font_size=26,
            color=COL["sub"],
        ).next_to(eq, DOWN, buff=0.22)

        self.play(FadeIn(ramp_title, shift=DOWN * 0.12), run_time=0.7 * SPEED)
        self.play(FadeIn(eq), FadeIn(sub), run_time=0.7 * SPEED)

        axes_r = Axes(
            x_range=[1, max(2, kmax), max(1, kmax // 6)],
            y_range=[0, max(2, y_r_max), max(1, y_r_max // 6)],
            x_length=11.6,
            y_length=4.9,
            tips=False,
            axis_config={"include_numbers": False, "stroke_width": 2},
        ).to_edge(DOWN, buff=1.15)
        axes_r.shift(RIGHT * 0.40)

        axes_r.get_x_axis().add_numbers([1, kmax], font_size=22, num_decimal_places=0)
        axes_r.get_y_axis().add_numbers([0, y_r_max], font_size=22, num_decimal_places=0)

        xlab_r = Text("prime index k", font=FONT["serif"], font_size=26, color=COL["sub"]).next_to(
            axes_r.get_x_axis(), RIGHT, buff=0.20
        )
        ylab_r = Text("sans-space s(p)", font=FONT["serif"], font_size=26, color=COL["sub"]).rotate(PI / 2)
        ylab_r.next_to(axes_r.get_y_axis(), LEFT, buff=0.30)
        ylab_r.move_to(np.array([ylab_r.get_center()[0], axes_r.get_y_axis().get_center()[1], 0]))

        self.play(Create(axes_r), FadeIn(xlab_r), FadeIn(ylab_r), run_time=0.9 * SPEED)

        pts_r = VGroup(*[
            Dot(axes_r.c2p(x, y), radius=0.05, color=COL["green"])
            for x, y in zip(xs, ys)
        ])

        line_r = VMobject()
        line_r.set_points_as_corners([axes_r.c2p(x, y) for x, y in zip(xs, ys)])
        line_r.set_stroke(BLUE_B, width=4)

        self.play(Create(line_r), run_time=0.8 * SPEED)
        self.play(ShowIncreasingSubsets(pts_r), run_time=1.0 * SPEED, rate_func=linear)

        # Boundary markers at powers of two: first prime index where p >= 2^m
        boundary_lines = VGroup()
        boundary_labels = VGroup()

        m = 1
        while (1 << m) <= P_MAX_RAMP:
            B = 1 << m
            idx = None
            for i, p in enumerate(primes_list):
                if p >= B:
                    idx = i + 1
                    break
            if idx is not None:
                vline = DashedLine(
                    start=axes_r.c2p(idx, 0),
                    end=axes_r.c2p(idx, y_r_max),
                    dash_length=0.08
                ).set_stroke(WHITE, width=2)

                lab = Text(f"2^{m}", font=FONT["mono"], font_size=18, color=COL["text"])
                lab.rotate(PI / 2)
                lab.next_to(axes_r.c2p(idx, y_r_max), DOWN, buff=0.12)

                boundary_lines.add(vline)
                boundary_labels.add(lab)
            m += 1

        self.play(FadeIn(boundary_lines), FadeIn(boundary_labels), run_time=0.8 * SPEED)
        self.wait(0.9 * SPEED)

        ramp_close = Text(
            "Resets happen at powers of two (binary buckets), not because of primes.",
            font=FONT["serif"],
            font_size=28,
            color=COL["hi"],
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(ramp_close), run_time=0.7 * SPEED)
        self.wait(1.0 * SPEED)

        self.play(
            FadeOut(VGroup(ramp_title, eq, sub, axes_r, xlab_r, ylab_r, pts_r, line_r, boundary_lines, boundary_labels, ramp_close)),
            run_time=0.8 * SPEED
        )

        # --------- Part 3: Density decay by bit bucket ----------
        MIN_BITS_B, MAX_BITS_B = 2, 16
        stats_b: List[BucketStats] = []
        prev_d: Optional[float] = None

        for bits in range(MIN_BITS_B, MAX_BITS_B + 1):
            start = 2 ** (bits - 1)
            end = (2 ** bits) - 1
            size = end - start + 1
            primes_count = sum(1 for n in range(start, end + 1) if is_prime(n))
            density = primes_count / size
            delta = None if prev_d is None else (density - prev_d)
            stats_b.append(BucketStats(bits, start, end, size, primes_count, density, delta))
            prev_d = density

        dens_vals = [s.density for s in stats_b]
        y_min = max(0.0, min(dens_vals) - 0.02)
        y_max2 = min(1.0, max(dens_vals) + 0.02)

        g_title = Text(
            "Prime Density Thins Out With Scale",
            font=FONT["serif"],
            weight=BOLD,
            font_size=42,
            color=COL["text"],
        ).to_edge(UP, buff=0.45)
        self.play(FadeIn(g_title, shift=DOWN * 0.12), run_time=0.8 * SPEED)

        axes2 = Axes(
            x_range=[MIN_BITS_B, MAX_BITS_B, 2],
            y_range=[y_min, y_max2, (y_max2 - y_min) / 5],
            x_length=11.6,
            y_length=5.1,
            tips=False,
            axis_config={"include_numbers": False, "stroke_width": 2},
        ).to_edge(DOWN, buff=1.35)
        axes2.shift(RIGHT * 0.40)

        axes2.get_x_axis().add_numbers(list(range(MIN_BITS_B, MAX_BITS_B + 1, 2)), font_size=22, num_decimal_places=0)
        axes2.get_y_axis().add_numbers(
            [round(y_min + i * (y_max2 - y_min) / 5, 2) for i in range(6)],
            font_size=22,
            num_decimal_places=2,
        )

        x_label2 = Text("bits", font=FONT["serif"], font_size=26, color=COL["sub"]).next_to(
            axes2.get_x_axis(), RIGHT, buff=0.20
        )
        y_label2 = Text("prime density", font=FONT["serif"], font_size=26, color=COL["sub"]).rotate(PI / 2)
        y_label2.next_to(axes2.get_y_axis(), LEFT, buff=0.30)
        y_label2.move_to(np.array([y_label2.get_center()[0], axes2.get_y_axis().get_center()[1], 0]))

        self.play(Create(axes2), FadeIn(x_label2), FadeIn(y_label2), run_time=0.9 * SPEED)

        pts = VGroup(*[
            Dot(axes2.c2p(s.bits, s.density), radius=0.05, color=COL["dot"])
            for s in stats_b
        ])

        line = VMobject(color=COL["line"])
        line.set_points_as_corners([axes2.c2p(s.bits, s.density) for s in stats_b])

        theory = VMobject(color=PURPLE_B)
        theory_pts = []
        for s in stats_b:
            mid = 0.5 * (s.start + s.end)
            theory_pts.append(axes2.c2p(s.bits, 1.0 / math.log(mid)))
        theory.set_points_smoothly(theory_pts)

        # Legend (top-right, forced below title)
        leg = VGroup(
            Dot(radius=0.05, color=COL["dot"]),
            Text("measured bucket density", font=FONT["serif"], font_size=20, color=COL["text"]),
        ).arrange(RIGHT, buff=0.18)

        leg2 = VGroup(
            Line(ORIGIN, RIGHT * 0.55, color=PURPLE_B, stroke_width=6),
            Text("~ 1/ln(n) guide", font=FONT["serif"], font_size=20, color=COL["text"]),
        ).arrange(RIGHT, buff=0.18)

        legend = VGroup(leg, leg2).arrange(DOWN, aligned_edge=LEFT, buff=0.10)

        legend_bg = RoundedRectangle(
            corner_radius=0.2,
            width=legend.width + 0.45,
            height=legend.height + 0.30,
            stroke_width=1.5,
            stroke_color=GREY_D,
        ).set_fill(color=BLACK, opacity=0.65)

        legend_pack = VGroup(legend_bg, legend).to_corner(UR, buff=0.35)

        desired_top = g_title.get_bottom()[1] - 0.18
        legend_pack.shift(UP * (desired_top - legend_pack.get_top()[1]))
        clamp_to_frame(legend_pack, margin=0.25)

        self.play(FadeIn(legend_pack), run_time=0.6 * SPEED)

        self.play(Create(line), run_time=0.9 * SPEED)
        self.play(ShowIncreasingSubsets(pts), run_time=1.1 * SPEED, rate_func=linear)
        self.play(Create(theory), run_time=0.9 * SPEED)
        self.wait(0.45 * SPEED)

        # Per-bucket thinning panel
        def pct_change(curr: float, prev: float) -> float:
            if prev == 0:
                return 0.0
            return 100.0 * (curr - prev) / prev

        changes = []
        for i, s in enumerate(stats_b):
            if i == 0:
                changes.append(0.0)
            else:
                changes.append(pct_change(s.density, stats_b[i - 1].density))

        panel2 = RoundedRectangle(
            corner_radius=0.25,
            width=4.95,
            height=2.10,
            stroke_width=2,
            stroke_color=GREY_B,
        ).to_corner(DR, buff=0.55)
        panel2.set_fill(BLACK, opacity=0.82)

        panel_title = Text("Per-bucket thinning", font=FONT["serif"], font_size=28, color=COL["hi"])
        panel_title.move_to(panel2.get_top() + DOWN * 0.36)

        readout = Text("b= 2   Δ%=+0.00", font=FONT["mono"], font_size=30, color=COL["text"])
        readout.move_to(panel2.get_center() + DOWN * 0.10)

        hint = Text("compare b to b−1", font=FONT["serif"], font_size=22, color=COL["sub"])
        hint.move_to(panel2.get_bottom() + UP * 0.33)

        panel_pack2 = VGroup(panel2, panel_title, readout, hint)
        panel_pack2.set_z_index(100)
        panel_pack2.shift(UP * 0.25 + LEFT * 0.10)
        clamp_to_frame(panel_pack2, margin=0.28)

        self.add(panel_pack2)
        self.play(FadeIn(panel_pack2), run_time=0.7 * SPEED)

        cursor = Dot(radius=0.07, color=YELLOW).set_z_index(200)
        cursor.move_to(axes2.c2p(stats_b[0].bits, stats_b[0].density))
        self.play(FadeIn(cursor), run_time=0.4 * SPEED)

        for i, s in enumerate(stats_b):
            new_pos = axes2.c2p(s.bits, s.density)
            val = changes[i]
            sign = "+" if val >= 0 else "−"
            txt = f"b={s.bits:>2}   Δ%={sign}{abs(val):5.2f}"

            new_readout = Text(txt, font=FONT["mono"], font_size=30, color=COL["text"])
            new_readout.move_to(readout.get_center())

            self.play(
                cursor.animate.move_to(new_pos),
                Transform(readout, new_readout),
                run_time=0.35 * SPEED,
                rate_func=linear,
            )

        close = Text(
            "Bigger buckets → fewer primes per number.\nThat thinning is the density decay you are seeing.",
            font=FONT["serif"],
            font_size=26,
            color=COL["text"],
        ).to_edge(DOWN, buff=0.35)
        clamp_to_frame(close, margin=0.25)

        self.play(FadeIn(close), run_time=0.8 * SPEED)
        self.wait(1.2 * SPEED)

        self.play(
            FadeOut(VGroup(axes2, x_label2, y_label2, pts, line, theory, legend_pack, panel_pack2, cursor, close, g_title)),
            run_time=0.8 * SPEED
        )

        # --------- End card ----------
        end_card = VGroup(
            Text("Thanks For Watching!", font=FONT["serif"], weight=BOLD, font_size=78, color=COL["text"]),
            Text("- ONOJK123", font=FONT["serif"], font_size=48, color=COL["hi"]),
        ).arrange(DOWN, buff=0.35).center()

        frame = RoundedRectangle(
            corner_radius=0.35,
            width=min(config.frame_width - 1.2, end_card.width + 1.2),
            height=min(config.frame_height - 1.2, end_card.height + 1.0),
            stroke_width=2,
            stroke_color=GREY_D,
        ).set_fill(BLACK, opacity=0.35)

        end_pack = VGroup(frame, end_card).center()

        self.play(FadeIn(end_pack, shift=UP * 0.10), run_time=0.9 * SPEED)
        self.wait(1.8 * SPEED)
        self.play(FadeOut(end_pack), run_time=0.8 * SPEED)
