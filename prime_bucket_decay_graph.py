# prime_bucket_decay_graph.py
#
# Manim Community v0.19.x
# Run:
#   python -m manim -pqh prime_bucket_decay_graph.py PrimeBucketDecayGraph

from __future__ import annotations

import math
from typing import List, Tuple

from manim import *


# ------------------ fast prime counting ------------------

def sieve_is_prime(n: int) -> List[bool]:
    """Simple sieve up to n inclusive."""
    if n < 1:
        return [False] * (n + 1)
    is_p = [True] * (n + 1)
    is_p[0] = False
    if n >= 1:
        is_p[1] = False
    r = int(math.isqrt(n))
    for p in range(2, r + 1):
        if is_p[p]:
            step = p
            start = p * p
            is_p[start : n + 1 : step] = [False] * (((n - start) // step) + 1)
    return is_p


def prefix_prime_counts(is_p: List[bool]) -> List[int]:
    """prefix[i] = #primes <= i"""
    pref = [0] * len(is_p)
    running = 0
    for i, v in enumerate(is_p):
        running += 1 if v else 0
        pref[i] = running
    return pref


def primes_in_range(prefix: List[int], a: int, b: int) -> int:
    """#primes in [a, b] inclusive"""
    if a <= 0:
        return prefix[b]
    return prefix[b] - prefix[a - 1]


def bucket_density(bits: int, prefix: List[int]) -> Tuple[int, int, float]:
    start = 2 ** (bits - 1)
    end = (2 ** bits) - 1
    size = end - start + 1
    pcount = primes_in_range(prefix, start, end)
    return start, end, pcount / size


def nice_step(max_val: float) -> float:
    """
    Pick a human-ish step for y ticks.
    This doesn't need to be perfect—just avoid 1.149999999 labels.
    """
    if max_val <= 0:
        return 1.0
    # candidate steps (scaled by powers of 10)
    candidates = [1, 0.5, 0.25, 0.2, 0.1, 0.05, 0.02, 0.01]
    power = 10 ** math.floor(math.log10(max_val))
    scaled = max_val / power
    # choose a step that gives ~4–7 ticks
    best = None
    best_score = 1e9
    for c in candidates:
        ticks = max_val / (c * power)
        score = abs(ticks - 5.0)
        if score < best_score:
            best_score = score
            best = c * power
    return float(best)


# ------------------ scene ------------------

class PrimeBucketDecayGraph(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        SPEED = 1.0  # animation timing multiplier

        MIN_BITS = 2
        MAX_BITS = 18  # safe + fast with sieve

        # ---------- compute data (FAST) ----------
        max_n = (2 ** MAX_BITS) - 1
        is_p = sieve_is_prime(max_n)
        pref = prefix_prime_counts(is_p)

        bits_list: List[int] = list(range(MIN_BITS, MAX_BITS + 1))
        densities: List[float] = []
        ranges: List[Tuple[int, int]] = []

        for b in bits_list:
            s, e, d = bucket_density(b, pref)
            ranges.append((s, e))
            densities.append(d)

        baseline = densities[0]
        percent_of_baseline = [100.0 * (d / baseline) for d in densities]

        # Heuristic curve: 1/ln(2^b)=1/(b ln 2), scaled to match baseline visually
        ln2 = math.log(2)
        heuristic_raw = [1.0 / (b * ln2) for b in bits_list]
        scale = baseline / heuristic_raw[0]
        heuristic = [scale * v for v in heuristic_raw]

        # ---------- title block (smaller, higher) ----------
        title = Text(
            "PRIME BUCKETS: DENSITY DECAY",
            font="DejaVu Serif",
            weight=BOLD,
            color=WHITE,
            font_size=44,
        ).set_opacity(0.95)
        subtitle = Text(
            "Density per bit-range bucket shrinks as numbers grow",
            font="DejaVu Serif",
            color=GREY_B,
            font_size=26,
        )

        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.12)
        title_group.to_edge(UP, buff=0.35)

        self.play(FadeIn(title_group, shift=UP * 0.2), run_time=0.8 * SPEED)
        self.wait(0.2 * SPEED)

        # ---------- axes layout ----------
        x_min = MIN_BITS
        x_max = MAX_BITS

        y1_max = max(densities) * 1.10
        y1_step = nice_step(y1_max)
        # snap y1_max upward to a clean multiple of y1_step
        y1_max = y1_step * math.ceil(y1_max / y1_step)

        left_axes = Axes(
            x_range=[x_min, x_max, 2],
            y_range=[0, y1_max, y1_step],
            x_length=5.6,
            y_length=4.3,
            axis_config={"color": GREY_B, "stroke_width": 2},
            tips=False,
        )
        right_axes = Axes(
            x_range=[x_min, x_max, 2],
            y_range=[0, 100, 20],
            x_length=5.6,
            y_length=4.3,
            axis_config={"color": GREY_B, "stroke_width": 2},
            tips=False,
        )

        # Put them in a row with real spacing
        graphs = VGroup(left_axes, right_axes).arrange(RIGHT, buff=1.0)
        graphs.next_to(title_group, DOWN, buff=0.55)

        # We will add our own numbers (avoid Manim auto-number overlap)
        for ax in [left_axes, right_axes]:
            ax.get_x_axis().add_numbers(
                *list(range(MIN_BITS, MAX_BITS + 1, 2)),
                font_size=22,
                num_decimal_places=0,
            )

        left_axes.get_y_axis().add_numbers(
            *[round(v, 3) for v in frange(0, y1_max, y1_step)],
            font_size=22,
            num_decimal_places=2,
        )

        right_axes.get_y_axis().add_numbers(
            0, 20, 40, 60, 80, 100,
            font_size=22,
            num_decimal_places=0,
        )

        left_label = Text(
            "Prime density per bucket",
            font="DejaVu Serif",
            color=WHITE,
            font_size=26,
        ).next_to(left_axes, UP, buff=0.18)

        right_label = Text(
            "% of 2-bit baseline",
            font="DejaVu Serif",
            color=WHITE,
            font_size=26,
        ).next_to(right_axes, UP, buff=0.18)

        self.play(Create(left_axes), Create(right_axes), run_time=0.9 * SPEED)
        self.play(FadeIn(left_label), FadeIn(right_label), run_time=0.5 * SPEED)

        # ---------- plot helpers ----------
        def pts_for(axes: Axes, ys: List[float]) -> List[np.ndarray]:
            return [axes.c2p(b, y) for b, y in zip(bits_list, ys)]

        left_points = pts_for(left_axes, densities)
        right_points = pts_for(right_axes, percent_of_baseline)
        heuristic_points = pts_for(left_axes, heuristic)

        left_line = VMobject().set_points_as_corners(left_points).set_stroke(width=4)
        right_line = VMobject().set_points_as_corners(right_points).set_stroke(width=4)
        heuristic_line = (
            VMobject()
            .set_points_as_corners(heuristic_points)
            .set_stroke(width=3, opacity=0.55)
        )

        left_dots = VGroup(*[Dot(p, radius=0.04) for p in left_points])
        right_dots = VGroup(*[Dot(p, radius=0.04) for p in right_points])

        self.play(Create(left_line), FadeIn(left_dots, lag_ratio=0.05), run_time=1.0 * SPEED)
        self.play(Create(right_line), FadeIn(right_dots, lag_ratio=0.05), run_time=1.0 * SPEED)

        heur_label = Text(
            "heuristic ~ 1/ln(n)",
            font="DejaVu Serif",
            color=GREY_B,
            font_size=22,
        ).next_to(left_axes, DOWN, buff=0.25).align_to(left_axes, LEFT)

        self.play(Create(heuristic_line), FadeIn(heur_label), run_time=0.7 * SPEED)

        # ---------- compact explanation block (no collision with axes) ----------
        expl = VGroup(
            Text("Interpretation", font="DejaVu Serif", weight=BOLD, font_size=30, color=WHITE),
            Text("Bucket b is [2^(b−1), 2^b−1].  Density(b) = primes / bucket_size.", font="DejaVu Serif", font_size=24, color=GREY_B),
            Text("Prime Number Theorem suggests density ≈ 1/ln(n), so it slowly decays.", font="DejaVu Serif", font_size=24, color=GREY_B),
            Text("Primes never stop… but the *fraction* of integers that are prime → 0.", font="DejaVu Serif", font_size=26, color=WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        box = RoundedRectangle(
            corner_radius=0.2,
            width=graphs.width,
            height=1.55,
            stroke_width=1.5,
            stroke_color=GREY_B,
        ).set_fill(BLACK, opacity=0.6)

        expl_group = VGroup(box, expl)
        expl_group.next_to(graphs, DOWN, buff=0.35)
        expl.move_to(box.get_center()).shift(LEFT * (box.width / 2 - 0.4)).align_to(box, LEFT)

        self.play(FadeIn(expl_group, shift=UP * 0.1), run_time=0.8 * SPEED)

        # ---------- moving markers + readout (kept small, top-left) ----------
        marker_left = Dot(left_points[0], radius=0.055)
        marker_right = Dot(right_points[0], radius=0.055)

        readout = Text(
            "",
            font="DejaVu Sans Mono",
            font_size=22,
            color=WHITE,
        )
        readout.to_corner(UL, buff=0.35).shift(DOWN * 1.05)  # sits under title area

        self.play(FadeIn(marker_left), FadeIn(marker_right), FadeIn(readout), run_time=0.5 * SPEED)

        for i, b in enumerate(bits_list):
            new_readout = Text(
                f"b={b:>2}   density={densities[i]:.3f}   ({percent_of_baseline[i]:.1f}% baseline)",
                font="DejaVu Sans Mono",
                font_size=22,
                color=WHITE,
            ).move_to(readout)

            anims = [Transform(readout, new_readout)]
            if i > 0:
                anims += [
                    marker_left.animate.move_to(left_points[i]),
                    marker_right.animate.move_to(right_points[i]),
                ]

            self.play(*anims, run_time=0.22 * SPEED)

        # ---------- final punchline ----------
        punch = Text(
            "Conclusion: bucket prime density decays toward 0 as b grows.",
            font="DejaVu Serif",
            weight=BOLD,
            font_size=30,
            color=YELLOW,
        )
        punch.next_to(expl_group, DOWN, buff=0.25)

        self.play(FadeIn(punch, shift=UP * 0.1), run_time=0.7 * SPEED)
        self.wait(1.0 * SPEED)


def frange(a: float, b: float, step: float):
    x = a
    # include b (within tolerance)
    while x <= b + 1e-9:
        yield x
        x += step

