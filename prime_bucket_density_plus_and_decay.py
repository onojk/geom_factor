# prime_bucket_density_plus_and_decay.py
#
# Manim Community v0.19.x  (tested for compatibility with 0.19.1)
#
# Run:
#   python -m manim -pqh prime_bucket_density_plus_and_decay.py PrimeBucketDensityPlusAndDecay
#
# Notes:
# - This version is SLOWER (global SPEED factor).
# - Fixes right-edge clipping of the asymptote label (hard clamp).
# - Fixes overlaps by using a consistent content-safe layout region and column layout.
# - Avoids bounding_box attribute issues by using get_left/right/top/bottom instead.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from manim import (
    Scene,
    VGroup,
    Group,
    Mobject,
    VMobject,
    Text,
    MathTex,
    Line,
    Rectangle,
    Axes,
    Dot,
    FadeIn,
    FadeOut,
    Transform,
    Write,
    Create,
    ReplacementTransform,
    ORIGIN,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    WHITE,
    BLACK,
    GREY_A,
    GREY_B,
    GREY_C,
    GREY_D,
    BLUE_B,
    PURPLE_B,
    YELLOW_B,
    GREEN_B,
    RED_B,
)


# ----------------------------
# Global look & timing
# ----------------------------
SERIF = "Noto Serif"  # available on most Linux installs; avoids Times New Roman warning
TITLE_COL = GREY_A
SUBTITLE_COL = GREY_C
RULE_COL = GREY_D
BODY_COL = GREY_B
ACCENT = PURPLE_B
ACCENT2 = YELLOW_B

# Slow-down factor: multiply all run-times and waits by this
SPEED = 3.00  # increase to slow more, e.g. 2.0


# Safe margins (in Manim units)
SAFE_LEFT = 0.60
SAFE_RIGHT = 0.60
SAFE_TOP = 0.45
SAFE_BOTTOM = 0.55


def rt(t: float) -> float:
    """Scale run_time by SPEED."""
    return max(0.01, t * SPEED)


def wt(t: float) -> float:
    """Scale wait time by SPEED."""
    return max(0.01, t * SPEED)


def wipe_keep(scene: Scene, keep: List[Mobject]) -> None:
    """Fade out everything except keep."""
    current = list(scene.mobjects)
    to_remove = [m for m in current if (m not in keep)]
    if to_remove:
        scene.play(FadeOut(Group(*to_remove)), run_time=rt(0.6))


def clamp_to_frame(mob: Mobject, left=SAFE_LEFT, right=SAFE_RIGHT, top=SAFE_TOP, bottom=SAFE_BOTTOM) -> None:
    """Shift mob so it stays inside the frame with margins."""
    # Frame bounds in Manim coordinates
    xL = -7.0 + left
    xR = 7.0 - right
    yT = 4.0 - top
    yB = -4.0 + bottom

    dx = 0.0
    dy = 0.0

    if mob.get_left()[0] < xL:
        dx += (xL - mob.get_left()[0])
    if mob.get_right()[0] > xR:
        dx -= (mob.get_right()[0] - xR)
    if mob.get_top()[1] > yT:
        dy -= (mob.get_top()[1] - yT)
    if mob.get_bottom()[1] < yB:
        dy += (yB - mob.get_bottom()[1])

    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
        mob.shift(dx * RIGHT + dy * UP)


def fit_width(mob: Mobject, max_width: float) -> None:
    if mob.width > max_width:
        mob.scale(max_width / mob.width)


def fit_height(mob: Mobject, max_height: float) -> None:
    if mob.height > max_height:
        mob.scale(max_height / mob.height)


# ----------------------------
# Prime utilities
# ----------------------------
def sieve_upto(n: int) -> List[bool]:
    """Return is_prime[0..n]."""
    if n < 1:
        return [False] * (n + 1)
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    p = 2
    while p * p <= n:
        if is_prime[p]:
            step = p
            start = p * p
            is_prime[start : n + 1 : step] = [False] * (((n - start) // step) + 1)
        p += 1
    return is_prime


def bucket_range(k: int) -> Tuple[int, int]:
    """k-bit bucket is [2^(k-1), 2^k - 1]."""
    lo = 1 << (k - 1)
    hi = (1 << k) - 1
    return lo, hi


def bucket_density(k: int, is_prime: List[bool]) -> float:
    lo, hi = bucket_range(k)
    width = hi - lo + 1
    primes = sum(1 for n in range(lo, hi + 1) if is_prime[n])
    return primes / width


# ----------------------------
# Title bar
# ----------------------------
def make_title_bar() -> VGroup:
    title = Text("PRIME NUMBER DISTRIBUTION", font=SERIF, weight="BOLD", color=TITLE_COL).scale(1.30)
    subtitle = Text("bit buckets • bucket offsets • density decay", font=SERIF, color=SUBTITLE_COL).scale(0.68)
    bar = VGroup(title, subtitle).arrange(DOWN, buff=0.12)
    bar.to_edge(UP, buff=0.20)
    fit_width(bar, 13.0)
    clamp_to_frame(bar)
    return bar


def make_title_rule(bar: VGroup) -> Line:
    y = bar.get_bottom()[1] - 0.20
    rule = Line(LEFT * 6.8, RIGHT * 6.8, color=RULE_COL, stroke_width=3).shift(UP * (y - 0.0))
    clamp_to_frame(rule)
    return rule


def content_top_y(rule: Line) -> float:
    return rule.get_bottom()[1] - 0.25


# ----------------------------
# Layout helpers
# ----------------------------
@dataclass
class ContentRegion:
    xL: float
    xR: float
    yT: float
    yB: float

    @property
    def width(self) -> float:
        return self.xR - self.xL

    @property
    def height(self) -> float:
        return self.yT - self.yB


def get_content_region(rule: Line) -> ContentRegion:
    xL = -7.0 + SAFE_LEFT
    xR = 7.0 - SAFE_RIGHT
    yT = content_top_y(rule)
    yB = -4.0 + SAFE_BOTTOM
    return ContentRegion(xL=xL, xR=xR, yT=yT, yB=yB)


def place_under_rule(mob: Mobject, rule: Line, align="left", y_buff=0.0) -> None:
    reg = get_content_region(rule)
    # Put mob top to region top
    mob.move_to(UP * (reg.yT - mob.height / 2 - y_buff))
    if align == "left":
        mob.shift(RIGHT * (reg.xL - mob.get_left()[0]))
    elif align == "center":
        mob.shift(RIGHT * ((reg.xL + reg.xR) / 2 - mob.get_center()[0]))
    elif align == "right":
        mob.shift(RIGHT * (reg.xR - mob.get_right()[0]))
    clamp_to_frame(mob)


def make_step_header(step_text: str, rule: Line) -> Text:
    h = Text(step_text, font=SERIF, weight="BOLD", color=TITLE_COL).scale(1.05)
    place_under_rule(h, rule, align="left", y_buff=0.10)
    return h


# ----------------------------
# Slide builders
# ----------------------------
def slide_intro(scene: Scene, bar: VGroup, rule: Line) -> None:
    wipe_keep(scene, keep=[bar, rule])

    h = Text("Idea", font=SERIF, weight="BOLD", color=TITLE_COL).scale(1.10)
    p1 = Text('Group integers by bit-length ("bit buckets").', font=SERIF, color=BODY_COL).scale(0.72)
    p2 = Text("Then measure how prime density changes as buckets grow.", font=SERIF, color=BODY_COL).scale(0.72)

    block = VGroup(h, p1, p2).arrange(DOWN, aligned_edge=LEFT, buff=0.30)
    place_under_rule(block, rule, align="left", y_buff=0.15)

    scene.play(FadeIn(block), run_time=rt(1.0))
    scene.wait(wt(1.2))


def slide_define_bucket(scene: Scene, bar: VGroup, rule: Line) -> None:
    wipe_keep(scene, keep=[bar, rule])

    h = make_step_header("1) Bit buckets", rule)

    t1 = Text("The k-bit bucket is the integers that use exactly k bits:", font=SERIF, color=BODY_COL).scale(0.70)
    eq = MathTex(r"[2^{k-1},\;2^k-1]").scale(1.05).set_color(ACCENT)
    t2 = Text("Bucket width grows exponentially:", font=SERIF, color=BODY_COL).scale(0.70)
    eq2 = MathTex(r"\text{width}=2^{k-1}").scale(0.95).set_color(ACCENT2)

    block = VGroup(t1, eq, t2, eq2).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
    # Put under header
    block.next_to(h, DOWN, aligned_edge=LEFT, buff=0.38)
    clamp_to_frame(block)

    scene.play(Write(h), run_time=rt(0.9))
    scene.play(FadeIn(block), run_time=rt(1.0))
    scene.wait(wt(1.4))


def make_bucket_table(k: int, is_prime: List[bool], rule: Line) -> VGroup:
    lo, hi = bucket_range(k)
    width = hi - lo + 1

    title = Text(f"{k}-BIT BUCKET", font=SERIF, weight="BOLD", color=TITLE_COL).scale(0.92)
    sub = Text(f"{lo} – {hi}   (width = {width})", font=SERIF, color=ACCENT).scale(0.80)

    # columns
    col1 = Text("DEC", font=SERIF, color=SUBTITLE_COL).scale(0.70)
    col2 = Text("BINARY", font=SERIF, color=SUBTITLE_COL).scale(0.70)
    col3 = Text("RESULT", font=SERIF, color=SUBTITLE_COL).scale(0.70)

    header = VGroup(col1, col2, col3).arrange(RIGHT, buff=1.25, aligned_edge=DOWN)
    header_rule = Line(LEFT, RIGHT, color=RULE_COL, stroke_width=2)
    header_rule.set_width(header.width)
    header_rule.next_to(header, DOWN, buff=0.15)

    rows = VGroup()
    for n in range(lo, hi + 1):
        dec = Text(str(n), font=SERIF, color=BODY_COL).scale(0.72)
        bin_s = format(n, "b")
        bin_t = Text(bin_s, font="Noto Sans Mono", color=BODY_COL).scale(0.78)

        if is_prime[n]:
            res = Text("PRIME", font="Noto Sans Mono", color=GREEN_B).scale(0.78)
        else:
            res = Text("COMPOSITE", font="Noto Sans Mono", color=RED_B).scale(0.78)

        row = VGroup(dec, bin_t, res).arrange(RIGHT, buff=1.25, aligned_edge=DOWN)
        row.align_to(header, LEFT)
        rows.add(row)

    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
    rows.next_to(header_rule, DOWN, aligned_edge=LEFT, buff=0.22)

    primes = sum(1 for n in range(lo, hi + 1) if is_prime[n])
    dens = primes / width
    foot = Text(f"{primes} primes out of {width}  →  density = {dens:.3f}", font=SERIF, color=TITLE_COL).scale(0.74)
    foot.next_to(rows, DOWN, aligned_edge=LEFT, buff=0.30)

    table = VGroup(title, sub, header, header_rule, rows, foot).arrange(DOWN, aligned_edge=LEFT, buff=0.22)

    # Ensure it fits inside content region
    reg = get_content_region(rule)
    max_w = reg.width * 0.55  # used in multi-table slide
    fit_width(table, max_w)

    return table


def slide_bucket_examples(scene: Scene, bar: VGroup, rule: Line, is_prime: List[bool]) -> None:
    wipe_keep(scene, keep=[bar, rule])

    h = make_step_header("2) Small buckets (examples)", rule)
    hint = Text("Watch density drop as bucket width grows.", font=SERIF, color=BODY_COL).scale(0.70)
    hint.next_to(h, DOWN, aligned_edge=LEFT, buff=0.25)
    clamp_to_frame(hint)

    t3 = make_bucket_table(3, is_prime, rule)
    t4 = make_bucket_table(4, is_prime, rule)

    # Two-column layout under hint: left table k=3, right table k=4
    reg = get_content_region(rule)
    y_start = hint.get_bottom()[1] - 0.28

    # Position left
    t3.move_to(UP * (y_start - t3.height / 2))
    t3.shift(RIGHT * (reg.xL - t3.get_left()[0]))

    # Position right
    t4.move_to(UP * (y_start - t4.height / 2))
    t4.shift(RIGHT * (reg.xR - t4.get_right()[0]))

    # If too tall, scale both a bit
    max_h = (y_start - reg.yB) * 0.98
    if max(t3.height, t4.height) > max_h:
        scale = max_h / max(t3.height, t4.height)
        t3.scale(scale)
        t4.scale(scale)
        # re-place after scaling
        t3.move_to(UP * (y_start - t3.height / 2))
        t3.shift(RIGHT * (reg.xL - t3.get_left()[0]))
        t4.move_to(UP * (y_start - t4.height / 2))
        t4.shift(RIGHT * (reg.xR - t4.get_right()[0]))

    clamp_to_frame(t3)
    clamp_to_frame(t4)

    scene.play(Write(h), run_time=rt(0.9))
    scene.play(FadeIn(hint), run_time=rt(0.7))
    scene.play(FadeIn(t3), FadeIn(t4), run_time=rt(1.2))
    scene.wait(wt(1.8))


def slide_offsets(scene: Scene, bar: VGroup, rule: Line) -> None:
    wipe_keep(scene, keep=[bar, rule])

    h = make_step_header("3) Bucket offsets", rule)
    p1 = Text("Inside a bucket, measure position from the start:", font=SERIF, color=BODY_COL).scale(0.70)
    eq = MathTex(r"\mathrm{offset}(n)=n-2^{k-1}").scale(1.05).set_color(ACCENT)
    p2 = Text("Offsets let you compare patterns across different k.", font=SERIF, color=BODY_COL).scale(0.70)

    block = VGroup(p1, eq, p2).arrange(DOWN, aligned_edge=LEFT, buff=0.30)
    block.next_to(h, DOWN, aligned_edge=LEFT, buff=0.40)

    # Clamp so equation never runs off-screen
    fit_width(block, 12.8)
    clamp_to_frame(block)

    scene.play(Write(h), run_time=rt(0.9))
    scene.play(FadeIn(block), run_time=rt(1.1))
    scene.wait(wt(1.6))


def slide_density_definition(scene: Scene, bar: VGroup, rule: Line) -> None:
    wipe_keep(scene, keep=[bar, rule])

    h = make_step_header("4) Bucket prime density", rule)
    p1 = Text("For each k-bit bucket, compute:", font=SERIF, color=BODY_COL).scale(0.70)
    eq = MathTex(
        r"\mathrm{density}(k)=\frac{\#\{\text{primes in bucket}\}}{\#\{\text{numbers in bucket}\}}"
    ).scale(0.92).set_color(ACCENT)
    p2 = Text("That gives one density value per bucket (one dot per k).", font=SERIF, color=BODY_COL).scale(0.70)

    block = VGroup(p1, eq, p2).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
    block.next_to(h, DOWN, aligned_edge=LEFT, buff=0.40)

    fit_width(block, 12.8)
    clamp_to_frame(block)

    scene.play(Write(h), run_time=rt(0.9))
    scene.play(FadeIn(block), run_time=rt(1.1))
    scene.wait(wt(1.6))


def build_density_plot(rule: Line, densities: List[float], k_min: int, k_max: int) -> Tuple[VGroup, MathTex]:
    reg = get_content_region(rule)

    # Plot area (left 70% width)
    plot_w = reg.width * 0.74
    plot_h = reg.height * 0.58

    # y-range based on data
    y_max = max(densities) if densities else 0.6
    y_max = max(0.55, min(0.95, y_max * 1.15))

    axes = Axes(
        x_range=[k_min, k_max, 2],
        y_range=[0, y_max, y_max / 6],
        x_length=plot_w,
        y_length=plot_h,
        axis_config={"stroke_color": GREY_A, "stroke_width": 2},
        tips=False,
    )

    # Put axes bottom-left inside region
    axes.to_corner(DOWN + LEFT)
    axes.shift(RIGHT * (reg.xL - axes.get_left()[0]))
    axes.shift(UP * (reg.yB - axes.get_bottom()[1]))
    axes.shift(UP * 0.25)  # lift off bottom safe margin a touch
    clamp_to_frame(axes)

    # Labels
    xlab = Text("bit-length k", font=SERIF, color=SUBTITLE_COL).scale(0.72)
    ylab = Text("prime density", font=SERIF, color=SUBTITLE_COL).scale(0.72).rotate(math.pi / 2)

    xlab.next_to(axes, DOWN, buff=0.25)
    ylab.next_to(axes, LEFT, buff=0.25)

    clamp_to_frame(xlab)
    clamp_to_frame(ylab)

    # Data points
    ks = list(range(k_min, k_max + 1))
    pts = [axes.c2p(k, densities[k - k_min]) for k in ks]

    dots = VGroup(*[Dot(p, radius=0.055, color=ACCENT2) for p in pts])

    # Smooth-ish polyline (Manim: set points as corners)
    empirical = VMobject(color=PURPLE_B, stroke_width=4)
    empirical.set_points_as_corners(pts)

    # Asymptote curve: ~ 1/(k ln 2)
    def approx_density(k: float) -> float:
        return 1.0 / (k * math.log(2.0))

    approx_pts = [axes.c2p(k, approx_density(k)) for k in ks]
    approx_curve = VMobject(color=ACCENT2, stroke_width=3)
    approx_curve.set_points_as_corners(approx_pts)

    # Label for approx curve (RIGHT side, clamped)
    asymp_label = MathTex(r"\approx \frac{1}{k\ln 2}").scale(1.05).set_color(ACCENT2)
    # Place near right edge, mid-height of plot
    asymp_label.move_to(RIGHT * (reg.xR - 1.2) + UP * (axes.get_center()[1] + 0.7))
    # Hard clamp to prevent clipping
    clamp_to_frame(asymp_label)

    plot = VGroup(axes, xlab, ylab, approx_curve, empirical, dots)
    return plot, asymp_label


def slide_density_plot(scene: Scene, bar: VGroup, rule: Line, is_prime: List[bool], k_min=2, k_max=18) -> None:
    wipe_keep(scene, keep=[bar, rule])

    h = make_step_header("5) Density decay across buckets", rule)
    sub = Text(
        f"Each dot is density(k) in bucket [2^(k−1), 2^k−1], for k = {k_min}…{k_max}.",
        font=SERIF,
        color=BODY_COL,
    ).scale(0.66)
    sub.next_to(h, DOWN, aligned_edge=LEFT, buff=0.25)
    clamp_to_frame(sub)

    densities = [bucket_density(k, is_prime) for k in range(k_min, k_max + 1)]
    plot, asymp_label = build_density_plot(rule, densities, k_min, k_max)

    # Ensure header/sub don't collide with plot
    reg = get_content_region(rule)
    y_start = sub.get_bottom()[1] - 0.25
    if plot.get_top()[1] > y_start:
        plot.shift(DOWN * (plot.get_top()[1] - y_start + 0.10))
        clamp_to_frame(plot)

    scene.play(Write(h), run_time=rt(0.9))
    scene.play(FadeIn(sub), run_time=rt(0.8))
    scene.play(Create(plot), run_time=rt(1.6))
    scene.play(FadeIn(asymp_label), run_time=rt(1.0))
    scene.wait(wt(1.8))


def slide_takeaway(scene: Scene, bar: VGroup, rule: Line, is_prime: List[bool], k_min=2, k_max=18) -> None:
    wipe_keep(scene, keep=[bar, rule])

    # Rebuild plot (so this slide is self-contained)
    densities = [bucket_density(k, is_prime) for k in range(k_min, k_max + 1)]
    plot, asymp_label = build_density_plot(rule, densities, k_min, k_max)

    # Plot column (left)
    plot.scale(0.95)
    plot.to_edge(LEFT, buff=0.70)
    plot.shift(DOWN * 0.25)
    clamp_to_frame(plot)

    # Takeaway column (right-center)
    take_title = Text("Takeaway", font=SERIF, weight="BOLD", color=TITLE_COL).scale(1.05)
    line1 = Text("Buckets widen exponentially.", font=SERIF, color=TITLE_COL).scale(0.82)
    line2 = Text("Primes thin roughly like 1/ln(n).", font=SERIF, color=TITLE_COL).scale(0.82)
    line3 = Text("So prime density decays with bit-length.", font=SERIF, color=TITLE_COL).scale(0.82)

    take_lines = VGroup(line1, line2, line3).arrange(DOWN, aligned_edge=LEFT, buff=0.30)
    take = VGroup(take_title, take_lines).arrange(DOWN, aligned_edge=LEFT, buff=0.40)

    take.next_to(plot, RIGHT, buff=0.85)
    take.align_to(plot, UP)
    take.shift(DOWN * 0.10)
    fit_width(take, 5.8)
    clamp_to_frame(take)

    # Asymptote label far right (clamped)
    asymp_label.to_edge(RIGHT, buff=0.60)
    asymp_label.shift(UP * 0.45)
    clamp_to_frame(asymp_label)

    scene.play(FadeIn(plot), run_time=rt(1.0))
    scene.play(FadeIn(take), run_time=rt(1.2))
    scene.play(FadeIn(asymp_label), run_time=rt(0.9))
    scene.wait(wt(2.0))


# ----------------------------
# Scene
# ----------------------------
class PrimeBucketDensityPlusAndDecay(Scene):
    def construct(self):
        # Precompute primes up to max bucket end
        k_min, k_max = 2, 18
        _, max_n = bucket_range(k_max)
        is_prime = sieve_upto(max_n)

        # Persistent title bar + rule
        bar = make_title_bar()
        rule = make_title_rule(bar)
        self.add(bar, rule)

        # Slides
        slide_intro(self, bar, rule)
        slide_define_bucket(self, bar, rule)
        slide_bucket_examples(self, bar, rule, is_prime)
        slide_offsets(self, bar, rule)
        slide_density_definition(self, bar, rule)
        slide_density_plot(self, bar, rule, is_prime, k_min=k_min, k_max=k_max)
        slide_takeaway(self, bar, rule, is_prime, k_min=k_min, k_max=k_max)

        # End hold (slower)
        self.wait(wt(1.6))
