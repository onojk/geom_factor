# prime_bucket_density_plus_and_decay.py
#
# Manim Community v0.19.1
# Run:
#   python -m manim -pqh prime_bucket_density_plus_and_decay.py PrimeBucketDensityPlusAndDecay
#
# Fixes in THIS rewrite:
# - Fixes the Text() crash: NEVER passes weight=None into Text()
# - Table has NO border/panel (your request)
# - Table columns are hard-aligned (no drift / no “not aligned with boundary” issues)
# - Table auto-picks font size per bucket so it NEVER runs off the bottom
# - Graph slide: simplified (no “too much for 1 slide”), no overlap with y-label/subtitle
# - Graph is placed with safe bottom margin so axis never clips

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from manim import *

# ----------------------------
# Timing + style
# ----------------------------

SLOW = 3.0

FONT_SERIF = "DejaVu Serif"
FONT_MONO = "DejaVu Sans Mono"

COL = {
    "bg": "#000000",
    "text": "#F2F2F2",
    "sub": "#A8A8A8",
    "accent": "#F1E05A",
    "prime": "#7CFF7C",
    "comp": "#FF6B6B",
    "purple": "#C8A2FF",
}

SAFE_SIDE_MARGIN = 0.80
SAFE_TOP_MARGIN = 0.50
SAFE_BOTTOM_MARGIN = 0.65


def slow(t: float) -> float:
    return t * SLOW


# ----------------------------
# Text helpers (CRITICAL: never pass weight=None)
# ----------------------------

def safe_text(
    s: str,
    *,
    font: str = FONT_SERIF,
    font_size: int = 48,
    color: str = COL["text"],
    weight: str | None = None,
    max_width: float | None = None,
) -> Text:
    if not s.strip():
        s = " "
    if weight is None:
        t = Text(s, font=font, font_size=font_size, color=color)
    else:
        t = Text(s, font=font, font_size=font_size, color=color, weight=weight)

    if max_width is not None and t.width > max_width:
        t.scale_to_fit_width(max_width)
    return t


def safe_title(s: str, *, font_size: int = 76) -> Text:
    # Use weight=BOLD explicitly (never None)
    return safe_text(
        s,
        font=FONT_SERIF,
        font_size=font_size,
        color=COL["text"],
        weight=BOLD,
        max_width=config.frame_width - SAFE_SIDE_MARGIN,
    )


# ----------------------------
# Math helpers
# ----------------------------

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


def offset_s(n: int) -> int:
    return n - (1 << (n.bit_length() - 1))


def sieve_pi(limit: int) -> List[int]:
    if limit < 1:
        return [0] * (limit + 1)
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            start = p * p
            sieve[start : limit + 1 : p] = b"\x00" * (((limit - start) // p) + 1)
    pi = [0] * (limit + 1)
    c = 0
    for i in range(limit + 1):
        if sieve[i]:
            c += 1
        pi[i] = c
    return pi


@dataclass
class BucketInfo:
    k: int
    lo: int
    hi: int
    width: int
    primes: List[int]

    @property
    def prime_count(self) -> int:
        return len(self.primes)

    @property
    def density(self) -> float:
        return (self.prime_count / self.width) if self.width else 0.0


def bucket_for_k(k: int) -> BucketInfo:
    lo = 2 ** (k - 1)
    hi = 2**k - 1
    primes = [n for n in range(lo, hi + 1) if is_prime(n)]
    return BucketInfo(k=k, lo=lo, hi=hi, width=(hi - lo + 1), primes=primes)


# ----------------------------
# Layout helpers
# ----------------------------

def make_title_bar() -> VGroup:
    title = safe_title("PRIME NUMBER DISTRIBUTION", font_size=76)
    subtitle = safe_text(
        "bit buckets • bucket offsets • density decay",
        font=FONT_SERIF,
        font_size=32,
        color=COL["sub"],
        max_width=config.frame_width - SAFE_SIDE_MARGIN,
    )
    bar = VGroup(title, subtitle).arrange(DOWN, buff=0.16)
    bar.to_edge(UP, buff=SAFE_TOP_MARGIN)
    return bar


def content_anchor_below(title_bar: VGroup) -> np.ndarray:
    return title_bar.get_bottom() + DOWN * 0.35


def slide_clear(scene: Scene, *mobs: Mobject, keep: List[Mobject] | None = None, rt: float = 0.6):
    if keep is None:
        keep = []
    if mobs:
        scene.play(*[FadeOut(m) for m in mobs], run_time=slow(rt))
        return
    to_remove = [m for m in scene.mobjects if m not in keep]
    if to_remove:
        scene.play(*[FadeOut(m) for m in to_remove], run_time=slow(rt))


def fit_between_y(group: Mobject, *, y_top: float, y_bottom: float):
    """Scale down if needed to fit in vertical window, then center inside it."""
    max_h = max(0.1, y_top - y_bottom)
    if group.height > max_h:
        group.scale_to_fit_height(max_h)
    y_center = (y_top + y_bottom) / 2
    group.move_to(np.array([group.get_center()[0], y_center, 0.0]))


# ----------------------------
# Table builder (NO BORDER; hard-aligned columns; auto font sizes)
# ----------------------------

def build_bucket_table(info: BucketInfo) -> VGroup:
    """
    Returns a VGroup: header + rows.
    Columns are aligned by fixed X positions so nothing drifts.
    """
    row_count = info.width

    # Auto font sizing so the table never falls off-screen.
    # k=2 -> 2 rows (big), k=4 -> 8 rows (smaller), etc.
    if row_count <= 2:
        fs_hdr, fs_row = 44, 44
    elif row_count <= 8:
        fs_hdr, fs_row = 40, 40
    elif row_count <= 16:
        fs_hdr, fs_row = 34, 34
    else:
        fs_hdr, fs_row = 30, 30

    # Fixed column centers (scene units)
    # Keep table comfortably away from left edge.
    x_dec = -2.8
    x_bin = -0.4
    x_res = 2.6

    hdr_dec = safe_text("DEC", font=FONT_MONO, font_size=fs_hdr, color=COL["sub"])
    hdr_bin = safe_text("BINARY", font=FONT_MONO, font_size=fs_hdr, color=COL["sub"])
    hdr_res = safe_text("RESULT", font=FONT_MONO, font_size=fs_hdr, color=COL["sub"])

    hdr_dec.move_to([x_dec, 0, 0])
    hdr_bin.move_to([x_bin, 0, 0])
    hdr_res.move_to([x_res, 0, 0])

    header = VGroup(hdr_dec, hdr_bin, hdr_res)

    # divider line under header
    line = Line(start=[x_dec - 1.2, -0.45, 0], end=[x_res + 1.6, -0.45, 0], stroke_width=2, color=COL["sub"])
    line.set_opacity(0.35)

    rows = VGroup()
    for n in range(info.lo, info.hi + 1):
        b = format(n, "b")
        prime = is_prime(n)
        row_col = COL["prime"] if prime else COL["comp"]

        t_dec = safe_text(str(n), font=FONT_MONO, font_size=fs_row, color=row_col)
        t_bin = safe_text(b, font=FONT_MONO, font_size=fs_row, color=row_col)
        t_res = safe_text("PRIME" if prime else "COMPOSITE", font=FONT_MONO, font_size=fs_row, color=row_col)

        t_dec.move_to([x_dec, 0, 0])
        t_bin.move_to([x_bin, 0, 0])
        t_res.move_to([x_res, 0, 0])

        rows.add(VGroup(t_dec, t_bin, t_res))

    rows.arrange(DOWN, buff=0.18).next_to(line, DOWN, buff=0.18)

    table = VGroup(header, line, rows)
    return table


# ----------------------------
# Scene
# ----------------------------

class PrimeBucketDensityPlusAndDecay(Scene):
    def construct(self):
        self.camera.background_color = COL["bg"]

        title_bar = make_title_bar()
        self.play(FadeIn(title_bar, shift=UP * 0.12), run_time=slow(0.9))
        anchor = content_anchor_below(title_bar)

        # ------------------------------------------------------------
        # Slide 1: definition
        # ------------------------------------------------------------
        s1_a = safe_text(
            "A bit-bucket groups numbers with the same bit-length.",
            font_size=42,
            color=COL["text"],
            max_width=config.frame_width - SAFE_SIDE_MARGIN,
        ).move_to(anchor).to_edge(LEFT, buff=0.8)

        s1_b = safe_text(
            "All k-bit numbers live in one contiguous block:",
            font_size=34,
            color=COL["sub"],
            max_width=config.frame_width - SAFE_SIDE_MARGIN,
        ).next_to(s1_a, DOWN, buff=0.28).align_to(s1_a, LEFT)

        s1_c = MathTex(r"[2^{k-1},\;2^k-1]", color=COL["accent"]).scale(1.35)
        s1_c.next_to(s1_b, DOWN, buff=0.35).align_to(s1_a, LEFT)

        self.play(Write(s1_a), run_time=slow(0.9))
        self.play(Write(s1_b), run_time=slow(0.8))
        self.play(Write(s1_c), run_time=slow(0.9))
        self.wait(slow(0.6))

        # ------------------------------------------------------------
        # Helper: show a bucket slide cleanly
        # ------------------------------------------------------------
        def show_bucket(k: int) -> Tuple[Mobject, Mobject, Mobject, Mobject]:
            info = bucket_for_k(k)

            hdr = safe_text(
                f"{k}-BIT BUCKET",
                font_size=60,
                color=COL["text"],
                weight=BOLD,
                max_width=config.frame_width - SAFE_SIDE_MARGIN,
            ).move_to(anchor).to_edge(LEFT, buff=0.8)

            rng = safe_text(
                f"{info.lo} – {info.hi}   (width = {info.width})",
                font_size=36,
                color=COL["purple"],
                max_width=config.frame_width - SAFE_SIDE_MARGIN,
            ).next_to(hdr, DOWN, buff=0.18).align_to(hdr, LEFT)

            table = build_bucket_table(info)

            # Place table under range, fit safely to bottom
            table.next_to(rng, DOWN, buff=0.40).to_edge(LEFT, buff=0.8)

            y_top = rng.get_bottom()[1] - 0.25
            y_bottom = -config.frame_height / 2 + SAFE_BOTTOM_MARGIN + 0.15
            fit_between_y(table, y_top=y_top, y_bottom=y_bottom)

            dens = safe_text(
                f"{info.prime_count} primes out of {info.width}  →  density = {info.density:.3f}",
                font=FONT_SERIF,
                font_size=44 if info.width <= 8 else 40,
                color=COL["text"],
                max_width=config.frame_width - SAFE_SIDE_MARGIN,
            )
            dens.to_edge(DOWN, buff=SAFE_BOTTOM_MARGIN)

            self.play(Write(hdr), run_time=slow(0.8))
            self.play(Write(rng), run_time=slow(0.7))
            self.play(FadeIn(table), run_time=slow(0.9))
            self.wait(slow(0.35))

            # density line gets its own moment, but never overlaps table
            self.play(Write(dens), run_time=slow(0.9))
            self.wait(slow(0.55))

            return hdr, rng, table, dens

        # ------------------------------------------------------------
        # Slide 2: 2-bit bucket
        # ------------------------------------------------------------
        slide_clear(self, s1_a, s1_b, s1_c, keep=[title_bar], rt=0.7)
        h2, r2, t2, d2 = show_bucket(2)

        # ------------------------------------------------------------
        # Slide 3: 4-bit bucket
        # ------------------------------------------------------------
        slide_clear(self, h2, r2, t2, d2, keep=[title_bar], rt=0.7)
        h4, r4, t4, d4 = show_bucket(4)

        # ------------------------------------------------------------
        # Slide 4: bucket offset
        # ------------------------------------------------------------
        slide_clear(self, h4, r4, t4, d4, keep=[title_bar], rt=0.7)

        off_hdr = safe_text(
            "Bucket offset",
            font_size=60,
            color=COL["text"],
            weight=BOLD,
            max_width=config.frame_width - SAFE_SIDE_MARGIN,
        ).move_to(anchor).to_edge(LEFT, buff=0.8)

        off_eq = MathTex(r"s(n)=n-2^{\lfloor \log_2 n \rfloor}", color=COL["accent"]).scale(1.25)
        off_eq.next_to(off_hdr, DOWN, buff=0.35).align_to(off_hdr, LEFT)

        off_explain = safe_text(
            "Reset each bucket to start at 0, so positions are comparable across buckets.",
            font_size=34,
            color=COL["sub"],
            max_width=config.frame_width - SAFE_SIDE_MARGIN,
        ).next_to(off_eq, DOWN, buff=0.35).align_to(off_hdr, LEFT)

        a, b = 11, 13
        off_example = safe_text(
            f"Example: s({a})={offset_s(a)}   and   s({b})={offset_s(b)}",
            font=FONT_MONO,
            font_size=38,
            color=COL["text"],
            max_width=config.frame_width - SAFE_SIDE_MARGIN,
        ).next_to(off_explain, DOWN, buff=0.42).align_to(off_hdr, LEFT)

        self.play(Write(off_hdr), run_time=slow(0.8))
        self.play(Write(off_eq), run_time=slow(0.9))
        self.play(Write(off_explain), run_time=slow(0.9))
        self.play(Write(off_example), run_time=slow(0.9))
        self.wait(slow(0.8))

        # ------------------------------------------------------------
        # Slide 5: density decay graph (clean; no overload)
        # ------------------------------------------------------------
        slide_clear(self, off_hdr, off_eq, off_explain, off_example, keep=[title_bar], rt=0.7)

        g_hdr = safe_text(
            "Prime density decays as bit-length grows",
            font_size=58,
            color=COL["text"],
            weight=BOLD,
            max_width=config.frame_width - SAFE_SIDE_MARGIN,
        ).move_to(anchor).to_edge(LEFT, buff=0.8)

        g_sub = safe_text(
            "Density = (primes in bucket) / (bucket width)",
            font_size=28,
            color=COL["sub"],
            max_width=config.frame_width - SAFE_SIDE_MARGIN,
        ).next_to(g_hdr, DOWN, buff=0.22).align_to(g_hdr, LEFT)

        self.play(Write(g_hdr), run_time=slow(0.9))
        self.play(Write(g_sub), run_time=slow(0.8))

        KMAX = 18
        max_n = 2**KMAX - 1
        pi = sieve_pi(max_n)

        xs = list(range(2, KMAX + 1))
        ys = []
        for kk in xs:
            lo = 2 ** (kk - 1)
            hi = 2**kk - 1
            primes_in_bucket = pi[hi] - pi[lo - 1]
            width = hi - lo + 1
            ys.append(primes_in_bucket / width)

        y_max = max(ys) * 1.12

        ax = Axes(
            x_range=[2, KMAX, 2],
            y_range=[0, y_max, 0.2],
            x_length=11.2,
            y_length=5.2,
            tips=False,
            axis_config={"include_numbers": True, "font_size": 26, "color": COL["sub"]},
        )

        ylab = safe_text("prime density", font_size=24, color=COL["sub"])
        xlab = safe_text("bit-length k", font_size=24, color=COL["sub"])

        # Group the graph objects so we can fit safely between subtitle and bottom
        dots = VGroup(*[Dot(ax.c2p(x, y), radius=0.055, color=COL["accent"]) for x, y in zip(xs, ys)])

        curve = VMobject(stroke_width=4, color=COL["accent"])
        curve.set_points_smoothly([ax.c2p(x, y) for x, y in zip(xs, ys)])

        ref = VMobject(stroke_width=3, color=COL["purple"])
        ref_pts = [ax.c2p(kk, 1.0 / (kk * math.log(2.0))) for kk in xs]
        ref.set_points_smoothly(ref_pts)

        graph = VGroup(ax, dots, curve, ref)

        # Place graph under subtitle with a safe bottom margin (prevents cut-off)
        graph.next_to(g_sub, DOWN, buff=0.55).to_edge(LEFT, buff=0.8)

        # Fit to safe window
        y_top = g_sub.get_bottom()[1] - 0.25
        y_bottom = -config.frame_height / 2 + SAFE_BOTTOM_MARGIN
        fit_between_y(graph, y_top=y_top, y_bottom=y_bottom)

        # Labels attached AFTER fit (so they won't overlap)
        ylab.next_to(ax, LEFT, buff=0.25).rotate(PI / 2)
        xlab.next_to(ax, DOWN, buff=0.20)

        self.play(Create(ax), run_time=slow(1.1))
        self.play(FadeIn(xlab), FadeIn(ylab), run_time=slow(0.6))
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.07), run_time=slow(0.9))
        self.play(Create(curve), run_time=slow(0.9))
        self.play(Create(ref), run_time=slow(0.9))

        # Tiny legend in corner (no giant label over the plot)
        legend = safe_text("purple: ~ 1/(k ln 2)", font_size=22, color=COL["purple"])
        legend.to_corner(DR, buff=0.60)

        self.play(FadeIn(legend), run_time=slow(0.6))
        self.wait(slow(1.0))

        # ------------------------------------------------------------
        # Slide 6: End card
        # ------------------------------------------------------------
        slide_clear(
            self,
            g_hdr, g_sub,
            ax, xlab, ylab, dots, curve, ref, legend,
            keep=[title_bar],
            rt=0.8,
        )

        thanks = safe_text(
            "Thanks For Watching!",
            font_size=90,
            color=COL["text"],
            weight=BOLD,
            max_width=config.frame_width - SAFE_SIDE_MARGIN,
        )
        sig = safe_text(
            "- ONOJK123",
            font_size=64,
            color=COL["accent"],
            max_width=config.frame_width - SAFE_SIDE_MARGIN,
        )
        end_grp = VGroup(thanks, sig).arrange(DOWN, buff=0.30).move_to(ORIGIN)
        if end_grp.width > (config.frame_width - SAFE_SIDE_MARGIN):
            end_grp.scale_to_fit_width(config.frame_width - SAFE_SIDE_MARGIN)

        self.play(Write(thanks), run_time=slow(0.9))
        self.play(Write(sig), run_time=slow(0.8))
        self.wait(slow(1.2))

        self.play(FadeOut(title_bar), run_time=slow(0.6))
        self.wait(slow(0.3))
