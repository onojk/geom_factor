# prime_bucket_density_plus_and_decay.py
#
# Manim Community v0.19.x (v0.19.1 compatible)
# Run:
#   python -m manim -pqh prime_bucket_density_plus_and_decay.py PrimeBucketDensityPlusAndDecay
#
# If you previously got "InvalidDataError ... partial_movie_file_list.txt", nuke old partials:
#   rm -rf media/videos/prime_bucket_density_plus_and_decay/1080p60/partial_movie_files/PrimeBucketDensityPlusAndDecay
#
# If you still see combine issues, try:
#   python -m manim -pqh --disable_caching prime_bucket_density_plus_and_decay.py PrimeBucketDensityPlusAndDecay

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from manim import *


# ==================== MATH ====================

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


def first_k_primes(k: int) -> List[int]:
    out: List[int] = []
    n = 2
    while len(out) < k:
        if is_prime(n):
            out.append(n)
        n += 1
    return out


def bit_width(n: int) -> int:
    return max(1, n.bit_length())


def bucket_start(n: int) -> int:
    return 1 << (bit_width(n) - 1)


def bucket_start_for_bits(bits: int) -> int:
    return 1 << (bits - 1)


def bucket_end_for_bits(bits: int) -> int:
    return (1 << bits) - 1


def s_offset_in_bucket(n: int) -> int:
    return n - bucket_start(n)


def bin_str(n: int, width: int) -> str:
    return format(n, "b").zfill(width)


@dataclass
class BucketStats:
    bits: int
    start: int
    end: int
    size: int
    primes: int
    density: float


def bucket_stats(bits: int) -> BucketStats:
    start = bucket_start_for_bits(bits)
    end = bucket_end_for_bits(bits)
    size = end - start + 1
    primes = sum(1 for n in range(start, end + 1) if is_prime(n))
    dens = primes / size
    return BucketStats(bits=bits, start=start, end=end, size=size, primes=primes, density=dens)


# ==================== STYLE ====================

COL = {
    "bg": BLACK,
    "text": WHITE,
    "sub": GREY_B,
    "prime": GREEN_C,
    "comp": RED_C,
    "hi": YELLOW,
    "accent": PURPLE_B,
    "line": GREY_B,
    "dot": BLUE_B,
    "panel": GREY_D,
}

FONT = {"serif": "DejaVu Serif", "mono": "DejaVu Sans Mono"}


def make_title(text: str) -> Text:
    return Text(text, font=FONT["serif"], weight=BOLD, font_size=56, color=COL["text"]).set_opacity(0.88)


def make_subtitle(text: str) -> Text:
    return Text(text, font=FONT["serif"], font_size=30, color=COL["sub"]).set_opacity(0.92)


def scale_down_to_fit(mob: Mobject, max_w: float, max_h: float) -> None:
    if mob.width <= 0 or mob.height <= 0:
        return
    s = min(max_w / mob.width, max_h / mob.height, 1.0)
    mob.scale(s)


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
        dy += (bot_lim - mob.get_bottom()[1])
    if mob.get_top()[1] > top_lim:
        dy -= (mob.get_top()[1] - top_lim)

    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
        mob.shift(np.array([dx, dy, 0.0]))


# ==================== UI ====================

def make_bucket_table(bits: int) -> VGroup:
    start = bucket_start_for_bits(bits)
    end = bucket_end_for_bits(bits)

    hdr = VGroup(
        Text("DEC", font=FONT["mono"], font_size=26, color=COL["sub"]),
        Text("BINARY", font=FONT["mono"], font_size=26, color=COL["sub"]),
        Text("RESULT", font=FONT["mono"], font_size=26, color=COL["sub"]),
    ).arrange(RIGHT, buff=1.2, aligned_edge=DOWN)

    rows: List[VGroup] = []
    pad = len(str(end))
    for n in range(start, end + 1):
        p = is_prime(n)
        c = COL["prime"] if p else COL["comp"]
        rows.append(
            VGroup(
                Text(f"{n}".rjust(pad), font=FONT["mono"], font_size=22, color=c),
                Text(bin_str(n, bits), font=FONT["mono"], font_size=22, color=c),
                Text("PRIME" if p else "COMPOSITE", font=FONT["mono"], font_size=22, color=c, weight=BOLD if p else NORMAL),
            ).arrange(RIGHT, buff=1.2, aligned_edge=DOWN)
        )

    body = VGroup(*rows).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
    pack = VGroup(hdr, body).arrange(DOWN, buff=0.22, aligned_edge=LEFT)

    frame = RoundedRectangle(
        corner_radius=0.25,
        width=pack.width + 0.6,
        height=pack.height + 0.5,
        stroke_width=2,
        stroke_color=COL["panel"],
    ).set_fill(BLACK, opacity=0.45)

    return VGroup(frame, pack).center()


def make_readout(stats: List[BucketStats], tracker: ValueTracker) -> Text:
    n = len(stats)
    idx = int(round(tracker.get_value()))
    idx = max(0, min(n - 1, idx))
    b = stats[idx].bits
    if idx == 0:
        val = 0.0
    else:
        prev = stats[idx - 1].density
        cur = stats[idx].density
        val = 0.0 if prev == 0 else 100.0 * (cur - prev) / prev

    sign = "+" if val >= 0 else "−"
    txt = f"b={b:>2}   Δ%={sign}{abs(val):5.2f}"
    return Text(txt, font=FONT["mono"], font_size=30, color=WHITE)


# ==================== SCENE ====================

class PrimeBucketDensityPlusAndDecay(Scene):
    def construct(self):
        self.camera.background_color = COL["bg"]
        SPEED = 1.0

        # ------------------ Intro ------------------
        title = make_title("PRIME NUMBER DISTRIBUTION").to_edge(UP, buff=0.55)
        sub = make_subtitle("Bit buckets • bucket offsets • density decay").next_to(title, DOWN, buff=0.20)

        self.play(FadeIn(title, shift=DOWN * 0.12), FadeIn(sub, shift=DOWN * 0.12), run_time=1.1 * SPEED)
        self.wait(0.35 * SPEED)

        rule = Text(
            "Bucket = numbers with the same bit-length",
            font=FONT["serif"],
            font_size=34,
            color=COL["hi"],
        ).to_edge(DOWN, buff=0.65)

        self.play(FadeIn(rule, shift=UP * 0.10), run_time=0.8 * SPEED)
        self.wait(0.45 * SPEED)

        # ------------------ Part A: a few example bucket tables ------------------
        example_bits = [2, 4, 8]

        # Build FIRST header/caption as real Text (not empty)
        first_stats = bucket_stats(example_bits[0])

        bucket_hdr = Text(
            f"{example_bits[0]}-BIT BUCKET",
            font=FONT["serif"],
            font_size=44,
            color=COL["text"],
            weight=BOLD,
        ).to_corner(UR, buff=0.55)

        bucket_cap = Text(
            f"{first_stats.start:,} – {first_stats.end:,}   (width={example_bits[0]} bits)",
            font=FONT["serif"],
            font_size=28,
            color=COL["accent"],
        ).next_to(bucket_hdr, DOWN, buff=0.15).align_to(bucket_hdr, RIGHT)

        self.add(bucket_hdr, bucket_cap)

        table_group: Optional[VGroup] = None

        for i, b in enumerate(example_bits):
            st = bucket_stats(b)

            new_hdr = Text(
                f"{b}-BIT BUCKET",
                font=FONT["serif"],
                font_size=44,
                color=COL["text"],
                weight=BOLD,
            ).to_corner(UR, buff=0.55)

            new_cap = Text(
                f"{st.start:,} – {st.end:,}   (width={b} bits)",
                font=FONT["serif"],
                font_size=28,
                color=COL["accent"],
            ).next_to(new_hdr, DOWN, buff=0.15).align_to(new_hdr, RIGHT)

            table = make_bucket_table(b)
            max_w = config.frame_width - 1.2
            max_h = config.frame_height - 3.6
            scale_down_to_fit(table, max_w, max_h)
            table.move_to(ORIGIN + DOWN * 0.30)

            metrics = VGroup(
                Text(
                    f"{st.primes} primes out of {st.size}  →  density = {st.density:.3f}",
                    font=FONT["serif"],
                    font_size=34,
                    color=COL["text"],
                ),
                Text(
                    "Inside a bucket: binary width is constant.",
                    font=FONT["serif"],
                    font_size=28,
                    color=COL["sub"],
                ),
            ).arrange(DOWN, buff=0.18, aligned_edge=LEFT).to_edge(LEFT, buff=0.65).shift(DOWN * 0.25)

            if i == 0:
                table_group = VGroup(table, metrics)
                self.play(
                    Transform(bucket_hdr, new_hdr),
                    Transform(bucket_cap, new_cap),
                    FadeIn(table),
                    FadeIn(metrics),
                    run_time=1.0 * SPEED,
                )
            else:
                assert table_group is not None
                old_table, old_metrics = table_group[0], table_group[1]
                self.play(
                    Transform(bucket_hdr, new_hdr),
                    Transform(bucket_cap, new_cap),
                    FadeTransform(old_table, table),
                    FadeTransform(old_metrics, metrics),
                    run_time=1.0 * SPEED,
                )
                table_group = VGroup(table, metrics)

            self.wait(0.5 * SPEED)

        self.play(FadeOut(table_group), FadeOut(bucket_hdr), FadeOut(bucket_cap), FadeOut(rule), run_time=0.7 * SPEED)

        # ------------------ Part B: bucket offset s(n) ------------------
        btitle = Text("Bucket Offset (\"digits sans space\")", font=FONT["serif"], font_size=44, color=COL["text"], weight=BOLD)
        btitle.to_edge(UP, buff=0.55)

        eq = MathTex(r"s(n)=n-2^{\lfloor \log_2(n)\rfloor}", color=COL["hi"]).scale(1.15).next_to(btitle, DOWN, buff=0.25)
        expl = Text(
            "Distance from n to the start of its power-of-two bucket.",
            font=FONT["serif"],
            font_size=28,
            color=COL["sub"],
        ).next_to(eq, DOWN, buff=0.18)

        self.play(FadeIn(btitle, shift=DOWN * 0.12), FadeIn(eq, shift=DOWN * 0.12), FadeIn(expl, shift=DOWN * 0.12), run_time=1.0 * SPEED)
        self.wait(0.4 * SPEED)

        k = 250
        ps = first_k_primes(k)
        ys = [s_offset_in_bucket(p) for p in ps]

        boundaries_idx: List[int] = []
        last_bucket = bucket_start(ps[0])
        for idx, p in enumerate(ps):
            bs = bucket_start(p)
            if bs != last_bucket:
                boundaries_idx.append(idx)
                last_bucket = bs

        y_max = max(ys) if ys else 10

        axes = Axes(
            x_range=[1, k, 25],
            y_range=[0, max(8, int(y_max * 1.05)), max(1, int(max(8, int(y_max * 1.05)) / 5))],
            x_length=11.5,
            y_length=4.8,
            tips=False,
            axis_config={"stroke_width": 2},
        ).to_edge(DOWN, buff=1.05)

        xlab = Text("prime index", font=FONT["serif"], font_size=24, color=COL["sub"]).next_to(axes.get_x_axis(), RIGHT, buff=0.18)
        ylab = Text("s(p)", font=FONT["serif"], font_size=24, color=COL["sub"]).rotate(PI / 2).next_to(axes.get_y_axis(), LEFT, buff=0.28)

        pts = VGroup(*[
            Dot(axes.c2p(i + 1, ys[i]), radius=0.045, color=COL["prime"])
            for i in range(k)
        ])

        ramp = VMobject(color=BLUE_B, stroke_width=3).set_points_as_corners(
            [axes.c2p(i + 1, ys[i]) for i in range(k)]
        )

        bound_lines = VGroup()
        for bi in boundaries_idx:
            x = bi + 1
            ln = Line(axes.c2p(x, 0), axes.c2p(x, max(ys) * 1.02), color=COL["hi"], stroke_width=3).set_opacity(0.95)
            bound_lines.add(ln)

        legend = VGroup(
            VGroup(Dot(radius=0.055, color=COL["prime"]), Text("primes", font=FONT["serif"], font_size=22, color=COL["text"])).arrange(RIGHT, buff=0.18),
            VGroup(Line(ORIGIN, RIGHT * 0.6, color=BLUE_B, stroke_width=6), Text("connect s(p)", font=FONT["serif"], font_size=22, color=COL["text"])).arrange(RIGHT, buff=0.18),
            VGroup(Line(ORIGIN, RIGHT * 0.6, color=COL["hi"], stroke_width=6), Text("bucket boundary", font=FONT["serif"], font_size=22, color=COL["text"])).arrange(RIGHT, buff=0.18),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        legend_bg = RoundedRectangle(corner_radius=0.2, width=legend.width + 0.5, height=legend.height + 0.35, stroke_width=1.5, stroke_color=GREY_D).set_fill(BLACK, opacity=0.55)
        legend_pack = VGroup(legend_bg, legend).to_corner(UR, buff=0.45).shift(DOWN * 0.9)
        clamp_to_frame(legend_pack, margin=0.25)

        self.play(Create(axes), FadeIn(xlab), FadeIn(ylab), run_time=0.9 * SPEED)
        self.play(Create(ramp), FadeIn(pts), FadeIn(bound_lines), FadeIn(legend_pack), run_time=1.1 * SPEED)
        self.wait(0.5 * SPEED)

        wiggle = Text(
            "Prime gaps are irregular → primes land unevenly inside each bucket.",
            font=FONT["serif"],
            font_size=26,
            color=COL["text"],
        ).to_edge(DOWN, buff=0.35)

        self.play(FadeIn(wiggle, shift=UP * 0.10), run_time=0.7 * SPEED)
        self.wait(0.8 * SPEED)

        self.play(FadeOut(VGroup(btitle, eq, expl, axes, xlab, ylab, pts, ramp, bound_lines, legend_pack, wiggle)), run_time=0.75 * SPEED)

        # ------------------ Part C: density decay vs bits ------------------
        g_title = Text("Prime Density Thins Out With Scale", font=FONT["serif"], weight=BOLD, font_size=46, color=COL["text"]).to_edge(UP, buff=0.55)
        g_sub = Text("Density = (#primes in bucket) / (bucket size)", font=FONT["serif"], font_size=28, color=COL["sub"]).next_to(g_title, DOWN, buff=0.18)

        self.play(FadeIn(g_title, shift=DOWN * 0.12), FadeIn(g_sub, shift=DOWN * 0.12), run_time=1.0 * SPEED)

        MIN_BITS, MAX_BITS = 2, 16
        stats = [bucket_stats(bits) for bits in range(MIN_BITS, MAX_BITS + 1)]
        dens_vals = [s.density for s in stats]
        y_min = max(0.0, min(dens_vals) - 0.02)
        y_max2 = min(1.0, max(dens_vals) + 0.02)

        axes2 = Axes(
            x_range=[MIN_BITS, MAX_BITS, 2],
            y_range=[y_min, y_max2, (y_max2 - y_min) / 5],
            x_length=11.5,
            y_length=5.2,
            tips=False,
            axis_config={"include_numbers": True, "stroke_width": 2},
        ).to_edge(DOWN, buff=1.05)

        axes2.get_x_axis().set_color(GREY_B)
        axes2.get_y_axis().set_color(GREY_B)

        x_label2 = Text("bits", font=FONT["serif"], font_size=24, color=COL["sub"]).next_to(axes2.get_x_axis(), RIGHT, buff=0.20)
        y_label2 = Text("prime density", font=FONT["serif"], font_size=24, color=COL["sub"]).rotate(PI / 2).next_to(axes2.get_y_axis(), LEFT, buff=0.30)

        pts2 = VGroup(*[
            Dot(axes2.c2p(s.bits, s.density), radius=0.06, color=COL["dot"])
            for s in stats
        ])

        line2 = VMobject(color=COL["line"], stroke_width=3).set_points_as_corners(
            [axes2.c2p(s.bits, s.density) for s in stats]
        )

        theory = VMobject(color=COL["accent"], stroke_width=3)
        theory_pts = []
        for s in stats:
            mid = 0.5 * (s.start + s.end)
            theory_pts.append(axes2.c2p(s.bits, 1.0 / math.log(mid)))
        theory.set_points_smoothly(theory_pts)

        leg = VGroup(
            VGroup(Dot(radius=0.06, color=COL["dot"]), Text("measured density", font=FONT["serif"], font_size=22, color=COL["text"])).arrange(RIGHT, buff=0.18),
            VGroup(Line(ORIGIN, RIGHT * 0.6, color=COL["accent"], stroke_width=6), Text("~ 1/ln(n) guide", font=FONT["serif"], font_size=22, color=COL["text"])).arrange(RIGHT, buff=0.18),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.10)

        leg_bg = RoundedRectangle(corner_radius=0.2, width=leg.width + 0.5, height=leg.height + 0.35, stroke_width=1.5, stroke_color=GREY_D).set_fill(BLACK, opacity=0.55)
        leg_pack = VGroup(leg_bg, leg).to_corner(UR, buff=0.45).shift(DOWN * 0.85)
        clamp_to_frame(leg_pack, margin=0.25)

        self.play(Create(axes2), FadeIn(x_label2), FadeIn(y_label2), run_time=0.9 * SPEED)
        self.play(Create(line2), FadeIn(pts2), Create(theory), FadeIn(leg_pack), run_time=1.1 * SPEED)
        self.wait(0.5 * SPEED)

        cursor = Dot(radius=0.08, color=COL["hi"]).set_z_index(10)
        tracker = ValueTracker(0.0)

        def interp_point(t: float) -> np.ndarray:
            n = len(stats)
            if n == 1:
                return axes2.c2p(stats[0].bits, stats[0].density)
            t = max(0.0, min(float(n - 1), t))
            i = int(math.floor(t))
            if i >= n - 1:
                i = n - 2
                frac = 1.0
            else:
                frac = t - i
            p0 = axes2.c2p(stats[i].bits, stats[i].density)
            p1 = axes2.c2p(stats[i + 1].bits, stats[i + 1].density)
            return p0 * (1 - frac) + p1 * frac

        cursor.add_updater(lambda m: m.move_to(interp_point(tracker.get_value())))

        panel = RoundedRectangle(corner_radius=0.25, width=5.4, height=2.1, stroke_width=2, stroke_color=GREY_B).to_corner(DR, buff=0.55)
        panel.set_fill(BLACK, opacity=0.75)

        panel_title = Text("Per-bucket thinning", font=FONT["serif"], font_size=28, color=COL["hi"])
        panel_title.move_to(panel.get_top() + DOWN * 0.36)

        readout = always_redraw(lambda: make_readout(stats, tracker).move_to(panel.get_center() + DOWN * 0.05))
        hint = Text("compare bucket b to b−1", font=FONT["serif"], font_size=22, color=COL["sub"]).move_to(panel.get_bottom() + UP * 0.33)

        panel_pack = VGroup(panel, panel_title, readout, hint).set_z_index(20)
        clamp_to_frame(panel_pack, margin=0.28)

        self.add(cursor)
        self.play(FadeIn(panel_pack), FadeIn(cursor), run_time=0.7 * SPEED)
        self.play(tracker.animate.set_value(len(stats) - 1), run_time=2.2 * SPEED, rate_func=linear)
        self.wait(0.6 * SPEED)

        close = Text(
            "Bigger buckets → fewer primes per number.\nThat’s the density decay.",
            font=FONT["serif"],
            font_size=28,
            color=COL["text"],
        ).to_edge(DOWN, buff=0.35)

        self.play(FadeIn(close, shift=UP * 0.10), run_time=0.7 * SPEED)
        self.wait(0.9 * SPEED)

        self.play(FadeOut(VGroup(g_title, g_sub, axes2, x_label2, y_label2, pts2, line2, theory, leg_pack, panel_pack, cursor, close)), run_time=0.8 * SPEED)

        # ------------------ End card ------------------
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
        self.wait(1.4 * SPEED)
        self.play(FadeOut(end_pack), run_time=0.8 * SPEED)
