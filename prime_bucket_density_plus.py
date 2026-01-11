# prime_bucket_density_plus.py
#
# Manim Community v0.19.x (works with 0.19.1)
# Run:
#   python -m manim -pqh prime_bucket_density_plus.py PrimeBucketDensityPlus
#
# Fixes in THIS version:
# - Title never overlaps bucket header: header placed first, title scaled to remaining width
# - Header is nudged down slightly below title line
# - No empty/blank Text() objects ever created
# - No get_opacity() usage (avoids NoneType + deprecation warnings)
# - Caption visibility tracked with self.caption_visible
# - Caption never overlaps table (table top limit depends on caption visibility)
# - Tables only scale DOWN (never up)
# - Slowed down ~6× with SPEED = 6.0
# - Text animations (Write/Transform/Fade) with slower pacing

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

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
    delta_density: float | None
    single_prime_buckets_so_far: int
    buckets_so_far: int


# ==================== STYLE ====================

COL = {
    "bg": BLACK,
    "text": WHITE,
    "sub": GREY_B,
    "prime": GREEN_C,
    "comp": RED_C,
    "accent": PURPLE_B,
    "hi": YELLOW,
}

FONT = {
    "serif": "DejaVu Serif",
    "mono": "DejaVu Sans Mono",
}

LAY = {
    "top": 0.60,
    "bottom": 0.70,
    "side": 0.70,
    "gap_caption": 0.22,  # between top row and caption
    "gap_table": 0.30,    # between caption and table
    "col_gap": 1.20,
    "row_gap": 0.15,
    "hdr_gap": 0.28,
    "title_header_gap": 0.35,  # horizontal gap between title area and header area
    "header_drop": 0.18,       # push header down so it's not on same baseline as title
}


# ==================== COMPONENTS ====================

class BucketHeader(VGroup):
    def __init__(self, bits: int, start: int, end: int):
        super().__init__()
        t1 = Text(
            f"{bits}-BIT BUCKET",
            font=FONT["serif"],
            weight=BOLD,
            color=COL["text"],
            font_size=50,
        )
        t2 = Text(
            f"{start:,} – {end:,}",
            font=FONT["serif"],
            color=COL["accent"],
            font_size=34,
        )
        self.add(t1, t2).arrange(DOWN, aligned_edge=RIGHT, buff=0.10)


class PrimeTable(VGroup):
    def __init__(self, rows: List[Tuple[str, str, str, bool]], bits: int):
        super().__init__()
        self.rows = rows
        self.bits = bits
        self._build()

    def _sizes_for_bits(self) -> Tuple[int, int]:
        if self.bits <= 3:
            return (28, 28)
        if self.bits <= 5:
            return (24, 24)
        if self.bits <= 7:
            return (20, 20)
        return (18, 18)

    def _choose_cols(self, n: int) -> int:
        # slightly more aggressive split so 7-bit and 8-bit stay safe
        if n <= 12:
            return 1
        if n <= 24:
            return 2
        return 3

    def _build_column(self, chunk: List[Tuple[str, str, str, bool]]) -> VGroup:
        header_size, row_size = self._sizes_for_bits()

        h_dec = Text("DEC", font=FONT["mono"], color=COL["sub"], font_size=header_size)
        h_bin = Text("BINARY", font=FONT["mono"], color=COL["sub"], font_size=header_size)
        h_res = Text("RESULT", font=FONT["mono"], color=COL["sub"], font_size=header_size)
        hdr = VGroup(h_dec, h_bin, h_res).arrange(RIGHT, buff=1.4, aligned_edge=DOWN)

        max_dec = max((len(dec) for dec, _, _, _ in chunk), default=3)
        max_bin = max((len(bi) for _, bi, _, _ in chunk), default=6)
        dec_w = max(max_dec, 3)
        bin_w = max(max_bin, 6)

        row_mobs = []
        for dec, bi, res, is_p in chunk:
            c = COL["prime"] if is_p else COL["comp"]
            t_dec = Text(dec.rjust(dec_w), font=FONT["mono"], color=c, font_size=row_size)
            t_bin = Text(bi.rjust(bin_w), font=FONT["mono"], color=c, font_size=row_size)
            t_res = Text(res, font=FONT["mono"], color=c, font_size=row_size, weight=BOLD if is_p else NORMAL)
            row_mobs.append(VGroup(t_dec, t_bin, t_res).arrange(RIGHT, buff=1.2, aligned_edge=DOWN))

        body = VGroup(*row_mobs).arrange(DOWN, buff=LAY["row_gap"], aligned_edge=LEFT)
        return VGroup(hdr, body).arrange(DOWN, buff=LAY["hdr_gap"], aligned_edge=LEFT)

    def _build(self):
        n = len(self.rows)
        cols = self._choose_cols(n)
        chunk_size = (n + cols - 1) // cols
        chunks = [self.rows[i * chunk_size:(i + 1) * chunk_size] for i in range(cols)]

        col_groups = [self._build_column(ch) for ch in chunks]
        self.add(*col_groups)
        if len(col_groups) > 1:
            self.arrange(RIGHT, buff=LAY["col_gap"], aligned_edge=UP)


class MetricsPanel(VGroup):
    def __init__(self, st: BucketStats):
        super().__init__()
        summary = Text(
            f"Bucket {st.bits}: {st.primes} primes out of {st.size}  →  density = {st.density:.3f}",
            font=FONT["serif"],
            color=COL["text"],
            font_size=40,
        )
        era = Text(
            "Bit bucket era = prime density era",
            font=FONT["serif"],
            color=COL["hi"],
            font_size=34,
        )
        rate = st.single_prime_buckets_so_far / st.buckets_so_far
        single = Text(
            f"Single-prime buckets so far: {st.single_prime_buckets_so_far}/{st.buckets_so_far}  →  rate = {rate:.3f}",
            font=FONT["serif"],
            color=BLUE_B,
            font_size=32,
        )
        if st.delta_density is None:
            thin = Text(
                "Per-bucket thinning: (no previous bucket yet)",
                font=FONT["serif"],
                color=COL["hi"],
                font_size=32,
            )
        else:
            sign = "+" if st.delta_density >= 0 else "−"
            thin = Text(
                f"Per-bucket thinning: Δdensity = {sign}{abs(st.delta_density):.3f}",
                font=FONT["serif"],
                color=COL["hi"],
                font_size=32,
            )

        self.add(summary, era, single, thin).arrange(DOWN, aligned_edge=LEFT, buff=0.22)


# ==================== SCENE ====================

class PrimeBucketDensityPlus(Scene):
    def construct(self):
        self.camera.background_color = COL["bg"]
        SPEED = 6.0

        MIN_BITS, MAX_BITS = 2, 8
        prev_density: Optional[float] = None
        buckets_so_far = 0
        single_prime_buckets_so_far = 0

        self.caption_visible = False

        # ----- Title -----
        self.main_title = Text(
            "PRIME NUMBER DISTRIBUTION",
            font=FONT["serif"],
            weight=BOLD,
            color=COL["text"],
            font_size=56,
        ).set_opacity(0.80)

        # initial placement (will be re-fit per-bucket once header exists)
        self._place_title_ul(max_w=config.frame_width - 2 * LAY["side"])
        self.play(FadeIn(self.main_title, shift=DOWN * 0.15), run_time=0.8 * SPEED)
        self.wait(0.25 * SPEED)

        # Caption: always real text, starts hidden
        caption = Text("•", font=FONT["serif"], color=COL["sub"], font_size=30).set_opacity(0.0)
        self.add(caption)

        for bits in range(MIN_BITS, MAX_BITS + 1):
            start = 2 ** (bits - 1)
            end = (2 ** bits) - 1
            size = end - start + 1

            rows: List[Tuple[str, str, str, bool]] = []
            primes = 0
            for n in range(start, end + 1):
                p = is_prime(n)
                primes += int(p)
                rows.append((str(n), bin_str(n, bits), "PRIME" if p else "COMPOSITE", p))

            density = primes / size
            delta = None if prev_density is None else (density - prev_density)

            buckets_so_far += 1
            if primes == 1:
                single_prime_buckets_so_far += 1

            st = BucketStats(
                bits=bits,
                start=start,
                end=end,
                size=size,
                primes=primes,
                density=density,
                delta_density=delta,
                single_prime_buckets_so_far=single_prime_buckets_so_far,
                buckets_so_far=buckets_so_far,
            )

            header, table = self._show_bucket_table(rows, st, caption, SPEED)
            self._show_bucket_metrics(st, header, table, caption, SPEED)

            prev_density = density

        self.wait(0.4 * SPEED)

    # ---------- helpers ----------

    def _scale_down_to_fit(self, mob: Mobject, max_w: float, max_h: float) -> None:
        if mob.width <= 0 or mob.height <= 0:
            return
        s = min(max_w / mob.width, max_h / mob.height, 1.0)
        mob.scale(s)

    def _place_title_ul(self, max_w: float) -> None:
        if self.main_title.width > max_w:
            self.main_title.scale(max_w / self.main_title.width)
        self.main_title.to_corner(UL, buff=LAY["side"])
        # ensure exact top margin look
        self.main_title.to_edge(UP, buff=LAY["top"])

    def _place_title_and_header_no_overlap(self, header: Mobject) -> None:
        """
        Place header first (UR), drop it slightly,
        then scale/title-fit into remaining width on the left.
        """
        # Place header top-right and nudge it down
        header.to_corner(UR, buff=LAY["side"])
        header.to_edge(UP, buff=LAY["top"])
        header.shift(DOWN * LAY["header_drop"])

        # Compute remaining width for title:
        left_edge = -config.frame_width / 2 + LAY["side"]
        right_limit = header.get_left()[0] - LAY["title_header_gap"]
        max_w = max(1.0, right_limit - left_edge)

        # Fit title into that width, and lock UL/top
        self._place_title_ul(max_w=max_w)

    def _caption_y_from_toprow(self, header: Mobject, cap_height: float) -> float:
        # use bottom of whichever sits LOWER: title or header
        top_row_bottom_y = min(self.main_title.get_bottom()[1], header.get_bottom()[1])
        return top_row_bottom_y - LAY["gap_caption"] - cap_height / 2

    def _set_caption(self, caption: Text, header: Mobject, text: str, speed: float) -> None:
        if not text.strip():
            if self.caption_visible:
                self.play(caption.animate.set_opacity(0.0), run_time=0.35 * speed)
                self.caption_visible = False
            return

        new_cap = Text(text, font=FONT["serif"], color=COL["sub"], font_size=30).set_opacity(0.90)

        max_w = config.frame_width - 2 * LAY["side"]
        if new_cap.width > max_w:
            new_cap.scale(max_w / new_cap.width)

        y = self._caption_y_from_toprow(header, new_cap.height)
        new_cap.move_to(np.array([0.0, y, 0.0]))

        if not self.caption_visible:
            caption.set_opacity(0.90)
            self.caption_visible = True

        self.play(Transform(caption, new_cap), run_time=0.55 * speed)

    def _table_top_limit(self, caption: Text, header: Mobject) -> float:
        if not self.caption_visible:
            top_row_bottom_y = min(self.main_title.get_bottom()[1], header.get_bottom()[1])
            return top_row_bottom_y - LAY["gap_caption"] - LAY["gap_table"]
        return caption.get_bottom()[1] - LAY["gap_table"]

    # ---------- phases ----------

    def _show_bucket_table(
        self,
        rows: List[Tuple[str, str, str, bool]],
        st: BucketStats,
        caption: Text,
        speed: float,
    ) -> Tuple[BucketHeader, PrimeTable]:

        header = BucketHeader(st.bits, st.start, st.end)

        # IMPORTANT: place header first, then fit title to remaining space
        self._place_title_and_header_no_overlap(header)

        self._set_caption(
            caption,
            header,
            f"Bucket: {st.start:,} to {st.end:,}   •   fixed width = {st.bits} bits",
            speed,
        )

        table = PrimeTable(rows, st.bits)

        safe_left = -config.frame_width / 2 + LAY["side"]
        safe_right = config.frame_width / 2 - LAY["side"]
        safe_bottom = -config.frame_height / 2 + LAY["bottom"]
        top_limit = self._table_top_limit(caption, header)

        avail_w = safe_right - safe_left
        avail_h = max(top_limit - safe_bottom, 0.8)

        self._scale_down_to_fit(table, avail_w, avail_h)

        center_y = safe_bottom + avail_h / 2
        table.move_to(np.array([0.0, center_y, 0.0]))

        self.play(Write(header), run_time=0.9 * speed)

        self._set_caption(caption, header, "Green = prime   •   Red = composite", speed)
        self.play(FadeIn(table, shift=UP * 0.10), run_time=1.0 * speed)
        self.wait(0.45 * speed)

        self._set_caption(caption, header, "Binary column is constant width inside this bucket.", speed)
        self.wait(0.35 * speed)

        return header, table

    def _show_bucket_metrics(
        self,
        st: BucketStats,
        header: BucketHeader,
        table: PrimeTable,
        caption: Text,
        speed: float,
    ):
        self._set_caption(caption, header, "Now summarize: density + thinning signal", speed)

        metrics = MetricsPanel(st)
        max_w = config.frame_width - 2 * LAY["side"]
        max_h = config.frame_height * 0.62
        self._scale_down_to_fit(metrics, max_w, max_h)
        metrics.center().shift(DOWN * 0.1)

        self.play(FadeOut(table, shift=DOWN * 0.15), FadeOut(header, shift=UP * 0.15), run_time=0.75 * speed)
        self.play(Write(metrics), run_time=1.0 * speed)
        self.wait(0.8 * speed)
        self.play(FadeOut(metrics, shift=DOWN * 0.15), run_time=0.7 * speed)

        self._set_caption(caption, header, "", speed)

