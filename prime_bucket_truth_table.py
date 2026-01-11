# prime_bucket_truth_table.py
# Manim Community v0.19.1
#
# Render:
#   python -m manim -pqh prime_bucket_truth_table.py PrimeBucketTruthTable
#   python -m manim -pql prime_bucket_truth_table.py PrimeBucketTruthTable

from manim import *
import math

# ---------- math helpers ----------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    d = 3
    while d <= r:
        if n % d == 0:
            return False
        d += 2
    return True


def bucket_range(bits: int):
    # "bit bucket" = numbers with exactly `bits` bits (leading bit = 1)
    start = 1 << (bits - 1)
    end = (1 << bits) - 1
    return start, end


# ---------- visual helpers ----------

MONO = "DejaVu Sans Mono"
GREEN = "#7CFF6B"   # readable green on black
RED = "#FF5A5A"     # readable red on black
DIM = "#CFCFCF"

def make_bucket_header(bits: int) -> VGroup:
    start, end = bucket_range(bits)
    start_bin = format(start, f"0{bits}b")
    end_bin = format(end, f"0{bits}b")

    # main bucket title
    t1 = Text(f"{bits}-bit bucket", font="DejaVu Serif").scale(1.25)
    # range line
    t2 = Text(f"{start_bin} – {end_bin}  (dec {start}–{end})", font="DejaVu Serif").scale(0.7)

    header = VGroup(t1, t2).arrange(DOWN, buff=0.15)
    header.to_edge(UP, buff=0.6)
    return header


def make_density_block(bits: int, prev_density: float | None) -> VGroup:
    start, end = bucket_range(bits)
    size = end - start + 1
    primes = sum(1 for n in range(start, end + 1) if is_prime(n))
    density = primes / size if size else 0.0

    # per-bucket thinning explanation (not generic)
    line1 = Text(
        f"Bucket {bits}: primes = {primes} out of {size}  →  density = {density:.3f} ({density*100:.1f}%)",
        font="DejaVu Serif"
    ).scale(0.58)

    if prev_density is None:
        line2 = Text(
            "When you add 1 bit, the bucket size doubles. Prime count rises, but density already starts thinning.",
            font="DejaVu Serif"
        ).scale(0.50)
    else:
        delta = density - prev_density
        sign = "+" if delta >= 0 else "−"
        line2 = Text(
            f"Compared to bucket {bits-1}: density change = {sign}{abs(delta):.3f}. "
            "Bigger buckets mean more candidates, but primes don’t keep up proportionally.",
            font="DejaVu Serif"
        ).scale(0.50)

    block = VGroup(line1, line2).arrange(DOWN, buff=0.12)
    block.to_edge(DOWN, buff=0.55)
    return block, density


def make_table(bits: int, page_start: int | None = None, page_len: int | None = None) -> VGroup:
    """
    Creates a table that fits on screen. For big buckets, call this multiple times (pagination)
    to avoid off-screen cutoffs.
    """
    start, end = bucket_range(bits)
    nums = list(range(start, end + 1))

    if page_start is not None and page_len is not None:
        nums = nums[page_start:page_start + page_len]

    dec_width = len(str(end))  # for leading zeros in decimal column

    # Column x positions (fixed) for clean alignment:
    #   dec | b_{bits-1} ... b0 | result
    # Use monospace and fixed spacing.
    x_dec = -5.4
    x_bits_start = -2.9
    bit_step = 0.55
    x_result = x_bits_start + bit_step * bits + 1.0

    # Header labels
    dec_lab = Text("dec", font=MONO).scale(0.55).set_color(DIM).move_to([x_dec, 1.6, 0])

    bit_labs = VGroup()
    for i in range(bits):
        bname = f"b{bits-1-i}"
        t = Text(bname, font=MONO).scale(0.55).set_color(DIM)
        t.move_to([x_bits_start + i * bit_step, 1.6, 0])
        bit_labs.add(t)

    res_lab = Text("result", font=MONO).scale(0.55).set_color(DIM).move_to([x_result, 1.6, 0])

    header = VGroup(dec_lab, bit_labs, res_lab)

    # Rows
    rows = VGroup()
    y0 = 1.15
    row_step = 0.43

    for r, n in enumerate(nums):
        y = y0 - r * row_step

        # decimal with leading zeros (e.g., 08, 09 in the 4-bit bucket)
        dec_str = f"{n:0{dec_width}d}"
        dec_t = Text(dec_str, font=MONO).scale(0.60).move_to([x_dec, y, 0])

        b = format(n, f"0{bits}b")
        bits_t = VGroup()
        for i, ch in enumerate(b):
            t = Text(ch, font=MONO).scale(0.60)
            t.move_to([x_bits_start + i * bit_step, y, 0])
            bits_t.add(t)

        prime = is_prime(n)
        res_str = "PRIME (TRUE)" if prime else "COMPOSITE (FALSE)"
        res_t = Text(res_str, font=MONO).scale(0.60).move_to([x_result, y, 0])

        # Color entire row consistently
        color = GREEN if prime else RED
        dec_t.set_color(color)
        for t in bits_t:
            t.set_color(color)
        res_t.set_color(color)

        row = VGroup(dec_t, bits_t, res_t)
        rows.add(row)

    table = VGroup(header, rows)

    # Fit table into frame height safely (prevents off-screen cutoffs)
    # Leave room for bucket header (top) and density block (bottom).
    max_height = 4.6  # tuned for default 16:9 frame
    if table.height > max_height:
        table.scale(max_height / table.height)

    # Place table between header and density block region
    table.move_to(ORIGIN)
    table.shift(UP * 0.2)
    return table


# ---------- scene ----------

class PrimeBucketTruthTable(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        # Persistent small label (top-left)
        watermark = Text("Prime Buckets", font="DejaVu Serif").scale(0.6).set_opacity(0.55)
        watermark.to_corner(UL, buff=0.35)
        self.add(watermark)

        # We’ll show buckets 1 through 7. (7-bit = 64 rows, paginated.)
        prev_density = None

        for bits in range(1, 8):
            header = make_bucket_header(bits)

            # Decide pagination to prevent off-screen table (6-bit and 7-bit are big)
            start, end = bucket_range(bits)
            size = end - start + 1

            # Small buckets: single page
            if size <= 16:
                pages = [(0, size)]
            else:
                # paginate in chunks of 16 rows (fits well)
                chunk = 16
                pages = [(i, chunk) for i in range(0, size, chunk)]

            # Show header
            self.play(FadeIn(header, shift=DOWN * 0.2), run_time=0.6)

            # Build density block (per-bucket thinning, with comparison)
            density_block, prev_density = make_density_block(bits, prev_density)

            # Show pages
            table = None
            for pi, (ps, plen) in enumerate(pages):
                new_table = make_table(bits, page_start=ps, page_len=plen)

                if table is None:
                    table = new_table
                    self.play(FadeIn(table), run_time=0.6)
                    self.play(FadeIn(density_block), run_time=0.6)
                else:
                    # IMPORTANT: vanish page label elements before changing pages
                    # (avoids “text writes over text”)
                    self.play(Transform(table, new_table), run_time=0.6)

                # Small pause per page
                self.wait(0.6)

                # If more pages remain, add a subtle “scroll” feeling (no overlap)
                if pi < len(pages) - 1:
                    self.play(table.animate.shift(UP * 0.2).set_opacity(0.85), run_time=0.25)
                    self.play(table.animate.shift(DOWN * 0.2).set_opacity(1.0), run_time=0.25)

            # Explain “thinning” per bucket is already shown in density block.
            # Now clean stage BEFORE moving to next bucket to avoid overlaps.
            self.wait(0.4)
            self.play(
                FadeOut(table),
                FadeOut(density_block),
                FadeOut(header),
                run_time=0.6
            )
            self.wait(0.15)

        # Closing thought (no overlap: everything is cleared first)
        closing = VGroup(
            Text("Primality isn’t something you detect — it’s something that survives division.", font="DejaVu Serif").scale(0.72),
            Text("A number is prime only if every possible mod node fails.", font="DejaVu Serif").scale(0.72),
            Text("As long as division takes real work, certifying primality can’t be absorbed into a finite pattern.", font="DejaVu Serif").scale(0.62),
        ).arrange(DOWN, buff=0.35).move_to(ORIGIN)

        self.play(FadeIn(closing), run_time=0.9)
        self.wait(2.0)
        self.play(FadeOut(closing), run_time=0.6)
        self.wait(0.2)
