from manim import *
import math

class PrimeBucketDensity(Scene):
    def is_prime(self, n):
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    def bucket_table(self, bits):
        start = 2 ** (bits - 1)
        end = 2 ** bits - 1
        rows = []

        primes = 0
        for n in range(start, end + 1):
            binary = format(n, f"0{bits}b")
            prime = self.is_prime(n)
            if prime:
                primes += 1

            row = [
                Text(f"{n:>2}", font="Monospace", font_size=28),
                Text(" ".join(binary), font="Monospace", font_size=28),
                Text(
                    "PRIME (TRUE)" if prime else "COMPOSITE (FALSE)",
                    font="Monospace",
                    font_size=28,
                    color=GREEN if prime else RED
                )
            ]
            rows.append(row)

        table = VGroup()
        for r in rows:
            row_group = VGroup(*r).arrange(RIGHT, buff=0.6)
            table.add(row_group)

        table.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        return table, primes, (end - start + 1), start, end

    def construct(self):
        title = Text("Prime Buckets", font_size=44).to_corner(UL)
        self.play(FadeIn(title))
        self.wait(1)

        last_density = None

        for bits in range(2, 6):
            bucket_title = Text(
                f"{bits}-bit bucket",
                font_size=48
            ).to_edge(UP)

            range_label = Text(
                f"{2**(bits-1)}–{2**bits - 1}",
                font_size=28
            ).next_to(bucket_title, DOWN)

            self.play(FadeIn(bucket_title), FadeIn(range_label))
            self.wait(0.8)

            table, prime_count, total, start, end = self.bucket_table(bits)
            table.scale(0.9)
            table.next_to(range_label, DOWN, buff=0.6)

            self.play(FadeIn(table))
            self.wait(1.2)

            density = prime_count / total
            density_text = Text(
                f"Bucket {bits}: {prime_count} primes out of {total}  →  density = {density:.3f}",
                font_size=28
            ).next_to(table, DOWN, buff=0.6)

            self.play(FadeIn(density_text))
            self.wait(1)

            if last_density is not None:
                change = density - last_density
                change_text = Text(
                    f"Density change from previous bucket: {change:+.3f}",
                    font_size=26,
                    color=YELLOW
                ).next_to(density_text, DOWN, buff=0.3)

                self.play(FadeIn(change_text))
                self.wait(1)
                self.play(FadeOut(change_text))

            last_density = density

            self.play(
                FadeOut(table),
                FadeOut(density_text),
                FadeOut(bucket_title),
                FadeOut(range_label)
            )

        explanation = VGroup(
            Text(
                "Prime density thins per bucket — not because primes disappear,",
                font_size=34
            ),
            Text(
                "but because each bucket introduces exponentially more candidates.",
                font_size=34
            ),
            Text(
                "Primality isn’t something you detect — it’s something that survives division.",
                font_size=34
            ),
            Text(
                "A number is prime only if every possible mod node fails.",
                font_size=34
            ),
            Text(
                "As long as division takes real work, the cost of certifying primality must grow.",
                font_size=34
            )
        ).arrange(DOWN, buff=0.5).to_edge(DOWN)

        self.play(FadeIn(explanation))
        self.wait(4)
