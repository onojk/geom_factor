from manim import *
import numpy as np

TIME_SCALE = 13.3  # bigger = slower


def first_nontrivial_divisor(n: int):
    """Smallest divisor in {2..n-1}, or None if prime (or n<2)."""
    if n < 2:
        return None
    for d in range(2, n):
        if n % d == 0:
            return d
    return None


def gate_font_sizes(n: int):
    """
    Aggressive text scaling to prevent overlap at N=10 and N=11.
    Returns (line1_font, line2_font, d_label_font).
    """
    if n <= 7:
        return 16, 16, 20      # roomy
    elif n <= 9:
        return 14, 14, 18      # compact
    elif n == 10:
        return 11, 11, 16      # tight
    else:  # n == 11
        return 9, 9, 15        # very tight


class LaserGatesTo11(Scene):
    def construct(self):
        safe = Rectangle(width=13.2, height=7.2).set_stroke(opacity=0).move_to(ORIGIN)

        # -----------------------------
        # TITLE
        # -----------------------------
        title = Text("Prime Roads → Laser Through Gates", font_size=56).move_to([0, 3.25, 0])
        subtitle = Text(
            "For each N, show ALL gates d = 1..N (even obvious clear ones).",
            font_size=28,
        ).move_to([0, 2.62, 0])

        self.play(
            FadeIn(title, shift=DOWN * 0.12),
            FadeIn(subtitle, shift=DOWN * 0.12),
            run_time=0.55 * TIME_SCALE,
        )
        self.wait(0.55 * TIME_SCALE)
        self.play(FadeOut(title), FadeOut(subtitle), run_time=0.35 * TIME_SCALE)

        # -----------------------------
        # HUD (two-line, wraps via \n)
        # -----------------------------
        HUD_Y = 2.35
        hud = VGroup().move_to([0, HUD_Y, 0])

        def set_hud(line1: str, line2: str = "", color=WHITE, rt=0.35):
            nonlocal hud
            new = VGroup(
                Text(line1, font_size=34),
                Text(line2, font_size=26).set_opacity(0.90)
                if line2
                else Text("", font_size=1).set_opacity(0),
            ).arrange(DOWN, buff=0.16)
            new.set_color(color)
            new.move_to([0, HUD_Y, 0])

            if len(hud) == 0:
                hud = new
                self.play(FadeIn(hud), run_time=rt * TIME_SCALE)
            else:
                self.play(ReplacementTransform(hud, new), run_time=rt * TIME_SCALE)
                hud = new

        set_hud(
            "All gates are wired in parallel and decide simultaneously.",
            "The laser sweep REVEALS results.\nThe first nontrivial block proves composite.",
            rt=0.55,
        )
        self.wait(0.55 * TIME_SCALE)

        # -----------------------------
        # STAGE
        # -----------------------------
        BEAM_Y = -0.85
        SOURCE_X = safe.get_left()[0] + 1.0
        SCREEN_X = safe.get_right()[0] - 1.0

        track = Line([SOURCE_X, BEAM_Y, 0], [SCREEN_X, BEAM_Y, 0], stroke_width=6).set_opacity(0.20)
        self.play(FadeIn(track), run_time=0.30 * TIME_SCALE)

        source = Dot(point=[SOURCE_X, BEAM_Y, 0], radius=0.12)
        source_label = Text("laser", font_size=22).next_to(source, DOWN, buff=0.14)

        screen = RoundedRectangle(width=0.95, height=2.6, corner_radius=0.2)
        screen.set_stroke(width=6)
        screen.set_fill(WHITE, opacity=0.05)
        screen.move_to([SCREEN_X, BEAM_Y, 0])
        screen_label = Text("screen", font_size=22).next_to(screen, UP, buff=0.12)

        # Screen shows current N
        screen_value = Text("", font_size=46, weight=BOLD).move_to(screen.get_center()).set_opacity(0.25)

        self.play(
            FadeIn(source),
            FadeIn(source_label),
            FadeIn(screen),
            FadeIn(screen_label),
            FadeIn(screen_value),
            run_time=0.35 * TIME_SCALE,
        )

        # -----------------------------
        # LASER (wavefront) — RED LASER
        # -----------------------------
        wave_x = ValueTracker(SOURCE_X)

        def beam_mob():
            x0 = SOURCE_X
            x1 = max(x0, wave_x.get_value())
            w = x1 - x0
            rect = Rectangle(width=max(0.001, w), height=0.18).set_stroke(width=0).set_fill(RED, opacity=0.28)
            rect.move_to([x0 + w / 2, BEAM_Y, 0])
            rect.set_z_index(-10)
            return rect

        beam = always_redraw(beam_mob)
        wave_tip = always_redraw(lambda: Dot(point=[wave_x.get_value(), BEAM_Y, 0], radius=0.06, color=RED))
        self.add(beam, wave_tip)

        # -----------------------------
        # GATE FACTORY — YELLOW RINGS
        # -----------------------------
        def make_gate(x: float, d: int, n: int):
            """
            Reverse bulb gate, always visible.
            Under text is TWO LINES:
              line 1: mod statement (or 'always clear')
              line 2: state word ('clear', 'pending', 'opaque')
            """
            outline = Circle(radius=0.24).set_stroke(color=YELLOW, width=5).set_fill(opacity=0)
            fill = Circle(radius=0.24).set_stroke(width=0).set_fill(YELLOW, opacity=0.06)

            outline.move_to([x, BEAM_Y, 0])
            fill.move_to([x, BEAM_Y, 0])

            fs1, fs2, fsd = gate_font_sizes(n)

            d_lbl = Text(str(d), font_size=fsd).next_to(outline, UP, buff=0.10)

            if d == 1 or d == n:
                line1 = Text("always clear", font_size=fs1).set_opacity(0.80)
                line2 = Text("clear", font_size=fs2).set_opacity(0.92)
            else:
                line1 = Text(f"{n} mod {d} = ?", font_size=fs1).set_opacity(0.75)
                line2 = Text("pending", font_size=fs2).set_opacity(0.55)

            status_group = VGroup(line1, line2).arrange(DOWN, buff=0.05)
            status_group.next_to(outline, DOWN, buff=0.18)

            # 0 fill, 1 outline, 2 label, 3 status_group
            return VGroup(fill, outline, d_lbl, status_group)

        # -----------------------------
        # ONE N RUN (2..11)
        # -----------------------------
        def run_n(n: int):
            nonlocal hud

            screen_value.become(
                Text(str(n), font_size=46, weight=BOLD).move_to(screen.get_center()).set_opacity(0.25)
            )

            set_hud(
                f"Testing N = {n}",
                "SHOW all gates (even obvious clear ones).\nThe first nontrivial block proves composite.",
                rt=0.45,
            )
            self.wait(0.15 * TIME_SCALE)

            if hasattr(self, "gates"):
                self.play(FadeOut(self.gates), run_time=0.25 * TIME_SCALE)

            ds = list(range(1, n + 1))
            gates = VGroup()

            left = SOURCE_X + 1.35
            right = SCREEN_X - 1.35
            xs = np.linspace(left, right, len(ds))

            for x, d in zip(xs, ds):
                gates.add(make_gate(x, d, n))

            self.gates = gates
            self.play(FadeIn(gates, shift=UP * 0.08), run_time=0.35 * TIME_SCALE)

            wave_x.set_value(SOURCE_X)
            blocked = False

            for i, d in enumerate(ds):
                gate = gates[i]
                fill = gate[0]
                status_group = gate[3]
                gx = gate.get_center()[0]

                self.play(wave_x.animate.set_value(gx + 0.10), run_time=0.28 * TIME_SCALE)

                if d == 1 or d == n:
                    self.play(fill.animate.set_fill(YELLOW, opacity=0.14), run_time=0.12 * TIME_SCALE)
                    self.play(fill.animate.set_fill(YELLOW, opacity=0.06), run_time=0.10 * TIME_SCALE)
                    continue

                fs1, fs2, _ = gate_font_sizes(n)
                r = n % d

                if r != 0:
                    new_status = VGroup(
                        Text(f"{n} mod {d} ≠ 0", font_size=fs1).set_opacity(0.92),
                        Text("clear", font_size=fs2).set_opacity(0.98),
                    ).arrange(DOWN, buff=0.05).move_to(status_group)

                    self.play(
                        fill.animate.set_fill(YELLOW, opacity=0.14),
                        Transform(status_group, new_status),
                        run_time=0.16 * TIME_SCALE,
                    )
                    self.play(fill.animate.set_fill(YELLOW, opacity=0.06), run_time=0.10 * TIME_SCALE)

                else:
                    new_status = VGroup(
                        Text(f"{n} mod {d} = 0", font_size=fs1).set_opacity(1.0),
                        Text("opaque", font_size=fs2).set_opacity(1.0),
                    ).arrange(DOWN, buff=0.05).move_to(status_group)

                    self.play(
                        fill.animate.set_fill(RED, opacity=0.78),
                        Transform(status_group, new_status),
                        screen.animate.set_fill(opacity=0.02),
                        run_time=0.22 * TIME_SCALE,
                    )

                    other = n // d
                    set_hud("COMPOSITE", f"Blocked at d = {d} (so {d} × {other} = {n}).", color=RED, rt=0.45)
                    blocked = True
                    break

            if not blocked:
                self.play(wave_x.animate.set_value(SCREEN_X - 0.55), run_time=0.35 * TIME_SCALE)
                self.play(
                    screen.animate.set_fill(opacity=0.12),
                    screen_value.animate.set_opacity(1.0),
                    run_time=0.20 * TIME_SCALE,
                )
                set_hud("PRIME", "The laser passed every gate and reached the screen.", color=GREEN, rt=0.50)

            self.wait(0.35 * TIME_SCALE)

            self.play(
                screen.animate.set_fill(opacity=0.05),
                screen_value.animate.set_opacity(0.25),
                run_time=0.15 * TIME_SCALE,
            )

        # Run N = 2..11
        for n in range(2, 12):
            run_n(n)

        # -----------------------------
        # Clean up before end cards
        # -----------------------------
        if hasattr(self, "gates"):
            self.play(FadeOut(self.gates), run_time=0.35 * TIME_SCALE)

        self.play(FadeOut(hud), run_time=0.25 * TIME_SCALE)
        hud = VGroup()

        self.play(FadeOut(source), FadeOut(source_label), FadeOut(track), run_time=0.30 * TIME_SCALE)
        self.play(FadeOut(screen), FadeOut(screen_label), FadeOut(screen_value), run_time=0.30 * TIME_SCALE)

        self.remove(beam, wave_tip)

        # -----------------------------
        # FINAL TAKEAWAY SLIDE (phrase updated)
        # -----------------------------
        takeaway = VGroup(
            Text("Final takeaway", font_size=52),
            Text("If the setup follows these rules:", font_size=30),
            Text(
                "• all gates are wired in parallel and decide simultaneously\n"
                "• each gate turns opaque if N mod d = 0\n"
                "• the laser reveals the result",
                font_size=28,
            ),
            Text("then:", font_size=30),
            Text("ALL CLEAR  ⟺  N is PRIME", font_size=38, weight=BOLD).set_color(GREEN),
        ).arrange(DOWN, buff=0.28)

        takeaway.move_to([0, 0.45, 0])
        self.play(FadeIn(takeaway, shift=DOWN * 0.12), run_time=0.65 * TIME_SCALE)
        self.wait(1.25 * TIME_SCALE)
        self.play(FadeOut(takeaway, shift=UP * 0.10), run_time=0.45 * TIME_SCALE)

        # -----------------------------
        # THANK YOU / SIGNATURE
        # -----------------------------
        thanks = Text("Thank you for watching!", font_size=48)
        signature = Text("- ONOJK123", font_size=30).set_opacity(0.85)

        end_card = VGroup(thanks, signature).arrange(DOWN, buff=0.32)
        end_card.move_to([0, 0.25, 0])

        self.play(FadeIn(end_card, shift=UP * 0.12), run_time=0.55 * TIME_SCALE)
        self.wait(1.75 * TIME_SCALE)
        self.play(FadeOut(end_card), run_time=0.45 * TIME_SCALE)

