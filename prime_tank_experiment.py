from manim import *
import math
import numpy as np

# ============================================================
# Prime Tank Experiment (clean layout, slower pacing, pauses)
# - Primes pause 2s, non-primes pause 1s (at each whole number)
# - Animations slowed (SLOW multiplier)
# - Tank frame stays fixed; "scale" readout grows
# - Water looks like it's flowing (gentle shimmer pulse)
# - Divisor test is *presented* as instant (quick flash sweep)
# ============================================================

SLOW = 4.0                 # 4× slower animations (not waits)
PRIME_PAUSE = 2.0          # seconds
NONPRIME_PAUSE = 1.0       # seconds

TARGET_PRIMES = 12         # how many primes to showcase

# --- timing helpers (animations only) ---
def rt(x: float) -> float:
    return x * SLOW

# --- math helpers ---
def first_forbidden_divisor(n: int):
    """Return (d_hit, limit). If prime: d_hit=None. We test d=2..floor(sqrt(n))."""
    if n < 2:
        return (None, 1)
    limit = int(math.isqrt(n))
    for d in range(2, limit + 1):
        if n % d == 0:
            return (d, limit)
    return (None, limit)

def compute_run_numbers(target_primes: int):
    """Precompute numbers from 2 up to n_end where we have `target_primes` primes."""
    primes_seen = 0
    n = 2
    nums = []
    while primes_seen < target_primes:
        d_hit, _ = first_forbidden_divisor(n)
        if d_hit is None:
            primes_seen += 1
        nums.append(n)
        n += 1
    return nums

# --- layout helpers ---
def fit_width(mob: Mobject, max_w: float):
    if mob.width > max_w:
        mob.scale_to_fit_width(max_w)

def fit_height(mob: Mobject, max_h: float):
    if mob.height > max_h:
        mob.scale_to_fit_height(max_h)

class PrimeTankExperiment(Scene):
    def construct(self):
        # ----------------------------
        # Precompute the "story length"
        # ----------------------------
        numbers = compute_run_numbers(TARGET_PRIMES)
        n_start = numbers[0]
        n_end = numbers[-1]

        # ----------------------------
        # Safe frame + columns
        # ----------------------------
        fw = config.frame_width
        fh = config.frame_height
        margin = 0.55

        SAFE = Rectangle(width=fw - 2 * margin, height=fh - 2 * margin).set_stroke(opacity=0).move_to(ORIGIN)

        divider_x = 1.85
        divider = Line(
            [divider_x, SAFE.get_top()[1], 0],
            [divider_x, SAFE.get_bottom()[1], 0],
            stroke_width=2,
        ).set_opacity(0.20)

        LEFT_COL = Rectangle(
            width=(divider_x - SAFE.get_left()[0]) - 0.25,
            height=SAFE.height
        ).set_stroke(opacity=0)
        LEFT_COL.move_to([(SAFE.get_left()[0] + divider_x) / 2, SAFE.get_center()[1], 0])

        RIGHT_COL = Rectangle(
            width=(SAFE.get_right()[0] - divider_x) - 0.25,
            height=SAFE.height
        ).set_stroke(opacity=0)
        RIGHT_COL.move_to([(SAFE.get_right()[0] + divider_x) / 2, SAFE.get_center()[1], 0])

        self.add(divider)

        # ----------------------------
        # Title + teaser (storytelling)
        # ----------------------------
        title = Text("Prime Tank Experiment", font_size=54)
        subtitle = Text("A controller that only knows: POUR  or  PAUSE", font_size=28)

        fit_width(title, SAFE.width * 0.92)
        fit_width(subtitle, SAFE.width * 0.92)

        title.move_to([SAFE.get_center()[0], SAFE.get_top()[1] - 0.65, 0])
        subtitle.next_to(title, DOWN, buff=0.22)

        self.play(FadeIn(title, shift=DOWN * 0.15), FadeIn(subtitle, shift=DOWN * 0.15), run_time=rt(0.55))
        self.wait(0.55)

        teaser = Text(
            "We raise the water level through whole numbers: 2, 3, 4, 5, ...",
            font_size=26
        )
        fit_width(teaser, SAFE.width * 0.92)
        teaser.next_to(subtitle, DOWN, buff=0.35)

        teaser2 = Text(
            "At each level, the controller checks if a 'crash divisor' exists.",
            font_size=26
        )
        fit_width(teaser2, SAFE.width * 0.92)
        teaser2.next_to(teaser, DOWN, buff=0.18)

        note = Text(
            "In this illustration, the math check is instant.",
            font_size=24
        ).set_opacity(0.75)
        fit_width(note, SAFE.width * 0.92)
        note.next_to(teaser2, DOWN, buff=0.22)

        self.play(FadeIn(teaser, shift=DOWN * 0.1), run_time=rt(0.40))
        self.play(FadeIn(teaser2, shift=DOWN * 0.1), run_time=rt(0.40))
        self.play(FadeIn(note, shift=DOWN * 0.1), run_time=rt(0.35))

        self.wait(1.0)

        self.play(FadeOut(teaser), FadeOut(teaser2), FadeOut(note), run_time=rt(0.35))

        # ----------------------------
        # LEFT: Controller UI (fixed anchors, no overlap)
        # ----------------------------
        left_top = LEFT_COL.get_top()[1] - 0.55

        n_tracker = ValueTracker(n_start)

        left_header = Text("CONTROLLER", font_size=30).set_opacity(0.90)
        left_header.move_to([LEFT_COL.get_center()[0], left_top, 0])

        n_readout = always_redraw(
            lambda: MathTex(r"n =", str(int(n_tracker.get_value())), font_size=64)
        )
        n_readout.next_to(left_header, DOWN, buff=0.28)
        n_readout.align_to(LEFT_COL, LEFT).shift(RIGHT * 0.35)

        mode_label = Text("mode:", font_size=26).set_opacity(0.85)
        mode_value = Text("POUR", font_size=28).set_color(BLUE)

        mode_group = VGroup(mode_label, mode_value).arrange(RIGHT, buff=0.20)
        mode_group.next_to(n_readout, DOWN, buff=0.28)
        mode_group.align_to(LEFT_COL, LEFT).shift(RIGHT * 0.40)

        status_panel = RoundedRectangle(corner_radius=0.18, width=LEFT_COL.width * 0.95, height=2.25)
        status_panel.set_stroke(width=2, opacity=0.70).set_fill(opacity=0.04)
        status_panel.next_to(mode_group, DOWN, buff=0.35)
        status_panel.align_to(LEFT_COL, LEFT).shift(RIGHT * 0.20)

        status_line1 = Text("Waiting for a level…", font_size=30)
        status_line2 = Text("We only decide at whole-number heights.", font_size=22).set_opacity(0.85)
        fit_width(status_line1, status_panel.width * 0.92)
        fit_width(status_line2, status_panel.width * 0.92)

        status_text = VGroup(status_line1, status_line2).arrange(DOWN, buff=0.16)
        status_text.move_to(status_panel.get_center())

        # divisor HUD (bottom of left column)
        test_panel = RoundedRectangle(corner_radius=0.18, width=LEFT_COL.width * 0.95, height=1.75)
        test_panel.set_stroke(width=2, opacity=0.55).set_fill(opacity=0.03)
        test_panel.move_to([LEFT_COL.get_center()[0], LEFT_COL.get_bottom()[1] + 1.35, 0])

        test_title = Text("Instant check (shown as a flash)", font_size=20).set_opacity(0.70)
        test_title.next_to(test_panel.get_top(), DOWN, buff=0.16)

        d_label = Text("testing d =", font_size=22)
        d_value = Integer(2, font_size=26)
        checks_label = Text("checks =", font_size=22)
        checks_value = Integer(0, font_size=26)
        hud = VGroup(
            VGroup(d_label, d_value).arrange(RIGHT, buff=0.18),
            VGroup(checks_label, checks_value).arrange(RIGHT, buff=0.18),
        ).arrange(DOWN, buff=0.18)
        hud.move_to(test_panel.get_center()).shift(DOWN * 0.10)

        self.play(
            FadeIn(left_header, shift=DOWN * 0.10),
            FadeIn(n_readout, shift=DOWN * 0.10),
            FadeIn(mode_group, shift=DOWN * 0.10),
            FadeIn(status_panel),
            FadeIn(status_text),
            FadeIn(test_panel),
            FadeIn(test_title),
            FadeIn(hud),
            run_time=rt(0.55),
        )

        # ----------------------------
        # RIGHT: Tank (fixed frame + growing scale label)
        # ----------------------------
        tank_title = Text("TANK", font_size=34).set_opacity(0.90)

        tank_frame = RoundedRectangle(
            corner_radius=0.18,
            width=RIGHT_COL.width * 0.78,
            height=RIGHT_COL.height * 0.68
        )
        tank_frame.set_stroke(width=2, opacity=0.85).set_fill(opacity=0.03)
        tank_frame.move_to([RIGHT_COL.get_center()[0], RIGHT_COL.get_center()[1] - 0.25, 0])

        tank_title.next_to(tank_frame, UP, buff=0.22)

        # Scale readout (this is the “tank is growing” illusion)
        scale_tracker = ValueTracker(1.0)
        scale_text = always_redraw(
            lambda: Text(f"scale: ×{scale_tracker.get_value():.1f}", font_size=24).set_opacity(0.85)
        )
        scale_text.next_to(tank_title, RIGHT, buff=0.35)

        # fill tracker is normalized 0..1 across our known run
        fill_tracker = ValueTracker(0.0)

        inner = tank_frame.copy().scale(0.92)

        water_base_opacity = 0.52
        water_color = BLUE  # will become GREEN / RED by outcome
        flow_phase = 0.0

        water = always_redraw(lambda: Rectangle(
            width=inner.width,
            height=max(0.002, inner.height * fill_tracker.get_value()),
        ).set_stroke(width=0).set_fill(water_color, opacity=water_base_opacity)
          .align_to(inner, DOWN)
          .move_to(inner.get_bottom() + UP * (max(0.002, inner.height * fill_tracker.get_value()) / 2))
        )

        # add a subtle pulsing shimmer to suggest flow
        def shimmer_updater(dt):
            nonlocal flow_phase, water_base_opacity
            flow_phase += dt * 2.0  # speed of shimmer
            # small pulse in opacity
            water_base_opacity = 0.50 + 0.05 * np.sin(2 * np.pi * flow_phase)

        shimmer_driver = VMobject()
        shimmer_driver.add_updater(lambda m, dt: shimmer_updater(dt))
        self.add(shimmer_driver)

        level_label = always_redraw(
            lambda: Text(f"level: n = {int(n_tracker.get_value())}", font_size=24).set_opacity(0.90)
        )
        level_label.next_to(tank_frame, DOWN, buff=0.20)

        # a little "spout" indicator (visual cue for POUR/PAUSE)
        spout = Triangle().scale(0.12).rotate(-PI/2)
        spout.set_fill(BLUE, opacity=1).set_stroke(width=0)
        spout.move_to(tank_frame.get_top() + LEFT * (tank_frame.width * 0.22) + UP * 0.05)

        spout_label = Text("flow", font_size=18).set_opacity(0.70)
        spout_label.next_to(spout, UP, buff=0.10)

        self.play(
            FadeIn(tank_frame),
            FadeIn(tank_title, shift=DOWN * 0.10),
            FadeIn(scale_text, shift=DOWN * 0.10),
            FadeIn(water),
            FadeIn(level_label),
            FadeIn(spout),
            FadeIn(spout_label),
            run_time=rt(0.60),
        )

        self.wait(0.75)

        # ----------------------------
        # helper: update status text cleanly
        # ----------------------------
        def set_status(line1: str, line2: str, color=WHITE, opacity2=0.85):
            nonlocal status_text
            a = Text(line1, font_size=30).set_color(color)
            b = Text(line2, font_size=22).set_opacity(opacity2).set_color(color)
            fit_width(a, status_panel.width * 0.92)
            fit_width(b, status_panel.width * 0.92)
            new = VGroup(a, b).arrange(DOWN, buff=0.16).move_to(status_panel.get_center())
            self.play(ReplacementTransform(status_text, new), run_time=rt(0.30))
            status_text = new

        # helper: mode switch (POUR/PAUSE)
        def set_mode(text: str, color):
            nonlocal mode_value
            new_mode = Text(text, font_size=28).set_color(color)
            self.play(ReplacementTransform(mode_value, new_mode), run_time=rt(0.18))
            mode_value = new_mode
            mode_group[1] = mode_value

            # spout color indicates flow state
            self.play(spout.animate.set_fill(color, opacity=1), run_time=rt(0.12))

        # ----------------------------
        # MAIN LOOP (pour to each integer, then pause + decide)
        # ----------------------------
        primes_shown = 0
        prev_n = n_start - 1  # so first pour feels like "arriving at 2"
        for n in numbers:
            # scale grows slowly (illusion of larger world)
            # (gentle ramp, not explosive)
            target_scale = 1.0 + 0.06 * (n - n_start)
            self.play(scale_tracker.animate.set_value(target_scale), run_time=rt(0.22))

            # --- POUR from prev_n to n ---
            set_mode("POUR", BLUE)
            set_status("Pouring…", f"Rising toward the next whole number: {n}", color=BLUE)

            self.play(n_tracker.animate.set_value(n), run_time=rt(0.25))

            # normalized fill (steady growth across entire run)
            t = (n - n_start) / max(1, (n_end - n_start))
            self.play(fill_tracker.animate.set_value(t), run_time=rt(0.38))

            # --- ARRIVE at integer level -> PAUSE (sync & label) ---
            set_mode("PAUSE", YELLOW)
            set_status(
                "Arrived at a whole-number level.",
                "We pause here to synchronize and label the level.",
                color=YELLOW
            )

            # show the divisor check as a quick flash (math is "instant" but visible)
            d_hit, limit = first_forbidden_divisor(n)
            checks_value.set_value(0)
            d_value.set_value(2)

            # quick sweep end
            sweep_end = 2 if n < 4 else (limit if d_hit is None else d_hit)
            if sweep_end >= 2:
                for i, d in enumerate(range(2, sweep_end + 1), start=1):
                    self.play(
                        d_value.animate.set_value(d),
                        checks_value.animate.set_value(i),
                        run_time=rt(0.06),
                    )

            # decide outcome (color + pause duration)
            if d_hit is None:
                primes_shown += 1
                water_color = GREEN
                set_status(
                    "PRIME — clear sailing.",
                    f"No divisor exits between 2 and ⌊√n⌋ (checked up to {limit}).",
                    color=GREEN
                )
                self.wait(PRIME_PAUSE)  # prime pauses longer
            else:
                water_color = RED
                set_status(
                    "COMPOSITE — crash detected.",
                    f"Found an exit: d = {d_hit}  because  {n} mod {d_hit} = 0.",
                    color=RED
                )
                self.wait(NONPRIME_PAUSE)  # non-prime pauses too (shorter)

            prev_n = n

        # ----------------------------
        # Outro (locks in the intuition)
        # ----------------------------
        outro = Text("Primes are the levels where no new divisors appear\nbetween 1 and n.", font_size=28)
        fit_width(outro, SAFE.width * 0.92)
        outro.move_to([SAFE.get_center()[0], SAFE.get_bottom()[1] + 0.95, 0])

        self.play(FadeIn(outro, shift=UP * 0.15), run_time=rt(0.55))
        self.wait(2.0)

