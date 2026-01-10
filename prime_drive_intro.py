from manim import *
import math
import numpy as np

# ~20% faster than 16.0
TIME_SCALE = 13.3


def first_exit_divisor(n: int):
    """Return (d_hit, limit). If prime: d_hit=None."""
    if n < 2:
        return None, 1
    limit = int(math.isqrt(n))
    for d in range(2, limit + 1):
        if n % d == 0:
            return d, limit
    return None, limit


class PrimeDriveIntro(Scene):
    def construct(self):
        safe = Rectangle(width=13.2, height=7.2).set_stroke(opacity=0).move_to(ORIGIN)

        # -----------------------------
        # LAYOUT BANDS (guardrails)
        # -----------------------------
        ROAD_Y = -2.55

        YOU_BAND_DEPTH = 0.60
        YOU_Y = ROAD_Y - YOU_BAND_DEPTH

        MAP_Y_BOTTOM = ROAD_Y + 0.12
        MAP_Y_TOP = MAP_Y_BOTTOM + 0.95

        road_left = safe.get_left()[0] + 1.0
        road_right = safe.get_right()[0] - 1.0

        # -----------------------------
        # TITLE + SUBTITLE (extra gap)
        # -----------------------------
        title_y = 3.35
        subtitle_y = 2.62

        title = Text("Prime Roads", font_size=60).move_to([0, title_y, 0])
        subtitle = Text(
            "A number is prime when you reach the finish with no exits.",
            font_size=28,
        ).move_to([0, subtitle_y, 0])

        self.play(
            FadeIn(title, shift=DOWN * 0.15),
            FadeIn(subtitle, shift=DOWN * 0.15),
            run_time=0.55 * TIME_SCALE,
        )
        self.wait(0.30 * TIME_SCALE)

        # -----------------------------
        # HUD (two-line narration)
        # -----------------------------
        hud_y = 1.65
        hud = VGroup().move_to([0, hud_y, 0])

        def set_hud(line1: str, line2: str = "", color=WHITE, rt=0.35):
            nonlocal hud
            new = VGroup(
                Text(line1, font_size=34),
                Text(line2, font_size=26).set_opacity(0.90) if line2 else Text("", font_size=1).set_opacity(0),
            ).arrange(DOWN, buff=0.16)
            new.set_color(color)
            new.move_to([0, hud_y, 0])

            if len(hud) == 0:
                hud = new
                self.play(FadeIn(hud), run_time=rt * TIME_SCALE)
            else:
                self.play(ReplacementTransform(hud, new), run_time=rt * TIME_SCALE)
                hud = new

        # -----------------------------
        # ROAD (looks like a real road)
        # -----------------------------
        ROAD_WIDTH = 0.55
        LANE_DASH_LEN = 0.35
        LANE_DASH_GAP = 0.30

        road_body = Rectangle(
            width=(road_right - road_left),
            height=ROAD_WIDTH,
        ).set_fill(GREY_B, opacity=0.95).set_stroke(width=0)
        road_body.move_to([(road_left + road_right) / 2, ROAD_Y, 0])

        dash_y = ROAD_Y
        dashes = VGroup()
        x = road_left + LANE_DASH_LEN / 2
        while x < road_right:
            dash = Line(
                [x - LANE_DASH_LEN / 2, dash_y, 0],
                [x + LANE_DASH_LEN / 2, dash_y, 0],
                stroke_width=4,
            ).set_color(WHITE).set_opacity(0.90)
            dashes.add(dash)
            x += LANE_DASH_LEN + LANE_DASH_GAP

        edge_top = Line(
            [road_left, ROAD_Y + ROAD_WIDTH / 2, 0],
            [road_right, ROAD_Y + ROAD_WIDTH / 2, 0],
            stroke_width=2,
        ).set_color(GREY_C).set_opacity(0.60)

        edge_bottom = Line(
            [road_left, ROAD_Y - ROAD_WIDTH / 2, 0],
            [road_right, ROAD_Y - ROAD_WIDTH / 2, 0],
            stroke_width=2,
        ).set_color(GREY_C).set_opacity(0.60)

        road = VGroup(road_body, dashes, edge_top, edge_bottom)

        self.play(FadeIn(road), run_time=0.6 * TIME_SCALE)

        # Road caption UNDER road (moved DOWN a bit more)
        road_caption = Text("Map area (1 → n)", font_size=22).set_opacity(0.70)
        road_caption.next_to(road_body, DOWN, buff=0.18)
        road_caption.align_to(road_body, LEFT)
        road_caption.shift(RIGHT * 0.70)
        road_caption.set_y(ROAD_Y - ROAD_WIDTH / 2 - 0.22)  # <- lower than before

        self.play(FadeIn(road_caption), run_time=0.25 * TIME_SCALE)
        self.wait(0.15 * TIME_SCALE)

        # -----------------------------
        # TEST STRIP for d-markers (solves clustering)
        # -----------------------------
        # Lift the strip a bit, then put the caption above it with extra spacing.
        test_strip_y = MAP_Y_BOTTOM + 0.30
        test_left = road_left + 1.15
        test_right = road_left + 4.45  # width of test strip

        test_strip = Line([test_left, test_strip_y, 0], [test_right, test_strip_y, 0], stroke_width=5).set_opacity(0.35)

        test_caption = Text("Test strip (d = 2 → ⌊√n⌋)", font_size=18).set_opacity(0.60)
        # Caption ABOVE strip, pushed up (“two lines above”) so d-labels have room
        test_caption.next_to(test_strip, UP, buff=0.34).align_to(test_strip, LEFT)

        self.play(FadeIn(test_strip), FadeIn(test_caption), run_time=0.35 * TIME_SCALE)
        self.wait(0.15 * TIME_SCALE)

        # -----------------------------
        # MINI CAR (procedural icon)
        # -----------------------------
        def make_car_icon():
            car_body = RoundedRectangle(width=0.42, height=0.20, corner_radius=0.06)\
                .set_fill(BLUE_E, opacity=1.0).set_stroke(width=0)

            car_roof = RoundedRectangle(width=0.26, height=0.12, corner_radius=0.05)\
                .set_fill(BLUE_D, opacity=1.0).set_stroke(width=0)
            car_roof.next_to(car_body, UP, buff=-0.02)

            wheel_r = 0.045
            wheel_left = Circle(radius=wheel_r).set_fill(GREY_D, opacity=1).set_stroke(width=0)
            wheel_right = wheel_left.copy()

            wheel_left.move_to(car_body.get_bottom() + LEFT * 0.14 + UP * 0.01)
            wheel_right.move_to(car_body.get_bottom() + RIGHT * 0.14 + UP * 0.01)

            return VGroup(car_body, car_roof, wheel_left, wheel_right)

        car_icon = make_car_icon().move_to([road_left, ROAD_Y, 0])

        car_label = Text("car", font_size=20).set_opacity(0.85)
        car_label.next_to(car_icon, DOWN, buff=0.10)
        car_label.set_y(YOU_Y)  # hard clamp

        car_group = VGroup(car_icon, car_label)

        self.play(FadeIn(car_group, shift=UP * 0.10), run_time=0.35 * TIME_SCALE)
        self.wait(0.20 * TIME_SCALE)

        def clamp_car_label():
            car_label.set_y(YOU_Y)

        # -----------------------------
        # Helper: mapping 1..n to road x-position
        # -----------------------------
        def x_for_road_progress(t: float):
            t = max(0.0, min(1.0, t))
            margin_band = 0.86
            t = (1 - margin_band) / 2 + margin_band * t
            return road_left + t * (road_right - road_left)

        def finish_x_for_n(n: int):
            # Always render finish at far right (including n=2) for consistent storytelling.
            return x_for_road_progress(1.0)

        def x_for_divisor_on_main_road(d: int, n: int):
            # (d-1)/(n-1) maps 1->0, n->1
            if n <= 2:
                t = 0.0
            else:
                t = (d - 1) / (n - 1)
            return x_for_road_progress(t)

        def road_point_x(xv: float):
            return np.array([xv, ROAD_Y, 0])

        # -----------------------------
        # Overlap-avoid helpers (for map band objects)
        # -----------------------------
        def bbox(m):
            return (m.get_left()[0], m.get_right()[0], m.get_bottom()[1], m.get_top()[1])

        def intersects(a, b, pad=0.06):
            ax0, ax1, ay0, ay1 = bbox(a)
            bx0, bx1, by0, by1 = bbox(b)
            return not (ax1 < bx0 - pad or ax0 > bx1 + pad or ay1 < by0 - pad or ay0 > by1 + pad)

        def place_without_overlap(mob, obstacles, y_min, y_max, step=0.11, tries=16):
            mob.set_y(np.clip(mob.get_y(), y_min + 0.05, y_max - 0.05))

            offsets = [0.0]
            for k in range(1, tries + 1):
                offsets += [k * step, -k * step]

            x0 = mob.get_x()
            base_y = mob.get_y()

            for dy in offsets:
                mob.move_to([x0, np.clip(base_y + dy, y_min + 0.05, y_max - 0.05), 0])
                if all(not intersects(mob, obs) for obs in obstacles):
                    return mob

            mob.scale(0.92)
            return mob

        # -----------------------------
        # d MARKERS on TEST STRIP (no clustering)
        # -----------------------------
        def make_d_marker(d: int, limit: int):
            # map d in [2..limit] across the test strip
            if limit <= 2:
                t = 0.0
            else:
                t = (d - 2) / (limit - 2)

            x = test_left + t * (test_right - test_left)

            # Marker drops DOWN from the strip; label sits above the strip line.
            post = Line([x, test_strip_y, 0], [x, test_strip_y - 0.22, 0], stroke_width=4).set_opacity(0.55)
            label = Text(f"d = {d}", font_size=18).set_opacity(0.85)
            label.next_to(post, UP, buff=0.10)

            return VGroup(label, post)

        # -----------------------------
        # Finish line (ALWAYS far right)
        # -----------------------------
        def make_finish(n: int):
            x = finish_x_for_n(n)
            line = Line([x, ROAD_Y - 0.45, 0], [x, ROAD_Y + 0.45, 0], stroke_width=6)
            label = Text(f"finish: n = {n}", font_size=26).next_to(line, UP, buff=0.25)
            return VGroup(line, label)

        # Exit ramp: down-right from road at exit_x
        def make_exit_ramp(exit_x: float):
            return Line([exit_x, ROAD_Y, 0], [exit_x + 1.15, ROAD_Y - 0.95, 0], stroke_width=7).set_opacity(0.6)

        # -----------------------------
        # Checkered flag (procedural)
        # -----------------------------
        def make_checkered_flag():
            cell = 0.10
            cols, rows = 6, 4
            squares = VGroup()
            for r in range(rows):
                for c in range(cols):
                    sq = Square(side_length=cell).set_stroke(width=0).set_fill(opacity=1.0)
                    sq.set_opacity(0.95 if (r + c) % 2 == 0 else 0.25)
                    sq.move_to([c * cell, -r * cell, 0])
                    squares.add(sq)
            squares.center()
            pole = Line(UP * 0.35, DOWN * 0.35, stroke_width=6).set_opacity(0.65)
            return VGroup(squares, pole).arrange(RIGHT, buff=0.10)

        # -----------------------------
        # EXIT CALLOUT PANEL (light panel + dark text)
        # -----------------------------
        callout_anchor = np.array([safe.get_right()[0] - 2.3, 0.10, 0])
        callout_box = RoundedRectangle(width=4.7, height=1.75, corner_radius=0.18)\
            .set_stroke(GREY_C, opacity=0.65)\
            .set_fill(WHITE, opacity=0.92)
        callout_box.move_to(callout_anchor)

        callout_title = Text("Exit", font_size=26, color=BLACK)
        callout_title.move_to(callout_box.get_top() + DOWN * 0.30)

        callout_text = VGroup(Text("", font_size=1)).move_to(callout_box.get_center())
        exit_callout = VGroup(callout_box, callout_title, callout_text).set_opacity(0.0)
        self.add(exit_callout)

        def show_exit_callout(n: int, d: int):
            nonlocal callout_text
            new_text = VGroup(
                Text(f"d = {d}", font_size=26, color=BLACK),
                Text(f"{n} mod {d} = 0", font_size=22, color=BLACK).set_opacity(0.90),
                Text(f"A divisor exists between 1 and {n}.", font_size=20, color=BLACK).set_opacity(0.85),
            ).arrange(DOWN, buff=0.14)

            # Slightly lower so "Exit" header never crowds first line
            new_text.move_to(callout_box.get_center() + DOWN * 0.32)

            self.play(exit_callout.animate.set_opacity(1.0), run_time=0.18 * TIME_SCALE)
            self.play(ReplacementTransform(callout_text, new_text), run_time=0.22 * TIME_SCALE)
            callout_text = new_text

        def hide_exit_callout():
            self.play(exit_callout.animate.set_opacity(0.0), run_time=0.18 * TIME_SCALE)

        # -----------------------------
        # Utility: clear d markers
        # -----------------------------
        d_markers = VGroup()

        def clear_d_markers(rt=0.20):
            nonlocal d_markers
            if len(d_markers) > 0:
                self.play(FadeOut(d_markers), run_time=rt * TIME_SCALE)
                d_markers = VGroup()

        # -----------------------------
        # PROLOGUE: n = 2 (finish at end)
        # -----------------------------
        hide_exit_callout()
        clear_d_markers(rt=0.0)

        set_hud("Prologue: n = 2", "Special case: there are no exit tests yet.", rt=0.55)
        self.wait(0.25 * TIME_SCALE)

        n = 2
        finish = make_finish(n)
        self.play(FadeIn(finish), run_time=0.28 * TIME_SCALE)
        self.wait(0.18 * TIME_SCALE)

        set_hud("Testing exits (empty)", "d = 2 .. ⌊√2⌋ = 1  → no tests occur.", rt=0.55)
        self.wait(0.45 * TIME_SCALE)

        finish_x = finish_x_for_n(n)
        self.play(car_icon.animate.move_to(road_point_x(finish_x)), run_time=0.85 * TIME_SCALE)
        clamp_car_label()
        self.wait(0.15 * TIME_SCALE)

        flag = make_checkered_flag().scale(1.6)
        flag.next_to(finish[0], UP, buff=0.25)
        flag.set_y(min(flag.get_y(), MAP_Y_TOP - 0.08))
        place_without_overlap(flag, obstacles=[title, subtitle, hud, finish[1]], y_min=MAP_Y_BOTTOM, y_max=MAP_Y_TOP)

        self.play(FadeIn(flag, shift=UP * 0.08), run_time=0.33 * TIME_SCALE)

        set_hud("Prime (by default here)", "You reach the finish; there were no exits to test.", color=GREEN, rt=0.55)
        self.wait(0.85 * TIME_SCALE)

        self.play(FadeOut(VGroup(finish, flag)), run_time=0.25 * TIME_SCALE)
        self.play(car_icon.animate.move_to(road_point_x(road_left)), run_time=0.65 * TIME_SCALE)
        clamp_car_label()
        self.wait(0.30 * TIME_SCALE)

        # -----------------------------
        # MAIN SHOW
        # -----------------------------
        prime_targets = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        max_n = prime_targets[-1]

        set_hud(
            "Now the real test begins (n ≥ 3)",
            "Exits can be anywhere on the road — but we only need to TEST d = 2..⌊√n⌋.",
            rt=0.65,
        )
        self.wait(0.70 * TIME_SCALE)

        set_hud(
            "Why the test strip is limited",
            "If n has any divisor at all, one divisor is always ≤ √n (so we’ll find an exit early).",
            rt=0.65,
        )
        self.wait(0.85 * TIME_SCALE)

        for n in range(3, max_n + 1):
            clear_d_markers(rt=0.15)
            hide_exit_callout()

            finish = make_finish(n)
            self.play(FadeIn(finish), run_time=0.20 * TIME_SCALE)

            set_hud(f"Road to n = {n}", "We start at 1 and try to reach the finish line.", rt=0.38)
            self.wait(0.28 * TIME_SCALE)

            d_hit, limit = first_exit_divisor(n)

            set_hud(
                f"Testing exits: d = 2 .. ⌊√{n}⌋ = {limit}",
                "If any test succeeds (remainder = 0), we take the exit early → composite.",
                rt=0.50,
            )
            self.wait(0.30 * TIME_SCALE)

            # show d-markers on the TEST STRIP (spread out by limit)
            if limit >= 2:
                for d in range(2, limit + 1):
                    m = make_d_marker(d, limit).set_opacity(0.35)
                    d_markers.add(m)

                self.play(FadeIn(d_markers), run_time=0.33 * TIME_SCALE)
                self.wait(0.20 * TIME_SCALE)

                for d in range(2, limit + 1):
                    marker = d_markers[d - 2]
                    self.play(marker.animate.set_opacity(1.0), run_time=0.14 * TIME_SCALE)

                    if n % d == 0:
                        set_hud(
                            f"Exit found at d = {d}",
                            f"{n} mod {d} = 0  → a divisor exists between 1 and {n}.",
                            color=RED,
                            rt=0.55,
                        )
                        self.wait(0.35 * TIME_SCALE)

                        # exit notch position ON THE MAIN ROAD (still proportional to d/n)
                        exit_x = x_for_divisor_on_main_road(d, n)

                        # notch on BOTTOM of road (avoids test strip region)
                        pole_height = 0.24
                        road_bottom = ROAD_Y - ROAD_WIDTH / 2

                        exit_pole = Line(
                            [exit_x, road_bottom, 0],
                            [exit_x, road_bottom - pole_height, 0],
                            stroke_width=5,
                        ).set_opacity(0.55)

                        exit_number = Text(str(d), font_size=20).set_opacity(0.90)
                        exit_number.next_to(exit_pole, RIGHT, buff=0.08)
                        exit_number.set_y(max(exit_number.get_y(), YOU_Y + 0.10))

                        self.play(
                            Create(exit_pole),
                            FadeIn(exit_number, shift=DOWN * 0.05),
                            run_time=0.30 * TIME_SCALE,
                        )

                        show_exit_callout(n, d)

                        ramp = make_exit_ramp(exit_x)
                        self.play(Create(ramp), run_time=0.30 * TIME_SCALE)

                        # car hugs road to exit point, then exits
                        self.play(car_icon.animate.move_to(road_point_x(exit_x)), run_time=0.80 * TIME_SCALE)
                        clamp_car_label()
                        self.wait(0.18 * TIME_SCALE)

                        self.play(car_icon.animate.move_to(ramp.get_end()), run_time=0.85 * TIME_SCALE)
                        clamp_car_label()

                        set_hud("Composite", "One exit is enough: you do NOT reach the finish flag.", color=RED, rt=0.55)
                        self.wait(0.85 * TIME_SCALE)

                        self.play(FadeOut(VGroup(ramp, exit_pole, exit_number, finish)), run_time=0.25 * TIME_SCALE)
                        self.play(car_icon.animate.move_to(road_point_x(road_left)), run_time=0.80 * TIME_SCALE)
                        clamp_car_label()
                        break
                    else:
                        set_hud(
                            f"No exit at d = {d}",
                            f"{n} mod {d} ≠ 0  (there is a remainder)",
                            rt=0.38,
                        )
                        self.wait(0.28 * TIME_SCALE)
                        self.play(marker.animate.set_opacity(0.45), run_time=0.10 * TIME_SCALE)

            if d_hit is None:
                set_hud(
                    "No exits appeared",
                    f"So there is NO divisor between 1 and {n} — the road stays clear.",
                    color=GREEN,
                    rt=0.55,
                )
                self.wait(0.30 * TIME_SCALE)

                finish_x = finish_x_for_n(n)
                self.play(car_icon.animate.move_to(road_point_x(finish_x)), run_time=0.98 * TIME_SCALE)
                clamp_car_label()

                flag = make_checkered_flag().scale(1.6)
                flag.next_to(finish[0], UP, buff=0.25)
                flag.set_y(min(flag.get_y(), MAP_Y_TOP - 0.08))
                place_without_overlap(
                    flag,
                    obstacles=[title, subtitle, hud, finish[1], road_caption, test_caption] + list(d_markers),
                    y_min=MAP_Y_BOTTOM,
                    y_max=MAP_Y_TOP,
                )

                self.play(FadeIn(flag, shift=UP * 0.08), run_time=0.35 * TIME_SCALE)

                set_hud(
                    "Prime",
                    "You only know it’s prime when you reach the finish with no exits.",
                    color=GREEN,
                    rt=0.55,
                )
                self.wait(1.45 * TIME_SCALE)

                if n in prime_targets:
                    stamp = Text("✓ prime finish", font_size=34).set_color(GREEN)
                    stamp.next_to(finish, DOWN, buff=0.45)
                    self.play(FadeIn(stamp, shift=UP * 0.12), run_time=0.35 * TIME_SCALE)
                    self.wait(0.60 * TIME_SCALE)
                    self.play(FadeOut(stamp), run_time=0.25 * TIME_SCALE)

                self.play(FadeOut(VGroup(finish, flag)), run_time=0.25 * TIME_SCALE)
                self.play(car_icon.animate.move_to(road_point_x(road_left)), run_time=0.85 * TIME_SCALE)
                clamp_car_label()
                self.wait(0.30 * TIME_SCALE)

        # -----------------------------
        # Ending definition (no HUD overlap)
        # -----------------------------
        clear_d_markers(rt=0.20)
        hide_exit_callout()

        set_hud(
            "Rigorous definition (with the story intact)",
            "Now we say the same idea in one clean card.",
            rt=0.65,
        )
        self.wait(0.55 * TIME_SCALE)

        self.play(FadeOut(hud), run_time=0.35 * TIME_SCALE)
        hud = VGroup()

        ending = VGroup(
            Text("In the road story:", font_size=32),
            Text("Start = 1 and Finish = n are always there.", font_size=28),
            Text("Exits can exist anywhere (any divisor 1 < d < n).", font_size=28),
            Text("But we only test d = 2 .. ⌊√n⌋ (a composite always has a small divisor).", font_size=28),
            Text("Prime = you reach the flag with no exits.", font_size=30).set_color(GREEN),
        ).arrange(DOWN, buff=0.18)

        ending.move_to([0, 1.10, 0])

        self.play(FadeIn(ending, shift=DOWN * 0.10), run_time=0.60 * TIME_SCALE)
        self.wait(1.55 * TIME_SCALE)

        # -----------------------------
        # Final PRIME victory card (big flag + thanks)
        # -----------------------------
        self.wait(0.45 * TIME_SCALE)
        self.play(FadeOut(ending), run_time=0.45 * TIME_SCALE)

        final_flag = make_checkered_flag().scale(3.2)
        final_flag.move_to([0, 0.75, 0])

        thank_you = Text("Thank you for watching!", font_size=44).set_color(GREEN)
        signature = Text("- ONOJK123", font_size=28).set_opacity(0.85)

        text_block = VGroup(thank_you, signature).arrange(DOWN, buff=0.28)
        text_block.next_to(final_flag, DOWN, buff=0.45)

        self.play(FadeIn(final_flag, shift=UP * 0.15), run_time=0.55 * TIME_SCALE)
        self.wait(0.22 * TIME_SCALE)
        self.play(FadeIn(thank_you, shift=UP * 0.10), run_time=0.40 * TIME_SCALE)
        self.wait(0.18 * TIME_SCALE)
        self.play(FadeIn(signature, shift=UP * 0.08), run_time=0.35 * TIME_SCALE)

        self.wait(2.10 * TIME_SCALE)

