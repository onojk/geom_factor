# prime_gaps_full_circle_8bit.py
#
# PRIME GAPS encoded around a full circle with vertical+horizontal symmetry,
# plus an INSCRIBED EQUILATERAL TRIANGLE whose vertices display (x,y),
# AND an "event" system when triangle vertices pass over prime instances:
#   - 1 vertex on prime (single)
#   - 2 vertices on primes simultaneously (double)
#   - 3 vertices on primes simultaneously (triple)
#
# Controls:
#   ESC   quit
#   P     toggle showing prime dots on base circle
#   L     toggle gap labels (quadrant 0)
#   G     toggle ghost ticks for all integers in bucket
#   T     toggle triangle
#   LEFT/RIGHT rotate triangle (hold)
#   UP/DN increase/decrease gap scale
#   S/A   more/less smoothing
#   [ ]   label density
#   +/-   change dot size
#   ,/.   decrease/increase HIT tolerance (degrees)
#   R     reset

import math
import bisect
import pygame

W, H = 980, 780
FPS = 60

BG = (8, 8, 12)
WHITE = (235, 235, 245)
SUB = (160, 160, 180)

CIRCLE = (60, 60, 80)
AXIS = (50, 50, 70)
ARC = (120, 140, 220)

PRIME_DOT = (80, 255, 120)
GHOST_TICK = (255, 180, 80)
TICK_ALPHA = 55

GAP_LINE = (170, 140, 255)
GAP_NODE = (255, 230, 90)

TRI_EDGE = (120, 220, 255)
TRI_VERT = (255, 230, 90)
TRI_HIT  = (255, 160, 90)   # vertex highlight when "on prime"

# Bucket
START, END = 128, 255
DEN = END - START  # 127


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


def primes_in_bucket(a: int, b: int):
    return [n for n in range(a, b + 1) if is_prime(n)]


def theta_for(n: int) -> float:
    # quarter arc mapping: START -> 0, END -> pi/2
    t = (n - START) / DEN
    t = max(0.0, min(1.0, t))
    return t * (math.pi / 2.0)


def point(cx, cy, r, theta):
    x = cx + r * math.cos(theta)
    y = cy - r * math.sin(theta)
    return int(x), int(y)


def smooth_series(vals, passes=2):
    if len(vals) < 3 or passes <= 0:
        return vals[:]
    out = vals[:]
    for _ in range(passes):
        tmp = out[:]
        for i in range(1, len(out) - 1):
            tmp[i] = 0.25 * out[i - 1] + 0.5 * out[i] + 0.25 * out[i + 1]
        out = tmp
    return out


def rotate_point_about(cx, cy, px, py, ang):
    dx = px - cx
    dy = py - cy
    ca = math.cos(ang)
    sa = math.sin(ang)
    rx = dx * ca - dy * sa
    ry = dx * sa + dy * ca
    return int(cx + rx), int(cy + ry)


def draw_vertex_label(screen, font, x, y, ox=10, oy=-18, color=(235, 235, 245)):
    txt = font.render(f"({x}, {y})", True, color)
    screen.blit(txt, (x + ox, y + oy))


def norm_angle(a: float) -> float:
    a = a % (2.0 * math.pi)
    return a


def circ_dist(a: float, b: float) -> float:
    """Smallest circular distance between angles a and b (radians)."""
    d = abs(a - b) % (2.0 * math.pi)
    return min(d, 2.0 * math.pi - d)


def nearest_angle_distance(sorted_angles, a: float) -> float:
    """Return distance (radians) from angle a to nearest in sorted_angles."""
    if not sorted_angles:
        return 9e9
    a = norm_angle(a)
    i = bisect.bisect_left(sorted_angles, a)
    candidates = []
    if 0 <= i < len(sorted_angles):
        candidates.append(sorted_angles[i])
    if i - 1 >= 0:
        candidates.append(sorted_angles[i - 1])
    # wrap neighbors
    candidates.append(sorted_angles[0])
    candidates.append(sorted_angles[-1])
    return min(circ_dist(a, c) for c in candidates)


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("8-bit bucket PRIME GAPS on full circle + triangle events")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont("dejavusansmono", 22, True)
    font = pygame.font.SysFont("dejavusansmono", 16)
    font_s = pygame.font.SysFont("dejavusansmono", 14)

    cx, cy = W // 2, int(H * 0.62)
    R = 220

    primes = primes_in_bucket(START, END)

    # Gap series
    P = primes[:]
    gaps = [P[i + 1] - P[i] for i in range(len(P) - 1)] if len(P) >= 2 else []
    primes_for_gaps = P[:-1]
    thetas = [theta_for(p) for p in primes_for_gaps]

    # Precompute prime angles for FULL circle (mirrored quadrants)
    # Each prime p gives base theta in [0, pi/2], then add q*pi/2 for q=0..3
    prime_angles = []
    for p in primes:
        base = theta_for(p)
        for q in range(4):
            prime_angles.append(norm_angle(base + q * (math.pi / 2.0)))
    prime_angles.sort()

    show_prime_dots = True
    show_labels = True
    show_ghost = False

    show_triangle = True
    tri_phi = 0.0
    tri_step = math.radians(2.0)  # per frame while holding

    dot_r = 4
    label_stride = 3

    gap_scale = 10.0
    smooth_passes = 2

    # Prime-hit tolerance in degrees -> radians
    hit_tol_deg = 1.2
    def hit_tol_rad():
        return math.radians(hit_tol_deg)

    # Event state
    prev_hit = [False, False, False]
    event_timer = 0
    event_text = ""
    event_color = WHITE

    ghost_surf = pygame.Surface((W, H), pygame.SRCALPHA)

    gaps_used = []
    rads = []
    pts_quadrant = []

    def rebuild_curve():
        nonlocal gaps_used, rads, pts_quadrant
        if not gaps:
            gaps_used = []
            rads = []
            pts_quadrant = []
            return

        gaps_used = gaps[:]
        mean_gap = sum(gaps_used) / len(gaps_used)
        centered = [g - mean_gap for g in gaps_used]
        centered = smooth_series(centered, passes=smooth_passes)

        rads = [R + gap_scale * c for c in centered]
        pts_quadrant = [point(cx, cy, r, th) for r, th in zip(rads, thetas)]

    rebuild_curve()

    def trigger_event(txt, col):
        nonlocal event_timer, event_text, event_color
        event_timer = 45  # frames
        event_text = txt
        event_color = col

    def triangle_vertex_angles():
        # equilateral triangle
        a0 = tri_phi
        a1 = tri_phi + 2.0 * math.pi / 3.0
        a2 = tri_phi + 4.0 * math.pi / 3.0
        return [norm_angle(a0), norm_angle(a1), norm_angle(a2)]

    def draw_triangle(hit_flags):
        angs = triangle_vertex_angles()
        v0 = point(cx, cy, R, angs[0])
        v1 = point(cx, cy, R, angs[1])
        v2 = point(cx, cy, R, angs[2])

        pygame.draw.polygon(screen, TRI_EDGE, [v0, v1, v2], width=3)

        # vertices (highlight if hitting prime)
        cols = [TRI_HIT if hit_flags[i] else TRI_VERT for i in range(3)]
        pygame.draw.circle(screen, cols[0], v0, 7)
        pygame.draw.circle(screen, cols[1], v1, 7)
        pygame.draw.circle(screen, cols[2], v2, 7)

        # center mark
        pygame.draw.circle(screen, (90, 90, 120), (cx, cy), 3)

        # coordinate labels
        draw_vertex_label(screen, font_s, v0[0], v0[1])
        draw_vertex_label(screen, font_s, v1[0], v1[1])
        draw_vertex_label(screen, font_s, v2[0], v2[1])

    running = True
    while running:
        clock.tick(FPS)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_p:
                    show_prime_dots = not show_prime_dots
                elif e.key == pygame.K_l:
                    show_labels = not show_labels
                elif e.key == pygame.K_g:
                    show_ghost = not show_ghost
                elif e.key == pygame.K_t:
                    show_triangle = not show_triangle
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    dot_r = min(14, dot_r + 1)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    dot_r = max(1, dot_r - 1)
                elif e.key == pygame.K_LEFTBRACKET:
                    label_stride = min(12, label_stride + 1)
                elif e.key == pygame.K_RIGHTBRACKET:
                    label_stride = max(1, label_stride - 1)
                elif e.key == pygame.K_UP:
                    gap_scale = min(60.0, gap_scale + 1.0)
                    rebuild_curve()
                elif e.key == pygame.K_DOWN:
                    gap_scale = max(0.0, gap_scale - 1.0)
                    rebuild_curve()
                elif e.key == pygame.K_s:
                    smooth_passes = min(10, smooth_passes + 1)
                    rebuild_curve()
                elif e.key == pygame.K_a:
                    smooth_passes = max(0, smooth_passes - 1)
                    rebuild_curve()
                elif e.key == pygame.K_COMMA:  # ,
                    hit_tol_deg = max(0.2, hit_tol_deg - 0.2)
                elif e.key == pygame.K_PERIOD:  # .
                    hit_tol_deg = min(10.0, hit_tol_deg + 0.2)
                elif e.key == pygame.K_r:
                    dot_r = 4
                    label_stride = 3
                    gap_scale = 10.0
                    smooth_passes = 2
                    show_prime_dots = True
                    show_labels = True
                    show_ghost = False
                    show_triangle = True
                    tri_phi = 0.0
                    hit_tol_deg = 1.2
                    prev_hit = [False, False, False]
                    event_timer = 0
                    rebuild_curve()

        # Hold-to-rotate using LEFT/RIGHT
        keys = pygame.key.get_pressed()
        if show_triangle:
            if keys[pygame.K_LEFT]:
                tri_phi -= tri_step
            if keys[pygame.K_RIGHT]:
                tri_phi += tri_step

        # ---- hit detection + event triggering ----
        hit_flags = [False, False, False]
        if show_triangle and prime_angles:
            angs = triangle_vertex_angles()
            tol = hit_tol_rad()
            for i in range(3):
                d = nearest_angle_distance(prime_angles, angs[i])
                hit_flags[i] = (d <= tol)

            # edge-trigger: only trigger when entering a hit state
            newly = [hit_flags[i] and not prev_hit[i] for i in range(3)]
            count_hit = sum(1 for b in hit_flags if b)
            count_new = sum(1 for b in newly if b)

            if count_new > 0:
                if count_hit == 3:
                    trigger_event("EVENT: TRIPLE PRIME ALIGNMENT (3 vertices)", (255, 230, 90))
                elif count_hit == 2:
                    trigger_event("EVENT: DOUBLE PRIME ALIGNMENT (2 vertices)", (180, 220, 255))
                elif count_hit == 1:
                    idx = newly.index(True) + 1  # first newly-hit vertex
                    trigger_event(f"EVENT: VERTEX {idx} HIT PRIME", (120, 255, 170))

            prev_hit = hit_flags[:]

        if event_timer > 0:
            event_timer -= 1

        # ---- drawing ----
        screen.fill(BG)

        # Base circle
        pygame.draw.circle(screen, CIRCLE, (cx, cy), R, 2)

        # Axes
        pygame.draw.line(screen, AXIS, (cx - R - 80, cy), (cx + R + 80, cy), 2)
        pygame.draw.line(screen, AXIS, (cx, cy - R - 80), (cx, cy + R + 80), 2)

        # Quadrant arc guides
        rect = pygame.Rect(cx - R, cy - R, 2 * R, 2 * R)
        for q in range(4):
            a0 = q * (math.pi / 2.0)
            a1 = a0 + (math.pi / 2.0)
            pygame.draw.arc(screen, ARC, rect, a0, a1, 3)

        # Ghost ticks (all integers in bucket)
        if show_ghost:
            ghost_surf.fill((0, 0, 0, 0))
            for n in range(START, END + 1):
                th = theta_for(n)
                for q in range(4):
                    thq = th + q * (math.pi / 2.0)
                    x1, y1 = point(cx, cy, R - 6, thq)
                    x2, y2 = point(cx, cy, R + 6, thq)
                    pygame.draw.line(
                        ghost_surf,
                        (*GHOST_TICK, TICK_ALPHA),
                        (x1, y1),
                        (x2, y2),
                        1,
                    )
            screen.blit(ghost_surf, (0, 0))

        # Prime dots on base circle
        if show_prime_dots:
            for p in primes:
                th = theta_for(p)
                for q in range(4):
                    thq = th + q * (math.pi / 2.0)
                    x, y = point(cx, cy, R, thq)
                    pygame.draw.circle(screen, PRIME_DOT, (x, y), max(2, dot_r - 1))

        # Gap curve
        if len(pts_quadrant) >= 2:
            for q in range(4):
                ang = q * (math.pi / 2.0)
                ptsq = [rotate_point_about(cx, cy, x, y, ang) for (x, y) in pts_quadrant]
                pygame.draw.lines(screen, GAP_LINE, False, ptsq, 3)

                for (x, y) in ptsq[::2]:
                    pygame.draw.circle(screen, GAP_NODE, (x, y), 2)

            # labels only on quadrant 0
            if show_labels:
                for i, (g, th) in enumerate(zip(gaps_used, thetas)):
                    if i % label_stride != 0:
                        continue
                    r_here = rads[i]
                    x, y = point(cx, cy, r_here + 16, th)
                    lab = font_s.render(f"{int(round(g))}", True, WHITE)
                    screen.blit(lab, (x + 4, y - 8))

        # Triangle overlay
        if show_triangle:
            draw_triangle(hit_flags)

        # HUD
        screen.blit(font_big.render("PRIME GAPS on full circle + inscribed triangle (events)", True, WHITE), (24, 18))
        screen.blit(font.render(f"bucket=[128..255] primes={len(primes)} gaps={len(gaps)}", True, SUB), (24, 48))
        screen.blit(font.render(f"gap_scale={gap_scale:.1f}px   smoothing={smooth_passes}   label_stride={label_stride}", True, SUB), (24, 70))
        screen.blit(font.render(f"prime-hit tolerance = {hit_tol_deg:.1f}°   ( , / . to adjust )", True, SUB), (24, 92))
        screen.blit(
            font.render(
                "P primes • L labels • G ghost • T triangle • LEFT/RIGHT rotate tri • UP/DN scale • S/A smooth • [ ] density • +/- dot • R reset • ESC quit",
                True,
                SUB,
            ),
            (24, 114),
        )

        # Event overlay
        if event_timer > 0:
            # little translucent banner
            banner = pygame.Surface((W, 44), pygame.SRCALPHA)
            banner.fill((0, 0, 0, 165))
            screen.blit(banner, (0, H - 54))
            screen.blit(font_big.render(event_text, True, event_color), (24, H - 46))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
