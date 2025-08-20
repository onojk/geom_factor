import math
def geo_score_for_x(N, x, theta=math.tau/7):
    A = int(math.isqrt(N)) + x
    Q = A*A - N
    angle = (theta*A) % math.tau
    resonance = min(angle, math.tau - angle)
    return 1.0 / (1.0 + abs(Q)**0.2) + 0.1 * (1.0 - resonance / math.pi)
