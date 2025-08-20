#!/usr/bin/env python3
"""
Minimal ECM (Lenstra) Stage-1 fallback
- Random short Weierstrass over Z/nZ
- Detects non-invertible steps via gcd to return a factor
- Args: B1 (stage-1), number of curves, seed, max_seconds
"""
from math import gcd
import random, time

def _build_M(B1: int) -> int:
    sieve = [True]*(B1+1)
    sieve[:2] = [False, False]
    for i in range(2, int(B1**0.5)+1):
        if sieve[i]:
            for j in range(i*i, B1+1, i):
                sieve[j] = False
    M = 1
    for p, is_p in enumerate(sieve):
        if not is_p:
            continue
        pk = p
        while pk * p <= B1:
            pk *= p
        M *= pk
    return M

def _inv(a, n):
    g = gcd(a, n)
    if g > 1:
        return -g
    return pow(a, -1, n)

def _ec_add(P, Q, a, n):
    if P is None: return Q, None
    if Q is None: return P, None
    x1, y1 = P; x2, y2 = Q
    if (x1 - x2) % n == 0 and (y1 + y2) % n == 0:
        return None, None
    if (x1, y1) == (x2, y2):
        num = (3 * x1 * x1 + a) % n
        den = (2 * y1) % n
    else:
        num = (y2 - y1) % n
        den = (x2 - x1) % n
    inv = _inv(den, n)
    if isinstance(inv, int) and inv < 0:
        return None, -inv
    lam = (num * inv) % n
    x3 = (lam * lam - x1 - x2) % n
    y3 = (lam * (x1 - x3) - y1) % n
    return (x3, y3), None

def _ec_mul(k, P, a, n):
    R = None
    Q = P
    while k > 0:
        if k & 1:
            R, g = _ec_add(R, Q, a, n)
            if g: return None, g
        k >>= 1
        if k:
            Q, g = _ec_add(Q, Q, a, n)
            if g: return None, g
    return R, None

def _random_curve(n, rng):
    while True:
        x = rng.randrange(2, n-1)
        y = rng.randrange(2, n-1)
        a = rng.randrange(2, n-1)
        b = (y*y - (x*x % n)*x - a*x) % n
        disc = (-16 * ((4*pow(a,3,n) + 27*pow(b,2,n)) % n)) % n
        g = gcd(disc, n)
        if 1 < g < n:
            return None, None, g
        return (a, b), (x, y), None

def ecm(n: int, B1: int = 50000, curves: int = 200, seed: int | None = None, max_seconds: float | None = None):
    if n % 2 == 0: return 2
    if n % 3 == 0: return 3
    if seed is None:
        seed = 0xEC11
    rng = random.Random(seed)
    M = _build_M(max(1000, int(B1)))
    t0 = time.time()
    for _ in range(curves):
        if max_seconds is not None and time.time() - t0 > max_seconds:
            return None
        curve, P, g = _random_curve(n, rng)
        if g: return g
        a, _b = curve
        _, g = _ec_mul(M, P, a, n)
        if g: return g
    return None

if __name__ == "__main__":
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 91
    f = ecm(N, B1=5000, curves=100, seed=1234, max_seconds=2.0)
    print(f or "None")
