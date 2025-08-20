#!/usr/bin/env python3
"""
Minimal ECM (Lenstra) Stage-1 fallback
- Random short Weierstrass curves over Z/nZ
- Detects non-invertible steps via gcd to return a factor
- Use B1 (stage-1 bound) and number of random curves
- Deterministic RNG via seed for reproducibility
"""
from math import gcd
import random

def _build_M(B1: int) -> int:
    # product of p^e for primes p<=B1, p^e<=B1
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
    # modular inverse; if doesn't exist, return gcd factor signal as negative
    g = gcd(a, n)
    if g > 1:
        return -g
    try:
        return pow(a, -1, n)
    except ValueError:
        # Fallback if Python doesn't support pow(...,-1, n)
        # (Shouldn't happen on 3.8+, but keep safe.)
        # Use extended Euclid
        t, newt, r, newr = 0, 1, n, a % n
        while newr != 0:
            q = r // newr
            t, newt = newt, t - q * newt
            r, newr = newr, r - q * newr
        if r > 1:
            return -r
        if t < 0:
            t += n
        return t

def _ec_add(P, Q, a, n):
    # Short Weierstrass y^2 = x^3 + a x + b mod n
    # Points P=(x1,y1), Q=(x2,y2), None = infinity
    if P is None:
        return Q, None
    if Q is None:
        return P, None
    x1, y1 = P
    x2, y2 = Q
    if (x1 - x2) % n == 0 and (y1 + y2) % n == 0:
        return None, None  # P + (-P) = O
    if (x1, y1) == (x2, y2):
        num = (3 * x1 * x1 + a) % n
        den = (2 * y1) % n
    else:
        num = (y2 - y1) % n
        den = (x2 - x1) % n
    inv = _inv(den, n)
    if isinstance(inv, int) and inv < 0:
        return None, -inv  # return factor via "hint"
    lam = (num * inv) % n
    x3 = (lam * lam - x1 - x2) % n
    y3 = (lam * (x1 - x3) - y1) % n
    return (x3, y3), None

def _ec_mul(k, P, a, n):
    # double-and-add with gcd leak detection
    R = None
    Q = P
    while k > 0:
        if k & 1:
            R, g = _ec_add(R, Q, a, n)
            if g:  # found factor
                return None, g
        k >>= 1
        if k:
            Q, g = _ec_add(Q, Q, a, n)
            if g:
                return None, g
    return R, None

def _random_curve(n, rng):
    # Choose random x,y,a; derive b so that P lies on curve
    while True:
        x = rng.randrange(2, n-1)
        y = rng.randrange(2, n-1)
        a = rng.randrange(2, n-1)
        # b = y^2 - x^3 - a x  (mod n)
        b = (y*y - (x*x % n)*x - a*x) % n
        # Discriminant Δ = -16(4a^3 + 27b^2); if gcd(Δ, n) > 1 we may already get a factor
        disc = (-16 * ( (4*pow(a,3,n) + 27*pow(b,2,n)) % n )) % n
        g = gcd(disc, n)
        if 1 < g < n:
            return None, None, g
        # Return curve y^2 = x^3 + a x + b with point P=(x,y)
        return (a, b), (x, y), None

def ecm(n: int, B1: int = 50000, curves: int = 200, seed: int | None = None):
    """
    Try to find a nontrivial factor of n using ECM Stage-1.
    Returns factor or None.
    """
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if seed is None:
        seed = 0xEC11
    rng = random.Random(seed)
    M = _build_M(max(1000, int(B1)))  # ensure reasonable lower bound

    for _ in range(curves):
        curve, P, g = _random_curve(n, rng)
        if g:  # lucky discriminant gcd
            return g
        a, b = curve
        # scalar multiply by smooth exponent M
        _, g = _ec_mul(M, P, a, n)
        if g:
            return g
    return None

if __name__ == "__main__":
    # quick smoke test
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 91
    f = ecm(N, B1=5000, curves=100, seed=1234)
    print(f or "None")
