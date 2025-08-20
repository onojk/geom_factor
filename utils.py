import math

def small_primes_up_to(n, limit=10000):
    """
    Generate up to `limit` primes, ignoring n if it's huge.
    Prevents OverflowError and memory exhaustion.
    """
    # hard cap on sieve size
    max_bound = 2000000  # enough for first ~148k primes
    bound = min(n, max_bound)

    sieve = [True] * (bound + 1)
    sieve[0:2] = [False, False]
    primes = []

    for p in range(2, bound + 1):
        if sieve[p]:
            primes.append(p)
            if len(primes) >= limit:
                break
            step_start = p * p
            if step_start > bound:
                continue
            for m in range(step_start, bound + 1, p):
                sieve[m] = False
    return primes

def tonelli_shanks(n, p):
    """
    Tonelli–Shanks: solve r^2 ≡ n (mod p) for odd prime p.
    Returns one root r in [0, p-1], or None if no solution.
    """
    n %= p
    if n == 0:
        return 0
    if p == 2:
        return n
    if pow(n, (p - 1) // 2, p) != 1:
        return None

    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1

    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1

    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)

    while t != 1:
        i = 1
        t2i = pow(t, 2, p)
        while i < m and t2i != 1:
            t2i = pow(t2i, 2, p)
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        r = (r * b) % p
        t = (t * pow(b, 2, p)) % p
        c = pow(b, 2, p)
        m = i
    return r
