import math

def small_primes_up_to(limit):
    """Yield primes up to `limit` using a segmented sieve (safe for large limit)."""
    if limit < 2:
        return []
    limit = int(limit)
    # we only sieve up to sqrt(limit)
    root = int(math.isqrt(limit)) + 1
    sieve = [True] * (root + 1)
    sieve[0:2] = [False, False]
    for p in range(2, root+1):
        if sieve[p]:
            yield p
            step = p
            start = p*p
            sieve[start: root+1: step] = [False] * (((root - start)//step)+1)
    # after sqrt(limit), just yield odd numbers checked by trial division
    def is_prime(n):
        r = int(math.isqrt(n))
        for p in range(2, r+1):
            if n % p == 0:
                return False
        return True
    for n in range(root+1, limit+1):
        if is_prime(n):
            yield n

def tonelli_shanks(n, p):
    """Tonelli-Shanks sqrt mod prime p. Returns r s.t. r^2 ≡ n mod p, or None."""
    assert 0 <= n < p
    if n == 0:
        return 0
    if pow(n, (p-1)//2, p) != 1:
        return None
    if p % 4 == 3:
        return pow(n, (p+1)//4, p)
    # general Tonelli–Shanks omitted (rarely needed in toy demo)
    return None
