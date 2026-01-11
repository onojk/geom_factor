#!/usr/bin/env python3
import math
import matplotlib.pyplot as plt

def sieve(n: int):
    """Return boolean list is_prime[0..n]."""
    if n < 2:
        return [False] * (n + 1)
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(n)) + 1):
        if is_prime[p]:
            step = p
            start = p * p
            is_prime[start:n+1:step] = [False] * (((n - start) // step) + 1)
    return is_prime

def minmax(xs):
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]

def main():
    max_bucket = 31

    # Bucket maxima: 2^k - 1
    buckets = list(range(1, max_bucket + 1))
    max_vals = [(1 << k) - 1 for k in buckets]

    # We'll compute primes up to max of bucket 31 (which is huge) — so instead,
    # do a *practical* overlay: use buckets up to a reasonable limit (e.g., 20),
    # OR compute P_k using a prime-count approximation (pi(x)).
    #
    # Since you asked 1..31, we'll do an approximation for P_k so the shapes overlay.
    # Approximation: pi(x) ~ x / ln(x)  (good enough for "shape" comparison)
    def pi_approx(x: int) -> float:
        if x < 2:
            return 0.0
        return x / math.log(x)

    prime_counts = []
    for k in buckets:
        lo = 1 << (k - 1)
        hi = (1 << k) - 1
        # approximate primes in [lo, hi] as pi(hi) - pi(lo-1)
        prime_counts.append(pi_approx(hi) - pi_approx(lo - 1))

    # Normalize for shape overlay
    max_norm = minmax(max_vals)
    primes_norm = minmax(prime_counts)

    plt.figure()
    plt.plot(buckets, max_norm, marker="o", label="Normalized bucket max (2^k-1)")
    plt.plot(buckets, primes_norm, marker="o", label="Normalized prime count per bucket")
    plt.xlabel("Bucket (bit-length k)")
    plt.ylabel("Normalized value (0..1)")
    plt.title("Shape overlay (normalized)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
