# geom_factor/viz/primes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class PrimeSieveResult:
    max_n: int
    is_prime: bytearray  # index i => 1 if prime else 0

    def primes_up_to(self) -> List[int]:
        return [i for i in range(2, self.max_n + 1) if self.is_prime[i]]


def sieve_upto(n: int) -> PrimeSieveResult:
    """
    Fast sieve of Eratosthenes up to n (inclusive).
    Returns a bytearray is_prime of length n+1 (0/1).
    """
    if n < 1:
        return PrimeSieveResult(max_n=n, is_prime=bytearray(b"\x00") * (max(0, n) + 1))

    is_prime = bytearray(b"\x01") * (n + 1)
    is_prime[0:2] = b"\x00\x00"

    # eliminate evens > 2
    for i in range(4, n + 1, 2):
        is_prime[i] = 0
    if n >= 2:
        is_prime[2] = 1

    p = 3
    while p * p <= n:
        if is_prime[p]:
            step = 2 * p
            start = p * p
            for k in range(start, n + 1, step):
                is_prime[k] = 0
        p += 2

    return PrimeSieveResult(max_n=n, is_prime=is_prime)
