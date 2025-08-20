#!/usr/bin/env python3
import math

def _build_M(B1: int) -> int:
    # L = lcm(1..B1) via product of p^k for primes p<=B1 with p^k<=B1
    M = 1
    sieve = [True]*(B1+1)
    sieve[:2] = [False, False]
    for i in range(2, int(B1**0.5)+1):
        if sieve[i]:
            for j in range(i*i, B1+1, i):
                sieve[j] = False
    for p, is_p in enumerate(sieve):
        if not is_p: continue
        pk = p
        while pk * p <= B1:
            pk *= p
        M *= pk
    return M

def pollard_p1(n: int, B1: int = 100000, base: int = 2) -> int | None:
    if n % 2 == 0: return 2
    if n % 3 == 0: return 3
    a = base % n
    g = math.gcd(a, n)
    if 1 < g < n: return g
    M = _build_M(B1)
    aM = pow(a, M, n)
    g = math.gcd(aM-1, n)
    if 1 < g < n: return g
    return None

def pollard_p1_multi(n: int, B1: int = 100000, bases=(2,3,5,7,11)) -> int | None:
    for a in bases:
        g = pollard_p1(n, B1=B1, base=a)
        if g: return g
    return None
