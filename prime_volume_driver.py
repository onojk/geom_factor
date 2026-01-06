#!/usr/bin/env python3
"""
prime_volume_driver.py

Demonstrate that a rectangular prism (or "fill to a height") built from
integer side lengths that are primes > 1 cannot have a prime integer volume.

Also shows the only way to get a prime integer volume from L*W*H is when
two dimensions are 1 and the third is prime (if you allow 1, which is not prime).
"""

from __future__ import annotations
import argparse
from math import isqrt
from typing import List, Tuple


def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return n == 2
    r = isqrt(n)
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True


def primes_up_to(limit: int) -> List[int]:
    return [n for n in range(2, limit + 1) if is_prime(n)]


def scan_rectangular_prisms(prime_limit: int, max_h: int, allow_one: bool) -> List[Tuple[int, int, int, int]]:
    """
    Return list of (L, W, H, V) where L,W,H are in allowed set (primes up to prime_limit, plus 1 if allow_one)
    and V = L*W*H is prime.
    """
    ps = primes_up_to(prime_limit)
    dims = ([1] if allow_one else []) + ps

    hits = []
    for L in dims:
        for W in dims:
            for H in dims:
                if H > max_h:
                    continue
                V = L * W * H
                if is_prime(V):
                    hits.append((L, W, H, V))
    return hits


def fill_to_height_demo(base_p: int, base_q: int, max_h: int) -> None:
    """
    Model "filling" a base of prime sides base_p and base_q up to height h, integer steps.
    V(h) = (base_p * base_q) * h. Shows whether any V(h) can be prime.
    """
    base_area = base_p * base_q
    print(f"Base sides: {base_p}, {base_q} (both prime > 1)")
    print(f"Base area = {base_area} (composite because it's a product of two >1 integers)")
    print(f"Volume at height h: V(h) = {base_area} * h\n")

    any_prime = False
    for h in range(1, max_h + 1):
        V = base_area * h
        if is_prime(V):
            any_prime = True
            print(f"FOUND prime volume? h={h}, V={V}")
            break

    if not any_prime:
        print("No prime volumes found for any integer height in the scanned range.")
        print("Reason: V(h) always has factor base_area (>1), so V(h) cannot be prime.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Driver: prime sides vs prime integer volume demonstration.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Mode 1: "fill to height"
    ap_fill = sub.add_parser("fill", help="Demonstrate why stopping at 'prime height' doesn't yield prime volume.")
    ap_fill.add_argument("--p", type=int, required=True, help="Prime base side p (e.g., 3,5,7,...)")
    ap_fill.add_argument("--q", type=int, required=True, help="Prime base side q (e.g., 3,5,7,...)")
    ap_fill.add_argument("--max-h", type=int, default=200, help="Max integer height to scan (default: 200)")

    # Mode 2: brute force rectangular prisms
    ap_scan = sub.add_parser("scan", help="Scan L*W*H for prime V under dimension constraints.")
    ap_scan.add_argument("--prime-limit", type=int, default=50, help="Use primes up to this value (default: 50)")
    ap_scan.add_argument("--max-h", type=int, default=50, help="Max H to allow (default: 50)")
    ap_scan.add_argument("--allow-one", action="store_true",
                         help="Allow dimension=1 (not prime) to show the only way to get prime volume.")

    args = ap.parse_args()

    if args.cmd == "fill":
        if not is_prime(args.p) or not is_prime(args.q):
            raise SystemExit("Error: --p and --q must both be prime integers > 1.")
        fill_to_height_demo(args.p, args.q, args.max_h)

    elif args.cmd == "scan":
        hits = scan_rectangular_prisms(args.prime_limit, args.max_h, args.allow_one)
        print(f"Scan results: prime_limit={args.prime_limit}, max_h={args.max_h}, allow_one={args.allow_one}")
        if not hits:
            print("No (L,W,H) found such that V=L*W*H is prime under these constraints.")
            if not args.allow_one:
                print("This matches the rule: if all dimensions are primes (>1), V is composite (product of >1 integers).")
            else:
                print("Try increasing --prime-limit / --max-h if you expected to see (1,1,p).")
        else:
            print(f"FOUND {len(hits)} hits. Showing up to first 30:")
            for (L, W, H, V) in hits[:30]:
                print(f"L={L:>2} W={W:>2} H={H:>2}  -> V={V}")
            print("\nInterpretation:")
            print("- If allow_one=False, hits should be empty.")
            print("- If allow_one=True, hits will essentially be permutations of (1,1,prime).")


if __name__ == "__main__":
    main()
