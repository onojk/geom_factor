#!/usr/bin/env python3
import math
import csv
from dataclasses import dataclass
from typing import List, Tuple


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


@dataclass
class BucketRow:
    k: int
    start_dec: int
    end_dec: int
    size: int
    primes: int
    density: float
    mersenne: int
    mersenne_is_prime: bool


def bucket_stats(k: int) -> Tuple[int, int, int]:
    # canonical bucket k: [2^(k-1), 2^k - 1]
    start_dec = 1 << (k - 1)
    end_dec = (1 << k) - 1
    size = end_dec - start_dec + 1
    return start_dec, end_dec, size


def make_rows(k_min: int, k_max: int) -> List[BucketRow]:
    rows: List[BucketRow] = []
    for k in range(k_min, k_max + 1):
        start_dec, end_dec, size = bucket_stats(k)
        primes = sum(1 for n in range(start_dec, end_dec + 1) if is_prime(n))
        density = primes / size if size else 0.0
        mersenne = (1 << k) - 1
        rows.append(
            BucketRow(
                k=k,
                start_dec=start_dec,
                end_dec=end_dec,
                size=size,
                primes=primes,
                density=density,
                mersenne=mersenne,
                mersenne_is_prime=is_prime(mersenne),
            )
        )
    return rows


def write_csv(rows: List[BucketRow], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "start_dec", "end_dec", "size", "primes", "density", "2^k-1", "2^k-1 prime?"])
        for r in rows:
            w.writerow([
                r.k,
                r.start_dec,
                r.end_dec,
                r.size,
                r.primes,
                f"{r.density:.6f}",
                r.mersenne,
                "Y" if r.mersenne_is_prime else "N",
            ])


def write_md(rows: List[BucketRow], path: str) -> None:
    lines = []
    lines.append("| k | bucket start (dec) | bucket end (dec) | size | primes | density | 2^k-1 | 2^k-1 prime? |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for r in rows:
        lines.append(
            f"| {r.k} | {r.start_dec} | {r.end_dec} | {r.size} | {r.primes} | {r.density:.6f} | {r.mersenne} | {'Y' if r.mersenne_is_prime else 'N'} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--out_csv", default="bucket_table.csv")
    ap.add_argument("--out_md", default="bucket_table.md")
    args = ap.parse_args()

    rows = make_rows(args.kmin, args.kmax)
    write_csv(rows, args.out_csv)
    write_md(rows, args.out_md)

    print("WROTE:", args.out_csv)
    print("WROTE:", args.out_md)
