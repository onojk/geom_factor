#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from coil3d import Coil3DParams, dP
from signature_driver import primes_upto


def arc_length(n0: float, n1: float, par: Coil3DParams, steps: int = 200) -> float:
    """
    Numerical arc length of P(t) from t=n0 to t=n1 using Simpson-like rule.
    """
    ts = np.linspace(n0, n1, steps)
    speeds = np.array([np.linalg.norm(dP(t, par)) for t in ts])
    return np.trapz(speeds, ts)


def main():
    ap = argparse.ArgumentParser(description="Arc length of coil segments between consecutive primes")
    ap.add_argument("--pairs", type=int, default=5000)
    ap.add_argument("--prime-limit", type=int, default=2000000)
    ap.add_argument("--steps", type=int, default=200)

    # coil params
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=0.02)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.1)

    args = ap.parse_args()

    par = Coil3DParams(
        a=args.a,
        b=args.b,
        omega=args.omega,
        c=args.c,
    )

    primes = primes_upto(args.prime_limit)
    if primes.size < args.pairs + 2:
        raise SystemExit("Increase --prime-limit")

    ps = primes[: args.pairs + 1]
    gaps = ps[1:] - ps[:-1]

    lengths = np.empty(args.pairs)
    for i in range(args.pairs):
        lengths[i] = arc_length(ps[i], ps[i+1], par, steps=args.steps)

    # ---- scatter plots ----
    plt.figure(figsize=(10,5))
    plt.scatter(range(args.pairs), lengths, s=6, alpha=0.6)
    plt.xlabel("Prime index k")
    plt.ylabel("Arc length L_k")
    plt.title("Arc length of coil segments between consecutive primes")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,6))
    plt.scatter(gaps, lengths, s=8, alpha=0.6)
    plt.xlabel("Prime gap")
    plt.ylabel("Arc length L_k")
    plt.title("Arc length vs prime gap")
    plt.tight_layout()
    plt.show()

    # ---- normalization check ----
    ratio = lengths / gaps
    plt.figure(figsize=(6,5))
    plt.scatter(gaps, ratio, s=8, alpha=0.6)
    plt.xlabel("Prime gap")
    plt.ylabel("L_k / gap")
    plt.title("Normalized arc length per gap")
    plt.tight_layout()
    plt.show()

    print(
        f"L_k stats: mean={lengths.mean():.6g}, std={lengths.std():.6g}, "
        f"min={lengths.min():.6g}, max={lengths.max():.6g}"
    )


if __name__ == "__main__":
    main()
