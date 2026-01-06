#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from coil3d import Coil3DParams, P, angle_between
from signature_driver import primes_upto


def edge_angles(E: np.ndarray):
    # azimuth in xy plane, elevation from xy plane
    ex, ey, ez = E[:,0], E[:,1], E[:,2]
    phi = np.arctan2(ey, ex)
    psi = np.arctan2(ez, np.sqrt(ex*ex + ey*ey))
    return phi, psi


def main():
    ap = argparse.ArgumentParser(description="Prime-lattice angles: connect prime points on the coil (ignore coil itself).")
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--prime-limit", type=int, default=20000000)

    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=0.02)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.1)

    ap.add_argument("--bins", type=int, default=120)
    args = ap.parse_args()

    par = Coil3DParams(a=args.a, b=args.b, omega=args.omega, c=args.c)

    primes = primes_upto(args.prime_limit)
    if primes.size < args.pairs + 2:
        raise SystemExit("Increase --prime-limit")

    ps = primes[: args.pairs + 1]

    # prime points on coil
    Q = np.array([P(int(p), par) for p in ps], dtype=float)  # (pairs+1, 3)
    E = Q[1:] - Q[:-1]                                       # edges (pairs, 3)

    # turning angles between consecutive edges
    turn = np.array([angle_between(E[i-1], E[i]) for i in range(1, len(E))], dtype=float)

    # heading angles of edges
    phi, psi = edge_angles(E)

    print(f"turning angle stats: mean={turn.mean():.6g}, std={turn.std():.6g}, min={turn.min():.6g}, max={turn.max():.6g}")

    # Scatter of headings
    plt.figure(figsize=(7,6))
    plt.scatter(phi, psi, s=6, alpha=0.5)
    plt.xlabel("edge azimuth φ (radians)")
    plt.ylabel("edge elevation ψ (radians)")
    plt.title("Prime-lattice edge headings (φ, ψ)")
    plt.tight_layout()
    plt.show()

    # Turning-angle histogram
    plt.figure(figsize=(9,5))
    plt.hist(turn[np.isfinite(turn)], bins=args.bins, density=True, alpha=0.7)
    plt.xlabel("turning angle θ (radians)")
    plt.ylabel("density")
    plt.title("Prime-lattice turning angles between successive prime edges")
    plt.tight_layout()
    plt.show()

    # Turn angle over index (first 2000)
    m = min(2000, turn.size)
    plt.figure(figsize=(10,4))
    plt.plot(turn[:m])
    plt.xlabel("k")
    plt.ylabel("θ")
    plt.title("Prime-lattice turning angles θ_k (first segment)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
