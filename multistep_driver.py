#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from coil3d import Coil3DParams, angle_between
from signature_driver import primes_upto, signature_for_step


def compute_turning_angles(dphi: np.ndarray, dpsi: np.ndarray) -> np.ndarray:
    """Turning angles of the trajectory S_k=(dphi,dpsi) using successive velocity vectors."""
    S = np.stack([dphi, dpsi], axis=1)
    V = S[1:] - S[:-1]          # velocities
    # turning angles between V_k and V_{k+1}
    thetas = np.array([angle_between(V[i], V[i+1]) for i in range(len(V)-1)], dtype=float)
    return thetas


def main():
    ap = argparse.ArgumentParser(description="Multi-step fingerprint: turning-angle coherence in (Δphi,Δpsi)")
    ap.add_argument("--pairs", type=int, default=20000, help="Number of consecutive prime pairs")
    ap.add_argument("--prime-limit", type=int, default=20000000, help="Sieve limit")
    ap.add_argument("--control-gmax", type=int, default=500, help="Max even gap for random-gap control")
    ap.add_argument("--seed", type=int, default=0)

    # coil params (same family you’ve been using)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=0.02)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.1)

    ap.add_argument("--bins", type=int, default=80, help="Histogram bins")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    coil_par = Coil3DParams(a=args.a, b=args.b, omega=args.omega, c=args.c)

    # ---- primes and true gaps ----
    primes = primes_upto(args.prime_limit)
    if primes.size < args.pairs + 2:
        raise SystemExit("Increase --prime-limit")

    ps = primes[: args.pairs + 1]
    gaps = ps[1:] - ps[:-1]

    dphi = np.empty(args.pairs)
    dpsi = np.empty(args.pairs)
    for i in range(args.pairs):
        dphi[i], dpsi[i] = signature_for_step(int(ps[i]), int(gaps[i]), coil_par)

    theta_prime = compute_turning_angles(dphi, dpsi)

    # ---- control A: random integers + random even gaps ----
    rng = np.random.default_rng(args.seed)
    ns = rng.integers(int(ps[0]), int(ps[-1]), size=args.pairs)
    evens = rng.integers(1, args.control_gmax // 2 + 1, size=args.pairs) * 2

    cdphi = np.empty(args.pairs)
    cdpsi = np.empty(args.pairs)
    for i in range(args.pairs):
        cdphi[i], cdpsi[i] = signature_for_step(int(ns[i]), int(evens[i]), coil_par)

    theta_rand = compute_turning_angles(cdphi, cdpsi)

    # ---- control B: same prime gaps but shuffled order (kills memory) ----
    sgaps = gaps.copy()
    rng.shuffle(sgaps)

    sdphi = np.empty(args.pairs)
    sdpsi = np.empty(args.pairs)
    # Use the same prime starting points (or random ints); key is the *gap order* is destroyed.
    # We’ll keep n=ps[i] so only gap order changes.
    for i in range(args.pairs):
        sdphi[i], sdpsi[i] = signature_for_step(int(ps[i]), int(sgaps[i]), coil_par)

    theta_shuf = compute_turning_angles(sdphi, sdpsi)

    # ---- summary stats ----
    def summarize(name: str, x: np.ndarray):
        x = x[np.isfinite(x)]
        return (f"{name}: n={x.size}, mean={float(np.mean(x)):.6g}, std={float(np.std(x)):.6g}, "
                f"p05={float(np.quantile(x,0.05)):.6g}, p50={float(np.quantile(x,0.50)):.6g}, "
                f"p95={float(np.quantile(x,0.95)):.6g}")

    print(summarize("TURN(primes)", theta_prime))
    print(summarize("TURN(random even gaps)", theta_rand))
    print(summarize("TURN(shuffled prime gaps)", theta_shuf))

    # A simple “coherence” score: lower variance and tighter IQR suggests more structured turning.
    def coherence(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        iqr = float(np.quantile(x,0.75) - np.quantile(x,0.25))
        return iqr

    cp = coherence(theta_prime)
    cr = coherence(theta_rand)
    cs = coherence(theta_shuf)
    print(f"IQR coherence (lower=tighter): primes={cp:.6g}, rand={cr:.6g}, shuf={cs:.6g}")

    if args.no_plots:
        return

    # ---- histograms ----
    plt.figure(figsize=(10,5))
    plt.hist(theta_prime[np.isfinite(theta_prime)], bins=args.bins, alpha=0.55, density=True, label="Primes")
    plt.hist(theta_shuf[np.isfinite(theta_shuf)], bins=args.bins, alpha=0.45, density=True, label="Shuffled prime gaps")
    plt.hist(theta_rand[np.isfinite(theta_rand)], bins=args.bins, alpha=0.35, density=True, label="Random even gaps")
    plt.title("Turning-angle distribution in signature trajectory (Δphi, Δpsi)")
    plt.xlabel("turning angle θ (radians)")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- time series sample (first 2000) ----
    m = min(2000, theta_prime.size)
    plt.figure(figsize=(10,4))
    plt.plot(theta_prime[:m], label="Primes")
    plt.plot(theta_shuf[:m], label="Shuffled gaps", alpha=0.8)
    plt.title("Turning angles θ_k (first segment)")
    plt.xlabel("k")
    plt.ylabel("θ")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
