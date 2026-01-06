#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import math
import matplotlib.pyplot as plt

from coil3d import Coil3DParams, P, dP, azimuth_elevation, wrap_pi


def primes_upto(limit: int) -> np.ndarray:
    """Simple sieve, returns primes <= limit as int numpy array."""
    if limit < 2:
        return np.array([], dtype=int)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i:limit+1:i] = False
    return np.nonzero(sieve)[0].astype(int)


def signature_for_step(n: int, g: int, par: Coil3DParams) -> tuple[float, float]:
    """
    Two-angle signature:
      Δphi = phi_chord - phi_tangent (wrapped to (-pi, pi])
      Δpsi = psi_chord - psi_tangent (wrapped to (-pi, pi])
    where chord is P(n+g)-P(n) and tangent is P'(n).
    """
    T = dP(float(n), par)
    C = P(float(n + g), par) - P(float(n), par)

    phi_T, psi_T = azimuth_elevation(T)
    phi_C, psi_C = azimuth_elevation(C)

    dphi = wrap_pi(phi_C - phi_T)
    dpsi = wrap_pi(psi_C - psi_T)
    return dphi, dpsi


def main():
    ap = argparse.ArgumentParser(description="Prime tangent/chord signature experiment (2-angle version)")
    ap.add_argument("--prime-limit", type=int, default=200000, help="Sieve limit for primes")
    ap.add_argument("--n-primes", type=int, default=2000, help="How many primes to analyze (uses consecutive primes)")
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=0.02)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.1)
    ap.add_argument("--control", action="store_true", help="Also plot control: random integers + random even gaps")
    ap.add_argument("--control-gmax", type=int, default=200, help="Max even gap for control sampling")
    args = ap.parse_args()

    par = Coil3DParams(a=args.a, b=args.b, omega=args.omega, c=args.c)

    primes = primes_upto(args.prime_limit)
    if primes.size < args.n_primes + 2:
        raise SystemExit(f"Not enough primes under limit={args.prime_limit} for n_primes={args.n_primes}. Increase --prime-limit.")

    # Use consecutive primes p_k -> p_{k+1}
    ps = primes[: args.n_primes + 1]
    gaps = ps[1:] - ps[:-1]

    dphi = np.empty(args.n_primes, dtype=float)
    dpsi = np.empty(args.n_primes, dtype=float)

    for i in range(args.n_primes):
        p = int(ps[i])
        g = int(gaps[i])
        dphi[i], dpsi[i] = signature_for_step(p, g, par)

    k = np.arange(1, args.n_primes + 1)

    # Plot Δphi
    plt.figure(figsize=(10,5))
    plt.scatter(k, dphi, s=8, alpha=0.4, label="Primes: Δphi (chord - tangent)")
    plt.xlabel("Prime step index k")
    plt.ylabel("Δphi (radians, wrapped)")
    plt.title("Prime signature: azimuth difference Δphi")
    plt.tight_layout()
    plt.show()

    # Plot Δpsi
    plt.figure(figsize=(10,5))
    plt.scatter(k, dpsi, s=8, alpha=0.4, label="Primes: Δpsi (chord - tangent)")
    plt.xlabel("Prime step index k")
    plt.ylabel("Δpsi (radians, wrapped)")
    plt.title("Prime signature: elevation difference Δpsi")
    plt.tight_layout()
    plt.show()

    # 2D scatter: (Δphi, Δpsi)
    plt.figure(figsize=(7,7))
    plt.scatter(dphi, dpsi, s=8, alpha=0.35, label="Primes")
    plt.xlabel("Δphi")
    plt.ylabel("Δpsi")
    plt.title("Prime signature cloud in (Δphi, Δpsi)")
    plt.tight_layout()
    plt.show()

    if args.control:
        rng = np.random.default_rng(0)
        # sample random integers in same numeric range as primes used
        n_min = int(ps[0])
        n_max = int(ps[-1])
        n_ctrl = args.n_primes

        ns = rng.integers(n_min, n_max, size=n_ctrl)
        # random even gaps up to gmax (>=2)
        evens = rng.integers(1, (args.control_gmax // 2) + 1, size=n_ctrl) * 2

        dphi_c = np.empty(n_ctrl, dtype=float)
        dpsi_c = np.empty(n_ctrl, dtype=float)
        for i in range(n_ctrl):
            dphi_c[i], dpsi_c[i] = signature_for_step(int(ns[i]), int(evens[i]), par)

        plt.figure(figsize=(7,7))
        plt.scatter(dphi, dpsi, s=8, alpha=0.35, label="Primes")
        plt.scatter(dphi_c, dpsi_c, s=8, alpha=0.35, label="Control (random n, random even gap)")
        plt.xlabel("Δphi")
        plt.ylabel("Δpsi")
        plt.title("Prime vs control signature cloud")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Print quick stats (helpful for eyeballing “envelopes” later)
    def stats(name, x):
        return f"{name}: mean={float(np.mean(x)):.4g}, std={float(np.std(x)):.4g}, min={float(np.min(x)):.4g}, max={float(np.max(x)):.4g}"

    print(stats("Δphi", dphi))
    print(stats("Δpsi", dpsi))
    print("Gap stats: mean={:.4g}, std={:.4g}, min={}, max={}".format(
        float(np.mean(gaps)), float(np.std(gaps)), int(np.min(gaps)), int(np.max(gaps))
    ))


if __name__ == "__main__":
    main()
