#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from coil3d import Coil3DParams, P, dP, angle_between
from signature_driver import primes_upto


def build_sequence_primes(primes: np.ndarray, pairs: int) -> np.ndarray:
    return primes[:pairs+1].astype(np.int64)


def build_sequence_shuffle_gaps(primes: np.ndarray, pairs: int, rng: np.random.Generator, start: int | None = None) -> np.ndarray:
    """
    Control: keep the same multiset of prime gaps, but randomize order.
    Sequence is strictly increasing but is NOT the prime sequence.
    """
    ps = primes[:pairs+1].astype(np.int64)
    gaps = (ps[1:] - ps[:-1]).astype(np.int64)
    rng.shuffle(gaps)
    if start is None:
        start = int(ps[0])
    seq = np.empty(pairs+1, dtype=np.int64)
    seq[0] = start
    for i in range(pairs):
        seq[i+1] = seq[i] + gaps[i]
    return seq


def build_sequence_random_even_gaps(pairs: int, rng: np.random.Generator, start: int = 3, gmax: int = 200) -> np.ndarray:
    """
    Baseline control: random even gaps in [2, gmax] (inclusive).
    """
    if gmax < 2:
        raise ValueError("gmax must be >= 2")
    # sample even gaps
    evens = np.arange(2, gmax+1, 2, dtype=np.int64)
    gaps = rng.choice(evens, size=pairs, replace=True)
    seq = np.empty(pairs+1, dtype=np.int64)
    seq[0] = int(start)
    for i in range(pairs):
        seq[i+1] = seq[i] + int(gaps[i])
    return seq


def tangent_chord_angles(seq: np.ndarray, par: Coil3DParams) -> np.ndarray:
    """
    alpha_k = angle between tangent at n_k and chord from n_k to n_{k+1}
    """
    n0 = seq[:-1]
    n1 = seq[1:]
    alphas = np.empty(len(n0), dtype=float)
    for i, (a, b) in enumerate(zip(n0, n1)):
        T = dP(float(a), par)
        E = P(float(b), par) - P(float(a), par)
        alphas[i] = angle_between(T, E)
    return alphas


def turning_angles_on_lattice(seq: np.ndarray, par: Coil3DParams) -> np.ndarray:
    """
    theta_k = angle between successive chords E_{k-1}, E_k
    """
    Q = np.array([P(float(n), par) for n in seq], dtype=float)
    E = Q[1:] - Q[:-1]
    if len(E) < 2:
        return np.array([], dtype=float)
    thetas = np.array([angle_between(E[i-1], E[i]) for i in range(1, len(E))], dtype=float)
    return thetas


def edge_headings(seq: np.ndarray, par: Coil3DParams):
    """
    headings of chords in spherical-ish coordinates:
      phi = atan2(dy, dx)
      psi = atan2(dz, sqrt(dx^2+dy^2))
    """
    Q = np.array([P(float(n), par) for n in seq], dtype=float)
    E = Q[1:] - Q[:-1]
    ex, ey, ez = E[:,0], E[:,1], E[:,2]
    phi = np.arctan2(ey, ex)
    psi = np.arctan2(ez, np.sqrt(ex*ex + ey*ey))
    return phi, psi


def summarize(name: str, x: np.ndarray):
    x = x[np.isfinite(x)]
    if x.size == 0:
        print(f"{name}: empty")
        return
    q05, q50, q95 = np.quantile(x, [0.05, 0.5, 0.95])
    iqr = np.quantile(x, 0.75) - np.quantile(x, 0.25)
    print(f"{name}: n={x.size}, mean={x.mean():.6g}, std={x.std():.6g}, p05={q05:.6g}, p50={q50:.6g}, p95={q95:.6g}, IQR={iqr:.6g}")


def main():
    ap = argparse.ArgumentParser(description="Honest test: tangent–chord angles at primes vs controls (gap-shuffle, random even gaps).")
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--prime-limit", type=int, default=20000000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--control-gmax", type=int, default=200)
    ap.add_argument("--bins", type=int, default=140)

    # coil params
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=0.02)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.1)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    par = Coil3DParams(a=args.a, b=args.b, omega=args.omega, c=args.c)

    primes = primes_upto(args.prime_limit)
    if primes.size < args.pairs + 2:
        raise SystemExit("Increase --prime-limit (not enough primes for --pairs)")

    seq_pr = build_sequence_primes(primes, args.pairs)
    seq_sh = build_sequence_shuffle_gaps(primes, args.pairs, rng, start=int(seq_pr[0]))
    seq_re = build_sequence_random_even_gaps(args.pairs, rng, start=int(seq_pr[0]), gmax=args.control_gmax)

    # --- tangent–chord angles ---
    a_pr = tangent_chord_angles(seq_pr, par)
    a_sh = tangent_chord_angles(seq_sh, par)
    a_re = tangent_chord_angles(seq_re, par)

    summarize("alpha(primes)", a_pr)
    summarize("alpha(shuffled gaps)", a_sh)
    summarize("alpha(random even gaps)", a_re)

    # Histogram comparison
    plt.figure(figsize=(10,5))
    plt.hist(a_pr, bins=args.bins, density=True, alpha=0.55, label="Primes")
    plt.hist(a_sh, bins=args.bins, density=True, alpha=0.45, label="Shuffled prime gaps")
    plt.hist(a_re, bins=args.bins, density=True, alpha=0.35, label="Random even gaps")
    plt.xlabel("tangent–chord angle α (radians)")
    plt.ylabel("density")
    plt.title("Honest test: tangent–chord angle distributions")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- optional: lattice turning angles ---
    t_pr = turning_angles_on_lattice(seq_pr, par)
    t_sh = turning_angles_on_lattice(seq_sh, par)
    t_re = turning_angles_on_lattice(seq_re, par)

    summarize("turn(primes)", t_pr)
    summarize("turn(shuffled gaps)", t_sh)
    summarize("turn(random even gaps)", t_re)

    plt.figure(figsize=(10,5))
    plt.hist(t_pr, bins=args.bins, density=True, alpha=0.55, label="Primes")
    plt.hist(t_sh, bins=args.bins, density=True, alpha=0.45, label="Shuffled prime gaps")
    plt.hist(t_re, bins=args.bins, density=True, alpha=0.35, label="Random even gaps")
    plt.xlabel("lattice turning angle θ (radians)")
    plt.ylabel("density")
    plt.title("Lattice turning-angle distributions (control check)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- optional: edge heading clouds (phi, psi) ---
    phi_pr, psi_pr = edge_headings(seq_pr, par)
    phi_sh, psi_sh = edge_headings(seq_sh, par)
    phi_re, psi_re = edge_headings(seq_re, par)

    plt.figure(figsize=(7,6))
    plt.scatter(phi_pr, psi_pr, s=4, alpha=0.35, label="Primes")
    plt.scatter(phi_sh, psi_sh, s=4, alpha=0.25, label="Shuffled prime gaps")
    plt.scatter(phi_re, psi_re, s=4, alpha=0.20, label="Random even gaps")
    plt.xlabel("edge azimuth φ (radians)")
    plt.ylabel("edge elevation ψ (radians)")
    plt.title("Prime-lattice edge headings (φ, ψ) vs controls")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
