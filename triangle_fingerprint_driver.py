#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

from coil3d import Coil3DParams, P
from signature_driver import primes_upto


# ----------------------------
# Helpers
# ----------------------------
def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def build_sequence_primes(primes: np.ndarray, triplets: int) -> np.ndarray:
    # need triplets + 2 nodes (k-1, k, k+1) for each center k
    return primes[: triplets + 2].astype(np.int64)


def build_sequence_shuffle_gaps(primes: np.ndarray, triplets: int, rng: np.random.Generator, start: int | None = None) -> np.ndarray:
    ps = primes[: triplets + 2].astype(np.int64)
    gaps = (ps[1:] - ps[:-1]).astype(np.int64)
    rng.shuffle(gaps)
    if start is None:
        start = int(ps[0])
    seq = np.empty(triplets + 2, dtype=np.int64)
    seq[0] = start
    for i in range(triplets + 1):
        seq[i + 1] = seq[i] + int(gaps[i])
    return seq


def build_sequence_random_even_gaps(triplets: int, rng: np.random.Generator, start: int = 3, gmax: int = 200) -> np.ndarray:
    if gmax < 2:
        raise ValueError("gmax must be >= 2")
    seq = np.empty(triplets + 2, dtype=np.int64)
    seq[0] = int(start)
    for i in range(triplets + 1):
        g = int(rng.integers(1, (gmax // 2) + 1) * 2)  # even
        seq[i + 1] = seq[i] + g
    return seq


def summarize(name: str, arr: np.ndarray) -> None:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        print(f"{name}: empty")
        return
    q05, q50, q95 = np.quantile(arr, [0.05, 0.5, 0.95])
    q25, q75 = np.quantile(arr, [0.25, 0.75])
    iqr = q75 - q25
    print(
        f"{name}: n={arr.size}, mean={arr.mean():.6g}, std={arr.std():.6g}, "
        f"p05={q05:.6g}, p50={q50:.6g}, p95={q95:.6g}, IQR={iqr:.6g}"
    )


# ----------------------------
# Triangle fingerprint at a node
# ----------------------------
def triangle_metrics_at_node(Qm1: np.ndarray, Q0: np.ndarray, Qp1: np.ndarray):
    """
    Bird's-eye plane spanned by incoming and outgoing chords at the node.

    Incoming: E- = Q0 - Qm1
    Outgoing: E+ = Qp1 - Q0

    Normalize by setting ex along E- and ey along E+ component orthogonal to E-.

    In that 2D basis, v- is on x-axis.
    v+ has coordinates (x,y).
      rho = |y|/|x|  = tan(theta)
      theta = atan(rho) in degrees
      d = min(|theta-30|, |theta-60|)

    Also compute the "swap" triangle:
      ex2 along E+, project E- into its plane basis, giving rho2, theta2, d2.
    """
    Em = Q0 - Qm1
    Ep = Qp1 - Q0

    ex = unit(Em)
    if float(np.linalg.norm(ex)) < 1e-12:
        return None

    # ey from outgoing orthogonal component
    ey_raw = Ep - float(np.dot(Ep, ex)) * ex
    ey = unit(ey_raw)
    if float(np.linalg.norm(ey)) < 1e-12:
        return None

    x = float(np.dot(Ep, ex))
    y = float(np.dot(Ep, ey))

    # avoid division blowups when x ~ 0
    if abs(x) < 1e-12:
        rho = float("inf")
        theta = 90.0
    else:
        rho = abs(y) / abs(x)
        theta = math.degrees(math.atan(rho))

    d = min(abs(theta - 30.0), abs(theta - 60.0))

    # swap orientation: ex2 along outgoing, project incoming
    ex2 = unit(Ep)
    if float(np.linalg.norm(ex2)) < 1e-12:
        return None

    ey2_raw = Em - float(np.dot(Em, ex2)) * ex2
    ey2 = unit(ey2_raw)
    if float(np.linalg.norm(ey2)) < 1e-12:
        return None

    x2 = float(np.dot(Em, ex2))
    y2 = float(np.dot(Em, ey2))

    if abs(x2) < 1e-12:
        rho2 = float("inf")
        theta2 = 90.0
    else:
        rho2 = abs(y2) / abs(x2)
        theta2 = math.degrees(math.atan(rho2))

    d2 = min(abs(theta2 - 30.0), abs(theta2 - 60.0))

    return rho, theta, d, rho2, theta2, d2


def compute_for_sequence(seq: np.ndarray, par: Coil3DParams):
    """
    Computes triangle metrics at each center node seq[k] for k=1..len-2.
    Returns arrays.
    """
    n = len(seq)
    m = n - 2
    rho = np.empty(m, dtype=float)
    theta = np.empty(m, dtype=float)
    d = np.empty(m, dtype=float)
    rho2 = np.empty(m, dtype=float)
    theta2 = np.empty(m, dtype=float)
    d2 = np.empty(m, dtype=float)

    keep = 0
    for i, k in enumerate(range(1, n - 1)):
        Qm1 = P(float(seq[k - 1]), par)
        Q0 = P(float(seq[k]), par)
        Qp1 = P(float(seq[k + 1]), par)

        out = triangle_metrics_at_node(Qm1, Q0, Qp1)
        if out is None:
            continue
        r, th, dd, r2, th2, dd2 = out
        rho[keep] = r
        theta[keep] = th
        d[keep] = dd
        rho2[keep] = r2
        theta2[keep] = th2
        d2[keep] = dd2
        keep += 1

    return {
        "rho": rho[:keep],
        "theta": theta[:keep],
        "d": d[:keep],
        "rho2": rho2[:keep],
        "theta2": theta2[:keep],
        "d2": d2[:keep],
    }


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Triangle fingerprint: bird's-eye normalized plane at each node; measure closeness to 30/60/90 right-triangle angles."
    )
    ap.add_argument("--triplets", type=int, default=5000, help="How many center-nodes (needs triplets+2 nodes).")
    ap.add_argument("--prime-limit", type=int, default=2_000_000, help="Sieve limit for primes_upto().")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--control", action="store_true", help="Include shuffled-gap and random-even-gap controls.")
    ap.add_argument("--control-gmax", type=int, default=200)
    ap.add_argument("--bins", type=int, default=120)
    ap.add_argument("--no-plot", action="store_true")

    # Coil parameters (match your defaults)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=0.02)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.1)

    args = ap.parse_args()

    par = Coil3DParams(a=args.a, b=args.b, omega=args.omega, c=args.c)
    rng = np.random.default_rng(args.seed)

    primes = primes_upto(args.prime_limit)
    if primes.size < args.triplets + 5:
        raise SystemExit("Increase --prime-limit; not enough primes for requested --triplets.")

    seq_pr = build_sequence_primes(primes, args.triplets)
    feat_pr = compute_for_sequence(seq_pr, par)

    print("\nTriangle fingerprint (primes):")
    summarize("theta (deg) incoming->outgoing", feat_pr["theta"])
    summarize("d = min(|theta-30|,|theta-60|)", feat_pr["d"])
    summarize("theta2 (deg) outgoing->incoming", feat_pr["theta2"])
    summarize("d2 = min(|theta2-30|,|theta2-60|)", feat_pr["d2"])

    feat_sh = feat_re = None
    if args.control:
        seq_sh = build_sequence_shuffle_gaps(primes, args.triplets, rng, start=int(seq_pr[0]))
        seq_re = build_sequence_random_even_gaps(args.triplets, rng, start=int(seq_pr[0]), gmax=int(args.control_gmax))
        feat_sh = compute_for_sequence(seq_sh, par)
        feat_re = compute_for_sequence(seq_re, par)

        print("\nTriangle fingerprint (shuffled prime gaps control):")
        summarize("theta (deg) incoming->outgoing", feat_sh["theta"])
        summarize("d = min(|theta-30|,|theta-60|)", feat_sh["d"])
        summarize("theta2 (deg) outgoing->incoming", feat_sh["theta2"])
        summarize("d2 = min(|theta2-30|,|theta2-60|)", feat_sh["d2"])

        print("\nTriangle fingerprint (random even gaps control):")
        summarize("theta (deg) incoming->outgoing", feat_re["theta"])
        summarize("d = min(|theta-30|,|theta-60|)", feat_re["d"])
        summarize("theta2 (deg) outgoing->incoming", feat_re["theta2"])
        summarize("d2 = min(|theta2-30|,|theta2-60|)", feat_re["d2"])

    if args.no_plot:
        return

    # Plot theta histograms with 30/60 guides
    def plot_theta_hist(title: str, pr: np.ndarray, sh: np.ndarray | None, re: np.ndarray | None):
        plt.figure(figsize=(10, 5))
        plt.hist(pr[np.isfinite(pr)], bins=args.bins, density=True, alpha=0.7, label="Primes")
        if sh is not None:
            plt.hist(sh[np.isfinite(sh)], bins=args.bins, density=True, alpha=0.5, label="Shuffled gaps")
        if re is not None:
            plt.hist(re[np.isfinite(re)], bins=args.bins, density=True, alpha=0.4, label="Random even gaps")
        plt.axvline(30.0, linestyle="--")
        plt.axvline(60.0, linestyle="--")
        plt.xlabel("theta (degrees)")
        plt.ylabel("density")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    plot_theta_hist("Theta (incoming->outgoing) with 30°/60° guides",
                    feat_pr["theta"],
                    feat_sh["theta"] if feat_sh else None,
                    feat_re["theta"] if feat_re else None)

    plot_theta_hist("Theta2 (outgoing->incoming) with 30°/60° guides",
                    feat_pr["theta2"],
                    feat_sh["theta2"] if feat_sh else None,
                    feat_re["theta2"] if feat_re else None)

    # Plot d histograms (closeness to 30/60)
    def plot_d_hist(title: str, pr: np.ndarray, sh: np.ndarray | None, re: np.ndarray | None):
        plt.figure(figsize=(10, 5))
        plt.hist(pr[np.isfinite(pr)], bins=args.bins, density=True, alpha=0.7, label="Primes")
        if sh is not None:
            plt.hist(sh[np.isfinite(sh)], bins=args.bins, density=True, alpha=0.5, label="Shuffled gaps")
        if re is not None:
            plt.hist(re[np.isfinite(re)], bins=args.bins, density=True, alpha=0.4, label="Random even gaps")
        plt.xlabel("d (degrees)")
        plt.ylabel("density")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    plot_d_hist("Closeness d = min(|theta-30|, |theta-60|)",
                feat_pr["d"],
                feat_sh["d"] if feat_sh else None,
                feat_re["d"] if feat_re else None)

    plot_d_hist("Closeness d2 = min(|theta2-30|, |theta2-60|)",
                feat_pr["d2"],
                feat_sh["d2"] if feat_sh else None,
                feat_re["d2"] if feat_re else None)


if __name__ == "__main__":
    main()
