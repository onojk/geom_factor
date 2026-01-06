#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from coil3d import Coil3DParams, P, dP
from signature_driver import primes_upto


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def build_sequence_primes(primes: np.ndarray, pairs: int) -> np.ndarray:
    return primes[: pairs + 1].astype(np.int64)


def build_sequence_shuffle_gaps(primes: np.ndarray, pairs: int, rng: np.random.Generator, start: int | None = None) -> np.ndarray:
    ps = primes[: pairs + 1].astype(np.int64)
    gaps = (ps[1:] - ps[:-1]).astype(np.int64)
    rng.shuffle(gaps)
    if start is None:
        start = int(ps[0])
    seq = np.empty(pairs + 1, dtype=np.int64)
    seq[0] = start
    for i in range(pairs):
        seq[i + 1] = seq[i] + int(gaps[i])
    return seq


def build_sequence_random_even_gaps(pairs: int, rng: np.random.Generator, start: int = 3, gmax: int = 200) -> np.ndarray:
    if gmax < 2:
        raise ValueError("gmax must be >= 2")
    seq = np.empty(pairs + 1, dtype=np.int64)
    seq[0] = int(start)
    for i in range(pairs):
        g = int(rng.integers(1, (gmax // 2) + 1) * 2)
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


def peak_split_features_for_pair(n0: int, n1: int, par: Coil3DParams, steps: int = 250):
    A = P(float(n0), par)
    B = P(float(n1), par)

    chord = B - A
    ex = unit(chord)

    TA = unit(dP(float(n0), par))
    ey_raw = TA - float(np.dot(TA, ex)) * ex
    ey = unit(ey_raw)

    if float(np.linalg.norm(ey)) < 1e-9:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(ex, fallback))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        ey = unit(fallback - float(np.dot(fallback, ex)) * ex)

    ts = np.linspace(float(n0), float(n1), steps)
    Q = np.array([P(t, par) for t in ts], dtype=float)
    D = Q - A

    x = D @ ex
    y = D @ ey

    x_end = float(np.dot(B - A, ex))
    if abs(x_end) < 1e-12:
        return float("nan"), float("nan"), float("nan"), float(np.linalg.norm(chord))

    u = x / x_end

    mask = (u >= 0.0) & (u <= 1.0)
    u = u[mask]
    y = y[mask]
    if u.size < 5:
        return float("nan"), float("nan"), float("nan"), float(np.linalg.norm(chord))

    if float(np.max(y)) < abs(float(np.min(y))):
        y = -y

    idx = int(np.argmax(y))
    u_star = float(u[idx])
    peak_h = float(y[idx])
    chord_len = float(np.linalg.norm(chord))

    denom = max(1e-12, 1.0 - u_star)
    ratio = float(u_star / denom)

    return u_star, ratio, peak_h, chord_len


def compute_features_for_sequence(seq: np.ndarray, par: Coil3DParams, steps: int):
    pairs = len(seq) - 1
    u_star = np.empty(pairs, dtype=float)
    ratio = np.empty(pairs, dtype=float)
    peak_h = np.empty(pairs, dtype=float)
    chord_len = np.empty(pairs, dtype=float)

    for k in range(pairs):
        u, r, h, c = peak_split_features_for_pair(int(seq[k]), int(seq[k + 1]), par, steps=steps)
        u_star[k] = u
        ratio[k] = r
        peak_h[k] = h
        chord_len[k] = c

    return {
        "u_star": u_star,
        "ratio": ratio,
        "peak_h": peak_h,
        "chord_len": chord_len,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Peak-split fingerprint: bird's-eye coil arc between consecutive nodes; peak (slope=0) splits chord."
    )
    ap.add_argument("--pairs", type=int, default=5000)
    ap.add_argument("--prime-limit", type=int, default=2_000_000)
    ap.add_argument("--steps", type=int, default=250)

    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=0.02)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.1)

    ap.add_argument("--control", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--control-gmax", type=int, default=200)

    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--bins", type=int, default=120)

    args = ap.parse_args()

    par = Coil3DParams(a=args.a, b=args.b, omega=args.omega, c=args.c)

    primes = primes_upto(args.prime_limit)
    if primes.size < args.pairs + 2:
        raise SystemExit("Increase --prime-limit; not enough primes for requested --pairs.")

    rng = np.random.default_rng(args.seed)

    seq_pr = build_sequence_primes(primes, args.pairs)
    feat_pr = compute_features_for_sequence(seq_pr, par, steps=args.steps)

    print("\nPeak-split fingerprint (primes):")
    summarize("u_star (split fraction)", feat_pr["u_star"])
    summarize("ratio u/(1-u)", feat_pr["ratio"])
    summarize("peak height", feat_pr["peak_h"])
    summarize("chord length", feat_pr["chord_len"])

    feat_sh = feat_re = None

    if args.control:
        seq_sh = build_sequence_shuffle_gaps(primes, args.pairs, rng=rng, start=int(seq_pr[0]))
        seq_re = build_sequence_random_even_gaps(args.pairs, rng=rng, start=int(seq_pr[0]), gmax=int(args.control_gmax))
        feat_sh = compute_features_for_sequence(seq_sh, par, steps=args.steps)
        feat_re = compute_features_for_sequence(seq_re, par, steps=args.steps)

        print("\nPeak-split fingerprint (shuffled prime gaps control):")
        summarize("u_star (split fraction)", feat_sh["u_star"])
        summarize("ratio u/(1-u)", feat_sh["ratio"])
        summarize("peak height", feat_sh["peak_h"])
        summarize("chord length", feat_sh["chord_len"])

        print("\nPeak-split fingerprint (random even gaps control):")
        summarize("u_star (split fraction)", feat_re["u_star"])
        summarize("ratio u/(1-u)", feat_re["ratio"])
        summarize("peak height", feat_re["peak_h"])
        summarize("chord length", feat_re["chord_len"])

    if args.no_plot:
        return

    k = np.arange(args.pairs)

    plt.figure(figsize=(10, 5))
    plt.scatter(k, feat_pr["u_star"], s=6, alpha=0.6, label="Primes")
    if args.control and feat_sh is not None and feat_re is not None:
        plt.scatter(k, feat_sh["u_star"], s=6, alpha=0.35, label="Shuffled prime gaps")
        plt.scatter(k, feat_re["u_star"], s=6, alpha=0.35, label="Random even gaps")
    plt.xlabel("Pair index k")
    plt.ylabel("u* (peak split fraction on chord)")
    plt.title("Peak split location u* (bird's-eye arc peak)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.hist(feat_pr["u_star"][np.isfinite(feat_pr["u_star"])], bins=args.bins, density=True, alpha=0.7, label="Primes")
    if args.control and feat_sh is not None and feat_re is not None:
        plt.hist(feat_sh["u_star"][np.isfinite(feat_sh["u_star"])], bins=args.bins, density=True, alpha=0.5, label="Shuffled prime gaps")
        plt.hist(feat_re["u_star"][np.isfinite(feat_re["u_star"])], bins=args.bins, density=True, alpha=0.5, label="Random even gaps")
    plt.xlabel("u*")
    plt.ylabel("density")
    plt.title("Distribution of peak split fraction u*")
    plt.legend()
    plt.tight_layout()
    plt.show()

    clip_max = float(np.nanquantile(feat_pr["ratio"], 0.99)) if np.isfinite(feat_pr["ratio"]).any() else 10.0
    clip_max = max(5.0, clip_max)

    plt.figure(figsize=(10, 5))
    pr_ratio = np.clip(feat_pr["ratio"][np.isfinite(feat_pr["ratio"])], 0, clip_max)
    plt.hist(pr_ratio, bins=args.bins, density=True, alpha=0.7, label="Primes")
    if args.control and feat_sh is not None and feat_re is not None:
        sh_ratio = np.clip(feat_sh["ratio"][np.isfinite(feat_sh["ratio"])], 0, clip_max)
        re_ratio = np.clip(feat_re["ratio"][np.isfinite(feat_re["ratio"])], 0, clip_max)
        plt.hist(sh_ratio, bins=args.bins, density=True, alpha=0.5, label="Shuffled prime gaps")
        plt.hist(re_ratio, bins=args.bins, density=True, alpha=0.5, label="Random even gaps")
    plt.xlabel(f"ratio u/(1-u) (clipped at {clip_max:.3g})")
    plt.ylabel("density")
    plt.title("Distribution of peak split ratio")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
