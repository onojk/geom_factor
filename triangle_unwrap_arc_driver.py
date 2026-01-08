#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

from coil3d import Coil3DParams, P, dP
from signature_driver import primes_upto


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


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


def build_sequence_primes(primes: np.ndarray, pairs: int) -> np.ndarray:
    return primes[: pairs + 1].astype(np.int64)


def build_sequence_shuffle_gaps(
    primes: np.ndarray, pairs: int, rng: np.random.Generator, start: int | None = None
) -> np.ndarray:
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


def build_sequence_random_even_gaps(
    pairs: int, rng: np.random.Generator, start: int = 3, gmax: int = 200
) -> np.ndarray:
    if gmax < 2:
        raise ValueError("gmax must be >= 2")
    seq = np.empty(pairs + 1, dtype=np.int64)
    seq[0] = int(start)
    for i in range(pairs):
        g = int(rng.integers(1, (gmax // 2) + 1) * 2)  # even
        seq[i + 1] = seq[i] + g
    return seq


def canonical_triangle_axes(A: np.ndarray, B: np.ndarray, tA: np.ndarray, tB: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    AB is hypotenuse of a 45-45-90 triangle. Define a deterministic plane:

      ex = unit(B-A)   (along AB)
      ey = unit(ex x t_avg)  (perpendicular direction for triangle "height")

    where t_avg is average of endpoint tangents.
    Triangle apex is at midpoint +/- (L/2)*ey.
    Returns (ex, ey, L).
    """
    v = B - A
    L = float(np.linalg.norm(v))
    if L < 1e-12:
        return np.zeros(3), np.zeros(3), 0.0

    ex = unit(v)
    t_avg = unit(unit(tA) + unit(tB))

    # if t_avg degenerates, fall back to global z
    if float(np.linalg.norm(t_avg)) < 1e-12:
        t_avg = np.array([0.0, 0.0, 1.0])

    ey = unit(np.cross(ex, t_avg))

    # if ex || t_avg, cross ~ 0; pick another fallback axis
    if float(np.linalg.norm(ey)) < 1e-12:
        fallback = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(ex, fallback))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        ey = unit(np.cross(ex, fallback))

    return ex, ey, L


def unwrap_arc_metrics(n0: int, n1: int, par: Coil3DParams, steps: int = 300) -> tuple[float, float, float, float]:
    """
    For consecutive nodes at parameters n0->n1:

    1) Build canonical 45-45-90 triangle plane with AB as hypotenuse.
    2) Unwrap coil segment P(t) into that plane using:
          x = (P(t)-A)·ex,  y = (P(t)-A)·ey
       so A=(0,0), B=(L,0).
    3) Force y to be "above" AB by flipping if needed.
    4) Compute:
       - u*   = x_at_peak / L
       - yMax = max(y)
       - area = ∫ y dx (trapezoid; only y>=0)
       - sym  = area_left / area_right split at peak x*
    """
    A = P(float(n0), par)
    B = P(float(n1), par)
    tA = dP(float(n0), par)
    tB = dP(float(n1), par)

    ex, ey, L = canonical_triangle_axes(A, B, tA, tB)
    if L <= 0.0 or float(np.linalg.norm(ex)) < 1e-12 or float(np.linalg.norm(ey)) < 1e-12:
        return float("nan"), float("nan"), float("nan"), float("nan")

    ts = np.linspace(float(n0), float(n1), steps)
    Q = np.array([P(t, par) for t in ts], dtype=float)
    D = Q - A

    x = D @ ex
    y = D @ ey

    # normalize x to [0,L] and keep points inside
    mask = (x >= -1e-9) & (x <= L + 1e-9)
    x = x[mask]
    y = y[mask]
    if x.size < 10:
        return float("nan"), float("nan"), float("nan"), float("nan")

    # make "above" AB: flip so peak is positive
    if float(np.nanmax(y)) < abs(float(np.nanmin(y))):
        y = -y

    # clip to y>=0 for area comparisons (we care about bulge above chord)
    y_pos = np.maximum(y, 0.0)

    idx = int(np.argmax(y_pos))
    x_star = float(x[idx])
    y_max = float(y_pos[idx])

    # sort by x for integration stability
    order = np.argsort(x)
    xs = x[order]
    ys = y_pos[order]

    # trapezoid integral over [0,L]
    area = float(np.trapz(ys, xs))

    # left/right area split at x_star
    left_mask = xs <= x_star
    right_mask = xs >= x_star

    area_left = float(np.trapz(ys[left_mask], xs[left_mask])) if np.count_nonzero(left_mask) >= 2 else 0.0
    area_right = float(np.trapz(ys[right_mask], xs[right_mask])) if np.count_nonzero(right_mask) >= 2 else 0.0

    sym = float("nan")
    if area_right > 1e-12:
        sym = area_left / area_right

    u_star = float("nan") if L <= 1e-12 else (x_star / L)

    return u_star, y_max, area, sym


def compute_for_sequence(seq: np.ndarray, par: Coil3DParams, steps: int) -> dict[str, np.ndarray]:
    pairs = len(seq) - 1
    u_star = np.empty(pairs, dtype=float)
    y_max = np.empty(pairs, dtype=float)
    area = np.empty(pairs, dtype=float)
    sym = np.empty(pairs, dtype=float)

    for k in range(pairs):
        u, ym, a, s = unwrap_arc_metrics(int(seq[k]), int(seq[k + 1]), par, steps=steps)
        u_star[k] = u
        y_max[k] = ym
        area[k] = a
        sym[k] = s

    return {"u_star": u_star, "y_max": y_max, "area": area, "sym": sym}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unwrap each prime-to-prime coil segment into a canonical 45-45-90 triangle plane (AB is hypotenuse) and compare arc metrics."
    )
    ap.add_argument("--pairs", type=int, default=5000)
    ap.add_argument("--prime-limit", type=int, default=2_000_000)
    ap.add_argument("--steps", type=int, default=300, help="Samples along each arc for unwrapping metrics.")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--control-gmax", type=int, default=200)
    ap.add_argument("--bins", type=int, default=120)
    ap.add_argument("--no-plot", action="store_true")

    # Coil params
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=0.02)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.1)

    args = ap.parse_args()

    par = Coil3DParams(a=args.a, b=args.b, omega=args.omega, c=args.c)
    rng = np.random.default_rng(args.seed)

    primes = primes_upto(args.prime_limit)
    if primes.size < args.pairs + 2:
        raise SystemExit("Increase --prime-limit; not enough primes for requested --pairs.")

    seq_pr = build_sequence_primes(primes, args.pairs)
    feat_pr = compute_for_sequence(seq_pr, par, steps=args.steps)

    print("\nTriangle-unwrapped arc metrics (primes):")
    summarize("u_star (peak position along AB)", feat_pr["u_star"])
    summarize("y_max (peak height above AB)", feat_pr["y_max"])
    summarize("area = ∫ y dx (above AB)", feat_pr["area"])
    summarize("sym = area_left/area_right", feat_pr["sym"])

    feat_sh = feat_re = None
    if args.control:
        seq_sh = build_sequence_shuffle_gaps(primes, args.pairs, rng=rng, start=int(seq_pr[0]))
        seq_re = build_sequence_random_even_gaps(args.pairs, rng=rng, start=int(seq_pr[0]), gmax=int(args.control_gmax))

        feat_sh = compute_for_sequence(seq_sh, par, steps=args.steps)
        feat_re = compute_for_sequence(seq_re, par, steps=args.steps)

        print("\nTriangle-unwrapped arc metrics (shuffled prime gaps control):")
        summarize("u_star (peak position along AB)", feat_sh["u_star"])
        summarize("y_max (peak height above AB)", feat_sh["y_max"])
        summarize("area = ∫ y dx (above AB)", feat_sh["area"])
        summarize("sym = area_left/area_right", feat_sh["sym"])

        print("\nTriangle-unwrapped arc metrics (random even gaps control):")
        summarize("u_star (peak position along AB)", feat_re["u_star"])
        summarize("y_max (peak height above AB)", feat_re["y_max"])
        summarize("area = ∫ y dx (above AB)", feat_re["area"])
        summarize("sym = area_left/area_right", feat_re["sym"])

    if args.no_plot:
        return

    k = np.arange(args.pairs)

    def scatter_metric(title: str, y_pr: np.ndarray, y_sh: np.ndarray | None, y_re: np.ndarray | None, ylabel: str):
        plt.figure(figsize=(10, 5))
        plt.scatter(k, y_pr, s=6, alpha=0.6, label="Primes")
        if y_sh is not None:
            plt.scatter(k, y_sh, s=6, alpha=0.35, label="Shuffled gaps")
        if y_re is not None:
            plt.scatter(k, y_re, s=6, alpha=0.35, label="Random even gaps")
        plt.xlabel("Pair index k")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def hist_metric(title: str, y_pr: np.ndarray, y_sh: np.ndarray | None, y_re: np.ndarray | None, xlabel: str):
        plt.figure(figsize=(10, 5))
        plt.hist(y_pr[np.isfinite(y_pr)], bins=args.bins, density=True, alpha=0.7, label="Primes")
        if y_sh is not None:
            plt.hist(y_sh[np.isfinite(y_sh)], bins=args.bins, density=True, alpha=0.5, label="Shuffled gaps")
        if y_re is not None:
            plt.hist(y_re[np.isfinite(y_re)], bins=args.bins, density=True, alpha=0.4, label="Random even gaps")
        plt.xlabel(xlabel)
        plt.ylabel("density")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    y_sh_u = feat_sh["u_star"] if (args.control and feat_sh is not None) else None
    y_re_u = feat_re["u_star"] if (args.control and feat_re is not None) else None
    y_sh_h = feat_sh["y_max"] if (args.control and feat_sh is not None) else None
    y_re_h = feat_re["y_max"] if (args.control and feat_re is not None) else None
    y_sh_a = feat_sh["area"] if (args.control and feat_sh is not None) else None
    y_re_a = feat_re["area"] if (args.control and feat_re is not None) else None
    y_sh_s = feat_sh["sym"] if (args.control and feat_sh is not None) else None
    y_re_s = feat_re["sym"] if (args.control and feat_re is not None) else None

    scatter_metric("u* peak location along AB", feat_pr["u_star"], y_sh_u, y_re_u, "u*")
    hist_metric("Distribution of u* (peak position)", feat_pr["u_star"], y_sh_u, y_re_u, "u*")

    scatter_metric("Peak height y_max above AB", feat_pr["y_max"], y_sh_h, y_re_h, "y_max")
    # log-hist for y_max can be helpful; keep linear for now.

    scatter_metric("Area under arc above AB (∫ y dx)", feat_pr["area"], y_sh_a, y_re_a, "area")
    hist_metric("Distribution of area (∫ y dx)", feat_pr["area"], y_sh_a, y_re_a, "area")

    # Sym ratio can be heavy-tailed; plot clipped histogram
    def clip_for_hist(arr: np.ndarray, q: float = 0.99) -> tuple[np.ndarray, float]:
        a = arr[np.isfinite(arr)]
        if a.size == 0:
            return a, 10.0
        hi = float(np.quantile(a, q))
        hi = max(5.0, hi)
        return np.clip(a, 0, hi), hi

    plt.figure(figsize=(10, 5))
    prc, hi = clip_for_hist(feat_pr["sym"])
    plt.hist(prc, bins=args.bins, density=True, alpha=0.7, label="Primes")
    if y_sh_s is not None:
        shc, _ = clip_for_hist(y_sh_s)
        plt.hist(shc, bins=args.bins, density=True, alpha=0.5, label="Shuffled gaps")
    if y_re_s is not None:
        rec, _ = clip_for_hist(y_re_s)
        plt.hist(rec, bins=args.bins, density=True, alpha=0.4, label="Random even gaps")
    plt.xlabel(f"sym = area_left/area_right (clipped ≤ {hi:.3g})")
    plt.ylabel("density")
    plt.title("Distribution of symmetry ratio")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
