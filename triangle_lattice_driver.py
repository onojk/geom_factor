#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from coil3d import Coil3DParams, P


# -----------------------
# helpers
# -----------------------
def wrap_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def summarize(name: str, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        print(f"{name}: (no finite values)")
        return
    p05, p50, p95 = np.percentile(x, [5, 50, 95])
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    print(
        f"{name}: n={x.size}, mean={x.mean():.6g}, std={x.std():.6g}, "
        f"p05={p05:.6g}, p50={p50:.6g}, p95={p95:.6g}, IQR={iqr:.6g}"
    )


def primes_upto(limit: int) -> np.ndarray:
    """Simple sieve, returns primes <= limit as numpy array."""
    if limit < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    p = 2
    while p * p <= limit:
        if sieve[p]:
            sieve[p * p : limit + 1 : p] = False
        p += 1
    return np.flatnonzero(sieve).astype(np.int64)


def build_sequence_primes(primes: np.ndarray, N: int) -> np.ndarray:
    return primes[:N].astype(np.int64)


def build_sequence_shuffle_gaps(primes: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    """Keep same multiset of prime gaps, randomize order. Monotone but not primes."""
    ps = primes[:N].astype(np.int64)
    gaps = (ps[1:] - ps[:-1]).astype(np.int64)
    rng.shuffle(gaps)
    seq = np.empty(N, dtype=np.int64)
    seq[0] = int(ps[0])
    for i in range(N - 1):
        seq[i + 1] = seq[i] + int(gaps[i])
    return seq


def build_sequence_random_even_gaps(
    N: int, rng: np.random.Generator, start: int = 3, gmax: int = 200
) -> np.ndarray:
    """Baseline control: random even gaps in [2, gmax]."""
    if gmax < 2:
        raise ValueError("gmax must be >= 2")
    evens = np.arange(2, gmax + 1, 2, dtype=np.int64)
    seq = np.empty(N, dtype=np.int64)
    seq[0] = int(start)
    gaps = rng.choice(evens, size=N - 1, replace=True)
    for i in range(N - 1):
        seq[i + 1] = seq[i] + int(gaps[i])
    return seq


# -----------------------
# metrics: XY + Z height
# -----------------------
def triangle_lattice_metrics(seq: np.ndarray, par: Coil3DParams) -> dict[str, np.ndarray]:
    """
    For each node k: point Q_k = P(seq[k]) in 3D.

    We treat the "taller triangle" idea as:
      - base plane triangle from origin to projection (x,y)
      - height z as the vertical leg
      - full 3D radius R = sqrt(x^2+y^2+z^2)

    Metrics returned:

    2D / bird's-eye:
      r_xy, theta, dr_xy, dtheta(wrapped), theta_unwrapped, dtheta_unwrapped

    Height / 3D:
      z, dz
      R3 = sqrt(x^2+y^2+z^2), dR3
      slope_z = dz / dr_xy   (how fast height rises per radial movement)
      pitch_angle = atan2(dz, dr_xy)   (angle of step relative to xy plane)

    Triangle angles (origin, bird's-eye):
      angle_to_x = |atan2(y, x)| in [0, pi]
      (and we also provide cos/sin/tan of theta)
    """
    Q = np.array([P(float(n), par) for n in seq], dtype=float)  # (N,3)
    x = Q[:, 0]
    y = Q[:, 1]
    z = Q[:, 2]

    # bird's-eye radial + heading
    r_xy = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    dr_xy = np.diff(r_xy)
    dtheta_w = wrap_pi(np.diff(theta))
    theta_u = np.unwrap(theta)
    dtheta_u = np.diff(theta_u)

    # 3D radius and height change
    R3 = np.sqrt(x * x + y * y + z * z)
    dR3 = np.diff(R3)
    dz = np.diff(z)

    # avoid div by 0
    eps = 1e-12
    slope_z = dz / np.maximum(np.abs(dr_xy), eps)
    pitch_angle = np.arctan2(dz, np.maximum(dr_xy, eps))  # signed

    # triangle “legs” in XY relative to axes
    sin_abs = np.abs(np.sin(theta))
    cos_abs = np.abs(np.cos(theta))
    tan_raw = np.abs(np.tan(theta))
    tan_abs = np.clip(tan_raw, 0.0, 50.0)

    angle_to_x = np.abs(theta)  # wrapped [-pi,pi] -> abs gives [0,pi]

    return {
        "x": x,
        "y": y,
        "z": z,
        "r_xy": r_xy,
        "theta": theta,
        "theta_u": theta_u,
        "dr_xy": dr_xy,
        "dtheta_w": dtheta_w,
        "dtheta_u": dtheta_u,
        "R3": R3,
        "dR3": dR3,
        "dz": dz,
        "slope_z": slope_z,
        "pitch_angle": pitch_angle,
        "sin_abs": sin_abs,
        "cos_abs": cos_abs,
        "tan_abs": tan_abs,
        "tan_raw": tan_raw,
        "angle_to_x": angle_to_x,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Triangle lattice fingerprint with z-height factor")
    ap.add_argument("--nodes", type=int, default=5000, help="Number of nodes (points) to analyze")
    ap.add_argument("--prime-limit", type=int, default=2_000_000, help="Sieve limit for primes")
    ap.add_argument("--control", action="store_true", help="Run controls too")
    ap.add_argument("--control-gmax", type=int, default=200, help="Max even gap for random-even control")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed for controls")
    ap.add_argument("--no-plot", action="store_true", help="Disable plots")

    # coil params
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=0.02)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.1)

    ap.add_argument("--bins", type=int, default=120)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    par = Coil3DParams(a=args.a, b=args.b, omega=args.omega, c=args.c)

    primes = primes_upto(args.prime_limit)
    if primes.size < args.nodes + 10:
        raise SystemExit("Increase --prime-limit (not enough primes for requested --nodes).")

    seq_pr = build_sequence_primes(primes, args.nodes)
    feat_pr = triangle_lattice_metrics(seq_pr, par)

    print("\nTriangle lattice (PRIMES):")
    summarize("r_xy", feat_pr["r_xy"])
    summarize("theta (wrapped)", feat_pr["theta"])
    summarize("dr_xy", feat_pr["dr_xy"])
    summarize("dtheta_w", feat_pr["dtheta_w"])
    summarize("dtheta_u", feat_pr["dtheta_u"])
    summarize("z", feat_pr["z"])
    summarize("dz", feat_pr["dz"])
    summarize("R3", feat_pr["R3"])
    summarize("dR3", feat_pr["dR3"])
    summarize("pitch_angle (rad)", feat_pr["pitch_angle"])
    summarize("slope_z = dz/dr_xy", feat_pr["slope_z"])

    feat_sh = feat_re = None
    if args.control:
        seq_sh = build_sequence_shuffle_gaps(primes, args.nodes, rng)
        feat_sh = triangle_lattice_metrics(seq_sh, par)

        seq_re = build_sequence_random_even_gaps(args.nodes, rng, start=int(seq_pr[0]), gmax=args.control_gmax)
        feat_re = triangle_lattice_metrics(seq_re, par)

        print("\nTriangle lattice (SHUFFLED PRIME GAPS control):")
        summarize("r_xy", feat_sh["r_xy"])
        summarize("dtheta_w", feat_sh["dtheta_w"])
        summarize("dtheta_u", feat_sh["dtheta_u"])
        summarize("dz", feat_sh["dz"])
        summarize("pitch_angle (rad)", feat_sh["pitch_angle"])
        summarize("slope_z = dz/dr_xy", feat_sh["slope_z"])

        print("\nTriangle lattice (RANDOM EVEN GAPS control):")
        summarize("r_xy", feat_re["r_xy"])
        summarize("dtheta_w", feat_re["dtheta_w"])
        summarize("dtheta_u", feat_re["dtheta_u"])
        summarize("dz", feat_re["dz"])
        summarize("pitch_angle (rad)", feat_re["pitch_angle"])
        summarize("slope_z = dz/dr_xy", feat_re["slope_z"])

    if args.no_plot:
        return

    # ---- plots ----
    def hist3(x_pr, x_sh, x_re, title, xlabel, bins=args.bins):
        plt.figure(figsize=(10, 5))
        plt.hist(x_pr[np.isfinite(x_pr)], bins=bins, density=True, alpha=0.55, label="Primes")
        if x_sh is not None:
            plt.hist(x_sh[np.isfinite(x_sh)], bins=bins, density=True, alpha=0.45, label="Shuffled gaps")
        if x_re is not None:
            plt.hist(x_re[np.isfinite(x_re)], bins=bins, density=True, alpha=0.35, label="Random even gaps")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # distributions that matter for “triangle lattice”
    hist3(feat_pr["dtheta_w"],
          feat_sh["dtheta_w"] if feat_sh is not None else None,
          feat_re["dtheta_w"] if feat_re is not None else None,
          "Heading change per step (wrapped) Δθ",
          "Δθ (radians)")

    hist3(feat_pr["pitch_angle"],
          feat_sh["pitch_angle"] if feat_sh is not None else None,
          feat_re["pitch_angle"] if feat_re is not None else None,
          "Pitch angle per step (atan2(dz, dr_xy))",
          "pitch angle (radians)")

    hist3(feat_pr["slope_z"],
          feat_sh["slope_z"] if feat_sh is not None else None,
          feat_re["slope_z"] if feat_re is not None else None,
          "Vertical slope per step (dz/dr_xy)",
          "dz/dr_xy")

    # scatter: (theta vs z) shows rotation + height “lattice”
    plt.figure(figsize=(7, 6))
    plt.scatter(feat_pr["theta"], feat_pr["z"], s=6, alpha=0.55, label="Primes")
    if feat_sh is not None:
        plt.scatter(feat_sh["theta"], feat_sh["z"], s=6, alpha=0.35, label="Shuffled gaps")
    if feat_re is not None:
        plt.scatter(feat_re["theta"], feat_re["z"], s=6, alpha=0.25, label="Random even gaps")
    plt.xlabel("theta (radians)")
    plt.ylabel("z (height)")
    plt.title("Triangle lattice: rotation vs height (theta, z)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # scatter: r_xy vs z (cone-ish unwrapping view)
    plt.figure(figsize=(7, 6))
    plt.scatter(feat_pr["r_xy"], feat_pr["z"], s=6, alpha=0.55, label="Primes")
    if feat_sh is not None:
        plt.scatter(feat_sh["r_xy"], feat_sh["z"], s=6, alpha=0.35, label="Shuffled gaps")
    if feat_re is not None:
        plt.scatter(feat_re["r_xy"], feat_re["z"], s=6, alpha=0.25, label="Random even gaps")
    plt.xlabel("r_xy")
    plt.ylabel("z (height)")
    plt.title("Triangle lattice: radial vs height (r_xy, z)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
