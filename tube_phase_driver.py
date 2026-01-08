#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

from coil3d import Coil3DParams, P, dP, angle_between
from signature_driver import primes_upto


# ----------------------------
# Basic vector helpers
# ----------------------------
def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def wrap_pi(a: np.ndarray) -> np.ndarray:
    # wrap to (-pi, pi]
    return (a + np.pi) % (2*np.pi) - np.pi


# ----------------------------
# Frenet frame (numerical)
# ----------------------------
def frenet_frame(t: float, par: Coil3DParams, eps: float = 1e-3):
    """
    Returns (T, N, B, kappa) at parameter t using numerical differentiation.

    T = unit tangent
    N = unit normal (from dT/dt)
    B = binormal
    kappa ~ curvature proxy = ||dT/dt|| / ||dP/dt||  (dimensionless-ish here)
    """
    v = dP(t, par)
    Tv = unit(v)
    if norm(Tv) == 0.0:
        return None

    # numerical derivative of T
    Tp = unit(dP(t + eps, par))
    Tm = unit(dP(t - eps, par))
    dT = (Tp - Tm) / (2*eps)
    dTn = norm(dT)
    if dTn < 1e-10:
        # near-zero curvature or numeric degeneracy
        return None

    N = dT / dTn
    B = np.cross(Tv, N)
    Bn = norm(B)
    if Bn < 1e-10:
        return None
    B = B / Bn

    speed = norm(v)
    kappa = dTn / max(speed, 1e-12)
    return Tv, N, B, kappa


# ----------------------------
# Tube phase / hugging angle
# ----------------------------
def tube_angle_at_node(n0: int, n1: int, par: Coil3DParams, eps: float = 1e-3):
    """
    Define theta at node n0 using outgoing chord to n1.

    Let Q0=P(n0), Q1=P(n1), E=Q1-Q0.
    Compute local Frenet frame (T,N,B) at n0.
    Project chord onto normal plane: E_perp = E - (E·T)T
    theta = atan2(E_perp·B, E_perp·N)   in (-pi, pi]
    """
    fr = frenet_frame(float(n0), par, eps=eps)
    if fr is None:
        return None

    T, N, B, kappa = fr
    Q0 = P(float(n0), par)
    Q1 = P(float(n1), par)
    E = Q1 - Q0

    # perpendicular component in normal plane
    E_perp = E - float(np.dot(E, T)) * T
    if norm(E_perp) < 1e-12:
        return None

    x = float(np.dot(E_perp, N))
    y = float(np.dot(E_perp, B))
    theta = math.atan2(y, x)  # (-pi, pi]
    return theta


# ----------------------------
# Controls
# ----------------------------
def build_prime_seq(primes: np.ndarray, pairs: int) -> np.ndarray:
    return primes[:pairs+1].astype(np.int64)

def build_shuffle_gap_seq(primes: np.ndarray, pairs: int, rng: np.random.Generator) -> np.ndarray:
    ps = primes[:pairs+1].astype(np.int64)
    gaps = (ps[1:] - ps[:-1]).astype(np.int64)
    rng.shuffle(gaps)
    seq = np.empty(pairs+1, dtype=np.int64)
    seq[0] = int(ps[0])
    for i in range(pairs):
        seq[i+1] = seq[i] + gaps[i]
    return seq

def build_random_even_gap_seq(pairs: int, start: int, gmax: int, rng: np.random.Generator) -> np.ndarray:
    evens = np.arange(2, gmax+1, 2, dtype=np.int64)
    gaps = rng.choice(evens, size=pairs, replace=True)
    seq = np.empty(pairs+1, dtype=np.int64)
    seq[0] = int(start)
    for i in range(pairs):
        seq[i+1] = seq[i] + int(gaps[i])
    return seq


# ----------------------------
# Circular stats
# ----------------------------
def circ_mean_resultant(theta: np.ndarray):
    """
    Returns (mu, R) where:
      mu is circular mean angle
      R is mean resultant length in [0,1] (1 = perfectly concentrated)
    """
    z = np.exp(1j * theta)
    m = np.mean(z)
    mu = float(np.angle(m))
    R = float(np.abs(m))
    return mu, R

def circ_autocorr(theta: np.ndarray, max_lag: int = 50):
    """
    Circular autocorrelation using complex representation.
    Returns lags array and correlation magnitudes.
    """
    z = np.exp(1j * theta)
    z = z - np.mean(z)
    denom = np.mean(np.abs(z)**2)
    if denom <= 0:
        lags = np.arange(1, max_lag+1)
        return lags, np.zeros_like(lags, dtype=float)

    corrs = []
    for lag in range(1, max_lag+1):
        if lag >= len(z):
            corrs.append(0.0)
            continue
        c = np.mean(z[:-lag] * np.conj(z[lag:])) / denom
        corrs.append(float(np.abs(c)))
    return np.arange(1, max_lag+1), np.array(corrs, dtype=float)


# ----------------------------
# Main
# ----------------------------
def compute_theta_series(seq: np.ndarray, par: Coil3DParams, eps: float) -> np.ndarray:
    thetas = []
    for a, b in zip(seq[:-1], seq[1:]):
        th = tube_angle_at_node(int(a), int(b), par, eps=eps)
        if th is not None and np.isfinite(th):
            thetas.append(th)
    return np.array(thetas, dtype=float)


def summarize(name: str, theta: np.ndarray):
    theta = theta[np.isfinite(theta)]
    if theta.size == 0:
        print(f"{name}: empty")
        return
    mu, R = circ_mean_resultant(theta)
    var = 1.0 - R
    q05, q50, q95 = np.quantile(theta, [0.05, 0.5, 0.95])
    print(f"{name}: n={theta.size}, circ_mean={mu:.6g}, R={R:.6g}, circ_var={var:.6g}, "
          f"p05={q05:.6g}, p50={q50:.6g}, p95={q95:.6g}")


def main():
    ap = argparse.ArgumentParser(
        description="Tube phase experiment: define a 0–2π angle around the coil tube using the outgoing chord direction in the local Frenet frame."
    )
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--prime-limit", type=int, default=20000000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--control-gmax", type=int, default=200)
    ap.add_argument("--bins", type=int, default=120)
    ap.add_argument("--eps", type=float, default=1e-3, help="Finite-difference step for Frenet frame")

    # coil params (match your defaults)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=0.02)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.1)

    ap.add_argument("--max-lag", type=int, default=80)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    par = Coil3DParams(a=args.a, b=args.b, omega=args.omega, c=args.c)

    primes = primes_upto(args.prime_limit)
    if primes.size < args.pairs + 2:
        raise SystemExit("Increase --prime-limit (not enough primes for --pairs).")

    seq_pr = build_prime_seq(primes, args.pairs)
    seq_sh = build_shuffle_gap_seq(primes, args.pairs, rng)
    seq_re = build_random_even_gap_seq(args.pairs, start=int(seq_pr[0]), gmax=args.control_gmax, rng=rng)

    th_pr = compute_theta_series(seq_pr, par, eps=args.eps)
    th_sh = compute_theta_series(seq_sh, par, eps=args.eps)
    th_re = compute_theta_series(seq_re, par, eps=args.eps)

    # Wrap to (-pi,pi] explicitly
    th_pr = wrap_pi(th_pr)
    th_sh = wrap_pi(th_sh)
    th_re = wrap_pi(th_re)

    summarize("theta(primes)", th_pr)
    summarize("theta(shuffled gaps)", th_sh)
    summarize("theta(random even gaps)", th_re)

    # Histograms (wrapped)
    plt.figure(figsize=(10,5))
    plt.hist(th_pr, bins=args.bins, density=True, alpha=0.55, label="Primes")
    plt.hist(th_sh, bins=args.bins, density=True, alpha=0.45, label="Shuffled prime gaps")
    plt.hist(th_re, bins=args.bins, density=True, alpha=0.35, label="Random even gaps")
    plt.xlabel("tube angle θ (radians, wrapped)")
    plt.ylabel("density")
    plt.title("Tube-phase angle distribution (Frenet-frame normal-plane heading)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Circular autocorrelation magnitude
    lags, ac_pr = circ_autocorr(th_pr, max_lag=args.max_lag)
    _, ac_sh = circ_autocorr(th_sh, max_lag=args.max_lag)
    _, ac_re = circ_autocorr(th_re, max_lag=args.max_lag)

    plt.figure(figsize=(10,4))
    plt.plot(lags, ac_pr, label="Primes")
    plt.plot(lags, ac_sh, label="Shuffled prime gaps")
    plt.plot(lags, ac_re, label="Random even gaps")
    plt.xlabel("lag")
    plt.ylabel("|circular autocorr|")
    plt.title("Tube-phase circular autocorrelation magnitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Cumulative unwrapped phase (first 2000) for visual “spiral around tube”
    def unwrap_series(theta: np.ndarray, m: int = 2000):
        t = theta[:m].copy()
        return np.unwrap(t)

    m = min(2000, len(th_pr), len(th_sh), len(th_re))
    if m >= 10:
        plt.figure(figsize=(10,4))
        plt.plot(unwrap_series(th_pr, m), label="Primes")
        plt.plot(unwrap_series(th_sh, m), label="Shuffled prime gaps", alpha=0.9)
        plt.plot(unwrap_series(th_re, m), label="Random even gaps", alpha=0.9)
        plt.xlabel("k")
        plt.ylabel("unwrapped θ")
        plt.title("Unwrapped tube-phase (first segment)")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
