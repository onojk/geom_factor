#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# primes helper (prefer repo)
# -----------------------------
def sieve_primes_upto(limit: int) -> np.ndarray:
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


def get_primes_upto() -> Callable[[int], np.ndarray]:
    try:
        import signature_driver  # type: ignore

        if hasattr(signature_driver, "primes_upto"):
            return signature_driver.primes_upto  # type: ignore
        if hasattr(signature_driver, "primes_up_to"):
            return signature_driver.primes_up_to  # type: ignore
    except Exception:
        pass
    return sieve_primes_upto


# -----------------------------
# coil P() compatibility
# -----------------------------
@dataclass
class CoilParams:
    a: float = 1.0
    b: float = 0.02
    omega: float = 0.5
    c: float = 0.1


def resolve_coil_params_class():
    """
    Try to use the repo's param class if present; otherwise keep our dataclass.
    """
    try:
        import coil3d  # type: ignore

        for name in ("Coil3DParams", "Coil3dParams", "Params", "CoilParams"):
            if hasattr(coil3d, name):
                return getattr(coil3d, name)
    except Exception:
        pass
    return CoilParams


def resolve_P_caller(par_obj) -> Callable[[float], np.ndarray]:
    """
    Build a callable f(n)->(x,y,z) that adapts to coil3d.P signature.
    Tries:
      P(n, par)
      P(n)
      P(n, a, b, omega, c)
    """
    import coil3d  # type: ignore

    P = coil3d.P  # type: ignore

    # Try to inspect signature, but don't trust it entirely.
    sig = None
    try:
        sig = inspect.signature(P)
    except Exception:
        sig = None

    def try_call(n: float) -> np.ndarray:
        # 1) P(n, par)
        try:
            q = P(float(n), par_obj)
            return np.asarray(q, dtype=float)
        except Exception:
            pass
        # 2) P(n)
        try:
            q = P(float(n))
            return np.asarray(q, dtype=float)
        except Exception:
            pass
        # 3) P(n, a,b,omega,c)
        try:
            q = P(float(n), float(par_obj.a), float(par_obj.b), float(par_obj.omega), float(par_obj.c))
            return np.asarray(q, dtype=float)
        except Exception as e:
            # Provide a helpful error once
            msg = "Unable to call coil3d.P with any supported signature.\n"
            msg += f"Attempted: P(n, par), P(n), P(n, a,b,omega,c)\n"
            if sig is not None:
                msg += f"Detected signature: {sig}\n"
            msg += f"Last error: {e}\n"
            raise TypeError(msg) from e

    return try_call


# -----------------------------
# sequences (primes + controls)
# -----------------------------
def build_sequence_primes(primes: np.ndarray, pairs: int) -> np.ndarray:
    return primes[: pairs + 1].astype(np.int64)


def build_sequence_shuffle_gaps(primes: np.ndarray, pairs: int, rng: np.random.Generator) -> np.ndarray:
    ps = primes[: pairs + 1].astype(np.int64)
    gaps = (ps[1:] - ps[:-1]).astype(np.int64)
    rng.shuffle(gaps)
    seq = np.empty(pairs + 1, dtype=np.int64)
    seq[0] = int(ps[0])
    for i in range(pairs):
        seq[i + 1] = seq[i] + int(gaps[i])
    return seq


def build_sequence_random_even(pairs: int, rng: np.random.Generator, start: int = 3, gmax: int = 200) -> np.ndarray:
    if gmax < 2:
        raise ValueError("gmax must be >= 2")
    seq = np.empty(pairs + 1, dtype=np.int64)
    seq[0] = int(start)
    gaps = rng.integers(1, (gmax // 2) + 1, size=pairs, dtype=np.int64) * 2
    for i in range(pairs):
        seq[i + 1] = seq[i] + int(gaps[i])
    return seq


# -----------------------------
# cone volume logic
# -----------------------------
def cone_k_fit(z: np.ndarray, r: np.ndarray) -> float:
    """
    Fit r ≈ k z for a single cone slope k using least squares:
      k = (z·r)/(z·z), ignoring tiny z
    """
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    mask = np.isfinite(z) & np.isfinite(r) & (np.abs(z) > 1e-12)
    if mask.sum() < 3:
        return float("nan")
    zz = z[mask]
    rr = r[mask]
    return float(np.dot(zz, rr) / np.dot(zz, zz))


def stats(name: str, x: np.ndarray) -> None:
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


def compute_volume_metrics(seq: np.ndarray, Pcall: Callable[[float], np.ndarray], mode: str) -> dict[str, np.ndarray]:
    """
    mode:
      - 'fit'     : single cone slope k fitted from (z,r) then r_cone=k*z
      - 'measured': use measured r at each node directly
    """
    Q = np.array([Pcall(float(n)) for n in seq], dtype=float)  # (pairs+1,3)
    x, y, z = Q[:, 0], Q[:, 1], Q[:, 2]
    r = np.sqrt(x * x + y * y)

    if mode == "fit":
        k = cone_k_fit(z, r)
        r_cone = k * z
        V = (math.pi / 3.0) * (r_cone ** 2) * z
        k_series = np.full_like(z, k, dtype=float)
    elif mode == "measured":
        V = (math.pi / 3.0) * (r ** 2) * z
        k_series = np.where(np.abs(z) > 1e-12, r / z, np.nan)
    else:
        raise ValueError("mode must be 'fit' or 'measured'")

    dV = np.diff(V)
    dz = np.diff(z)
    dr = np.diff(r)

    # normalized to remove cubic growth if cone-ish: V / z^3
    z3 = np.where(np.abs(z) > 1e-12, z ** 3, np.nan)
    V_norm = V / z3

    # another useful normalization: dV / (z^2) ~ const * dz if perfect cone
    z2_mid = np.where(np.abs(z[:-1]) > 1e-12, (z[:-1] ** 2), np.nan)
    dV_over_z2 = dV / z2_mid

    return {
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "k": k_series,
        "V": V,
        "dV": dV,
        "dz": dz,
        "dr": dr,
        "V_norm": V_norm,
        "dV_over_z2": dV_over_z2,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fill a cone with water up to each prime point; compare volumes + controls")
    ap.add_argument("--pairs", type=int, default=5000)
    ap.add_argument("--prime-limit", type=int, default=2_000_000)

    ap.add_argument("--cone-mode", choices=["fit", "measured"], default="fit",
                    help="fit: single cone r=kz; measured: use r at each point directly")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--control-gmax", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--bins", type=int, default=100)

    # coil params
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=0.02)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.1)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    primes_upto = get_primes_upto()
    primes = primes_upto(args.prime_limit)
    if primes.size < args.pairs + 2:
        raise SystemExit("Increase --prime-limit (not enough primes for --pairs)")

    # params + P caller
    ParamClass = resolve_coil_params_class()
    try:
        par = ParamClass(a=args.a, b=args.b, omega=args.omega, c=args.c)
    except Exception:
        # fallback to our dataclass if repo class init differs
        par = CoilParams(a=args.a, b=args.b, omega=args.omega, c=args.c)

    import coil3d  # type: ignore
    if not hasattr(coil3d, "P"):
        raise SystemExit("coil3d.P not found. Check your repo.")

    Pcall = resolve_P_caller(par)

    # primes
    seq_pr = build_sequence_primes(primes, args.pairs)
    feat_pr = compute_volume_metrics(seq_pr, Pcall, args.cone_mode)

    print(f"\nCone fill volume ({args.cone_mode.upper()}) — PRIMES")
    stats("z", feat_pr["z"])
    stats("r", feat_pr["r"])
    stats("k (r/z)", feat_pr["k"])
    stats("V", feat_pr["V"])
    stats("dV", feat_pr["dV"])
    stats("V_norm = V/z^3", feat_pr["V_norm"])
    stats("dV_over_z2 = dV/z^2", feat_pr["dV_over_z2"])

    feat_sh = feat_re = None
    if args.control:
        seq_sh = build_sequence_shuffle_gaps(primes, args.pairs, rng)
        feat_sh = compute_volume_metrics(seq_sh, Pcall, args.cone_mode)
        print(f"\nCone fill volume ({args.cone_mode.upper()}) — SHUFFLED PRIME GAPS control")
        stats("V", feat_sh["V"])
        stats("dV", feat_sh["dV"])
        stats("V_norm = V/z^3", feat_sh["V_norm"])
        stats("dV_over_z2 = dV/z^2", feat_sh["dV_over_z2"])

        seq_re = build_sequence_random_even(args.pairs, rng, start=int(seq_pr[0]), gmax=args.control_gmax)
        feat_re = compute_volume_metrics(seq_re, Pcall, args.cone_mode)
        print(f"\nCone fill volume ({args.cone_mode.upper()}) — RANDOM EVEN GAPS control")
        stats("V", feat_re["V"])
        stats("dV", feat_re["dV"])
        stats("V_norm = V/z^3", feat_re["V_norm"])
        stats("dV_over_z2 = dV/z^2", feat_re["dV_over_z2"])

    if args.no_plot:
        return

    # ---------------- plotting ----------------
    # 1) V vs index
    plt.figure(figsize=(10, 5))
    plt.plot(feat_pr["V"], label="Primes")
    if feat_sh is not None:
        plt.plot(feat_sh["V"], alpha=0.7, label="Shuffled gaps")
    if feat_re is not None:
        plt.plot(feat_re["V"], alpha=0.5, label="Random even gaps")
    plt.title("Cone fill volume V_k (cumulative) vs step index k")
    plt.xlabel("k (pair index)")
    plt.ylabel("V")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) dV histogram
    plt.figure(figsize=(10, 5))
    plt.hist(feat_pr["dV"][np.isfinite(feat_pr["dV"])], bins=args.bins, density=True, alpha=0.6, label="Primes")
    if feat_sh is not None:
        plt.hist(feat_sh["dV"][np.isfinite(feat_sh["dV"])], bins=args.bins, density=True, alpha=0.45, label="Shuffled gaps")
    if feat_re is not None:
        plt.hist(feat_re["dV"][np.isfinite(feat_re["dV"])], bins=args.bins, density=True, alpha=0.35, label="Random even gaps")
    plt.title("ΔV distribution")
    plt.xlabel("ΔV = V_{k+1} - V_k")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) normalized V/z^3 (should be ~constant for a perfect cone)
    plt.figure(figsize=(10, 5))
    plt.plot(feat_pr["V_norm"], label="Primes")
    if feat_sh is not None:
        plt.plot(feat_sh["V_norm"], alpha=0.7, label="Shuffled gaps")
    if feat_re is not None:
        plt.plot(feat_re["V_norm"], alpha=0.5, label="Random even gaps")
    plt.title("Normalized volume V / z^3 vs k (cone sanity check)")
    plt.xlabel("k")
    plt.ylabel("V / z^3")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4) dV / z^2 (should be ~const * dz for perfect cone)
    plt.figure(figsize=(10, 5))
    plt.hist(feat_pr["dV_over_z2"][np.isfinite(feat_pr["dV_over_z2"])], bins=args.bins, density=True, alpha=0.6, label="Primes")
    if feat_sh is not None:
        plt.hist(feat_sh["dV_over_z2"][np.isfinite(feat_sh["dV_over_z2"])], bins=args.bins, density=True, alpha=0.45, label="Shuffled gaps")
    if feat_re is not None:
        plt.hist(feat_re["dV_over_z2"][np.isfinite(feat_re["dV_over_z2"])], bins=args.bins, density=True, alpha=0.35, label="Random even gaps")
    plt.title("ΔV / z^2 distribution (removes cubic growth)")
    plt.xlabel("ΔV / z^2")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
