#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List

import numpy as np

# ---------- prime helpers ----------
def primes_upto_simple(limit: int) -> np.ndarray:
    if limit < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    p = 2
    while p * p <= limit:
        if sieve[p]:
            sieve[p*p : limit+1 : p] = False
        p += 1
    return np.flatnonzero(sieve).astype(np.int64)

try:
    from signature_driver import primes_upto as primes_upto_repo  # type: ignore
except Exception:
    primes_upto_repo = None

def primes_upto(limit: int) -> np.ndarray:
    if primes_upto_repo is not None:
        return primes_upto_repo(limit)
    return primes_upto_simple(limit)

def is_prime_trial(n: int, small_primes: np.ndarray) -> bool:
    if n < 2:
        return False
    for p in small_primes:
        pp = int(p)
        if pp * pp > n:
            break
        if n % pp == 0:
            return n == pp
    return True

# Wheel-30 residues for primes > 5
WHEEL30 = {1, 7, 11, 13, 17, 19, 23, 29}

# ---------- coil access ----------
def make_P_caller(a: float, b: float, omega: float, c: float) -> Callable[[float], Tuple[float, float, float]]:
    """
    Try several coil3d.P signatures:
      P(n)
      P(n, params)
      P(n, a, b, omega, c)
    Returns a callable Pn(n)->(x,y,z)
    """
    import coil3d  # local module in repo

    P = coil3d.P

    # Try P(n) first
    try:
        x, y, z = P(10.0)  # type: ignore
        return lambda n: tuple(P(float(n)))  # type: ignore
    except TypeError:
        pass
    except Exception:
        # If it failed for other reasons, keep trying other signatures.
        pass

    # Try params class
    Params = None
    for name in ("Coil3DParams", "Coil3dParams"):
        if hasattr(coil3d, name):
            Params = getattr(coil3d, name)
            break

    if Params is not None:
        try:
            par = Params(a=a, b=b, omega=omega, c=c)
            x, y, z = P(10.0, par)  # type: ignore
            return lambda n: tuple(P(float(n), par))  # type: ignore
        except TypeError:
            pass
        except Exception:
            pass

    # Try expanded args
    try:
        x, y, z = P(10.0, a, b, omega, c)  # type: ignore
        return lambda n: tuple(P(float(n), a, b, omega, c))  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Could not call coil3d.P with any known signature. "
            "Try opening coil3d.py and tell me how P is defined."
        ) from e

# ---------- geometry model: theta histogram ----------
@dataclass
class ThetaModel:
    bins: int
    edges: np.ndarray      # bin edges length bins+1
    prob: np.ndarray       # normalized probability per bin length bins
    thr: float             # probability threshold for “geometry-allowed” candidates

def theta_of_point(x: float, y: float) -> float:
    t = math.atan2(y, x)
    if t < 0:
        t += 2.0 * math.pi
    return t

def build_theta_model(primes: np.ndarray, Pn: Callable[[float], Tuple[float,float,float]],
                      bins: int, top_quantile: float) -> ThetaModel:
    thetas = np.empty(primes.size, dtype=np.float64)
    for i, p in enumerate(primes):
        x, y, _z = Pn(float(p))
        thetas[i] = theta_of_point(x, y)

    hist, edges = np.histogram(thetas, bins=bins, range=(0.0, 2.0*math.pi))
    prob = hist.astype(np.float64)
    prob = prob / max(prob.sum(), 1.0)

    # set threshold by quantile of nonzero bins (avoid zeros dominating)
    nz = prob[prob > 0]
    if nz.size == 0:
        thr = 0.0
    else:
        thr = float(np.quantile(nz, top_quantile))
    return ThetaModel(bins=bins, edges=edges, prob=prob, thr=thr)

def theta_bin(model: ThetaModel, theta: float) -> int:
    # np.searchsorted returns index in edges; clamp to [0,bins-1]
    j = int(np.searchsorted(model.edges, theta, side="right") - 1)
    if j < 0:
        return 0
    if j >= model.bins:
        return model.bins - 1
    return j

def theta_score(model: ThetaModel, theta: float) -> float:
    return float(model.prob[theta_bin(model, theta)])

# ---------- sieve-on-coil search ----------
def sieve_on_coil_find_next(primes_train: np.ndarray,
                            start_n: int,
                            how_many: int,
                            Pn: Callable[[float], Tuple[float,float,float]],
                            bins: int,
                            top_quantile: float,
                            trial_bound: int,
                            max_steps: int) -> dict:
    model = build_theta_model(primes_train, Pn, bins=bins, top_quantile=top_quantile)

    # trial primes up to bound
    small_pr = primes_upto(trial_bound)

    found: List[int] = []
    checked = 0
    wheel_pass = 0
    geom_pass = 0
    trial_tests = 0

    n = max(2, start_n + 1)

    # Make n odd to avoid even scanning (except n==2)
    if n > 2 and n % 2 == 0:
        n += 1

    for step in range(max_steps):
        checked += 1

        # basic wheel filter (skip 2,3,5 separately)
        if n in (2, 3, 5):
            found.append(n)
            if len(found) >= how_many:
                break
            n += 2
            continue

        if n % 2 == 0 or n % 3 == 0 or n % 5 == 0:
            n += 2
            continue

        if (n % 30) not in WHEEL30:
            n += 2
            continue
        wheel_pass += 1

        # geometry filter
        x, y, _z = Pn(float(n))
        th = theta_of_point(x, y)
        sc = theta_score(model, th)
        if sc < model.thr:
            n += 2
            continue
        geom_pass += 1

        # trial division confirmation (honest)
        trial_tests += 1
        if is_prime_trial(n, small_pr):
            found.append(n)
            if len(found) >= how_many:
                break

        n += 2

    return {
        "start_n": start_n,
        "found": found,
        "checked": checked,
        "wheel_pass": wheel_pass,
        "geom_pass": geom_pass,
        "trial_tests": trial_tests,
        "bins": bins,
        "thr": model.thr,
        "trial_bound": trial_bound,
        "max_steps": max_steps,
    }

# ---------- main ----------
def main() -> None:
    ap = argparse.ArgumentParser(description="Sieve-on-the-coil: wheel+geometry prioritization + honest trial division")
    ap.add_argument("--prime-limit", type=int, default=2_000_000, help="Training primes sieve limit")
    ap.add_argument("--train", type=int, default=50000, help="How many primes to train theta histogram on")
    ap.add_argument("--start", type=int, default=1_000_000, help="Search for primes strictly > start")
    ap.add_argument("--find", type=int, default=10, help="How many next primes to find")
    ap.add_argument("--bins", type=int, default=180, help="Theta histogram bins")
    ap.add_argument("--top-quantile", type=float, default=0.60, help="Keep bins with prob >= this quantile of nonzero bins (0..1)")
    ap.add_argument("--trial-bound", type=int, default=200000, help="Trial division primes up to this bound (>=sqrt(start) ideal)")
    ap.add_argument("--max-steps", type=int, default=5_000_000, help="Max candidate steps to scan")

    # coil params if needed by your P signature
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=0.02)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.1)

    args = ap.parse_args()

    Pn = make_P_caller(args.a, args.b, args.omega, args.c)

    primes = primes_upto(args.prime_limit)
    if primes.size < args.train + 10:
        print("Not enough primes for training; increase --prime-limit", file=sys.stderr)
        sys.exit(2)
    primes_train = primes[: args.train]

    res = sieve_on_coil_find_next(
        primes_train=primes_train,
        start_n=args.start,
        how_many=args.find,
        Pn=Pn,
        bins=args.bins,
        top_quantile=args.top_quantile,
        trial_bound=args.trial_bound,
        max_steps=args.max_steps,
    )

    print("\nSIEVE-ON-COIL RESULTS")
    print(f"start_n      = {res['start_n']}")
    print(f"found        = {res['found']}")
    print(f"checked      = {res['checked']}")
    print(f"wheel_pass   = {res['wheel_pass']}")
    print(f"geom_pass    = {res['geom_pass']}")
    print(f"trial_tests  = {res['trial_tests']}")
    print(f"bins         = {res['bins']}")
    print(f"theta_thr    = {res['thr']:.6g}")
    print(f"trial_bound  = {res['trial_bound']}")
    print(f"max_steps    = {res['max_steps']}")

if __name__ == "__main__":
    main()
