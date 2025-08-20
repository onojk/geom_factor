#!/usr/bin/env python3
"""
budget_driver.py
- Reads N from file or raw string (digits-only)
- Tries: trial division -> Pollard p-1 -> ECM
- ECM can use external GMP-ECM ('ecm') via --use-gmp, else pure-Python fallback
"""

import argparse, re, sys, subprocess, shutil, math, re as _re
from math import gcd

from pminus1 import pollard_p1
from ecm import ecm  # your stage-1 Python ECM

try:
    import sympy
except Exception:
    sympy = None

def read_number(source: str) -> int:
    try:
        with open(source, "r") as f:
            text = f.read()
    except FileNotFoundError:
        text = source
    digits = re.sub(r"[^0-9]", "", text)
    if not digits:
        raise ValueError("No digits found in input")
    return int(digits)

def trial_division(n: int, bound: int = 100000):
    if n % 2 == 0: return 2
    if n % 3 == 0: return 3
    if sympy is None:
        # simple 2*3 wheel
        p, step = 5, 2
        while p*p <= n and p <= bound:
            if n % p == 0:
                return p
            p += step
            step = 6 - step
        return None
    for p in sympy.primerange(5, bound+1):
        if n % p == 0:
            return p
    return None

def show_factor(n, f, tag):
    if f is None or f <= 1 or f >= n:
        return False
    print(f"[{tag}] factor found:", f)
    print("cofactor:", n // f)
    return True

def parse_ecm_output(n: int, out: str):
    """Grab any decimal token that actually divides n."""
    for tok in _re.findall(r"\b\d+\b", out):
        try:
            val = int(tok)
        except ValueError:
            continue
        if 1 < val < n and n % val == 0:
            return val
    return None

def run_gmp_ecm(n: int, B1: int, curves: int, B2: int | None = None, timeout: int = 60):
    """
    Call external 'ecm' if available. Returns factor or None.
    We use '-q' (quiet) and '-c curves'. Pass B1 (and optional B2).
    """
    if shutil.which("ecm") is None:
        return None  # signal to caller to use Python fallback

    cmd = ["ecm", "-q", "-c", str(curves)]
    if B2 and B2 > B1:
        cmd += [str(B1), str(B2)]
    else:
        cmd += [str(B1)]

    try:
        proc = subprocess.run(
            cmd,
            input=(str(n) + "\n").encode(),
            capture_output=True,
            text=False,
            timeout=None if timeout <= 0 else timeout,
        )
    except Exception as ex:
        print(f"[ecm] error: {ex}")
        return None

    out = (proc.stdout or b"") + b"\n" + (proc.stderr or b"")
    f = parse_ecm_output(n, out.decode())
    return f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="file containing N or a raw numeric string")
    ap.add_argument("--trial", type=int, default=100000, help="trial division bound")
    ap.add_argument("--p1", type=int, default=50000, help="Pollard p-1 B1 bound")
    ap.add_argument("--ecm-B1", type=int, default=50000, help="ECM stage-1 B1")
    ap.add_argument("--ecm-curves", type=int, default=800, help="ECM curves")
    ap.add_argument("--use-gmp", action="store_true",
                    help="use external GMP-ECM if available")
    ap.add_argument("--gmp-B2", type=int, default=0, help="optional ECM B2")
    args = ap.parse_args()

    N = read_number(args.source)
    print("Digits:", len(str(N)))

    # quick sanity gcds
    for c in (10**6+3, 10**6+33, 10**6+93):
        g = gcd(N, c)
        if g not in (0,1,N) and show_factor(N, g, f"gcd({c})"):
            return

    # trial division
    if args.trial > 0:
        f = trial_division(N, args.trial)
        if show_factor(N, f, "trial"):
            return

    # Pollard p-1
    if args.p1 > 0:
        f = pollard_p1(N, B1=args.p1, base=2)
        if show_factor(N, f, "p-1"):
            return

    # ECM
    if args.ecm_B1 > 0 and args.ecm_curves > 0:
        f = None
        if args.use_gmp:
            f = run_gmp_ecm(N, args.ecm_B1, args.ecm_curves, B2=args.gmp_B2)
            if show_factor(N, f, "gmp-ecm"):
                return
            if f is None:
                print("[gmp-ecm] no factor (or ecm not present); falling back to Python ECM.")
        f = ecm(N, B1=args.ecm_B1, curves=args.ecm_curves, seed=0xC0FFEE)
        if show_factor(N, f, "ecm"):
            return

    print("No factor found with current settings.")
if __name__ == "__main__":
    main()
