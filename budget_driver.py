#!/usr/bin/env python3
"""
budget_driver.py
- Reads N from file or raw string (digits-only)
- Tries: trial division -> Pollard p-1 (multi-base) -> ECM
- ECM can use external GMP-ECM ('ecm') via --use-gmp, else pure-Python fallback
- New: --max-seconds, --seed, --quiet, clean exit codes
"""
import argparse, re, sys, subprocess, shutil, time
from math import gcd

from pminus1 import pollard_p1, pollard_p1_multi
from ecm import ecm  # stage-1 Python ECM supports max_seconds

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
    # simple 2*3 wheel
    p, step = 5, 2
    while p*p <= n and p <= bound:
        if n % p == 0:
            return p
        p += step
        step = 6 - step
    return None

def show_factor(n, f, quiet):
    if f is None or f <= 1 or f >= n:
        return False
    if quiet:
        print(f"{f} {n//f}")
    else:
        print("[factor] ", f)
        print("cofactor:", n // f)
    return True

def parse_ecm_output(n: int, out: str):
    """Grab the first decimal token that actually divides n."""
    for tok in re.findall(r"\b\d+\b", out):
        val = int(tok)
        if 1 < val < n and n % val == 0:
            return val
    return None

def run_gmp_ecm(n: int, B1: int, curves: int, B2: int | None = None, timeout: int = 0, progress_chunk: int = 50):
    """Call external 'ecm' in chunks; respect timeout if >0 (seconds)."""
    if shutil.which("ecm") is None:
        return None
    total = curves
    done = 0
    started = time.time()
    while done < total:
        step = min(progress_chunk, total - done)
        cmd = ["ecm", "-q", "-c", str(step)]
        if B2 and B2 > B1:
            cmd += [str(B1), str(B2)]
        else:
            cmd += [str(B1)]
        try:
            # bytes in/out to avoid text encoding surprises
            remain = None
            if timeout and timeout > 0:
                elapsed = time.time() - started
                remain = max(1, int(timeout - elapsed))
                if remain <= 0:
                    return None
            proc = subprocess.run(
                cmd,
                input=(str(n) + "\n").encode(),
                capture_output=True,
                text=False,
                timeout=remain
            )
        except subprocess.TimeoutExpired:
            return None
        out = (proc.stdout or b"") + b"\n" + (proc.stderr or b"")
        f = parse_ecm_output(n, out.decode("ascii", "ignore"))
        if f:
            return f
        done += step
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="file containing N or a raw numeric string")
    ap.add_argument("--trial", type=int, default=100000, help="trial division bound")
    ap.add_argument("--p1", type=int, default=50000, help="Pollard p-1 B1 bound")
    ap.add_argument("--ecm-B1", type=int, default=50000, help="ECM stage-1 B1")
    ap.add_argument("--ecm-curves", type=int, default=800, help="ECM curves")
    ap.add_argument("--use-gmp", action="store_true", help="use external GMP-ECM if available")
    ap.add_argument("--gmp-B2", type=int, default=0, help="optional ECM B2")
    # NEW
    ap.add_argument("--max-seconds", type=int, default=0, help="overall wall-clock budget")
    ap.add_argument("--seed", type=int, default=0xC0FFEE, help="seed for Python ECM RNG")
    ap.add_argument("--quiet", action="store_true", help="machine-parsable: prints 'factor cofactor' only")
    args = ap.parse_args()

    start = time.time()
    def time_left():
        if args.max_seconds <= 0:
            return None
        left = args.max_seconds - (time.time() - start)
        return max(0, left)

    try:
        N = read_number(args.source)
    except Exception as e:
        if not args.quiet:
            print(f"Input error: {e}")
        sys.exit(2)

    if not args.quiet:
        print("Digits:", len(str(N)))

    # quick sanity gcds
    for c in (10**6+3, 10**6+33, 10**6+93):
        g = gcd(N, c)
        if g not in (0,1,N) and show_factor(N, g, args.quiet):
            sys.exit(0)

    # trial division
    if args.trial > 0:
        f = trial_division(N, args.trial)
        if show_factor(N, f, args.quiet):
            sys.exit(0)

    # Pollard p-1 (multi-base)
    if args.p1 > 0:
        f = pollard_p1_multi(N, B1=args.p1, bases=(2,3,5,7,11))
        if show_factor(N, f, args.quiet):
            sys.exit(0)

    # ECM
    if args.ecm_B1 > 0 and args.ecm_curves > 0:
        f = None
        if args.use_gmp:
            # hand remaining budget to subprocess; keep a floor to avoid 0s
            tl = time_left()
            tmo = int(tl) if tl is not None else 0
            f = run_gmp_ecm(N, args.ecm_B1, args.ecm_curves, B2=args.gmp_B2, timeout=tmo)
            if show_factor(N, f, args.quiet):
                sys.exit(0)
        # Python fallback with remaining budget
        tl = time_left()
        f = ecm(N, B1=args.ecm_B1, curves=args.ecm_curves, seed=args.seed,
                max_seconds=tl)
        if show_factor(N, f, args.quiet):
            sys.exit(0)

    if not args.quiet:
        print("No factor found with current settings.")
    sys.exit(1)

if __name__ == "__main__":
    main()
