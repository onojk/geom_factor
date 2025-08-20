#!/usr/bin/env python3
import argparse, random, json, os
from math import prod

# --- primality test (basic Miller-Rabin) ---
def is_probable_prime(n, k=16):
    if n < 2:
        return False
    small_primes = [2,3,5,7,11,13,17,19,23,29,31]
    if n in small_primes:
        return True
    for p in small_primes:
        if n % p == 0:
            return False
    # write n-1 as 2^r * d
    d, r = n-1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(k):
        a = random.randrange(2, n-2)
        x = pow(a, d, n)
        if x == 1 or x == n-1:
            continue
        for _ in range(r-1):
            x = pow(x, 2, n)
            if x == n-1:
                break
        else:
            return False
    return True

# --- checkpoint helpers ---
def save_ckpt(path, state):
    with open(path, "w") as f:
        json.dump(state, f)

def load_ckpt(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

# --- search function ---
def search_prime_near_target(T, wheel_primes, span=500000, expand_factor=2,
                             max_expansions=6, until_found=False,
                             ckpt_path=None, save_every=10000,
                             start_k=0, start_sign_index=0,
                             start_expansions=0):
    M = prod(wheel_primes)
    signs = [1, -1]
    k = start_k
    sign_index = start_sign_index
    expansions = start_expansions
    checked = 0

    while True:
        for _ in range(span):
            n = T + signs[sign_index] * k * M
            if n & 1:
                if is_probable_prime(n):
                    return n, signs[sign_index]*k, M, k, sign_index, expansions
            k += 1
            sign_index ^= 1
            checked += 1
            if ckpt_path and checked % save_every == 0:
                save_ckpt(ckpt_path, {
                    "T": str(T),
                    "wheel": wheel_primes,
                    "span": span,
                    "k": k,
                    "sign_index": sign_index,
                    "expansions": expansions
                })
        expansions += 1
        if not until_found and expansions > max_expansions:
            return None, None, M, k, sign_index, expansions
        span *= expand_factor
        print(f"[feedback] expanding search span -> {span}")

# --- main ---
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=str, help="Target integer (decimal)")
    p.add_argument("--wheel", type=str, default="2,3,5,7", help="Comma-separated wheel primes")
    p.add_argument("--span", type=int, default=500000, help="Initial span")
    p.add_argument("--expand-factor", type=int, default=2, help="Span multiplier")
    p.add_argument("--max-expansions", type=int, default=6, help="Max span doublings")
    p.add_argument("--random-near", type=int, default=0, help="Random offset within ±N")
    p.add_argument("--until-found", action="store_true", help="Expand until a prime is found")
    p.add_argument("--ckpt-file", type=str, default="tuner.ckpt", help="Checkpoint file ('' to disable)")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    p.add_argument("--save-every", type=int, default=10000, help="Save checkpoint every N candidates")

    args = p.parse_args()

    ck = load_ckpt(args.ckpt_file) if (args.resume and args.ckpt_file) else None

    if ck:
        # Resume
        T = int(ck["T"])
        wheel = [int(x) for x in ck["wheel"]]
        span = ck["span"]
        k = ck["k"]
        sign_index = ck["sign_index"]
        expansions = ck["expansions"]
        print(f"[resume] Loaded checkpoint: span={span}, k={k}, expansions={expansions}")
    else:
        if not args.target:
            raise SystemExit("Error: --target is required (or use --resume with a checkpoint).")
        T = int(args.target)
        if args.random_near > 0:
            off = random.randint(-args.random_near, args.random_near)
            T += off
            print(f"[randomize] Adjusted target by {off}, new T = {T}")
        wheel = [int(x) for x in args.wheel.split(",")]
        span = args.span
        k = 0
        sign_index = 0
        expansions = 0

    print("Target has", len(str(T)), "digits.")
    print("Wheel primes:", wheel)

    prime, delta, M, k, sign_index, expansions = search_prime_near_target(
        T, wheel,
        span=span,
        expand_factor=args.expand_factor,
        max_expansions=args.max_expansions,
        until_found=args.until_found,
        ckpt_path=(args.ckpt_file or None),
        save_every=args.save_every,
        start_k=k,
        start_sign_index=sign_index,
        start_expansions=expansions
    )

    if prime:
        direction = "exactly at T" if delta == 0 else ("above" if delta > 0 else "below")
        print("\n🚀 LANDING achieved 🚀")
        print("Step size M =", M)
        print("Offset delta =", delta, "-> direction:", direction)
        print("Probable prime n =", prime)
        print("Digits:", len(str(prime)))
        if args.ckpt_file:
            save_ckpt(args.ckpt_file, {
                "T": str(T),
                "wheel": wheel,
                "span": span,
                "k": k,
                "sign_index": sign_index,
                "expansions": expansions,
                "found": int(prime)
            })
    else:
        print("No prime found within max expansions.")
        if args.ckpt_file:
            print("🔄 Use --resume later to continue.")
