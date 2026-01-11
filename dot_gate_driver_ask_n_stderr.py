#!/usr/bin/env python3
"""
Dot Gate Driver — PLAIN TEXT

Behavior:
- Asks user for max N (1..5000) via stderr
- Outputs ONLY dots to stdout (safe for > file.txt)

Logic:
- Each dot represents a gate at position d = 1..N
- Stop early at first non-trivial divisor
- Primes output exactly N dots
"""

import sys


def render_line(n: int) -> str:
    dots = []

    for d in range(1, n + 1):
        # d = 1 always clear
        if d == 1:
            dots.append(".")
            continue

        # first non-trivial divisor -> stop
        if 2 <= d <= n - 1 and (n % d == 0):
            dots.append(".")
            break

        dots.append(".")

    return f"{n}: " + "".join(dots)


def ask_max_n() -> int:
    while True:
        print("Enter max N (1..5000): ", file=sys.stderr, end="", flush=True)
        raw = input().strip()
        try:
            n = int(raw)
            if 1 <= n <= 5000:
                return n
            else:
                print("Please enter a number between 1 and 5000.", file=sys.stderr)
        except ValueError:
            print("Please enter a valid integer.", file=sys.stderr)


def main():
    max_n = ask_max_n()

    for n in range(1, max_n + 1):
        print(render_line(n))


if __name__ == "__main__":
    main()
