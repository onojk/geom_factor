#!/usr/bin/env python3
"""
Dot Gate Visualizer — colored dots only (interactive)

Rules:
- Red '.'   = clear gate
- Green '.' = first non-trivial divisor (opaque) -> stop
- Primes produce EXACTLY n red dots
"""

import sys

USE_ANSI = True
RED   = "\033[31m" if USE_ANSI else ""
GREEN = "\033[32m" if USE_ANSI else ""
RESET = "\033[0m"  if USE_ANSI else ""


def render_line(n: int) -> str:
    dots = []

    for d in range(1, n + 1):
        # d = 1 is always clear
        if d == 1:
            dots.append(f"{RED}.{RESET}")
            continue

        # first non-trivial divisor → opaque → green dot → stop
        if 2 <= d <= n - 1 and n % d == 0:
            dots.append(f"{GREEN}.{RESET}")
            break

        # otherwise clear
        dots.append(f"{RED}.{RESET}")

    return f"{n}: " + "".join(dots)


def get_max_n():
    default = 100
    try:
        user = input(f"Enter max N (press Enter for {default}): ").strip()
        if user == "":
            return default
        return int(user)
    except ValueError:
        print("Invalid number. Using default.")
        return default


def main():
    # If argument supplied, use it; otherwise ask
    if len(sys.argv) > 1:
        try:
            max_n = int(sys.argv[1])
        except ValueError:
            print("Invalid argument. Exiting.")
            return
    else:
        max_n = get_max_n()

    for n in range(1, max_n + 1):
        print(render_line(n))


if __name__ == "__main__":
    main()
