#!/usr/bin/env python3
"""
Dot Gate Driver (up to 5000)

Dots only:
- red '.'   = clear gate
- green '.' = first non-trivial divisor (opaque) -> stop
- primes print EXACTLY N red dots

Prompts for max N (<=5000).
"""

USE_ANSI = True
RED   = "\033[31m" if USE_ANSI else ""
GREEN = "\033[32m" if USE_ANSI else ""
RESET = "\033[0m"  if USE_ANSI else ""


def render_line(n: int) -> str:
    dots = []

    for d in range(1, n + 1):
        if d == 1:
            dots.append(f"{RED}.{RESET}")
            continue

        # first non-trivial divisor -> green dot -> stop
        if 2 <= d <= n - 1 and (n % d == 0):
            dots.append(f"{GREEN}.{RESET}")
            break

        dots.append(f"{RED}.{RESET}")

    return f"{n}: " + "".join(dots)


def get_max_n() -> int:
    default = 1000
    raw = input(f"Enter max N (1..5000) [Enter for {default}]: ").strip()
    if raw == "":
        return default
    try:
        v = int(raw)
        if v < 1 or v > 5000:
            raise ValueError
        return v
    except ValueError:
        print("Please enter an integer between 1 and 5000.")
        raise SystemExit(1)


def main():
    max_n = get_max_n()

    print("\n(red '.' = clear) (green '.' = first divisor -> stop)\n")
    for n in range(1, max_n + 1):
        print(render_line(n))


if __name__ == "__main__":
    main()
