#!/usr/bin/env python3
"""
Dot Gate Driver — always asks for N

Dots only:
- red '.'   = clear gate
- green '.' = first non-trivial divisor (opaque) -> stop
- primes print EXACTLY N red dots

User is REQUIRED to enter max N (1..5000).
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

        # first non-trivial divisor → opaque → green dot → stop
        if 2 <= d <= n - 1 and (n % d == 0):
            dots.append(f"{GREEN}.{RESET}")
            break

        dots.append(f"{RED}.{RESET}")

    return f"{n}: " + "".join(dots)


def ask_max_n() -> int:
    while True:
        raw = input("Enter max N (1..5000): ").strip()
        try:
            n = int(raw)
            if 1 <= n <= 5000:
                return n
            else:
                print("Please enter a number between 1 and 5000.")
        except ValueError:
            print("Please enter a valid integer.")


def main():
    max_n = ask_max_n()
    print("\n(red '.' = clear) (green '.' = first divisor → stop)\n")

    for n in range(1, max_n + 1):
        print(render_line(n))


if __name__ == "__main__":
    main()
