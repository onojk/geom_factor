#!/usr/bin/env python3
"""
Full Node Laser-Gate Visualization (Green Dots)

For each N:
  - Print exactly N nodes (d = 1..N)
  - '.' = clear gate (green)
  - '0' = opaque gate (divisor)
  - d=1 and d=N are always clear
  - Primes have N green dots
"""

import math

# ===== ANSI colors =====
USE_ANSI = True
GREEN = "\033[32m" if USE_ANSI else ""
RED   = "\033[31m" if USE_ANSI else ""
RESET = "\033[0m"  if USE_ANSI else ""


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def gate_string(n: int):
    gates = []
    first_hit = None

    for d in range(1, n + 1):
        if d == 1 or d == n:
            gates.append(".")
        else:
            if n % d == 0:
                gates.append("0")
                if first_hit is None:
                    first_hit = d
            else:
                gates.append(".")

    return "".join(gates), first_hit


def render_line(n: int) -> str:
    gates, first_hit = gate_string(n)
    colored = ""

    for c in gates:
        if c == ".":
            colored += f"{GREEN}.{RESET}"
        else:
            colored += "0"

    header = f"N={n:>4} : "

    if is_prime(n):
        return f"{RED}{header}{colored}  => PRIME{RESET}"
    else:
        if first_hit is None:
            return f"{header}{colored}  => (n<2)"
        return f"{header}{colored}  => COMPOSITE (first 0 at d={first_hit})"


def main():
    START = 2
    END   = 1000

    print("\nFull Node Laser-Gate Visualization")
    print("Each character is one node d = 1..N")
    print("  green '.' = clear gate")
    print("  '0'       = opaque gate (divisor)")
    print("Primes have N green dots.\n")
    print("WARNING: Lines get very long near N=1000.\n")

    for n in range(START, END + 1):
        print(render_line(n))


if __name__ == "__main__":
    main()
