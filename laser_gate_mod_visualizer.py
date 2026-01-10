#!/usr/bin/env python3
"""
Laser Gate Mod Visualizer — Individual Gates Version

Legend:
  .  = clear gate  (N mod d != 0)
  0  = opaque gate (N mod d == 0)  -> STOP immediately

Rules:
  • Gates are tested in order d = 2 .. floor(sqrt(N))
  • Each gate is printed individually (no compression)
  • Output stops at the first opaque gate
  • Primes show the full gate run and are colored red
"""

import math

# ===== Terminal styling =====
USE_ANSI = True
RED   = "\033[31m" if USE_ANSI else ""
BOLD  = "\033[1m"  if USE_ANSI else ""
RESET = "\033[0m"  if USE_ANSI else ""


def render_line(n: int):
    if n < 2:
        return f"N={n} : (ignored)"

    limit = int(math.isqrt(n))
    gates = []
    hit = None

    for d in range(2, limit + 1):
        if n % d == 0:
            gates.append("0")
            hit = d
            break
        else:
            gates.append(".")

    gate_str = "".join(gates)

    header = f"N={n:>4}  d=2..⌊√N⌋={limit:>3} : "

    if hit is None:
        # PRIME → longest possible line
        return f"{RED}{header}{gate_str}   => PRIME{RESET}"
    else:
        # COMPOSITE → stopped at opaque
        return f"{header}{gate_str}   => COMPOSITE (hit d={hit})"


def main():
    START = 2
    END   = 1000

    print("\nLaser Gate Mod Visualizer")
    print("Each symbol is one gate:")
    print("  . = clear gate (laser passes)")
    print("  0 = opaque gate (laser stops)")
    print("Primes are longest lines and marked in red.\n")

    for n in range(START, END + 1):
        print(render_line(n))

    print("\nInterpretation:")
    print("• Composite numbers stop early when the laser hits an opaque gate.")
    print("• Prime numbers never block the laser, so they run to the end.\n")


if __name__ == "__main__":
    main()
