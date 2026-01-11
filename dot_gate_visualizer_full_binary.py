#!/usr/bin/env python3
"""
Dot Gate Visualizer — Full Binary Labels (up to 1000)

Each line:
  <binary(N)>:  [colored dots]

Dots:
  - Red '.'   = clear gate
  - Green '.' = first non-trivial divisor (opaque) -> stop
Prime rows:
  - exactly N red dots (no green dot)

Max N is limited to 1000.
"""

USE_ANSI = True
RED   = "\033[31m" if USE_ANSI else ""
GREEN = "\033[32m" if USE_ANSI else ""
RESET = "\033[0m"  if USE_ANSI else ""


def to_binary(n: int) -> str:
    return format(n, "b")  # full binary, no 0b, no fixed width


def render_line(n: int) -> str:
    dots = []
    for d in range(1, n + 1):
        if d == 1:
            dots.append(f"{RED}.{RESET}")
            continue

        # First non-trivial divisor -> opaque -> green dot -> stop
        if 2 <= d <= n - 1 and (n % d == 0):
            dots.append(f"{GREEN}.{RESET}")
            break

        dots.append(f"{RED}.{RESET}")

    return f"{to_binary(n)}: " + "".join(dots)


def get_max_n() -> int:
    default = 200
    raw = input(f"Enter max N (1..1000) [Enter for {default}]: ").strip()
    if raw == "":
        return default
    try:
        v = int(raw)
        if v < 1 or v > 1000:
            raise ValueError
        return v
    except ValueError:
        print("Please enter an integer between 1 and 1000.")
        raise SystemExit(1)


def main():
    max_n = get_max_n()
    for n in range(1, max_n + 1):
        print(render_line(n))


if __name__ == "__main__":
    main()
