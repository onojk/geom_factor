#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Gate:
    d: int
    status: str          # "CLEAR", "OPAQUE", "ALWAYS CLEAR"
    reason: str


def build_gates_for_n(n: int) -> List[Gate]:
    """
    Build gates d=1..n using your teaching rules:
      - d=1 and d=n are ALWAYS CLEAR (shown but ignored as evidence)
      - any interior d with n % d == 0 becomes OPAQUE
      - otherwise CLEAR
    """
    gates: List[Gate] = []
    for d in range(1, n + 1):
        if d == 1 or d == n:
            gates.append(Gate(d, "ALWAYS CLEAR", "trivial gate (ignored)"))
        else:
            if n % d == 0:
                gates.append(Gate(d, "OPAQUE", f"{n} mod {d} = 0"))
            else:
                gates.append(Gate(d, "CLEAR", f"{n} mod {d} ≠ 0"))
    return gates


def fire_laser(gates: List[Gate]) -> Optional[Gate]:
    """
    Laser travels through gates in order.
    It passes through CLEAR and ALWAYS CLEAR.
    It stops at the first OPAQUE gate (first nontrivial divisor).
    """
    for g in gates:
        if g.status == "OPAQUE":
            return g
    return None


def classify(n: int, hit: Optional[Gate]) -> str:
    if n < 2:
        return "NEITHER"
    return "COMPOSITE" if hit is not None else "PRIME"


def print_trace(n: int):
    gates = build_gates_for_n(n)

    # "Parallel evaluation" (conceptual): all statuses are already computed above.
    hit = fire_laser(gates)
    verdict = classify(n, hit)

    print(f"\n=== Testing N = {n} ===")
    print("Gates (d : status : reason)")
    for g in gates:
        print(f"{g.d:>2} : {g.status:<12} : {g.reason}")

    print("\nLaser result:")
    if hit:
        print(f"STOP at d = {hit.d}  ({hit.reason})  → {verdict}")
    else:
        print("PASSED ALL interior gates (no nontrivial divisor found) → PRIME")


def main():
    for n in range(2, 12):  # 2..11
        print_trace(n)


if __name__ == "__main__":
    main()
