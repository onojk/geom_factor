# geom_factor/viz/jumps.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class JumpStats:
    prime_count: int
    mean_gap: float | None
    median_gap: float | None
    max_gap: int | None
    last_gaps: List[int]
    last_gap_ratios: List[float]


def _median(xs: Sequence[int]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    m = len(s) // 2
    if len(s) % 2 == 1:
        return float(s[m])
    return (s[m - 1] + s[m]) / 2.0


def prime_gaps(primes: Sequence[int]) -> List[int]:
    return [primes[i + 1] - primes[i] for i in range(len(primes) - 1)]


def gap_ratios(gaps: Sequence[int]) -> List[float]:
    """
    ratio_i = gaps[i] / gaps[i-1] for i>=1, skipping divide-by-zero (not expected for prime gaps)
    """
    ratios: List[float] = []
    for i in range(1, len(gaps)):
        prev = gaps[i - 1]
        if prev == 0:
            ratios.append(float("inf"))
        else:
            ratios.append(gaps[i] / prev)
    return ratios


def compute_jump_stats(primes: Sequence[int], tail_k: int = 8) -> JumpStats:
    gaps = prime_gaps(primes)
    if gaps:
        mean = sum(gaps) / len(gaps)
        med = _median(gaps)
        mx = max(gaps)
    else:
        mean = med = None
        mx = None

    last = gaps[-tail_k:] if tail_k > 0 else []
    ratios = gap_ratios(last)

    return JumpStats(
        prime_count=len(primes),
        mean_gap=mean,
        median_gap=med,
        max_gap=mx,
        last_gaps=list(last),
        last_gap_ratios=list(ratios),
    )


def format_face_report(face: str, primes: Sequence[int], stats: JumpStats) -> str:
    lines = []
    lines.append(f"Face {face}: {stats.prime_count} primes")
    if stats.mean_gap is not None:
        lines.append(f"  mean gap:   {stats.mean_gap:.3f}")
        lines.append(f"  median gap: {stats.median_gap:.3f}" if stats.median_gap is not None else "  median gap: (none)")
        lines.append(f"  max gap:    {stats.max_gap}")
    else:
        lines.append("  gaps: (not enough primes to compute gaps)")

    if stats.last_gaps:
        lines.append(f"  last gaps:  {', '.join(map(str, stats.last_gaps))}")
        if stats.last_gap_ratios:
            lines.append(
                "  ratios:     " + ", ".join(f"{r:.3f}" for r in stats.last_gap_ratios)
            )
    return "\n".join(lines)


def compare_recent_jumps(stats_by_face: Dict[str, JumpStats]) -> str:
    """
    Quick cross-face comparison of last gap (if available).
    """
    items: List[Tuple[str, int]] = []
    for face, st in stats_by_face.items():
        if st.last_gaps:
            items.append((face, st.last_gaps[-1]))
    if not items:
        return "No recent jump comparison available (insufficient primes on faces)."

    items.sort(key=lambda t: t[1], reverse=True)
    lines = ["Recent jump comparison (last gap on each face):"]
    for face, g in items:
        lines.append(f"  {face}: {g}")
    return "\n".join(lines)
