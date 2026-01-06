from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict
import math

import numpy as np


# -----------------------------
# Primes + gaps
# -----------------------------
def first_primes(n: int) -> List[int]:
    """Return first n primes starting at 2."""
    if n <= 0:
        return []
    primes: List[int] = []
    x = 2
    while len(primes) < n:
        is_p = True
        if x == 2:
            is_p = True
        elif x % 2 == 0:
            is_p = False
        else:
            r = int(math.isqrt(x))
            d = 3
            while d <= r:
                if x % d == 0:
                    is_p = False
                    break
                d += 2
        if is_p:
            primes.append(x)
        x += 1
    return primes


def odd_primes_excluding_2(count: int) -> List[int]:
    """Return first `count` odd primes (3,5,7,...) excluding 2."""
    ps = first_primes(count + 50)  # overshoot
    out = [p for p in ps if p != 2]
    while len(out) < count:
        ps = first_primes(len(ps) + 200)
        out = [p for p in ps if p != 2]
    return out[:count]


def build_pairs_with_one(n_pairs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build n_pairs pairs starting with (1,3), then consecutive odd primes:
    (1,3), (3,5), (5,7), ...
    Returns (a, b, gaps).
    """
    nums = [1] + odd_primes_excluding_2(n_pairs)  # total n_pairs+1 numbers
    a = np.array(nums[:-1], dtype=int)
    b = np.array(nums[1:], dtype=int)
    gaps = b - a
    return a, b, gaps


# -----------------------------
# Area model (cone-like developable -> planar equivalent)
# -----------------------------
@dataclass(frozen=True)
class CoilAreaParams:
    psi_step_per_integer: float = 1.0  # radians per integer step: ψ = n*s
    beta: float = 1.0                  # r(ψ) = beta*ψ


def surface_area_pair(a: int, b: int, params: CoilAreaParams) -> float:
    """
    Intrinsic area for ideal cone (developable) equals developed planar area.
    Developed coil modeled as Archimedean spiral r(ψ)=beta*ψ with ψ=n*s.
    Boundary: spiral segment ψa->ψb plus straight chord closure.

    A = 1/2 * ( ∫ r(ψ)^2 dψ  - r_a r_b sin(Δψ) )
    With r=beta*ψ: ∫ r^2 dψ = beta^2 (ψb^3-ψa^3)/3.
    """
    s = params.psi_step_per_integer
    beta = params.beta

    psi1 = a * s
    psi2 = b * s
    if psi2 <= psi1:
        raise ValueError("Need b>a under ψ=n*s mapping.")

    r1 = beta * psi1
    r2 = beta * psi2
    dpsi = psi2 - psi1

    integral = (beta ** 2) * (psi2**3 - psi1**3) / 3.0
    A = 0.5 * (integral - r1 * r2 * math.sin(dpsi))
    return abs(A)


def surface_areas(n_pairs: int, params: CoilAreaParams) -> Dict[str, np.ndarray]:
    """Compute arrays for the first n_pairs (1–3, 3–5, ...)."""
    a, b, gaps = build_pairs_with_one(n_pairs)
    areas = np.array([surface_area_pair(int(ai), int(bi), params) for ai, bi in zip(a, b)], dtype=float)
    k = np.arange(1, n_pairs + 1, dtype=int)
    return {"k": k, "a": a, "b": b, "gaps": gaps, "areas": areas}


# -----------------------------
# Envelope extraction (local extrema)
# -----------------------------
def local_envelope_indices(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (upper_idx, lower_idx) for local maxima/minima using 1-step neighborhood:
      upper: y[i]>=y[i-1] and y[i]>=y[i+1]
      lower: y[i]<=y[i-1] and y[i]<=y[i+1]
    """
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    upper = []
    lower = []
    for i in range(1, y.size - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            upper.append(i)
        if y[i] <= y[i - 1] and y[i] <= y[i + 1]:
            lower.append(i)
    return np.array(upper, dtype=int), np.array(lower, dtype=int)


def fit_powerlaw(x: np.ndarray, y: np.ndarray, x_min: int = 1) -> Tuple[float, float]:
    """
    Fit y ≈ C * x^alpha via log-log regression for x>=x_min.
    Returns (C, alpha).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x >= x_min) & (x > 0) & (y > 0) & np.isfinite(y)
    X = np.log(x[mask])
    Y = np.log(y[mask])
    alpha, intercept = np.polyfit(X, Y, 1)
    C = float(np.exp(intercept))
    return C, float(alpha)


@dataclass(frozen=True)
class EnvelopeFits:
    Cu: float
    au: float
    Cl: float
    al: float

    def upper(self, k: np.ndarray) -> np.ndarray:
        return self.Cu * (k.astype(float) ** self.au)

    def lower(self, k: np.ndarray) -> np.ndarray:
        return self.Cl * (k.astype(float) ** self.al)

    def mid(self, k: np.ndarray) -> np.ndarray:
        u = self.upper(k)
        l = self.lower(k)
        return 0.5 * (u + l)

    def amp(self, k: np.ndarray) -> np.ndarray:
        u = self.upper(k)
        l = self.lower(k)
        return 0.5 * (u - l)


def fit_envelopes(k: np.ndarray, areas: np.ndarray, x_min_fit: int = 10) -> EnvelopeFits:
    """Extract local maxima/minima and fit power-law curves to each set."""
    up_idx, lo_idx = local_envelope_indices(areas)
    ku, Au = k[up_idx], areas[up_idx]
    kl, Al = k[lo_idx], areas[lo_idx]

    Cu, au = fit_powerlaw(ku, Au, x_min=x_min_fit)
    Cl, al = fit_powerlaw(kl, Al, x_min=x_min_fit)
    return EnvelopeFits(Cu=Cu, au=au, Cl=Cl, al=al)


# -----------------------------
# Prime-gap phase reconstruction
# -----------------------------
@dataclass(frozen=True)
class PhaseModel:
    s: float
    phi: float
    R: float = 1.0

    def phase_signal(self, gaps: np.ndarray) -> np.ndarray:
        g = gaps.astype(float)
        return self.R * np.sin(self.s * g + self.phi)


def fit_phase_model(
    gaps: np.ndarray,
    sin_est: np.ndarray,
    s_min: float = 0.05,
    s_max: float = 3.0,
    steps: int = 4000,
) -> Tuple[PhaseModel, float]:
    """
    Fit sin_est ≈ R*sin(s*g + phi) by grid-search over s.
    For each s, solve least squares sin_est ≈ a*sin(t)+b*cos(t).
    Convert to (R, phi). Return PhaseModel and best correlation score.
    """
    g = gaps.astype(float)
    y = sin_est.astype(float)

    best = {"score": -1e9, "s": None, "a": None, "b": None}

    for s in np.linspace(s_min, s_max, steps):
        t = s * g
        X = np.column_stack([np.sin(t), np.cos(t)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        c = np.corrcoef(y, pred)[0, 1]
        if np.isfinite(c) and float(c) > best["score"]:
            best.update(score=float(c), s=float(s), a=float(coef[0]), b=float(coef[1]))

    R = float(math.hypot(best["a"], best["b"]))
    phi = float(math.atan2(best["b"], best["a"]))
    model = PhaseModel(s=float(best["s"]), phi=phi, R=R)
    return model, float(best["score"])


def reconstruct_from_primes(
    n_pairs: int,
    params: CoilAreaParams = CoilAreaParams(),
    x_min_fit: int = 10,
    s_min: float = 0.05,
    s_max: float = 3.0,
    steps: int = 4000,
) -> Dict[str, np.ndarray]:
    """
    Pipeline:
      1) compute areas from prime pairs
      2) fit upper/lower envelope power laws (local extrema)
      3) compute sin_est = (A - mid)/amp (clipped)
      4) fit phase model sin_est ≈ R*sin(s*gap + phi)
      5) reconstruct A_hat = mid + amp * phase_signal
    """
    data = surface_areas(n_pairs, params)
    k = data["k"]
    gaps = data["gaps"]
    A = data["areas"]

    env = fit_envelopes(k, A, x_min_fit=x_min_fit)
    upper = env.upper(k)
    lower = env.lower(k)
    mid = 0.5 * (upper + lower)
    amp = 0.5 * (upper - lower)

    sin_est = (A - mid) / np.where(amp == 0, np.nan, amp)
    sin_est = np.clip(sin_est, -1.0, 1.0)

    phase, phase_corr = fit_phase_model(gaps, sin_est, s_min=s_min, s_max=s_max, steps=steps)
    A_hat = mid + amp * phase.phase_signal(gaps)

    out = dict(data)
    out.update({
        "A_upper_fit": upper,
        "A_lower_fit": lower,
        "A_mid_fit": mid,
        "A_amp_fit": amp,
        "sin_est": sin_est,
        "A_recon": A_hat,
        "env_Cu": np.array([env.Cu]),
        "env_au": np.array([env.au]),
        "env_Cl": np.array([env.Cl]),
        "env_al": np.array([env.al]),
        "phase_s": np.array([phase.s]),
        "phase_phi": np.array([phase.phi]),
        "phase_R": np.array([phase.R]),
        "phase_corr": np.array([phase_corr]),
    })
    return out
