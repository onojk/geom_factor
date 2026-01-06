from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import math


@dataclass(frozen=True)
class Coil3DParams:
    # Conical helix:
    # r(n) = a + b*n
    # theta(n) = omega*n
    # z(n) = c*n
    a: float = 1.0
    b: float = 0.02
    omega: float = 0.5
    c: float = 0.1


def P(n: float, par: Coil3DParams) -> np.ndarray:
    """3D position on the conical coil at parameter n (n can be non-integer)."""
    theta = par.omega * n
    r = par.a + par.b * n
    return np.array([r * math.cos(theta), r * math.sin(theta), par.c * n], dtype=float)


def dP(n: float, par: Coil3DParams) -> np.ndarray:
    """First derivative (tangent vector) P'(n)."""
    theta = par.omega * n
    r = par.a + par.b * n
    # r' = b, theta' = omega, z' = c
    dx = par.b * math.cos(theta) - r * par.omega * math.sin(theta)
    dy = par.b * math.sin(theta) + r * par.omega * math.cos(theta)
    dz = par.c
    return np.array([dx, dy, dz], dtype=float)


def safe_unit(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def azimuth_elevation(v: np.ndarray) -> tuple[float, float]:
    """
    Convert a 3D direction vector to:
      azimuth phi in XY plane: atan2(vy, vx) in (-pi, pi]
      elevation psi: atan2(vz, sqrt(vx^2+vy^2)) in (-pi/2, pi/2)
    """
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    phi = math.atan2(vy, vx)
    horiz = math.hypot(vx, vy)
    psi = math.atan2(vz, horiz)
    return phi, psi


def wrap_pi(x: float) -> float:
    """Wrap angle to (-pi, pi]."""
    x = (x + math.pi) % (2 * math.pi) - math.pi
    return x

def d2P(n: float, par: Coil3DParams) -> np.ndarray:
    """Second derivative P''(n) (curvature driver)."""
    theta = par.omega * n
    r = par.a + par.b * n

    # From:
    # dx = b*cosθ - r*ω*sinθ
    # dy = b*sinθ + r*ω*cosθ
    # dz = c
    #
    # Differentiate again w.r.t n:
    # d2x = -2*b*ω*sinθ - r*ω^2*cosθ
    # d2y =  2*b*ω*cosθ - r*ω^2*sinθ
    # d2z = 0
    w = par.omega
    d2x = -2.0 * par.b * w * math.sin(theta) - r * (w**2) * math.cos(theta)
    d2y =  2.0 * par.b * w * math.cos(theta) - r * (w**2) * math.sin(theta)
    d2z = 0.0
    return np.array([d2x, d2y, d2z], dtype=float)


def frenet_frame(n: float, par: Coil3DParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return unit tangent T_hat, unit normal N_hat, unit binormal B_hat at parameter n.

    N_hat is computed as:
      a = P''(n)
      a_perp = a - (a·T_hat) T_hat
      N_hat = a_perp / ||a_perp||
    """
    T = dP(n, par)
    T_hat = safe_unit(T)

    a = d2P(n, par)
    a_perp = a - float(np.dot(a, T_hat)) * T_hat
    N_hat = safe_unit(a_perp)

    B_hat = safe_unit(np.cross(T_hat, N_hat))
    return T_hat, N_hat, B_hat


def angle_between(u: np.ndarray, v: np.ndarray, eps: float = 1e-15) -> float:
    """Return angle in radians between nonzero vectors u and v in [0, pi]."""
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < eps or nv < eps:
        return float("nan")
    c = float(np.dot(u, v) / (nu * nv))
    c = max(-1.0, min(1.0, c))
    return math.acos(c)
