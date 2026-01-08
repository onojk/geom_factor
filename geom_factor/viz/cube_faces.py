# geom_factor/viz/cube_faces.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


FaceName = str


@dataclass(frozen=True)
class FacePoint:
    face: FaceName
    u: int
    v: int
    n: int
    xyz: Tuple[float, float, float]


FACES: List[FaceName] = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


def face_uv_to_xyz(face: FaceName, u: int, v: int) -> Tuple[float, float, float]:
    """
    Cube of side length 2 centered at origin:
      faces at x=±1, y=±1, z=±1
    Each face has (u,v) mapped onto the other two axes.
    """
    if face == "+X":
        return (1.0, float(u), float(v))
    if face == "-X":
        return (-1.0, float(u), float(v))
    if face == "+Y":
        return (float(u), 1.0, float(v))
    if face == "-Y":
        return (float(u), -1.0, float(v))
    if face == "+Z":
        return (float(u), float(v), 1.0)
    if face == "-Z":
        return (float(u), float(v), -1.0)
    raise ValueError(f"Unknown face: {face}")


def generate_face_grid(L: int) -> Dict[FaceName, List[Tuple[int, int]]]:
    """
    Returns a dict mapping face -> list of (u,v) lattice points.
    Ordering matters (used for scanline fill).
    """
    coords: Dict[FaceName, List[Tuple[int, int]]] = {}
    # scanline order: v from +L down to -L, u from -L to +L
    base = [(u, v) for v in range(L, -L - 1, -1) for u in range(-L, L + 1)]
    for f in FACES:
        coords[f] = list(base)
    return coords


def assign_integers_to_faces(
    L: int,
    max_n: int,
    mode: str = "round_robin",
) -> List[FacePoint]:
    """
    Build six face lattices and assign integers to lattice points.

    mode:
      - "round_robin": integer n goes to face index (n-1) mod 6; within that face, fill scanline.
      - "scanline": fill face +X then -X then +Y then -Y then +Z then -Z sequentially.
    """
    if L < 0:
        raise ValueError("L must be >= 0")
    if max_n < 1:
        return []

    grid = generate_face_grid(L)
    per_face_capacity = (2 * L + 1) ** 2
    total_capacity = per_face_capacity * 6
    if max_n > total_capacity:
        raise ValueError(
            f"max_n={max_n} exceeds cube-face lattice capacity {total_capacity} for L={L}. "
            f"Increase L or lower max_n."
        )

    face_indices = {f: i for i, f in enumerate(FACES)}
    face_fill_index = {f: 0 for f in FACES}

    points: List[FacePoint] = []

    if mode not in {"round_robin", "scanline"}:
        raise ValueError("mode must be 'round_robin' or 'scanline'")

    if mode == "round_robin":
        for n in range(1, max_n + 1):
            f = FACES[(n - 1) % 6]
            idx = face_fill_index[f]
            u, v = grid[f][idx]
            face_fill_index[f] += 1
            xyz = face_uv_to_xyz(f, u, v)
            points.append(FacePoint(face=f, u=u, v=v, n=n, xyz=xyz))
        return points

    # scanline: fill each face completely before moving to next
    n = 1
    for f in FACES:
        for (u, v) in grid[f]:
            if n > max_n:
                break
            xyz = face_uv_to_xyz(f, u, v)
            points.append(FacePoint(face=f, u=u, v=v, n=n, xyz=xyz))
            n += 1
        if n > max_n:
            break

    return points
