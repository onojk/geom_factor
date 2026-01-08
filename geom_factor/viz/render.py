# geom_factor/viz/render.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


@dataclass(frozen=True)
class RenderConfig:
    point_size: float = 10.0
    prime_size: float = 22.0
    alpha_composite: float = 0.10
    alpha_prime: float = 0.95
    show_axes: bool = True
    title: str = "Geom Factor — 6 Face Lattices (Cube) Prime Visualization"


def _face_color_map(faces: Sequence[str]) -> Dict[str, object]:
    """
    Use matplotlib default categorical palette (tab10). No hard-coded colors.
    """
    cmap = plt.get_cmap("tab10")
    return {face: cmap(i % 10) for i, face in enumerate(faces)}


def render_cube_faces(points, is_prime, faces: Sequence[str], cfg: Optional[RenderConfig] = None):
    """
    3D scatter of six face lattices with face-specific colors.
      - composites: faint
      - primes: strong
    """
    cfg = cfg or RenderConfig()
    face_colors = _face_color_map(faces)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot per-face so each face gets its own color
    for face in faces:
        fx_c, fy_c, fz_c = [], [], []
        fx_p, fy_p, fz_p = [], [], []
        for p in points:
            if p.face != face:
                continue
            x, y, z = p.xyz
            if 0 <= p.n < len(is_prime) and is_prime[p.n]:
                fx_p.append(x); fy_p.append(y); fz_p.append(z)
            else:
                fx_c.append(x); fy_c.append(y); fz_c.append(z)

        # composites first
        ax.scatter(
            fx_c, fy_c, fz_c,
            s=cfg.point_size,
            alpha=cfg.alpha_composite,
            c=[face_colors[face]],
            label=f"{face} (composite)"
        )
        # primes on top
        ax.scatter(
            fx_p, fy_p, fz_p,
            s=cfg.prime_size,
            alpha=cfg.alpha_prime,
            c=[face_colors[face]],
            label=f"{face} (prime)"
        )

    ax.set_title(cfg.title)

    # bounds from data
    xs = [p.xyz[0] for p in points]
    ys = [p.xyz[1] for p in points]
    zs = [p.xyz[2] for p in points]
    pad = 0.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_zlim(min(zs) - pad, max(zs) + pad)

    if not cfg.show_axes:
        ax.set_axis_off()

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Legend can get huge; keep it off by default. Uncomment if you want it.
    # ax.legend(loc="upper left", fontsize=8)

    return fig, ax


def render_jump_plot(gaps_by_face: Dict[str, List[int]], title: str = "Prime-gap jumps by face"):
    """
    Single 2D plot: x = gap index, y = gap size, one line per face.
    Uses matplotlib's default line colors automatically.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for face, gaps in gaps_by_face.items():
        if not gaps:
            continue
        xs = list(range(1, len(gaps) + 1))
        ax.plot(xs, gaps, label=face)

    ax.set_title(title)
    ax.set_xlabel("Gap index (within face)")
    ax.set_ylabel("Gap size (p[i+1] - p[i])")
    ax.legend()
    ax.grid(True, alpha=0.25)
    return fig, ax


def animate_growth(
    points,
    is_prime,
    faces: Sequence[str],
    max_n_final: int,
    step: int = 200,
    interval_ms: int = 50,
    cfg: Optional[RenderConfig] = None,
):
    """
    Animate "growth" by revealing points from n=1..max_n_final.
    IMPORTANT: This is *visual only*; primality labels come from is_prime already computed.

    step controls how many integers are added per frame.
    """
    cfg = cfg or RenderConfig()
    face_colors = _face_color_map(faces)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(cfg.title + " — Growth Animation")

    # Pre-create one scatter for composites and primes per face (so we can update offsets)
    scat_comp = {}
    scat_prime = {}
    for face in faces:
        scat_comp[face] = ax.scatter([], [], [], s=cfg.point_size, alpha=cfg.alpha_composite, c=[face_colors[face]])
        scat_prime[face] = ax.scatter([], [], [], s=cfg.prime_size, alpha=cfg.alpha_prime, c=[face_colors[face]])

    # bounds from full data
    xs = [p.xyz[0] for p in points[:max_n_final]]
    ys = [p.xyz[1] for p in points[:max_n_final]]
    zs = [p.xyz[2] for p in points[:max_n_final]]
    pad = 0.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_zlim(min(zs) - pad, max(zs) + pad)

    if not cfg.show_axes:
        ax.set_axis_off()

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Points are already in n=1..max order, so we can slice by prefix.
    frames = list(range(1, max_n_final + 1, step))
    if frames[-1] != max_n_final:
        frames.append(max_n_final)

    def update(n_now: int):
        # Collect per-face coords up to n_now
        for face in faces:
            comp = ([], [], [])
            prim = ([], [], [])
            for p in points[:n_now]:
                if p.face != face:
                    continue
                x, y, z = p.xyz
                if 0 <= p.n < len(is_prime) and is_prime[p.n]:
                    prim[0].append(x); prim[1].append(y); prim[2].append(z)
                else:
                    comp[0].append(x); comp[1].append(y); comp[2].append(z)

            # Update 3D scatter data (private API but standard practice)
            scat_comp[face]._offsets3d = comp
            scat_prime[face]._offsets3d = prim

        ax.set_title(cfg.title + f" — Growth Animation (n ≤ {n_now})")
        return list(scat_comp.values()) + list(scat_prime.values())

    anim = FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False)
    return fig, ax, anim
