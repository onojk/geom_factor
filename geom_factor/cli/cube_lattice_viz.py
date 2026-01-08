# geom_factor/cli/cube_lattice_viz.py
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List

from geom_factor.viz.cube_faces import FACES, assign_integers_to_faces
from geom_factor.viz.primes import sieve_upto
from geom_factor.viz.jumps import compute_jump_stats, format_face_report, compare_recent_jumps, prime_gaps
from geom_factor.viz.render import RenderConfig, render_cube_faces, render_jump_plot, animate_growth


NO_SHORTCUT_TEXT = (
    "Conclusion (why this saves no time):\n"
    "  • To color points as prime/composite, we still must compute primality up to max_n\n"
    "    (via sieve or primality tests). The cube/lattice does not provide a local geometric\n"
    "    rule that predicts the next prime without doing that discrete arithmetic.\n"
    "  • The 'jumps' (prime gaps) shown on each face are derived from primes already found,\n"
    "    so the visualization reorganizes information but does not reduce computational work.\n"
)


def build_reports(points, is_prime, tail_k: int) -> str:
    primes_by_face: Dict[str, List[int]] = defaultdict(list)
    for p in points:
        if 0 <= p.n < len(is_prime) and is_prime[p.n]:
            primes_by_face[p.face].append(p.n)

    stats_by_face = {}
    blocks = []
    for face in FACES:
        plist = primes_by_face.get(face, [])
        st = compute_jump_stats(plist, tail_k=tail_k)
        stats_by_face[face] = st
        blocks.append(format_face_report(face, plist, st))

    blocks.append("")
    blocks.append(compare_recent_jumps(stats_by_face))
    blocks.append("")
    blocks.append(NO_SHORTCUT_TEXT)
    return "\n\n".join(blocks), primes_by_face


def main():
    ap = argparse.ArgumentParser(
        description="Geom Factor: visualize primes on 6 cube-face lattices + jump analysis"
    )
    ap.add_argument("--L", type=int, default=25, help="lattice half-size per face (default: 25)")
    ap.add_argument("--max", dest="max_n", type=int, default=20000, help="max integer to place (default: 20000)")
    ap.add_argument("--mode", choices=["round_robin", "scanline"], default="round_robin", help="assignment mode")
    ap.add_argument("--tail", type=int, default=8, help="how many recent gaps to print per face (default: 8)")

    # Display / output
    ap.add_argument("--show", action="store_true", help="show plot windows")
    ap.add_argument("--save", type=str, default="", help="save 3D plot image (e.g., cube.png)")
    ap.add_argument("--save-jumps", type=str, default="", help="save jump plot image (e.g., jumps.png)")
    ap.add_argument("--no-axes", action="store_true", help="hide axes for cleaner render")
    ap.add_argument("--jumps-plot", action="store_true", help="show a 2D jump plot comparing faces")

    # Animation
    ap.add_argument("--animate", action="store_true", help="show growth animation instead of static 3D")
    ap.add_argument("--step", type=int, default=300, help="integers added per animation frame (default: 300)")
    ap.add_argument("--interval", type=int, default=50, help="animation frame interval ms (default: 50)")
    ap.add_argument("--save-anim", type=str, default="", help="save animation (e.g., anim.gif or anim.mp4)")

    args = ap.parse_args()

    capacity = 6 * (2 * args.L + 1) ** 2
    if args.max_n > capacity:
        print(f"Warning: --max {args.max_n} exceeds capacity {capacity} for L={args.L}. Clamping to {capacity}.")
        args.max_n = capacity

    points = assign_integers_to_faces(L=args.L, max_n=args.max_n, mode=args.mode)
    sieve = sieve_upto(args.max_n)

    report, primes_by_face = build_reports(points, sieve.is_prime, tail_k=args.tail)
    print(report)

    cfg = RenderConfig(show_axes=not args.no_axes)

    # Build jump plot data (gaps per face)
    gaps_by_face: Dict[str, List[int]] = {}
    for face in FACES:
        plist = primes_by_face.get(face, [])
        gaps_by_face[face] = prime_gaps(plist)

    import matplotlib.pyplot as plt

    if args.animate:
        fig3d, _ax3d, anim = animate_growth(
            points=points,
            is_prime=sieve.is_prime,
            faces=FACES,
            max_n_final=args.max_n,
            step=args.step,
            interval_ms=args.interval,
            cfg=cfg,
        )
        if args.save_anim:
            # Saving animation may require ffmpeg (mp4) or pillow (gif).
            # Try mp4 if you have ffmpeg, otherwise gif if you have pillow.
            out = args.save_anim
            print(f"Saving animation to: {out}")
            if out.lower().endswith(".gif"):
                anim.save(out, writer="pillow", fps=max(1, int(1000 / max(1, args.interval))))
            else:
                anim.save(out, writer="ffmpeg", fps=max(1, int(1000 / max(1, args.interval))))
    else:
        fig3d, _ax3d = render_cube_faces(points, sieve.is_prime, faces=FACES, cfg=cfg)
        if args.save:
            fig3d.savefig(args.save, dpi=200, bbox_inches="tight")
            print(f"\nSaved 3D plot to: {args.save}")

    if args.jumps_plot or args.save_jumps:
        figj, _axj = render_jump_plot(gaps_by_face, title="Prime-gap jumps by face (derived, not predictive)")
        if args.save_jumps:
            figj.savefig(args.save_jumps, dpi=200, bbox_inches="tight")
            print(f"Saved jump plot to: {args.save_jumps}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
