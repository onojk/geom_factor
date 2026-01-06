#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np

from prime_band import CoilAreaParams, reconstruct_from_primes
from prime_band_plot import plot_band, plot_phase_vs_gap, plot_residuals


def main():
    ap = argparse.ArgumentParser(description="Prime-band envelope + gap-phase reconstruction")
    ap.add_argument("--pairs", type=int, default=1000)
    ap.add_argument("--psi-step", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--x-min-fit", type=int, default=10)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--save-npz", type=str, default="", help="Save arrays to .npz")
    args = ap.parse_args()

    params = CoilAreaParams(psi_step_per_integer=args.psi_step, beta=args.beta)
    out = reconstruct_from_primes(args.pairs, params=params, x_min_fit=args.x_min_fit)

    Cu = float(out["env_Cu"][0]); au = float(out["env_au"][0])
    Cl = float(out["env_Cl"][0]); al = float(out["env_al"][0])
    s  = float(out["phase_s"][0]); phi = float(out["phase_phi"][0])
    R  = float(out["phase_R"][0]); corr_phase = float(out["phase_corr"][0])

    print("Envelope fits (power-law on local extrema):")
    print(f"  Upper: A_upper(k) ≈ {Cu:.6g} * k^{au:.6f}")
    print(f"  Lower: A_lower(k) ≈ {Cl:.6g} * k^{al:.6f}")
    print("Derived:")
    print("  Mid:   (Upper + Lower)/2")
    print("  Amp:   (Upper - Lower)/2")
    print("Phase model (gap -> band position):")
    print(f"  sin_est ≈ R*sin(s*gap + phi)")
    print(f"  s={s:.6f}, phi={phi:.6f}, R={R:.6f}, corr≈{corr_phase:.4f}")

    A = out["areas"]
    Ahat = out["A_recon"]
    resid = A - Ahat

    corr = float(np.corrcoef(A, Ahat)[0, 1])
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    print("Reconstruction metrics:")
    print(f"  corr(True,Recon)={corr:.6f}, MAE={mae:.6g}, RMSE={rmse:.6g}")

    if args.save_npz:
        np.savez(args.save_npz, **out)
        print(f"Saved: {args.save_npz}")

    if args.plot:
        plot_band(
            out["k"], out["areas"],
            out["A_upper_fit"], out["A_lower_fit"],
            mid=out["A_mid_fit"], recon=out["A_recon"],
            title=f"Prime-band model (pairs={args.pairs})"
        )
        plot_phase_vs_gap(out["gaps"], out["sin_est"])
        plot_residuals(out["k"], resid)


if __name__ == "__main__":
    main()
