#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from coil3d import (
    Coil3DParams,
    P,
    frenet_frame,
    angle_between,
)
from signature_driver import primes_upto, signature_for_step


def main():
    ap = argparse.ArgumentParser(
        description="Prime fingerprint table: (Δphi, Δpsi, κ) with control comparison"
    )
    ap.add_argument("--pairs", type=int, default=5000)
    ap.add_argument("--prime-limit", type=int, default=2000000)

    # coil params
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=0.02)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.1)

    # control
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--control-gmax", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    coil_par = Coil3DParams(
        a=args.a, b=args.b, omega=args.omega, c=args.c
    )

    primes = primes_upto(args.prime_limit)
    if primes.size < args.pairs + 2:
        raise SystemExit("Increase --prime-limit")

    ps = primes[: args.pairs + 1]
    gaps = ps[1:] - ps[:-1]

    # --- prime fingerprints ---
    dphi = np.empty(args.pairs)
    dpsi = np.empty(args.pairs)
    kappa = np.empty(args.pairs)

    for i in range(args.pairs):
        p = int(ps[i])
        g = int(gaps[i])

        dphi[i], dpsi[i] = signature_for_step(p, g, coil_par)

        C = P(float(p + g), coil_par) - P(float(p), coil_par)
        _, N_hat, _ = frenet_frame(float(p), coil_par)
        kappa[i] = angle_between(C, N_hat)

    # --- control fingerprints ---
    control = None
    if args.control:
        rng = np.random.default_rng(args.seed)
        ns = rng.integers(ps[0], ps[-1], size=args.pairs)
        evens = rng.integers(1, args.control_gmax // 2 + 1, size=args.pairs) * 2

        cdphi = np.empty(args.pairs)
        cdpsi = np.empty(args.pairs)
        ckappa = np.empty(args.pairs)

        for i in range(args.pairs):
            n = int(ns[i])
            g = int(evens[i])

            cdphi[i], cdpsi[i] = signature_for_step(n, g, coil_par)
            C = P(float(n + g), coil_par) - P(float(n), coil_par)
            _, N_hat, _ = frenet_frame(float(n), coil_par)
            ckappa[i] = angle_between(C, N_hat)

        control = (cdphi, cdpsi, ckappa)

    # --- plotting helper ---
    def scatter2(x, y, title, xlabel, ylabel, control_xy=None):
        plt.figure(figsize=(7, 7))
        plt.scatter(x, y, s=8, alpha=0.4, label="Primes")
        if control_xy is not None:
            cx, cy = control_xy
            plt.scatter(cx, cy, s=8, alpha=0.25, label="Control")
            plt.legend()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    # --- plots ---
    scatter2(
        dphi,
        dpsi,
        "Fingerprint cloud: (Δphi, Δpsi)",
        "Δphi",
        "Δpsi",
        None if control is None else (control[0], control[1]),
    )

    scatter2(
        dphi,
        kappa,
        "Fingerprint cloud: (Δphi, κ)",
        "Δphi",
        "κ (angle chord vs normal)",
        None if control is None else (control[0], control[2]),
    )

    scatter2(
        dpsi,
        kappa,
        "Fingerprint cloud: (Δpsi, κ)",
        "Δpsi",
        "κ (angle chord vs normal)",
        None if control is None else (control[1], control[2]),
    )

    # --- tightness metric ---
    Fp = np.stack([dphi, dpsi, kappa], axis=1)
    detP = float(np.linalg.det(np.cov(Fp.T)))
    print(f"Prime fingerprint covariance det ≈ {detP:.6g}")

    if control is not None:
        Fc = np.stack(control, axis=1)
        detC = float(np.linalg.det(np.cov(Fc.T)))
        print(f"Control fingerprint covariance det ≈ {detC:.6g}")
        print(f"Tightness ratio detP/detC ≈ {detP/detC:.6g}")


if __name__ == "__main__":
    main()
