from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_band(k, A, upper, lower, mid=None, recon=None, title="Prime-band model"):
    plt.figure(figsize=(10,6))
    plt.scatter(k, A, s=10, alpha=0.25, label="True")
    plt.plot(k, upper, linewidth=2, label="Upper fit")
    plt.plot(k, lower, linewidth=2, label="Lower fit")
    if mid is not None:
        plt.plot(k, mid, linewidth=2, label="Mid fit")
    if recon is not None:
        plt.scatter(k, recon, s=10, alpha=0.25, label="Reconstructed")
    plt.xlabel("Pair index k")
    plt.ylabel("Surface area A(k)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_phase_vs_gap(gaps, sin_est):
    plt.figure(figsize=(10,5))
    plt.scatter(gaps, sin_est, s=10, alpha=0.35)
    plt.xlabel("Prime gap g")
    plt.ylabel("Estimated phase signal (clipped)")
    plt.title("Recovered phase signal vs prime gap")
    plt.tight_layout()
    plt.show()

def plot_residuals(k, resid):
    plt.figure(figsize=(10,5))
    plt.scatter(k, resid, s=10, alpha=0.35)
    plt.xlabel("Pair index k")
    plt.ylabel("Residual (True - Recon)")
    plt.title("Residuals of reconstruction")
    plt.tight_layout()
    plt.show()
