# Geometric + Algorithmic Factorization Toolkit

⚠️ **Disclaimer**: Educational / research exploration only. Not suitable for real cryptographic attacks.

---

## Overview

This repository contains experimental code exploring whether **geometric embeddings and calculus-based signatures** can reveal structure in the distribution of prime numbers, and whether such structure could aid factorization or prime detection.

The core idea investigated here is:

> *If primes trace a structured geometric object, can local or multi-step geometric fingerprints distinguish primes from non-primes?*

To test this honestly, the project implements a sequence of increasingly strong geometric invariants and compares **true prime data** against carefully designed **control data**.

---

## What This Project Does

The toolkit provides:

* A smooth **3D conical–helical embedding** of the integers
* Local **calculus-based signatures** at each step:

  * Tangent direction
  * Relative chord angles (Δφ, Δψ)
  * Curvature and Frenet frame alignment
* **Planar surface-area band models** derived from prime gaps
* **Multi-step trajectory analysis** in signature space
* Direct comparison against:

  * Random even-gap controls
  * Shuffled prime-gap controls (same gap distribution, destroyed ordering)

All experiments are designed to be:

* falsifiable
* reproducible
* negative-result tolerant

---

## Summary of Experimental Findings

### 1. Single-Step Geometry

Local geometric quantities such as:

* tangent slope
* azimuth / elevation differences (Δφ, Δψ)
* curvature-normal alignment

**do exhibit strong structure**:

* banding
* hard geometric boundaries
* forbidden regions

However, these structures are **shared by primes and non-primes alike**.

**Conclusion:**

> Local geometric invariants define an *admissible phase space*, but do **not** uniquely identify primes.

---

### 2. Curvature and Frenet-Frame Signatures

Adding higher-order calculus information (Frenet normal, curvature alignment κ) confirms:

* curvature is a real geometric constraint
* but it is **not a prime discriminator**

Primes do **not** preferentially align with curvature in a way composites cannot.

---

### 3. Multi-Step Trajectory Fingerprints

To test whether **sequence memory** matters, the project analyzes the *trajectory* of primes in signature space:

* Treating successive (Δφ, Δψ) points as a curve
* Measuring **turning angles** between successive steps
* Comparing against:

  * shuffled prime gaps (same gaps, different order)
  * random even-gap controls

**Result:**

* Turning-angle distributions are nearly identical
* Shuffled gaps behave almost the same as true primes
* No exclusive “prime-only” regions emerge

**Conclusion:**

> Even multi-step local geometric trajectories do **not** encode primality.

---

## Core Conclusion

> **Geometry defines the stage; arithmetic writes the script.**

The experiments demonstrate that:

* Geometric embeddings impose strong constraints
* Primes trace structured paths within those constraints
* But **which path is taken is governed by global arithmetic**, not local or finite-depth geometry

No finite combination of smooth geometric or calculus-based invariants—single-step or multi-step—was found to uniquely characterize primes.

This aligns with deep results in number theory regarding the local pseudorandomness of primes.

---

## What This Toolkit Is Useful For

Although geometry alone does not select primes, this project **is still valuable** as:

* A research framework for falsifying geometric prime hypotheses
* A visualization tool for prime-gap structure
* A platform for hybrid experiments combining geometry + arithmetic
* An educational exploration of why local methods fail

The code is intentionally modular so future work can explore:

* nonlocal accumulation effects
* modular arithmetic overlays
* geometric visualizations of classical sieves
* factorization heuristics informed by structure (not prediction)

---

## Repository Structure

* `coil3d.py` — 3D integer embedding, derivatives, Frenet frame
* `signature_driver.py` — local angle signatures
* `prime_band.py` — planar surface-area / band models
* `fingerprint_driver.py` — single-step geometric fingerprint tests
* `multistep_driver.py` — trajectory / turning-angle analysis

---

## Research Philosophy

This project intentionally embraces **negative results**.

Showing *why* an idea fails—cleanly, empirically, and without hand-waving—is as important as showing success.

If you are interested in:

* experimental mathematics
* prime structure
* geometric intuition tested against reality

this repository is meant for you.

---

## License

MIT License
