# geom_factor

**Geometric Experiments in Prime Structure, Smoothness, and Coupling**

This repository contains a collection of exploratory visual and computational experiments that study **prime numbers through geometry**, rather than through direct arithmetic tests.

The goal is **not** to discover new primality tests or faster sieves, but to investigate how prime structure behaves under **global geometric constraints**, **smoothness**, and **coupling**.

---

## Core Ideas

### Primeness as Resistance
A prime number can be understood as a number that resists divisibility constraints.  
In these experiments, that resistance is visualized as **persistence under geometric constraint**.

### Smoothness
Smoothness here does **not** mean “numbers with small prime factors.”

Instead, smoothness means:

> **Behavior that remains continuous and coherent under imposed constraints.**

Prime gaps, persistence under rotation, and survival under multi-point coupling are treated as smoothness signals.

### Coupling
Coupling is introduced by rigid geometric relationships (e.g., vertices of a triangle fixed at 120° apart).  
Once coupled, no point can act independently — structure must survive **collectively** or disappear.

---

## Major Experiment Families

### 1. Bit Bucket Prime Towers
Files:
- `bit_bucket_prime_towers.py`
- `bit_bucket_prime_towers_slow.py`
- `bit_bucket_prime_towers_bucket_memory.py`
- `bit_bucket_prime_towers_4k_export.py`

Each bit-width interval (`[2^(k-1), 2^k - 1]`) is treated as a **bucket**.

Primes are drawn as **vertical towers**:
- height encodes persistence or structure
- buckets accumulate visually
- smoothness emerges across scale

These experiments emphasize **growth, decay, and memory** across increasing bit widths.

---

### 2. Bit Bucket Sieves (Visual)
File:
- `bit_bucket_sieve_dots.py`

A purely visual sieve:
- dots appear as candidates
- structure emerges without explicit primality labels
- shows how constraints eliminate composites

This emphasizes **constraint filtering**, not primality detection.

---

### 3. Prime Division & Carry Visuals
File:
- `prime_division_carry_strip.py`

Explores:
- odd/even structure
- carry propagation
- why division reveals compositeness faster than it confirms primeness

This is a **conceptual visualization**, not a numerical algorithm.

---

### 4. Prime Density & Decay by Bit Bucket
File:
- `prime_bucket_density_plus_and_decay.py`

Shows:
- prime density per bit bucket
- thinning as scale increases
- comparison with heuristic expectations (e.g. ~1 / ln(n))

This anchors the geometric work to known number-theoretic behavior.

---

### 5. Primes on Circles and Arcs
File:
- `primes_on_quarter_arc_8bit.py`

Maps primes from a single bit bucket onto:
- a quarter arc
- mirrored to a full circle

This converts linear ordering into **angular structure**.

---

### 6. Prime Gap Curves with Geometric Coupling
(extended in recent commits)

Primes are mapped to angles, and **prime gaps** are encoded as radial deviations, producing a smooth closed curve.

An **inscribed equilateral triangle** rotates continuously:
- vertices are rigidly coupled
- events trigger when vertices pass over prime instances
- single, double, and triple alignments are detected

These experiments study **global coherence** and **rare alignment events** under constraint.

---

## What This Project Is *Not*

- ❌ Not a primality test
- ❌ Not a factorization method
- ❌ Not a cryptographic attack
- ❌ Not a proof of prime randomness or non-randomness

These are **visual and structural probes**, closer to experimental mathematics than algorithm design.

---

## What This Project *Is*

- A geometric lens on prime structure
- A way to explore smoothness emerging from discreteness
- A sandbox for coupling, symmetry, and constraint
- A tool for generating new questions

---

## Future Directions

Potential next steps include:
- projecting primes as **towers on a sphere (prime globe)**
- comparing against randomized controls
- increasing polygon vertex counts to study coupling strength
- quantifying smoothness metrics formally
- exporting long-form videos for analysis

---

## Requirements

Most scripts use:
- Python 3.10+
- `pygame`
- (some earlier experiments use `manim`, but the current direction is pygame-based)

---

## Philosophy

> *Primes are discrete, but the constraints they survive are continuous.*

This project explores what survives when arithmetic structure is forced to live inside geometry.

---

## Author

Jonathan Kendall  
(ONOJK123)

Exploratory research, visualization, and conceptual development.
