# geom_factor

Structural experiments on primes, geometry, and factor constraints.

This repository explores a single guiding idea:

> **Primality is not a property that emerges from growth, volume, or accumulation.**
> It is a structural constraint that must be preserved at every step of a computation.

The code here treats primes as *invariants* and tests what kinds of mathematical or geometric constructions can — and cannot — produce them.

---

## Core Principle

A number is prime if and only if it has **exactly two positive divisors**: 1 and itself.

From this follows a strict rule that drives this project:

> **No computation intended to yield a prime may multiply more than two non-unit factors at any stage, unless the multiplication is immediately canceled.**

Once three or more independent factors greater than 1 exist simultaneously, compositeness is locked in and cannot be undone by later steps.

---

## What This Repository Demonstrates

### 1. Prime Volume Impossibility

Geometric volume formulas (e.g. length × width × height) necessarily multiply three independent quantities. If all dimensions exceed 1, the resulting volume is always composite.

This is not a numerical accident — it is a structural consequence of multiplication.

### 2. Immediate Cancellation Is the Only Escape

Expressions such as:

```
(a * b * c) / c
```

are allowed **only because** the extra factor is canceled *immediately*. Delayed cancellation or partial cancellation is insufficient.

### 3. Geometry Cannot Detect Primality

Processes based on:

* filling space
* increasing height
* accumulating volume
* iterating time steps

cannot certify primality. They create factors; primes forbid them.

---

## Included Drivers

### `prime_volume_driver.py`

Demonstrates that no rectangular prism with prime integer sides (>1) can have a prime integer volume.

### `prime_mult_rule_driver.py`

A structural analyzer that parses arithmetic expressions and flags any multiplication that violates the "no 3+ non-unit factors" rule unless immediately canceled.

### `cone_fill_volume_driver.py`

Extends the volume argument to conical and tapered geometries, showing that scaling and rational factors do not rescue primality.

### Other Drivers

Additional files explore lattice geometry, phase structure, triangle constructions, and factor fingerprints under similar constraints.

---

## What This Is *Not*

* This is not a prime-finding algorithm
* This is not a numeric sieve
* This is not probabilistic

Instead, it is a **structural proof environment**: if a construction violates the invariant, it cannot produce a prime — regardless of numeric size.

---

## Usage Example

```bash
python3 prime_volume_driver.py fill --p 3 --q 5 --max-h 500
```

Output:

```
No prime volumes found for any integer height in the scanned range.
Reason: base area > 1 forces compositeness.
```

---

## Conceptual Summary

* Multiplication creates divisors
* Primes forbid excess divisors
* Therefore primes cannot be reached by accumulation

> **Primes are preserved only by constructions that never create surplus factors.**

---

## Status

This repository is exploratory but internally consistent. Each driver encodes the same invariant from a different angle.

Future directions may include:

* symbolic proof extraction
* static analyzers for prime-safe expressions
* geometric visualizations of factor constraints

---

## License

Open source. Use, modify, and extend fre
