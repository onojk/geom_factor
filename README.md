# geom_factor

**geom_factor** is an experimental research and visualization project that explores **primality as accumulated division cost**, not as a static property.

Instead of treating primes as labels, this project treats primality as something that **survives a sequence of failed division attempts**. Every division check is explicit, visual, and time-ordered.

The core idea:
> **Primality isn’t detected — it’s earned by surviving division.**

---

## Core Concepts

### Prime Buckets
Numbers are grouped into **bit buckets** (binary length eras).  
Each bucket forms a distinct *prime density era*, where:

- The total search space grows exponentially
- Prime density thins
- Division cost increases
- No finite Boolean shortcut absorbs the cost

Each bucket has its own internal structure and behavior.

---

### Division as Process
Every number is tested by **explicit mod checks**:

- Each mod check is one unit of work
- Composite numbers fail early
- Prime numbers require exhausting all possible divisors up to √n

This makes primality a **process**, not a lookup.

---

## Prime Division Laser Visualization

The centerpiece of this repo is a **time-resolved division engine**:

### `prime_division_mode.py`

A pygame visualization where:

- **x-axis** = number `n`
- **y-axis** = division check index
- **Each dot** = one `n % d` operation
- Green dots = failed divisors (`n % d != 0`)
- Red dots = divisor hits (`n % d == 0`)
- Vertical columns form “division lasers”
- Primes are columns that **survive to the top**

Controls:
- `SPACE` tap → one division step
- `SPACE` hold → scroll through division steps
- `A` → auto-run
- `UP/DOWN` → adjust speed
- `LEFT/RIGHT` → zoom
- `R` → reset counter
- `C` → clear dots and start a new CSV
- `ESC` → quit

This makes **division effort visible** in real time.

---

## Why This Matters

- Composite detection is easy: one divisor ends the process
- Proving primality is hard: *all divisors must fail*
- As numbers grow, division cost grows
- Prime density thins but never vanishes
- No finite Boolean circuit can shortcut division indefinitely

This directly explains why:
- Primality testing has inherent computational cost
- RSA security depends on division hardness
- Prime patterns resist geometric or closed-form prediction

---

## Repository Structure

geom_factor/
├── prime_division_mode.py # Main interactive division visualization
├── prime_bucket_density.py # Prime density per bit bucket
├── prime_bucket_density_plus.py # Extended density analysis
├── prime_bucket_truth_table.py # Truth tables by bucket
├── dot_gate_driver_*.py # Binary / dot-logic experiments
├── gen_bucket_table.py # Bucket table generator
├── normalize_bucket_overlay.py # Overlay normalization
├── bucket_table.md # Bucket documentation
├── kicad_buckets/ # KiCad hardware explorations
├── exports/ # Rendered schematics / visuals
└── README.md

yaml
Copy code

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pygame numpy
Run the visualization:

bash
Copy code
python3 prime_division_mode.py
Philosophy
Irrationality is not infinite digits — it is infinite failure of numerator–denominator closure.

Primality is one manifestation of this principle.
It persists because division persists.

Status
This project is exploratory and intentionally experimental.
Expect rapid iteration, unconventional visualizations, and theory-driven code.

License
Open for research, learning, and exploration.

