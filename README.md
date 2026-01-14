geom_factor

Geometric, binary, and visual explorations of prime numbers

This repository explores prime numbers through structure, geometry, and computation, rather than treating them as isolated arithmetic objects. The core idea is that many properties of primes become clearer when viewed through binary representation, bit-length grouping, offset normalization, and visual density analysis.

The project combines:

number theory

binary arithmetic

geometric interpretation

algorithmic verification

Manim-based visualization

Core Concepts
Bit Buckets

Integers are grouped by bit-length into buckets of the form:

[
2
𝑘
−
1
,
  
2
𝑘
−
1
]
[2
k−1
,2
k
−1]

Each bucket:

has width 
2
𝑘
−
1
2
k−1

doubles in size with each increment of 
𝑘
k

provides a natural scale for comparing primes fairly

Offsets

For a number 
𝑛
n in the 
𝑘
k-bit bucket, its offset is:

offset
(
𝑛
)
=
𝑛
−
2
𝑘
−
1
offset(n)=n−2
k−1

Offsets normalize position within a bucket, allowing patterns to be compared across buckets without scale distortion.

Offsets separate:

scale (bucket index)

position (offset inside the bucket)

Prime Density

Prime density is defined per bucket as:

density
(
𝑘
)
=
#
primes in bucket
2
𝑘
−
1
density(k)=
2
k−1
#primes in bucket
	​


This makes visible how primes thin out as numbers grow larger.
Empirically, density decays roughly like:

1
𝑘
ln
⁡
2
kln2
1
	​


which aligns with the Prime Number Theorem expressed in bit-length terms.

Verification Cost & Carry Structure

Many scripts in this repo explore the idea that:

composite numbers are often easy to disprove

primes require exhaustive verification

binary carry propagation plays a structural role in divisibility tests

primes can be interpreted as numbers that resist internal factor alignment

This reframes primality as a verification process, not just a property.

Major Components
🎥 Manim Visualizations

High-quality mathematical animations built with Manim Community Edition, including:

bucket tables (2-bit, 3-bit, 4-bit, …)

offset-aligned prime layouts

prime density decay plots

asymptotic comparisons

explanatory slides designed for narration

Primary video script:

prime_bucket_density_plus_and_decay.py

📊 Density & Decay Analysis

Scripts that compute and visualize how prime density changes with scale, including:

density vs bucket index

empirical vs asymptotic curves

offset-based normalization

decay trend validation

🔢 Binary & Carry Experiments

Explorations of primes in binary space:

carry-strip visualizations

bit-bucket truth tables

verification step counting

binary pattern alignment

These scripts investigate why primes are computationally harder to confirm than composites.

📐 Geometric Interpretations

Several experiments reinterpret primes geometrically:

lattice-based layouts

arc and cone projections

spatial density thinning

structured but non-repeating patterns

Repository Structure (high-level)
geom_factor/
├── prime_bucket_density_plus_and_decay.py   # Main Manim explainer video
├── prime_bucket_density_*.py                # Density / decay experiments
├── prime_division_*.py                      # Verification & division logic
├── bit_bucket_*.py                          # Binary bucket & offset tools
├── arc_*.py                                 # Geometric projections
├── build_*.sh                               # Build / render helpers
└── README.md


(The repo is intentionally exploratory; not all scripts are “final outputs.”)

Requirements

Python 3.10+

Manim Community Edition (tested on 0.19.x)

NumPy

FFmpeg (for video output)

Philosophy

This project treats primes not as mystical objects, but as the result of:

binary structure

verification limits

exponential scaling

and information density constraints

Many visual patterns that look mysterious in base-10 become predictable when examined through binary buckets and offsets.

Status

This is an active exploratory research repo.

Scripts may evolve, be refactored, or be superseded as visual and conceptual clarity improves. The goal is insight, not minimal code.

Author

Jonathan Kendall
Explorations in mathematics, structure, and visualization.
