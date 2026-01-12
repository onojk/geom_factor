# geom_factor

Visual and computational experiments exploring **prime numbers through structure, space, and scale**.

This project focuses on how primes behave when integers are grouped by **binary bit-length (power-of-two buckets)**, how density changes across those buckets, and why structural representations organize the search space but do **not** predict primality.

The repo contains:
- Manim animations
- Pygame interactive visualizers
- Analytical scripts and CSV outputs
- Conceptual experiments around primes, density, and computational limits

---

## Core Idea

Integers naturally partition into **bit buckets**:

- Bucket 2 bits: 2–3  
- Bucket 3 bits: 4–7  
- Bucket 4 bits: 8–15  
- Bucket *b*: `[2^(b-1), 2^b − 1]`

Inside each bucket:
- Binary width is constant
- Structure is fixed
- **Prime density thins out as buckets grow**

This repo visualizes that thinning and shows why **geometry and structure organize numbers**, but **arithmetic alone determines primality**.

---

## Key Visualizations

### 1. Prime Bucket Density + Decay (Manim)

**File**
prime_bucket_density_plus_and_decay.py

markdown
Copy code

**Scene**
PrimeBucketDensityPlusAndDecay

markdown
Copy code

**What it shows**
- Tables of numbers grouped by bit-length
- Binary representations within a bucket
- Prime vs composite classification
- Prime density per bucket
- Density decay as bit-width increases
- A comparison curve (~ 1 / ln(n)) for intuition
- End card: *Thanks For Watching! – ONOJK123*

**Run (quick preview)**
```bash
python -m manim -pqh prime_bucket_density_plus_and_decay.py PrimeBucketDensityPlusAndDecay
Run (1080p)

bash
Copy code
python -m manim -p -r 1920,1080 prime_bucket_density_plus_and_decay.py PrimeBucketDensityPlusAndDecay
2. Prime Division Laser / Towers (Pygame)
Interactive visualization of trial division as vertical towers.

Green dots: n % d != 0

Red dots: divisor hit (n % d == 0)

Towers for primes grow taller (no early hit)

Composites terminate early

This makes the asymmetry of primality testing visible:

Composite → one witness stops everything

Prime → must survive all checks up to √n

Conceptual Results
Bit Buckets
Buckets define space, not prediction

Offsets inside buckets form clean ramps

Resets at powers of two are structural, not number-theoretic

Density Decay
Prime density decreases smoothly with scale

The decay is global, not local

Buckets thin even though internal structure stays constant

Why Structure Doesn’t Predict Primes
Removing shared bits (“sans space”) creates a coordinate system

That coordinate ramps cleanly

Primes merely sample that ramp irregularly

No geometric shortcut survives without embedding arithmetic

Troubleshooting (Important)
Overlapping Text in Manim
If text overlaps after edits or partial renders:

bash
Copy code
python -m manim -pqh --flush_cache prime_bucket_density_plus_and_decay.py PrimeBucketDensityPlusAndDecay
InvalidDataError: partial_movie_file_list.txt
This is not a code bug.
It happens when Manim’s partial MP4s or playlist become corrupted.

Fix (recommended):

bash
Copy code
rm -rf media/videos/prime_bucket_density_plus_and_decay/*/partial_movie_files/PrimeBucketDensityPlusAndDecay
python -m manim -pqh --flush_cache prime_bucket_density_plus_and_decay.py PrimeBucketDensityPlusAndDecay
If it still fails:

bash
Copy code
rm -rf media/videos/prime_bucket_density_plus_and_decay
python -m manim -pqh --flush_cache prime_bucket_density_plus_and_decay.py PrimeBucketDensityPlusAndDecay
Philosophy
Structure can organize numbers.
Density can thin predictably.
Geometry can visualize outcomes.

But primality itself remains an arithmetic fact.

No continuous process, geometric shortcut, or structural embedding predicts primes without explicitly performing computation.

Author
Jonathan Kendall
GitHub: onojk123
Project: geom_factor

License
Open source.
Use, modify, visualize, and build on these ideas freely.
