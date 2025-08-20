# geom_factor
Geometric + Algorithmic Factorization Toolkit 
============================================= This repository contains 
experimental code for exploring integer factorization using both classic number 
theory methods and geometric visualization ideas. ⚠️ Disclaimer: Educational / 
research exploration only. Not suitable for real cryptographic attacks.
## Features
- **budget_driver.py**: main orchestrator. Reads integer from file or raw string, 
  applies: 1. Trial division 2. Pollard’s p‑1 3. ECM (pure Python or external 
  GMP‑ECM if installed)
- **pminus1.py**: Pollard p‑1 stage‑1 algorithm - **ecm.py**: simple stage‑1 
elliptic curve method (toy) - RSA‑270 input helper (strip digits from Reddit 
block)
## Requirements
- Python 3.10+ - Packages: `sympy` (optional, for primes) - Optional: GMP‑ECM 
(`sudo apt install gmp-ecm`)
## Quick Start
```bash git clone https://github.com/onojk/geom_factor.git cd geom_factor
# prepare a test composite
python3 - <<'PY' import random, sympy as sp random.seed(3) p = sp.nextprime(10**11 
+ random.randrange(10**10)) q = sp.nextprime(10**11 + random.randrange(10**10)) n 
= p*q open("tinyN.txt","w").write(str(n)) print("Digits:", len(str(n))) PY
# run driver with trial, Pollard p‑1, ECM
python3 budget_driver.py tinyN.txt --trial 100000 --p1 50000 --ecm-B1 50000 
--ecm-curves 2000 --use-gmp ```
## Example: RSA‑270
```bash
# paste the Reddit block into raw_rsa270.txt
tr -cd '0-9' < raw_rsa270.txt | head -c 270 > rsa270.txt python3 budget_driver.py 
rsa270.txt --use-gmp --ecm-B1 1000000 --ecm-curves 10000 ``` (Real factorization 
of RSA‑270 is far beyond these parameters, but you can exercise the pipeline.)
## License
MIT License — for educational use only.
