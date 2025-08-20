# Quadratic Sieve (teaching/demo) with beefed-up parameters for very large N

import math, numpy as np
from collections import defaultdict

# we rely on your utils.py
from utils import small_primes_up_to as small_primes, tonelli_shanks

def build_factor_base(N, B):
    FB = []
    for p in small_primes(B):
        if p == 2:
            continue
        # N must be a quadratic residue mod p
        if pow(N % p, (p-1)//2, p) != 1:
            continue
        r = tonelli_shanks(N % p, p)
        if r is None:
            continue
        FB.append((p, r, (p - r) % p))
    return FB

def trial_divide_over_base(val, FB, LPB):
    """Return (factor dict, leftover). Accept large prime leftover <= LPB."""
    fac = defaultdict(int)
    tmp = abs(val)
    for p, _, _ in FB:
        while tmp % p == 0:
            fac[p] += 1
            tmp //= p
    return fac, tmp if tmp > 1 else 1

def sieve_and_collect(N, FB, A, xs, LPB, want=None, progress_every=5000):
    """Collect full relations and single-large-prime relations (singles)."""
    rels = []
    singles = {}
    ln_p = {p: math.log(p) for p,_,_ in FB}

    for i, x in enumerate(xs):
        Qx = (A + x) * (A + x) - N
        # log-sieve score
        score = math.log(abs(Qx))
        for p, r1, r2 in FB:
            # positions where (A+x) ≡ r (mod p)
            for r in (r1, r2):
                if (A + x) % p == r:
                    score -= ln_p[p]
        if score > 0.8 * math.log(abs(Qx)):
            continue  # unlikely smooth

        fac, leftover = trial_divide_over_base(Qx, FB, LPB)

        if leftover == 1:
            # full FB-smooth
            rels.append({"x": x, "fac": fac})
        elif leftover <= LPB:
            # one large prime
            if leftover in singles:
                # combine two singles with same large prime
                x2, fac2 = singles.pop(leftover)
                fac_merged = fac.copy()
                for p,e in fac2.items():
                    fac_merged[p] += e
                rels.append({"x": x, "fac": fac_merged})
            else:
                singles[leftover] = (x, fac)

        if progress_every and (i+1) % progress_every == 0:
            print(f"[QS] processed {i+1:>6} x-values, full_rels={len(rels)}, singles={len(singles)}")
            # stop early if we've reached our wish list
            if want is not None and len(rels) >= want:
                break

    return rels

def gf2_gauss_nullspace(mat):
    """Very small GF(2) nullspace finder (teaching-quality)."""
    m, n = mat.shape
    A = mat.copy()
    pivcol = [-1] * m
    r = 0
    for c in range(n):
        # find pivot
        pivot = None
        for rr in range(r, m):
            if A[rr, c]:
                pivot = rr
                break
        if pivot is None:
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        # eliminate
        for rr in range(m):
            if rr != r and A[rr, c]:
                A[rr, :] ^= A[r, :]
        pivcol[r] = c
        r += 1
        if r == m: break

    # back-substitute to build a nullspace vector (pick a free column if any)
    free_cols = [c for c in range(n) if c not in pivcol]
    if not free_cols:
        return []
    v = np.zeros(n, dtype=np.uint8)
    v[free_cols[0]] = 1
    for row in range(r-1, -1, -1):
        pc = pivcol[row]
        if pc == -1: continue
        s = 0
        for c in range(pc+1, n):
            s ^= (A[row, c] & v[c])
        v[pc] = s
    return [v]

def build_congruence(N, rels, FB_index, row_mask):
    X = 1
    exp_sum = defaultdict(int)
    for take, r in zip(row_mask, rels):
        if not take: continue
        X = (X * (r["x"] + int(math.isqrt(N))) ) % N  # approximate (A+x)
        for p,e in r["fac"].items():
            if p in FB_index:
                exp_sum[p] += e

    Y = 1
    for p, e in exp_sum.items():
        Y = (Y * pow(p, e//2, N)) % N
    return X % N, Y % N

def factor_via_geom_qs(N, extra_scale=5.0, want_extra=200, max_M=1_000_000):
    """
    QS with:
      - boosted B by a scale factor,
      - large prime bound LPB=1e15,
      - wide sieve interval M,
      - adaptive relation gathering,
      - simple GF(2) nullspace.
    Returns a nontrivial factor or None.
    """
    if N % 2 == 0:
        return 2

    logN = math.log(N)
    loglogN = math.log(logN)
    # base heuristic then multiply
    B0 = math.exp(0.5 * math.sqrt(logN * loglogN))
    B = int(B0 * extra_scale)
    if B < 10_000: B = 10_000

    # sieve width
    M = min(int(2.5 * B), max_M)
    # large prime bound
    LPB = int(1e15)

    A = int(math.isqrt(N))
    FB = build_factor_base(N, B)
    if not FB:
        return None
    FB_index = {p:i for i,(p,_,_) in enumerate(FB)}

    target = len(FB) + want_extra
    xs = list(range(-M, M+1))
    rels = sieve_and_collect(N, FB, A, xs, LPB, want=target, progress_every=10000)

    if len(rels) < max(10, len(FB)//2):
        return None

    # Build GF(2) matrix
    rows = []
    for r in rels:
        v = np.zeros(len(FB), dtype=np.uint8)
        for p,e in r["fac"].items():
            idx = FB_index.get(p)
            if idx is not None:
                v[idx] = (e & 1)
        rows.append(v)
    if not rows:
        return None
    mat = np.array(rows, dtype=np.uint8)

    nulls = gf2_gauss_nullspace(mat)
    if not nulls:
        return None
    row_mask = nulls[0]

    X, Y = build_congruence(N, rels, FB_index, row_mask)
    g = math.gcd((X - Y) % N, N)
    if 1 < g < N:
        return g
    g2 = math.gcd((X + Y) % N, N)
    if 1 < g2 < N:
        return g2
    return None
