#!/usr/bin/env python3
import math

# Terminal styling (bold). If your terminal doesn't like ANSI, set USE_ANSI = False.
USE_ANSI = True
BOLD = "\033[1m" if USE_ANSI else ""
RESET = "\033[0m" if USE_ANSI else ""

NZ = "!0"   # symbol for non-zero remainder
Z  = "0"    # symbol for zero remainder


def fmt_token(tok: str) -> str:
    """Make clusters of 0 visually bold."""
    if tok == Z:
        return f"{BOLD}{Z}{RESET}"
    return NZ


def tokens_for_n(n: int, mode: str = "sqrt"):
    """
    mode:
      - "sqrt": only test d = 2..floor(sqrt(n))  (untested d are deleted)
      - "full": test d = 2..n-1
    Returns: (limit, list_of_tokens, first_zero_divisor_or_None)
    """
    if n < 2:
        return 0, [], None

    if mode == "full":
        ds = range(2, n)
        limit = n - 1
    else:
        limit = int(math.isqrt(n))
        ds = range(2, limit + 1) if limit >= 2 else []

    toks = []
    first_hit = None
    for d in ds:
        if n % d == 0:
            toks.append(Z)
            if first_hit is None:
                first_hit = d
        else:
            toks.append(NZ)

    return limit, toks, first_hit


def compress_runs(toks):
    """
    Optional compression: collapse consecutive identical tokens into a run like:
      !0×5  or  0×2
    Still bolds '0' runs.
    """
    if not toks:
        return []

    out = []
    cur = toks[0]
    count = 1
    for t in toks[1:]:
        if t == cur:
            count += 1
        else:
            out.append((cur, count))
            cur = t
            count = 1
    out.append((cur, count))
    return out


def render_line(n: int, mode: str, compress: bool):
    limit, toks, first_hit = tokens_for_n(n, mode=mode)

    if mode == "sqrt":
        header = f"N={n:>2}  d=2..⌊√N⌋={limit:>2} : "
    else:
        header = f"N={n:>2}  d=2..N-1     : "

    if not toks:
        # Covers N=2 and N=3 in sqrt mode (no tests to show)
        body = "(no tests)"
    else:
        if compress:
            runs = compress_runs(toks)
            parts = []
            for tok, cnt in runs:
                if tok == Z:
                    parts.append(f"{BOLD}{Z}×{cnt}{RESET}")
                else:
                    parts.append(f"{NZ}×{cnt}")
            body = "  ".join(parts)
        else:
            body = " ".join(fmt_token(t) for t in toks)

    verdict = "PRIME" if first_hit is None else f"COMPOSITE (hit d={first_hit})"
    return header + body + "   => " + verdict


def main():
    # Adjust these defaults however you like
    start = 2
    end = 11

    mode = "sqrt"      # "sqrt" removes untested by not printing them; "full" prints all
    compress = False   # True shows runs like !0×k and 0×k instead of individual tokens

    print("\nMod-gate visualizer (0 vs !0). Untested checks are omitted.\n")
    for n in range(start, end + 1):
        print(render_line(n, mode=mode, compress=compress))

    print("\nLegend: 0 means N mod d = 0 (divisible); !0 means non-zero remainder.\n")


if __name__ == "__main__":
    main()
