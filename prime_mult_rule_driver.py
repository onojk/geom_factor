#!/usr/bin/env python3
"""
prime_mult_rule_driver.py

Driver for the rule:
  "Never multiply more than 2 potentially-nonunit factors (>1) at any point,
   unless the extra factor(s) are immediately canceled (e.g., (a*b*c)/c)."

This is a STRUCTURAL checker:
- It parses an expression (Python syntax) into an AST.
- It detects multiplication nodes with >=3 non-unit factors.
- It allows those nodes only if they are directly under a division numerator
  and at least one factor is canceled by the division denominator such that
  the remaining non-unit factor-count <= 2.

Supported:
- +, -, *, /, **, parentheses, integers, variables (names)
Not supported:
- function calls (sin(x)), comparisons, indexing, etc.
"""

from __future__ import annotations
import argparse
import ast
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ---------- Factor model ----------
# We treat these as "potentially non-unit (>1)" multiplicative atoms:
# - variable names (a, b, x, ...)
# - integer constants > 1
# We treat 0 and 1 and -1 as NOT adding "non-unit factor" count for this rule.
#
# This is a structural check, not a full algebra system.

def atom_key_for_int(n: int) -> Optional[str]:
    if abs(n) <= 1:
        return None
    return f"#{n}"  # keep sign; still a non-unit factor structurally


def factors_of_node(node: ast.AST) -> Counter:
    """
    Return multiplicative factors multiset for a node, *only* when the node
    is multiplicative-combinable (constants, names, multiplications, powers, divisions).

    If we hit addition/subtraction, we return an empty Counter and mark it "opaque"
    by raising ValueError — because sums don't preserve a simple factor multiset.
    """
    if isinstance(node, ast.Name):
        return Counter({node.id: 1})

    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        k = atom_key_for_int(node.value)
        return Counter({k: 1}) if k else Counter()

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        # -X : treat sign as not changing factor structure except for integer constants
        inner = node.operand
        if isinstance(inner, ast.Constant) and isinstance(inner.value, int):
            k = atom_key_for_int(-inner.value)
            return Counter({k: 1}) if k else Counter()
        # For -a, we ignore the -1 factor for this rule
        return factors_of_node(inner)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return factors_of_node(node.left) + factors_of_node(node.right)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        num = factors_of_node(node.left)
        den = factors_of_node(node.right)
        # cancellation: subtract common counts
        for k in list(num.keys()):
            c = min(num[k], den.get(k, 0))
            if c:
                num[k] -= c
                den[k] -= c
                if num[k] == 0:
                    del num[k]
                if den[k] == 0:
                    del den[k]
        # Remaining factors = numerator factors plus "inverse" denom factors (we keep as separate keys)
        # For the rule, denom factors also represent multiplicative structure, but we only need
        # numerator factor-count when checking "prime-producing" multiplication pressure.
        # Still, for completeness, represent denom leftovers as /k
        out = Counter(num)
        for k, v in den.items():
            out[f"/{k}"] += v
        return out

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        base_f = factors_of_node(node.left)
        # If exponent is a positive int constant, scale multiplicities
        exp = node.right
        if isinstance(exp, ast.Constant) and isinstance(exp.value, int) and exp.value >= 0:
            e = exp.value
            out = Counter()
            for k, v in base_f.items():
                out[k] += v * e
            return out
        # Unknown exponent: treat as opaque multiplicatively (could explode factors)
        raise ValueError("Opaque exponent: cannot safely factor")

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        raise ValueError("Opaque (addition/subtraction)")

    # Anything else: opaque
    raise ValueError(f"Unsupported/opaque node type: {type(node).__name__}")


def count_nonunit_factors(f: Counter) -> int:
    """
    Count how many potentially-nonunit factors are present (with multiplicity),
    ignoring any "/k" inverse markers.
    """
    total = 0
    for k, v in f.items():
        if k is None:
            continue
        if isinstance(k, str) and k.startswith("/"):
            continue
        total += v
    return total


# ---------- Violations ----------
@dataclass
class MulInfo:
    node: ast.BinOp
    factors: Counter
    nonunit_count: int
    allowed_by_immediate_cancel: bool
    reason: str


class MulRuleChecker(ast.NodeVisitor):
    def __init__(self):
        self.parent_stack: List[ast.AST] = []
        self.mul_nodes: List[MulInfo] = []

    def visit(self, node: ast.AST):
        self.parent_stack.append(node)
        super().visit(node)
        self.parent_stack.pop()

    def visit_BinOp(self, node: ast.BinOp):
        # visit children first
        self.visit(node.left)
        self.visit(node.right)

        if isinstance(node.op, ast.Mult):
            self._check_mul(node)

    def _check_mul(self, node: ast.BinOp):
        try:
            f = factors_of_node(node)
            ncnt = count_nonunit_factors(f)
        except ValueError:
            # opaque multiplication: still report as "unknown risk"
            self.mul_nodes.append(MulInfo(
                node=node,
                factors=Counter(),
                nonunit_count=-1,
                allowed_by_immediate_cancel=False,
                reason="Opaque multiplication (contains +/-, unknown exponent, or unsupported structure)"
            ))
            return

        if ncnt < 3:
            self.mul_nodes.append(MulInfo(
                node=node,
                factors=f,
                nonunit_count=ncnt,
                allowed_by_immediate_cancel=True,
                reason="OK (fewer than 3 non-unit factors)"
            ))
            return

        # ncnt >= 3: only allowed if immediate cancellation applies
        allowed, reason = self._immediate_cancel_allows(node, f)
        self.mul_nodes.append(MulInfo(
            node=node,
            factors=f,
            nonunit_count=ncnt,
            allowed_by_immediate_cancel=allowed,
            reason=reason
        ))

    def _immediate_cancel_allows(self, mul_node: ast.BinOp, mul_f: Counter) -> Tuple[bool, str]:
        """
        Allow if:
          - The parent is a division, and this mul_node is in the numerator (left),
          - AND denominator shares at least one factor with mul_f,
          - AND after canceling shared factors, remaining nonunit_count <= 2.
        """
        # Find direct parent
        if len(self.parent_stack) < 2:
            return False, "Violation: 3+ non-unit factors and no parent context"

        parent = self.parent_stack[-2]
        if not (isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div)):
            return False, "Violation: 3+ non-unit factors and not immediately canceled by /"

        # Must be numerator child
        if parent.left is not mul_node:
            return False, "Violation: 3+ non-unit factors only allowed in numerator of a cancelling division"

        # Compute denominator factors
        try:
            den_f = factors_of_node(parent.right)
        except ValueError:
            return False, "Violation: denominator is opaque; cannot confirm immediate cancellation"

        # Cancel common factors between mul_f and den_f (ignoring inverse markers)
        num = Counter(mul_f)
        den = Counter({k: v for k, v in den_f.items() if not (isinstance(k, str) and k.startswith("/"))})

        common = Counter()
        for k in list(num.keys()):
            if isinstance(k, str) and k.startswith("/"):
                continue
            c = min(num[k], den.get(k, 0))
            if c:
                common[k] = c
                num[k] -= c
                den[k] -= c
                if num[k] == 0:
                    del num[k]

        if sum(common.values()) == 0:
            return False, "Violation: 3+ non-unit factors but denominator cancels none of them"

        remaining = count_nonunit_factors(num)
        if remaining <= 2:
            return True, f"OK by immediate cancellation: canceled {dict(common)}; remaining non-unit factors={remaining}"
        return False, f"Violation: cancellation insufficient; remaining non-unit factors={remaining} (needs <=2)"


# ---------- Evaluation ----------
def safe_eval(expr: str, env: Dict[str, int]) -> int:
    """
    Evaluate expression with integer env using Python's eval on a restricted AST.
    We forbid attribute access, calls, comprehensions, etc.
    """
    node = ast.parse(expr, mode="eval")

    for n in ast.walk(node):
        if isinstance(n, (ast.Call, ast.Attribute, ast.Subscript, ast.Lambda, ast.Dict, ast.List, ast.Tuple,
                          ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Compare, ast.IfExp)):
            raise ValueError(f"Disallowed syntax: {type(n).__name__}")

    code = compile(node, "<expr>", "eval")
    return int(eval(code, {"__builtins__": {}}, dict(env)))


def parse_vars(kvs: List[str]) -> Dict[str, int]:
    env: Dict[str, int] = {}
    for s in kvs:
        if "=" not in s:
            raise ValueError(f"Bad --var '{s}', expected name=value")
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k.isidentifier():
            raise ValueError(f"Bad variable name '{k}'")
        env[k] = int(v)
    return env


# ---------- CLI ----------
def main() -> None:
    ap = argparse.ArgumentParser(description="Driver: enforce 'no 3+ non-unit factors unless immediately canceled' rule.")
    ap.add_argument("--expr", required=True, help='Expression, e.g. "(a*b*c)/c" or "L*W*H"')
    ap.add_argument("--var", action="append", default=[], help="Assign variable, e.g. --var a=3 (repeatable)")
    ap.add_argument("--show-ast", action="store_true", help="Print AST (debug)")

    args = ap.parse_args()
    env = parse_vars(args.var)

    tree = ast.parse(args.expr, mode="eval")
    if args.show_ast:
        print(ast.dump(tree, indent=2))

    checker = MulRuleChecker()
    checker.visit(tree.body)

    print(f"Expression: {args.expr}")
    if env:
        print(f"Vars: {env}")

    # Report multiplication nodes
    print("\nMultiplication checks:")
    any_violation = False
    for i, info in enumerate(checker.mul_nodes, 1):
        if info.nonunit_count == -1:
            status = "RISK"
            any_violation = True
        else:
            status = "OK" if info.allowed_by_immediate_cancel else "VIOLATION"
            if status == "VIOLATION":
                any_violation = True

        # Pretty factor summary
        if info.factors:
            factors_str = " * ".join([f"{k}^{v}" if v != 1 else str(k) for k, v in info.factors.items()])
        else:
            factors_str = "(opaque)"

        print(f"  [{i}] {status}: non-unit-factor-count={info.nonunit_count} | factors: {factors_str}")
        print(f"      {info.reason}")

    # Try evaluation
    try:
        val = safe_eval(args.expr, env)
        print(f"\nEvaluates to: {val}")
    except Exception as e:
        print(f"\nCould not evaluate numerically: {e}")

    print("\nResult:")
    if any_violation:
        print("  ❌ Rule violated (or opaque risk detected).")
    else:
        print("  ✅ Rule satisfied (no 3+ non-unit multiplication unless immediately canceled).")


if __name__ == "__main__":
    main()
