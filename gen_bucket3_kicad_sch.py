#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import re
import uuid

TEMPLATE = Path("~/projects/geom_factor/kicad_buckets/clean/bucket_buckets.kicad_sch").expanduser()
LIB74 = Path("/usr/share/kicad/symbols/74xx.kicad_sym")
OUT = Path("bucket3.kicad_sch")

def u() -> str:
    return str(uuid.uuid4())

def indent(block: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in block.splitlines())

def extract_symbol_block(kicad_sym_text: str, symbol_name: str) -> str:
    """
    Extract the full (symbol ...) block for a symbol from a KiCad .kicad_sym file.

    KiCad 7 .kicad_sym commonly stores symbols like:
      (symbol (name "74HC04") ...)
    Older forms may look like:
      (symbol "74HC04" ...)
    We support both.
    """
    # Try KiCad 7+ style first.
    patterns = [
        rf'\(symbol\s+\(name\s+"{re.escape(symbol_name)}"\)',   # (symbol (name "74HC04")
        rf'\(symbol\s+"{re.escape(symbol_name)}"\b',            # (symbol "74HC04"
    ]

    m = None
    for pat in patterns:
        m = re.search(pat, kicad_sym_text)
        if m:
            break

    if not m:
        # Helpful debug: show close matches
        near = sorted(set(re.findall(r'\(symbol\s+\(name\s+"([^"]+)"\)', kicad_sym_text)))
        close = [n for n in near if symbol_name.upper() in n.upper() or n.upper() in symbol_name.upper()]
        msg = f'Could not find symbol "{symbol_name}" in {LIB74}'
        if close:
            msg += f"\nClosest matches: {close[:20]}"
        raise ValueError(msg)

    start = m.start()

    # Balanced parentheses scan from start.
    depth = 0
    i = start
    while i < len(kicad_sym_text):
        ch = kicad_sym_text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return kicad_sym_text[start:i+1].strip()
        i += 1

    raise ValueError(f'Unbalanced parentheses while extracting "{symbol_name}"')

# ---- load template + libs ----
tmpl = TEMPLATE.read_text(encoding="utf-8").strip()
libtxt = LIB74.read_text(encoding="utf-8")

# IMPORTANT: In your install you proved SKiDL sees these names:
#   74HC04 and 74LS08
sym_74hc04 = extract_symbol_block(libtxt, "74HC04")
sym_74ls08 = extract_symbol_block(libtxt, "74LS08")

lib_symbols_section = "(lib_symbols\n" + indent(sym_74hc04, 2) + "\n" + indent(sym_74ls08, 2) + "\n)"

# ---- create instances ----
u1_uuid = u()
u2_uuid = u()

def symbol_instance(lib_id: str, ref: str, value: str, atx: int, aty: int, unit: int, uid: str) -> str:
    return f'''(symbol (lib_id "{lib_id}") (at {atx} {aty} 0) (unit {unit})
  (in_bom yes) (on_board yes)
  (uuid "{uid}")
  (property "Reference" "{ref}" (at {atx} {aty+7} 0) (effects (font (size 1.27 1.27))))
  (property "Value" "{value}" (at {atx} {aty-7} 0) (effects (font (size 1.27 1.27))))
  (property "Footprint" "" (at 0 0 0) (effects (font (size 1.27 1.27))) (hide yes))
)'''

# We'll start with plain symbol name IDs (no "74xx:" prefix) since we're embedding lib symbols.
U1 = symbol_instance("74HC04", "U1", "74HC04", 90, 100, 1, u1_uuid)
U2 = symbol_instance("74LS08", "U2", "74LS08", 130, 100, 1, u2_uuid)

def glabel(name: str, shape: str, x: int, y: int) -> str:
    return f'''(global_label "{name}" (shape {shape}) (at {x} {y} 0)
  (effects (font (size 1.27 1.27)))
)'''

def wire(x1: int, y1: int, x2: int, y2: int) -> str:
    return f'(wire (pts (xy {x1} {y1}) (xy {x2} {y2})))'

labels = "\n".join([
    glabel("A", "input", 60, 110),
    glabel("C", "input", 60, 90),
    glabel("C_BAR", "passive", 110, 90),
    glabel("F", "output", 170, 100),
])

wires = "\n".join([
    wire(65, 110, 120, 110),   # A bus to AND input zone
    wire(65, 90, 80, 90),      # C to NOT input zone
    wire(100, 90, 125, 100),   # NOT out toward AND input zone
    wire(150, 100, 165, 100),  # AND out to F label
])

sym_instances = f'''(symbol_instances
  (path "/{u1_uuid}" (reference "U1") (unit 1))
  (path "/{u2_uuid}" (reference "U2") (unit 1))
)'''

# ---- assemble final schematic ----
out = tmpl
out = out.replace("(lib_symbols)", lib_symbols_section)
out = re.sub(r"\(symbol_instances\)", sym_instances, out)

insert_block = "\n\n" + indent(U1, 2) + "\n\n" + indent(U2, 2) + "\n\n" + indent(labels, 2) + "\n\n" + indent(wires, 2) + "\n"
out = out[:-1] + insert_block + ")\n"

OUT.write_text(out, encoding="utf-8")
print(f"WROTE: {OUT.resolve()}")

