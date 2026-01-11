#!/usr/bin/env python3
from skidl import *

set_default_tool(KICAD)

# -------------------------
# Nets
# -------------------------
A = Net("A")        # MSB input
C = Net("C")        # LSB input
C_BAR = Net("C_BAR")
F = Net("F")        # output (1 = not prime)

# -------------------------
# Avoid "No footprint" failures in strict SKiDL builds
# -------------------------
set_default_footprint("")

# -------------------------
# Symbol library candidates (KiCad 7 on Ubuntu/Debian)
# -------------------------
LIBS = [
    "/usr/share/kicad/symbols/Logic_Gate.kicad_sym",
    "/usr/share/kicad/symbols/Device.kicad_sym",
]

NOT_CANDIDATES = [
    "NOT", "INV", "INVERTER", "NOT_Gate", "Inverter",
]

AND2_CANDIDATES = [
    "AND", "AND2", "AND2_Gate", "AND_Gate", "And", "And2",
]

def first_part_that_exists(libpaths, names, ref_prefix):
    """Return (part, libpath, name) for first symbol that loads."""
    last_err = None
    for lib in libpaths:
        for name in names:
            try:
                p = Part(lib, name, footprint="")
                # Give stable references (U1, U2...) rather than random tags.
                p.ref_prefix = ref_prefix
                return p, lib, name
            except Exception as e:
                last_err = e
    raise RuntimeError(f"Could not find any of {names} in libs {libpaths}. Last error: {last_err}")

# -------------------------
# Pick symbols that exist on *your* machine
# -------------------------
g_not, not_lib, not_name = first_part_that_exists(LIBS, NOT_CANDIDATES, "U")
g_and, and_lib, and_name = first_part_that_exists(LIBS, AND2_CANDIDATES, "U")

# -------------------------
# Wiring by pin numbers
# Assumption (true for most generic logic symbols):
#   NOT: 1=input, 2=output
#   AND2: 1=input, 2=input, 3=output
# -------------------------
C      += g_not[1]
C_BAR += g_not[2]

A      += g_and[1]
C_BAR += g_and[2]
F      += g_and[3]

# -------------------------
# Generate outputs (your SKiDL: no filename args)
# -------------------------
generate_netlist()
generate_bom()

print("WROTE: skidl.net")
print("WROTE: skidl_bom.csv")
print("USED NOT :", not_name, "from", not_lib)
print("USED AND2:", and_name, "from", and_lib)
