import os
import schemdraw
import schemdraw.elements as elm
import schemdraw.logic as logic

# Bucket 3 schematic
# Logic: F = A AND (NOT C)
# Output convention: 1 = not prime, 0 = prime

d = schemdraw.Drawing(scale=3)

# -------------------------
# Inputs
# -------------------------
# A (MSB)
d.add(elm.Dot().at((0, 4)).label('A (MSB)', loc='left'))
d.add(elm.Line().right(2))

# C (LSB)
d.add(elm.Dot().at((0, 0)).label('C (LSB)', loc='left'))
d.add(elm.Line().right(2))

# -------------------------
# NOT gate
# -------------------------
inv = logic.Not().at((2, 0)).right().label('74HC04')
inv.fill(False)
d.add(inv)

# Wire into NOT
d.add(elm.Line().at((2, 0)).left(0.8))

# -------------------------
# AND gate
# -------------------------
and1 = logic.And().at((6, 2)).right().label('74HC08')
and1.fill(False)
d.add(and1)

# NOT → AND (lower input region)
d.add(elm.Line().at((3.6, 0)).right(1.0))
d.add(elm.Line().up(2.0))
d.add(elm.Line().right(1.6))

# A → AND (upper input region)
d.add(elm.Line().at((2, 4)).right(4.0))

# -------------------------
# Output
# -------------------------
d.add(elm.Line().at((8.6, 2)).right(2.0))
d.add(elm.Dot().label('F (1 = not prime)', loc='right'))

# -------------------------
# Save SVG
# -------------------------
outfile = "bucket3.svg"
d.save(outfile)

print("SCHEMATIC WRITTEN TO:")
print(os.path.abspath(outfile))
