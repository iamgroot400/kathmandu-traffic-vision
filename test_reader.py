"""Self-checks for reader.reading_order with hand-made boxes (no models needed).

    python test_reader.py
"""
from reader import check_plate, reading_order, valid_prefix


def box(x, y, label, w=20, h=30):
    return (x, y, x + w, y + h, label, 0.9)


def text(boxes):
    return "".join(b[4] for b in reading_order(boxes))


# one row, given out of order
assert text([box(60, 10, "3"), box(10, 12, "1"), box(35, 8, "2")]) == "123", text([box(60, 10, "3"), box(10, 12, "1"), box(35, 8, "2")])

# two rows: top row first, each left to right; bottom row starts further left than the top row
two = [box(40, 70, "6"), box(30, 5, "B"), box(10, 72, "5"), box(55, 8, "A"), box(70, 68, "7")]
assert text(two) == "BA567", text(two)

# a slightly tilted single row must stay one row (each box a few pixels lower than the last)
tilt = [box(10 + 25 * i, 10 + 4 * i, str(i)) for i in range(6)][::-1]
assert text(tilt) == "012345", text(tilt)

# a tilted two-row plate
two_tilt = [box(10, 10, "a"), box(35, 14, "b"), box(60, 18, "c"), box(12, 60, "d"), box(37, 64, "e"), box(62, 68, "f")]
assert text(two_tilt[::-1]) == "abcdef", text(two_tilt[::-1])

# one box
assert text([box(5, 5, "x")]) == "x"
# plate format check
ok = lambda labels, conf=0.95: check_plate(labels, conf)[0]
assert ok(["बा", "२", "च", "१", "२", "३", "४"])                 # zonal: zone, lot, class, number
assert ok(["को", "१", "प", "५", "६"])                            # short number
assert ok(["प्र", "३", "०", "१", "च", "१", "२", "३", "४"])       # provincial reads as the same shape
assert not ok(["बा", "२", "च"])                                  # bottom row missed
assert not ok(["बा", "२", "च", "१", "२", "३", "४", "५"])         # five-digit number: extra character
assert not ok(["बा", "२", "१", "२", "३", "४"])                   # vehicle-class letter missed
assert not ok(["बा", "२", "च", "१", "२", "3", "४"])              # embossed digit inside a Devanagari plate
assert not ok(["बा", "२", "च", "१", "२", "३", "४"], conf=0.4)    # one unsure character
assert ok(["B", "A", "0", "1", "P", "A", "1", "2", "3", "4"])     # embossed: letters and digits
assert not ok(["1", "2", "3", "4"])                               # embossed digits only
assert not ok([])

# zone / province code at the start
for good in ["बा२च१२३४", "मे१प२३", "से५प१", "म३प४", "प्रदेश३०३००९प१२३४", "प्र३०३००९प१२३४",
             "बागमतीप्रदेश०६००९प१२३४", "बाप्रदेश०६०९प१२३४", "बाप्र०६००९प१२३४", "गण्डकीप्रदेश०४००९प१२३४"]:
    assert valid_prefix(good), good
for bad in ["वाप्र०६००९प१२३४", "ह२प१२३४", "बाच१२३४", "प्रदेशप१२", "x", ""]:
    assert not valid_prefix(bad), bad
assert not ok(["ह", "२", "च", "१", "२", "३", "४"])          # right shape, unknown zone

# read_vehicle with stand-in models: plate found and read; no plate found
import numpy as np
from types import SimpleNamespace as NS
from reader import read_vehicle

def fake(xyxy=(), names=None, top1=0):
    res = NS(boxes=NS(xyxy=NS(tolist=lambda: [list(b) for b in xyxy])), names=names, probs=NS(top1=top1, top1conf=0.9))
    return NS(predict=lambda *a, **k: [res])

frame = np.zeros((400, 600, 3), np.uint8)
plates = fake([(50, 200, 150, 240)])                               # plate inside the vehicle crop
chars = fake([(10 + 12 * i, 5, 20 + 12 * i, 30) for i in range(7)])  # 7 characters in one row
seq = iter("बा२च१२३४")
glyph_res = lambda: NS(names={0: next(seq)}, probs=NS(top1=0, top1conf=0.9))
glyphs = NS(predict=lambda *a, **k: [glyph_res()])
text_, check, crop = read_vehicle(frame, (100, 50, 400, 350), plates, chars, glyphs)
assert (text_, check) == ("बा२च१२३", "ok"), (text_, check)       # 7 boxes -> first 7 labels, valid layout
assert crop is not None and crop.shape[0] > 0
text_, check, crop = read_vehicle(frame, (100, 50, 400, 350), fake([]), chars, glyphs)
assert (text_, check, crop) == ("", "review: no plate found", None)
print("all checks pass")
