"""Self-checks for the decision logic and the light reader; no model or video needed.  python test_logic.py"""
import numpy as np

from light import cycle_colour, light_colour
from logic import Monitor, crossed, is_violation

LINE = ((0, 100), (200, 100))   # horizontal stop line; moving down (y grows) crosses it one way

# crossings
assert crossed((50, 90), (50, 110), *LINE) == -crossed((50, 110), (50, 90), *LINE) != 0
assert crossed((50, 90), (50, 95), *LINE) == 0          # approached, did not cross
assert crossed((300, 90), (300, 110), *LINE) == 0       # crossed beyond the end of the segment
FWD = crossed((50, 90), (50, 110), *LINE)               # moving down = entering the junction

# violation rule
assert is_violation(FWD, FWD, "red", 5.0, 1.0)          # clear red-light run
assert not is_violation(FWD, FWD, "green", 0.0, 1.0)
assert not is_violation(FWD, FWD, "red", 0.5, 1.0)      # inside the grace period
assert not is_violation(-FWD, FWD, "red", 5.0, 1.0)     # leaving the junction / reversing
assert not is_violation(FWD, FWD, "unknown", 0.0, 1.0)  # no evidence of red, no flag


def run(states, path):
    """one vehicle moving along path, one frame per entry, light as given; 10 fps"""
    m = Monitor(stop_line=LINE, forward=FWD, grace=1.0)
    for i, (s, p) in enumerate(zip(states, path)):
        m.update(i / 10, s, [(7, "car", p)])
    return m


down = [(50, 20 + 4 * i) for i in range(40)]            # crosses y=100 at frame 20 (t = 2 s)
assert len(run(["red"] * 40, down).violations) == 1     # red from the start (confirmed at frame 2)
assert len(run(["green"] * 40, down).violations) == 0
assert len(run(["green"] * 15 + ["red"] * 25, down).violations) == 0   # red confirmed 0.3 s before: grace
m = run(["red"] * 80, down + down[::-1])                # goes in, then backs out over the line
assert sum(m.counts.values()) == 2 and len(m.violations) == 1
assert len(run(["red"] * 3 + ["green"] + ["red"] * 36, down).violations) == 1   # one misread frame ignored

# light reader on synthetic frames (BGR)
f = np.zeros((100, 100, 3), np.uint8)
f[10:30, 40:60] = (0, 0, 255)
assert light_colour(f, (30, 0, 40, 100)) == "red"
f[:] = 0
f[70:90, 40:60] = (0, 255, 0)
assert light_colour(f, (30, 0, 40, 100)) == "green"
f[:] = 0
assert light_colour(f, (30, 0, 40, 100)) == "unknown"
assert [cycle_colour(t, red=30, green=20) for t in (0, 21, 30)] == ["green", "amber", "red"]
print("all checks pass")
