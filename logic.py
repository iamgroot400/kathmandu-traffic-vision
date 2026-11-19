"""Pure decision logic: line crossings, signal state, counts and red-light violations.
No video or model here, so all of it is testable with fake tracks (test_logic.py)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# passenger-car units per vehicle class (COCO names), for flow in PCU/h
PCU = {"bicycle": 0.5, "motorcycle": 0.5, "car": 1.0, "bus": 3.0, "truck": 3.0}


def _side(p, a, b) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossed(prev, cur, a, b) -> int:
    """+1 / -1 if the move prev -> cur crosses segment a-b (sign = which way), else 0."""
    s0, s1 = _side(prev, a, b), _side(cur, a, b)
    if s0 * s1 >= 0:                    # on the line, or stayed on one side (Monitor never stores on-line points)
        return 0
    if _side(a, prev, cur) * _side(b, prev, cur) > 0:   # crossed the infinite line beyond the segment
        return 0
    return 1 if s1 > 0 else -1


class Signal:
    """Debounced light state: a new colour must persist `hold` frames before it counts,
    so one misread frame does not flip red to green."""

    def __init__(self, hold: int = 3):
        self.hold, self.state, self.since = hold, "unknown", 0.0
        self._cand, self._n = None, 0

    def update(self, seen: str, t: float) -> str:
        if seen == self.state:
            self._cand, self._n = None, 0
        elif seen == self._cand:
            self._n += 1
            if self._n >= self.hold:
                self.state, self.since, self._cand, self._n = seen, t, None, 0
        else:
            self._cand, self._n = seen, 1
        return self.state


def is_violation(direction: int, forward: int, light: str, red_for: float, grace: float) -> bool:
    """Should a crossing of the stop line be flagged as running the red light?

    direction: +1/-1, which way the vehicle crossed the stop line
    forward:   the direction that means "entering the junction" (from config)
    light:     debounced signal state: "red", "amber", "green" or "unknown"
    red_for:   seconds the light has been red at the moment of crossing
    grace:     seconds after the light turns red during which a crossing is not flagged
    """
    # amber and unknown are never flagged: a wrong flag fines an innocent driver, a missed one costs little
    return direction == forward and light == "red" and red_for >= grace


@dataclass
class Monitor:
    stop_line: tuple
    forward: int = 1
    grace: float = 1.0
    signal: Signal = field(default_factory=Signal)
    counts: Counter = field(default_factory=Counter)      # (minute, class, "in"/"out") -> vehicles
    violations: list = field(default_factory=list)
    _last: dict = field(default_factory=dict)            # track id -> last point
    _done: set = field(default_factory=set)              # (track id, direction) already counted

    def update(self, t: float, seen_light: str, tracks) -> list:
        """tracks: iterable of (track_id, class_name, (x, y) ground point). Returns new violations."""
        light = self.signal.update(seen_light, t)
        red_for = t - self.signal.since if light == "red" else 0.0
        new = []
        for tid, cls, p in tracks:
            if _side(p, *self.stop_line) == 0:   # exactly on the line: wait until it is clearly past
                continue
            prev = self._last.get(tid)
            self._last[tid] = p
            if prev is None:
                continue
            d = crossed(prev, p, *self.stop_line)
            if not d or (tid, d) in self._done:
                continue
            self._done.add((tid, d))
            self.counts[int(t // 60), cls, "in" if d == self.forward else "out"] += 1
            if is_violation(d, self.forward, light, red_for, self.grace):
                v = {"time_s": round(t, 2), "track_id": tid, "class": cls, "light": light,
                     "red_for_s": round(red_for, 2)}
                self.violations.append(v)
                new.append(v)
        return new
