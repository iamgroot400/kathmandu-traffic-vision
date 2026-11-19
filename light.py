"""Read the traffic light's colour from a fixed box in the frame (the camera does not move)."""
import cv2
import numpy as np

# HSV ranges (OpenCV hue is 0-179); lit lamps are saturated and bright
RANGES = {"red": [((0, 100, 120), (10, 255, 255)), ((170, 100, 120), (179, 255, 255))],
          "amber": [((15, 100, 120), (35, 255, 255))],
          "green": [((40, 80, 120), (95, 255, 255))]}
MIN_SHARE = 0.02   # share of the box that must be lit in a colour to count


def light_colour(frame, roi) -> str:
    """'red' / 'amber' / 'green' for the most-lit colour in roi = (x, y, w, h), else 'unknown'."""
    x, y, w, h = roi
    hsv = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    share = {c: sum(int(np.count_nonzero(cv2.inRange(hsv, np.array(lo), np.array(hi)))) for lo, hi in rs) / hsv[..., 0].size
             for c, rs in RANGES.items()}
    best = max(share, key=share.get)
    return best if share[best] >= MIN_SHARE else "unknown"


def cycle_colour(t: float, red: float, green: float, amber: float = 3.0, offset: float = 0.0) -> str:
    """Fallback when the light is not in view: a fixed-time cycle (green, amber, red) from offset seconds."""
    k = (t - offset) % (green + amber + red)
    return "green" if k < green else "amber" if k < green + amber else "red"
