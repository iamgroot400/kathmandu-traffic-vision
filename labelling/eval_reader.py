"""Score the full plate reader (plate detector -> character detector -> classifier) against a label file.

    python labelling/eval_reader.py <labels.tsv> <photos dir> [--glyphs ...] [--conf 0.25 0.35 ...]

Prints, per character-detector confidence threshold: whole-plate exact match, number-part match, and how many
readings pass check_plate. Tune on training labels only; run the test labels once with the chosen setting.
"""
import argparse
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from reader import check_plate, read_plate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("labels"); ap.add_argument("photos")
ap.add_argument("--plates", default="results/plates_run1/plates_best.pt")
ap.add_argument("--chars", default="results/reader_run1/chars_best.pt")
ap.add_argument("--glyphs", default="results/reader_run3/glyphs_best.pt")
ap.add_argument("--conf", type=float, nargs="+", default=[0.25])
a = ap.parse_args()

from ultralytics import YOLO  # noqa: E402
plates, chars, glyphs = YOLO(a.plates), YOLO(a.chars), YOLO(a.glyphs)
lab = {k: v for k, v in (l.split("\t") for l in Path(a.labels).read_text(encoding="utf-8").splitlines() if "\t" in l)}
digits = lambda s: (re.findall(r"[०-९]+$", s) or [""])[0]


def crop_of(img):
    b = plates.predict(img, imgsz=1280, conf=0.2, verbose=False)[0].boxes.xyxy.tolist()
    if not b:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = min(v[0] for v in b), min(v[1] for v in b), max(v[2] for v in b), max(v[3] for v in b)
    px, py = (x2 - x1) * 0.08, (y2 - y1) * 0.15
    return img[int(max(0, y1 - py)):int(min(h, y2 + py)), int(max(0, x1 - px)):int(min(w, x2 + px))]


crops = {k: crop_of(cv2.imread(str(Path(a.photos) / f"{k}.jpg"))) for k in lab}
for c in a.conf:
    exact = num = passed = passed_exact = 0
    for k, truth in lab.items():
        labels, cf = read_plate(crops[k], chars, glyphs, min_conf=c) if crops[k] is not None else ([], 0.0)
        got = "".join(labels)
        exact += got == truth; num += digits(got) == digits(truth)
        if check_plate(labels, cf)[0]:
            passed += 1; passed_exact += got == truth
    n = len(lab)
    print(f"conf {c:.2f}: exact {exact / n:.1%}  number {num / n:.1%}  pass check {passed / n:.1%}  "
          f"exact among passing {passed_exact / max(1, passed):.1%}  (n={n})", flush=True)
