"""Read a number plate: plate crop -> character boxes (chars_best.pt) -> one label per box
(glyphs_best.pt) -> text in reading order. Models come from the train-plate-reader notebook.

    python reader.py plate.jpg [--chars chars_best.pt] [--glyphs glyphs_best.pt]
"""
from __future__ import annotations

import argparse
import re

import cv2

Box = tuple  # (x1, y1, x2, y2, label, conf)


def reading_order(boxes: list[Box]) -> list[Box]:
    """Return the character boxes in the order a person reads the plate.

    Nepali plates come in one row (most car plates) or two rows (most motorcycle plates, and the
    upper line of zone + lot over the number on older red plates). Read the top row left to right,
    then the bottom row left to right.
    """
    # New row where the gap between consecutive vertical centres exceeds half a character height:
    # scales with plate size, and a tilted row's small steady steps never reach it.
    heights = sorted(b[3] - b[1] for b in boxes)
    gap = heights[len(heights) // 2] / 2
    rows, last = [], None
    for b in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        c = (b[1] + b[3]) / 2
        if last is None or c - last > gap:
            rows.append([])
        rows[-1].append(b)
        last = c
    return [b for row in rows for b in sorted(row, key=lambda b: b[0])]


DEV_DIGITS = set("०१२३४५६७८९")
EMB_DIGITS = set("0123456789")
DEV_PLATE = re.compile(r"L+D+L+D{1,4}")   # zone (or province), lot, vehicle class, number: बा २ च १२३४

# shortcut: code lists written from memory of Nepali plate formats; confirm against the Department of
# Transport Management list before relying on the check for enforcement
ZONES = ["मे", "को", "स", "ज", "बा", "ना", "ग", "लु", "ध", "रा", "भे", "क", "से", "म"]   # 14 old zones
PROVINCES = ["को", "म", "बा", "ग", "लु", "क", "सु",                                    # abbreviations
             "कोशी", "मधेश", "बागमती", "गण्डकी", "लुम्बिनी", "कर्णाली", "सुदूरपश्चिम"]    # full names


def valid_prefix(text: str) -> bool:
    """Does a Devanagari reading start with a real zone or province code, followed by the lot digits?

    Accepted: zone + digit (बा२...), "प्रदेश" + digit (प्रदेश३...), "प्र" + digit (प्र३...),
    province + "प्रदेश" (बागमतीप्रदेश..., बाप्रदेश...), province abbreviation + "प्र" + digit (बाप्र०६...).
    """
    starts_digit = lambda rest: rest[:1] in DEV_DIGITS
    for p in ("प्रदेश", "प्र"):
        if text.startswith(p) and starts_digit(text[len(p):]):
            return True
    for p in PROVINCES:
        if text.startswith(p + "प्रदेश") or (text.startswith(p + "प्र") and starts_digit(text[len(p) + len("प्र"):])):
            return True
    return any(text.startswith(z) and starts_digit(text[len(z):]) for z in ZONES)


def check_plate(labels: list[str], conf: float, min_conf: float = 0.6) -> tuple[bool, str]:
    """Is this reading plausible enough to use without a person checking it? (ok, reason)

    Catches structure errors: missed or extra characters, a letter in a number slot, mixed scripts,
    and unsure characters. It cannot catch a look-alike swapped for another of the same kind
    (३ for ६, ना for बा): those only show up as low confidence, if at all.
    """
    if not labels:
        return False, "no characters read"
    if conf < min_conf:
        return False, f"unsure character (confidence {conf:.2f} < {min_conf})"
    dev = [lab for lab in labels if not (lab in EMB_DIGITS or lab.isascii())]
    if dev and len(dev) < len(labels):
        return False, "mixes Devanagari and embossed characters"
    if dev:
        shape = "".join("D" if lab in DEV_DIGITS else "L" for lab in labels)
        if not DEV_PLATE.fullmatch(shape):
            return False, f"not a Nepali plate layout (letters/digits: {shape})"
        if not valid_prefix("".join(labels)):
            return False, f"unknown zone or province code at the start ({''.join(labels)[:4]})"
        return True, "ok"
    # shortcut: embossed plates are only checked for digits and letters both being present;
    # add their exact layout once it is confirmed against real plates
    if not any(lab in EMB_DIGITS for lab in labels) or all(lab in EMB_DIGITS for lab in labels):
        return False, "embossed reading without both letters and digits"
    return True, "ok"


def read_plate(plate, chars, glyphs, min_conf: float = 0.55) -> tuple[list[str], float]:
    """Character labels in reading order, and the lowest confidence (a weak link makes the whole reading weak).

    min_conf 0.55: chosen on the 188 training plates (labelling/eval_reader.py); at the old 0.25 the character
    detector also boxed plate bolts and screws as characters.
    """
    boxes = []
    for x1, y1, x2, y2 in chars.predict(plate, imgsz=416, conf=min_conf, verbose=False)[0].boxes.xyxy.tolist():
        crop = plate[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            continue
        r = glyphs.predict(crop, imgsz=64, verbose=False)[0]
        label = r.names[r.probs.top1]
        if label != "flag":
            boxes.append((x1, y1, x2, y2, label.removeprefix("emb_"), float(r.probs.top1conf)))
    if not boxes:
        return [], 0.0
    ordered = reading_order(boxes)
    return [b[4] for b in ordered], min(b[5] for b in ordered)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image")
    ap.add_argument("--chars", default="chars_best.pt")
    ap.add_argument("--glyphs", default="glyphs_best.pt")
    a = ap.parse_args()
    from ultralytics import YOLO
    labels, conf = read_plate(cv2.imread(a.image), YOLO(a.chars), YOLO(a.glyphs))
    ok, why = check_plate(labels, conf)
    print(f"{''.join(labels)}  (lowest character confidence {conf:.2f})  {'ok' if ok else 'REVIEW: ' + why}")


if __name__ == "__main__":
    main()
