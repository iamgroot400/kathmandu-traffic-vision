#!/usr/bin/env python3
"""Count vehicles crossing a stop line and flag red-light runners, from a video file or camera.

    python watch.py --source junction.mp4 --pick          # once: click the stop line, box the light
    python watch.py --source junction.mp4 --save-video    # run

config.json:
    stop_line  [[x1, y1], [x2, y2]]  pixel coordinates
    forward    1 or -1               crossing direction that enters the junction (--pick asks)
    light_roi  [x, y, w, h] or null  box around the signal head
    cycle      {"red": s, "green": s, "amber": s, "offset": s} or null, used when light_roi is null
Writes out/counts.csv (vehicles and PCU per minute, class, direction), out/violations.csv and
out/snapshots/<time>_<id>.jpg (evidence frame with the vehicle boxed), <time>_<id>_vehicle.jpg (close-up of
the vehicle and its rider or driver), <time>_<id>_before1..5.jpg (vehicle close-ups from the seconds before the
violation, oldest first) and <time>_<id>_before1..5_scene.jpg (the whole uncropped frame at those same moments), <time>_<id>_card.jpg (frame, close-ups and the facts on one image) and
out/report.html (every violation with its card, for a person to review).
With --plates, each violation also gets the number plate read (plate), "ok" or "review: why" (plate_check)
and a close-up photo of the plate (plate_snapshot, out/snapshots/<time>_<id>_plate.jpg).
Plate numbers are personal data: out/ stays off git; use them only for flagged violations.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from light import cycle_colour, light_colour
from logic import PCU, Monitor
from reader import read_vehicle_best

COLOUR = {"red": (0, 0, 255), "amber": (0, 200, 255), "green": (0, 200, 0), "unknown": (160, 160, 160)}


def vehicle_crop(frame, box, pad: float = 0.15):
    """The vehicle with a margin, so a rider's head above the box and the plate below it are kept."""
    x1, y1, x2, y2 = box[:4]
    h, w = frame.shape[:2]
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return frame[int(max(0, y1 - py)):int(min(h, y2 + py)), int(max(0, x1 - px)):int(min(w, x2 + px))].copy()


def fit(img, w, h):
    """img scaled to fit inside w x h, on a dark background of exactly that size."""
    out = np.full((h, w, 3), 30, np.uint8)
    if img is None or img.size == 0:
        return out
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def evidence_card(frame, vehicle, plate, lines, before=()):
    """One image per violation: annotated frame on the left, vehicle and plate close-ups on the right, a strip of the
    close-ups taken before the violation (oldest first, at most 5), facts below.
    OpenCV cannot draw Devanagari, so the plate text is in violations.csv and report.html, not on the card."""
    rows = [np.hstack([fit(frame, 720, 480), np.vstack([fit(vehicle, 360, 320), fit(plate, 360, 160)])])]
    if before:
        rows.append(np.hstack([fit(b, 216, 200) for b in (list(before[-5:]) + [None] * 5)[:5]]))
    top = sum(r.shape[0] for r in rows)
    card = np.vstack(rows + [np.full((40 * len(lines) + 10, 1080, 3), 255, np.uint8)])
    for i, t in enumerate(lines):
        cv2.putText(card, t, (12, top + 35 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return card


def write_report(path: Path, violations) -> None:
    """A page for a person to review the flags: card, time, class, plate and whether the reading needs checking."""
    rows = "".join(
        f"<tr><td><a href='{html.escape(Path(v['card']).relative_to(path.parent).as_posix())}'>"
        f"<img src='{html.escape(Path(v['card']).relative_to(path.parent).as_posix())}' width='540'></a></td>"
        f"<td>{v['time_s']} s</td><td>{html.escape(v['class'])}</td><td>{v['red_for_s']} s</td>"
        f"<td>{html.escape(v.get('plate', '') or '-')}</td><td>{html.escape(v.get('plate_check', 'not read'))}</td></tr>"
        for v in violations)
    path.write_text("<!doctype html><meta charset='utf-8'><title>Red-light violations</title>"
                    "<style>body{font-family:sans-serif;margin:16px}td{padding:6px;border-bottom:1px solid #ccc}</style>"
                    f"<h1>Red-light violations ({len(violations)})</h1>"
                    "<p>Flags for a person to review, not automatic fines. Plates marked <i>review</i> need checking "
                    "against the close-up. Personal data: keep this folder private.</p>"
                    "<table><tr><th>Evidence</th><th>Time</th><th>Vehicle</th><th>Red for</th><th>Plate</th>"
                    f"<th>Plate check</th></tr>{rows}</table>", encoding="utf-8")


def pick(source: str, cfg_path: Path) -> None:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"cannot read {source}")
    pts = []
    win = "click the 2 ends of the stop line, then any key"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lambda e, x, y, *_: pts.append([x, y]) if e == cv2.EVENT_LBUTTONDOWN and len(pts) < 2 else None)
    while True:
        view = frame.copy()
        for p in pts:
            cv2.circle(view, tuple(p), 5, (0, 0, 255), -1)
        if len(pts) == 2:
            cv2.line(view, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), 2)
        cv2.imshow(win, view)
        if cv2.waitKey(30) != -1 and len(pts) == 2:
            break
    cv2.destroyWindow(win)
    roi = cv2.selectROI("drag a box around the signal head, Enter (Esc / empty = no light in view)", frame)
    cv2.destroyAllWindows()
    fwd = input("Do vehicles entering the junction move left-to-right / top-to-bottom across the line "
                "as you clicked it? Run once and check out/counts.csv 'in' vs 'out'; answer 1 or -1 [1]: ").strip()
    cfg = {"stop_line": pts, "forward": int(fwd or 1), "light_roi": list(roi) if roi[2] and roi[3] else None,
           "cycle": None}
    cfg_path.write_text(json.dumps(cfg, indent=1))
    print(f"wrote {cfg_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True, help="video file, stream URL, or camera index")
    ap.add_argument("--config", type=Path, default=Path("config.json"))
    ap.add_argument("--model", default="yolo11n.pt", help="any Ultralytics detector, e.g. results/vehicles_run1/vehicles_best.pt")
    ap.add_argument("--before", type=int, default=5, help="close-ups kept from before each violation (0 = none)")
    ap.add_argument("--before-gap", type=float, default=0.2, help="seconds between those close-ups")
    ap.add_argument("--grace", type=float, default=1.0, help="seconds after red begins before crossings count")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--pick", action="store_true", help="set the stop line and light box, then exit")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--plates", help="plate detector weights (plates_best.pt); turns on plate reading")
    ap.add_argument("--chars", default="results/reader_run4/chars_best.pt")
    ap.add_argument("--glyphs", default="results/reader_run3/glyphs_best.pt")
    a = ap.parse_args()
    if a.pick:
        return pick(a.source, a.config)

    from ultralytics import YOLO
    cfg = json.loads(a.config.read_text())
    line = tuple(tuple(p) for p in cfg["stop_line"])
    mon = Monitor(stop_line=line, forward=cfg.get("forward", 1), grace=a.grace)
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    (a.out / "snapshots").mkdir(parents=True, exist_ok=True)
    writer = None
    history, last_snap = deque(maxlen=a.before or 1), -1e9   # recent clean frames and their boxes, for the "before" close-ups
    readers = (YOLO(a.plates), YOLO(a.chars), YOLO(a.glyphs)) if a.plates else None

    det = YOLO(a.model)
    vehicles = {i: n for i, n in det.names.items() if n in PCU}   # class ids differ between COCO and our own model
    results = det.track(a.source, stream=True, persist=True, tracker="bytetrack.yaml",
                        classes=list(vehicles), conf=a.conf, verbose=False)
    for i, r in enumerate(results):
        frame, t = r.orig_img, i / fps
        if cfg.get("light_roi"):
            seen = light_colour(frame, cfg["light_roi"])
        elif cfg.get("cycle"):
            seen = cycle_colour(t, **cfg["cycle"])
        else:
            seen = "unknown"
        tracks, boxes = [], {}
        if r.boxes.id is not None:
            for (x1, y1, x2, y2), tid, c in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(), r.boxes.cls.int().tolist()):
                tracks.append((tid, vehicles[c], ((x1 + x2) / 2, y2)))   # bottom centre: where the vehicle meets the road
                boxes[tid] = (int(x1), int(y1), int(x2), int(y2), vehicles[c])
        new = mon.update(t, seen, tracks)
        clean = frame.copy() if new else None   # close-ups are cut before the boxes are drawn on
        due = a.before > 0 and t - last_snap >= a.before_gap
        pre = (clean if clean is not None else frame.copy()) if due else None

        light = mon.signal.state
        cv2.line(frame, line[0], line[1], COLOUR[light], 3)
        cv2.putText(frame, f"{light}  t={t:6.1f}s  vehicles={sum(mon.counts.values())}  violations={len(mon.violations)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOUR[light], 2)
        flagged = {v["track_id"] for v in mon.violations}
        for tid, (x1, y1, x2, y2, cls) in boxes.items():
            col = (0, 0, 255) if tid in flagged else (255, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, f"{cls} {tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        for v in new:
            stem = a.out / "snapshots" / f"{v['time_s']:08.2f}_{v['track_id']}"
            box = boxes[v["track_id"]]
            # the same vehicle (same track id) in the frames kept from before this one; frames it was not in are skipped
            raw_before = [(f, bx[v["track_id"]]) for _, f, bx in history if v["track_id"] in bx]
            crop = None
            if readers:
                # read the crossing frame and the close-ups from just before it, keep the most trustworthy reading
                v["plate"], v["plate_check"], crop = read_vehicle_best([(clean, box), *raw_before], *readers)
                v["plate_snapshot"] = ""
                if crop is not None:
                    v["plate_snapshot"] = f"{stem}_plate.jpg"
                    cv2.imwrite(v["plate_snapshot"], crop)
            v["snapshot"], v["vehicle_snapshot"], v["card"] = f"{stem}.jpg", f"{stem}_vehicle.jpg", f"{stem}_card.jpg"
            cv2.imwrite(v["snapshot"], frame)
            vehicle = vehicle_crop(clean, box)
            cv2.imwrite(v["vehicle_snapshot"], vehicle)
            before = [vehicle_crop(f, bx) for f, bx in raw_before]
            v["before_snapshots"] = ";".join(f"{stem}_before{k + 1}.jpg" for k in range(len(before)))
            v["before_scenes"] = ";".join(f"{stem}_before{k + 1}_scene.jpg" for k in range(len(raw_before)))
            for k, img in enumerate(before):
                cv2.imwrite(f"{stem}_before{k + 1}.jpg", img)                 # vehicle close-up
                cv2.imwrite(f"{stem}_before{k + 1}_scene.jpg", raw_before[k][0])  # whole uncropped frame, same moment
            facts = [f"RED-LIGHT VIOLATION  t={v['time_s']}s  {v['class']} #{v['track_id']}  red for {v['red_for_s']}s"]
            if readers:
                facts.append(f"plate: {v['plate_check']}"[:70])   # plate text itself: violations.csv, report.html
            if before:
                facts.append(f"strip: {len(before)} close-ups from the {len(before) * a.before_gap:.1f} s before, oldest first")
            cv2.imwrite(v["card"], evidence_card(frame, vehicle, crop, facts, before))
            print(f"VIOLATION t={v['time_s']}s {v['class']} #{v['track_id']} red for {v['red_for_s']}s"
                  + (f"  plate: {v['plate_check']}" if readers else ""))   # plate text stays out of the log
        if pre is not None:
            history.append((t, pre, dict(boxes)))
            last_snap = t
        if a.save_video:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(a.out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
        if a.show:
            cv2.imshow("traffic-watch", frame)
            if cv2.waitKey(1) == 27:
                break
    if writer:
        writer.release()

    with open(a.out / "counts.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["minute", "class", "direction", "vehicles", "pcu"])
        for (m, cls, d), n in sorted(mon.counts.items()):
            w.writerow([m, cls, d, n, n * PCU[cls]])
    with open(a.out / "violations.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["time_s", "track_id", "class", "light", "red_for_s", "snapshot", "vehicle_snapshot", "before_snapshots", "before_scenes", "card"]
                           + (["plate", "plate_check", "plate_snapshot"] if readers else []))
        w.writeheader()
        w.writerows(mon.violations)
    write_report(a.out / "report.html", mon.violations)
    print(f"{sum(mon.counts.values())} crossings, {len(mon.violations)} violations -> {a.out}")


if __name__ == "__main__":
    main()
