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
out/snapshots/<time>_<id>.jpg (evidence frame with the vehicle boxed).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from light import cycle_colour, light_colour
from logic import PCU, Monitor

VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}   # COCO class ids
COLOUR = {"red": (0, 0, 255), "amber": (0, 200, 255), "green": (0, 200, 0), "unknown": (160, 160, 160)}


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
    ap.add_argument("--model", default="yolo11n.pt", help="any Ultralytics detector, e.g. yolo12n.pt")
    ap.add_argument("--grace", type=float, default=1.0, help="seconds after red begins before crossings count")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--pick", action="store_true", help="set the stop line and light box, then exit")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--show", action="store_true")
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

    results = YOLO(a.model).track(a.source, stream=True, persist=True, tracker="bytetrack.yaml",
                                  classes=list(VEHICLES), conf=a.conf, verbose=False)
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
                tracks.append((tid, VEHICLES[c], ((x1 + x2) / 2, y2)))   # bottom centre: where the vehicle meets the road
                boxes[tid] = (int(x1), int(y1), int(x2), int(y2), VEHICLES[c])
        new = mon.update(t, seen, tracks)

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
            v["snapshot"] = str(a.out / "snapshots" / f"{v['time_s']:08.2f}_{v['track_id']}.jpg")
            cv2.imwrite(v["snapshot"], frame)
            print(f"VIOLATION t={v['time_s']}s {v['class']} #{v['track_id']} red for {v['red_for_s']}s")
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
    with open(a.out / "violations.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["time_s", "track_id", "class", "light", "red_for_s", "snapshot"])
        w.writeheader()
        w.writerows(mon.violations)
    print(f"{sum(mon.counts.values())} crossings, {len(mon.violations)} violations -> {a.out}")


if __name__ == "__main__":
    main()
