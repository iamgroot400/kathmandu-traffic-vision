"""Build the private Kaggle notebook that cuts plates out of 927 real Nepali vehicle photos
(nishapokharel/nepali-vehicles-number-plate-dataset, CC BY-NC 4.0, no labels) for hand labelling.

The plate detector (train-plates) finds the plates; each crop gets an id. The reader (run-1 character
detector + run-2 classifier, reader.py embedded below) also reads every crop, but its readings go to a
separate file that is NOT shown to the labeller, so the hand labels are not anchored to the model's guess.
The labeller reads the plate on each whole photo (downscaled), so the test does not depend on the
plate detector, which fragments close-up plates. Outputs: photos.zip, photos.csv, model_readings.csv.

    python kaggle/make_labelset.py     then  kaggle kernels push -p .  (from kaggle/labelset)
"""
import json
from pathlib import Path

from make_train import USER, code, md

READER = (Path(__file__).parent.parent / "reader.py").read_text(encoding="utf-8").split("\ndef main()")[0]

CUT = r'''
import csv, glob, zipfile
from pathlib import Path
import cv2
from ultralytics import YOLO
find = lambda name: sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
plates = YOLO(find("plates_best.pt")[0])
chars = YOLO(find("chars_best.pt")[0])
glyphs = YOLO([p for p in find("glyphs_best.pt") if "run-2" in p][0])     # run 2 classifier
photos = sorted(p for p in Path("/kaggle/input").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                and "nepali-vehicles-number-plate-dataset" in str(p))
print(len(photos), "photos")
out = Path("/kaggle/working/photos"); out.mkdir(exist_ok=True)
rows, reads = [], []
for k, photo in enumerate(photos):
    img = cv2.imread(str(photo))
    if img is None:
        continue
    pid = f"P{k + 1:04d}"
    h, w = img.shape[:2]
    s = 1100 / max(h, w)                                  # labelling copy: enough to read the plate
    cv2.imwrite(str(out / f"{pid}.jpg"), cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img,
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    rows.append({"id": pid, "photo": photo.name})
    boxes = plates.predict(img, imgsz=1280, conf=0.25, verbose=False)[0].boxes.data[:, :5].tolist()
    if not boxes:
        reads.append({"id": pid, "plates_found": 0, "reading": "", "min_conf": 0, "check": "no plate found"})
        continue
    # the model's answer for the photo: the most confident plate box, with a margin
    x1, y1, x2, y2, c = max(boxes, key=lambda b: b[4])
    px, py = (x2 - x1) * 0.08, (y2 - y1) * 0.15
    crop = img[int(max(0, y1 - py)):int(min(h, y2 + py)), int(max(0, x1 - px)):int(min(w, x2 + px))]
    labels, cf = read_plate(crop, chars, glyphs)
    ok, why = check_plate(labels, cf)
    reads.append({"id": pid, "plates_found": len(boxes), "reading": "".join(labels), "min_conf": round(cf, 3),
                  "check": "ok" if ok else why})
for name, data in [("photos.csv", rows), ("model_readings.csv", reads)]:
    with open(f"/kaggle/working/{name}", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(data[0]))
        wr.writeheader()
        wr.writerows(data)
with zipfile.ZipFile("/kaggle/working/photos.zip", "w") as z:
    for f in sorted(out.glob("*.jpg")):
        z.write(f, f.name)
import shutil; shutil.rmtree(out)
print(len(rows), "photos;", sum(r["plates_found"] > 1 for r in reads), "with more than one plate box;",
      sum(r["plates_found"] == 0 for r in reads), "with none;", sum(r["check"] == "ok" for r in reads), "readings pass the format check")
'''

TITLE = "Kathmandu traffic vision - plate label set"


def main():
    assert len(TITLE) <= 50, TITLE
    cells = [md(f"# {TITLE}\n\nCuts plates out of real Nepali vehicle photos for hand labelling; the model's "
                "readings are saved separately and not shown to the labeller."),
             code("!pip install -q ultralytics"), code(READER), code(CUT)]
    d = Path(__file__).parent / "labelset"
    d.mkdir(exist_ok=True)
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
          "language": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
    (d / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    slug = TITLE.lower().replace(" - ", "-").replace(" ", "-")
    meta = {"id": f"{USER}/{slug}", "title": TITLE, "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "competition_sources": [],
            "kernel_sources": [f"{USER}/kathmandu-traffic-vision-train-plates",
                               f"{USER}/kathmandu-traffic-vision-train-plate-reader",
                               f"{USER}/kathmandu-traffic-vision-plate-reader-run-2"],
            "dataset_sources": ["nishapokharel/nepali-vehicles-number-plate-dataset"]}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {d}  ({slug})")


if __name__ == "__main__":
    main()
