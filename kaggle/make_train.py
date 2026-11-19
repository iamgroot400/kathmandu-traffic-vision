"""Build the two private Kaggle GPU training notebooks:
  vehicles: motorcycle / car / bus / truck   (IDD + Indian traffic video + Nepali bike-and-car)
  plates:   one class `plate`                  (Kathmandu-valley plates + Nepali + Indian traffic video)
Class mappings come from the dataset inspection (kaggle/inspect-out/report.txt).

    python kaggle/make_train.py            then  kaggle kernels push -p kaggle/train-vehicles (and train-plates)
"""
import json
from pathlib import Path

USER = "aashikkhatri"

PREP = r'''
import os, random, yaml
from pathlib import Path
IN, OUT = Path("/kaggle/input/datasets"), Path("/kaggle/working/data")
VAL_LISTS = {}

def label_of(img):
    parts = list(img.parts)
    if "images" in parts:                                       # YOLO layout: .../images/... -> .../labels/...
        i = len(parts) - 1 - parts[::-1].index("images")
        p = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
        if p.exists():
            return p
    return img.with_suffix(".txt")                              # some sets keep labels next to images

def add(tag, split, images, names, mapping, repeat=1):
    """Link images into OUT/<split>, rewriting labels: keep only source classes in `mapping`
    (source name -> target id); boxes of other classes are dropped, the image stays."""
    d = OUT / split
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    n_box, kept = 0, []
    for img in images:
        rows, lab = [], label_of(img)
        if lab.exists():
            for ln in lab.read_text(errors="ignore").splitlines():
                parts = ln.split()
                if len(parts) != 5 or not parts[0].isdigit():      # skip polygons / malformed rows
                    continue
                k = int(parts[0])
                if k < len(names) and names[k] in mapping:
                    rows.append(" ".join([str(mapping[names[k]])] + parts[1:]))
        for r in range(repeat):
            stem = f"{tag}_{r}_{img.parent.name}_{img.stem}".replace(" ", "_")
            link = d / "images" / (stem + img.suffix.lower())
            if not link.exists():
                link.symlink_to(img)
            (d / "labels" / (stem + ".txt")).write_text("\n".join(rows))
            kept.append(str(link))
        n_box += len(rows)
    if split == "val":
        VAL_LISTS[tag] = kept
    print(f"{tag:<10} {split:<5} {len(images):>6} images x{repeat}  {n_box:>7} boxes kept")

def imgs(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

def split_random(images, val_share=0.15, seed=0):
    images = images[:]
    random.Random(seed).shuffle(images)
    n = int(len(images) * val_share)
    return images[n:], images[:n]

def write_yaml(names):
    (OUT / "data.yaml").write_text(yaml.safe_dump({"path": str(OUT), "train": "train/images", "val": "val/images",
                                                 "names": dict(enumerate(names))}))
    for tag, files in VAL_LISTS.items():          # per-source validation sets, to see where the model is weak
        (OUT / f"val_{tag}.txt").write_text("\n".join(files))
        (OUT / f"data_{tag}.yaml").write_text(yaml.safe_dump({"path": str(OUT), "train": "train/images",
                                                            "val": f"val_{tag}.txt", "names": dict(enumerate(names))}))
'''

VEHICLES = r'''
NAMES = ["motorcycle", "car", "bus", "truck"]
M, C, B, T = range(4)

idd = IN / "redzapdos123/indian-driving-dataset-detections-yolov11/IDDDetectionsYOLODataset"
idd_names = yaml.safe_load((idd / "data.yaml").read_text())["names"]
idd_map = {"motorcycle": M, "car": C, "caravan": C, "bus": B, "truck": T, "trailer": T}
sample = lambda xs, n: random.Random(0).sample(xs, min(n, len(xs)))
add("idd", "train", sample(imgs(idd / "train/images"), 12000), idd_names, idd_map)
add("idd", "val", sample(imgs(idd / "val/images"), 1500), idd_names, idd_map)

dau = IN / "daudshah/vehicle-detection-dataset"
dau_names = yaml.safe_load((dau / "data.yaml").read_text())["names"]
dau_map = {"motorbike": M, "car": C, "van": C, "bus": B, "truck": T}
add("india_vid", "train", imgs(dau / "train/images"), dau_names, dau_map)
add("india_vid", "val", imgs(dau / "valid/images"), dau_names, dau_map)

nep = IN / "kshitizgajurel042/nepali-bike-and-car-images-with-annotations/Final Data"
nep_names = ["car", "license_plate", "bike"]
nep_map = {"bike": M, "car": C}
add("nepal", "train", imgs(nep / "images/train"), nep_names, nep_map, repeat=3)   # small but local: weighted x3
add("nepal", "val", imgs(nep / "images/val"), nep_names, nep_map)
write_yaml(NAMES)
'''

PLATES = r'''
NAMES = ["plate"]

ish = IN / "ishworsubedii/vehicle-number-plate-datasetnepal/vehicle_number_plate_detection"
tr, va = split_random(imgs(ish / "images"))
add("kathmandu", "train", tr, ["plate"], {"plate": 0})
add("kathmandu", "val", va, ["plate"], {"plate": 0})

nep = IN / "kshitizgajurel042/nepali-bike-and-car-images-with-annotations/Final Data"
nep_names = ["car", "license_plate", "bike"]
add("nepal", "train", imgs(nep / "images/train"), nep_names, {"license_plate": 0}, repeat=2)
add("nepal", "val", imgs(nep / "images/val"), nep_names, {"license_plate": 0})

dee = IN / "deepakat002/indian-vehicle-number-plate-yolo-annotation"
tr, va = split_random(imgs(dee))
add("far_plates", "train", tr, ["number_plate"], {"number_plate": 0}, repeat=5)   # distant plates, like a junction camera
add("far_plates", "val", va, ["number_plate"], {"number_plate": 0})
write_yaml(NAMES)
'''

TRAIN = r'''
import glob, shutil
from ultralytics import YOLO
model = YOLO("{weights}")
model.train(data=str(OUT / "data.yaml"), epochs={epochs}, time={hours}, imgsz={imgsz}, batch={batch}, patience=12,
            project="/kaggle/working/runs", name="{name}", workers=4, seed=0, plots=True)
best = sorted(glob.glob("/kaggle/working/runs/{name}*/weights/best.pt"))[-1]
shutil.copy(best, "/kaggle/working/{name}_best.pt")
print("saved /kaggle/working/{name}_best.pt")
'''

EVAL = r'''
import pandas as pd
m = YOLO("/kaggle/working/{name}_best.pt")
rows = []
for tag in VAL_LISTS:
    r = m.val(data=str(OUT / f"data_{{tag}}.yaml"), imgsz={imgsz}, verbose=False, plots=False)
    row = {{"source": tag, "images": len(VAL_LISTS[tag]), "mAP50": round(r.box.map50, 3), "mAP50-95": round(r.box.map, 3)}}
    for i, n in enumerate(NAMES):
        if i < len(r.box.ap50):
            row[f"AP50 {{n}}"] = round(float(r.box.ap50[i]), 3)
    rows.append(row)
res = pd.DataFrame(rows)
res.to_csv("/kaggle/working/{name}_per_source.csv", index=False)
print(res.to_string(index=False))
shutil.rmtree(OUT)    # links only; keep the output small
'''

NOTEBOOKS = {
    "train-vehicles": dict(
        title="Kathmandu traffic vision - train vehicles", name="vehicles", weights="yolo11s.pt",
        epochs=60, hours=5.0, imgsz=640, batch=32, data=VEHICLES,
        intro="Fine-tunes YOLO11s (COCO-pretrained) to 4 classes for Kathmandu traffic: **motorcycle, car "
              "(incl. vans, jeeps, microbuses), bus, truck**. Data: Indian Driving Dataset (12,000-image sample; "
              "research use), Indian traffic-video frames (CC BY 4.0), and Nepali bike-and-car photos (weighted x3). "
              "Auto-rickshaws, bicycles, people and other classes are dropped. Validation is reported per source; "
              "none of these is Kathmandu junction footage, which is still needed for a real test."),
    "train-plates": dict(
        title="Kathmandu traffic vision - train plates", name="plates", weights="yolo11s.pt",
        epochs=60, hours=3.0, imgsz=640, batch=32, data=PLATES,
        intro="Fine-tunes YOLO11s to find **number plates** (one class, all Nepali formats). Data: "
              "Kathmandu-valley plate images, Nepali bike-and-car photos, and distant plates from Indian "
              "traffic video (weighted x5, closest to a junction camera). Reading the characters is a later step."),
}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def main():
    for key, c in NOTEBOOKS.items():
        assert len(c["title"]) <= 50, c["title"]
        fmt = {k: c[k] for k in ("name", "weights", "epochs", "hours", "imgsz", "batch")}
        cells = [md(f"# {c['title']}\n\n{c['intro']}"),
                 code("!pip install -q ultralytics\n!nvidia-smi --query-gpu=name,memory.total --format=csv"),
                 code(PREP), code(c["data"]), code(TRAIN.format(**fmt)), code(EVAL.format(**fmt))]
        d = Path(__file__).parent / key
        d.mkdir(exist_ok=True)
        nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
              "language": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
        (d / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
        slug = c["title"].lower().replace(" - ", "-").replace(" ", "-")
        meta = {"id": f"{USER}/{slug}", "title": c["title"], "code_file": "notebook.ipynb", "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "enable_internet": True, "competition_sources": [], "kernel_sources": [],
                "dataset_sources": ["redzapdos123/indian-driving-dataset-detections-yolov11",
                                    "daudshah/vehicle-detection-dataset",
                                    "kshitizgajurel042/nepali-bike-and-car-images-with-annotations",
                                    "ishworsubedii/vehicle-number-plate-datasetnepal",
                                    "deepakat002/indian-vehicle-number-plate-yolo-annotation"]}
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"wrote {d}  ({slug})")


if __name__ == "__main__":
    main()
