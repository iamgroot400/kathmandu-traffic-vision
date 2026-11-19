"""Build the Kaggle notebook that inspects every candidate dataset (format, classes, label counts),
so the training notebook's class mapping is written from facts.   python kaggle/make_inspect.py"""
import json
from pathlib import Path

USER = "aashikkhatri"
DATASETS = [
    # vehicles in road scenes
    "redzapdos123/indian-driving-dataset-detections-yolov11",
    "daudshah/vehicle-detection-dataset",
    "kshitizgajurel042/nepali-bike-and-car-images-with-annotations",
    "sdevkota007/vehicles-nepal",
    # plates in scenes (detection)
    "ishworsubedii/vehicle-number-plate-datasetnepal",
    "kshitizgajurel042/images-of-nepali-license-plate",
    "utsavbaral007/nepal-embossed-number-plate-dataset",
    "kritizsapkota/number-plates",
    "deepakat002/indian-vehicle-number-plate-yolo-annotation",
    # plate characters (reading)
    "inspiring-lab/nepali-vehicles-number-plate-dataset",
    "inspiring-lab/nepali-number-plate-characters-dataset",
    "theciceerguy/embossed-number-plate-detection-nepal-dataset",
]

INSPECT = r'''
import os, collections, itertools, json
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
META = {"data.yaml", "dataset.yaml", "classes.txt", "_annotations.coco.json", "obj.names", "notes.json", "readme.md",
        "readme.dataset.txt", "readme.roboflow.txt", "labels.json"}

def find_roots(base="/kaggle/input"):
    """dataset roots: dirs under /kaggle/input that are not just a wrapper around one more dir"""
    base = os.path.join(base, "datasets") if os.path.isdir(os.path.join(base, "datasets")) else base
    return [os.path.join(base, o, d) for o in sorted(os.listdir(base)) for d in sorted(os.listdir(os.path.join(base, o)))]

def inspect(root):
    ext, dirs, metas, yolo_ids, sample_txt, sample_xml, sample_json = collections.Counter(), collections.Counter(), [], collections.Counter(), None, None, None
    n_txt = 0
    for dp, dn, fn in os.walk(root):
        rel = os.path.relpath(dp, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        for f in fn:
            e = os.path.splitext(f)[1].lower()
            ext[e] += 1
            if depth <= 3:
                dirs[rel] += 1
            low = f.lower()
            fp = os.path.join(dp, f)
            if low in META or low.endswith(".yaml"):
                metas.append(fp)
            elif e == ".txt":
                n_txt += 1
                if n_txt <= 4000:
                    try:
                        lines = open(fp, errors="ignore").read().split("\n")
                        for ln in lines:
                            parts = ln.split()
                            if len(parts) >= 5 and parts[0].isdigit():
                                yolo_ids[int(parts[0])] += 1
                        if sample_txt is None and lines and lines[0].strip():
                            sample_txt = (fp, lines[:3])
                    except Exception:
                        pass
            elif e == ".xml" and sample_xml is None:
                sample_xml = (fp, open(fp, errors="ignore").read()[:600])
            elif e == ".json" and sample_json is None and os.path.getsize(fp) < 50_000_000:
                sample_json = (fp, open(fp, errors="ignore").read()[:600])
    print(f"\n{'=' * 100}\n{root}")
    print("files by type:", dict(ext.most_common(8)))
    print("images:", sum(v for k, v in ext.items() if k in IMG))
    print("folders (files directly inside, depth <= 3):")
    for k, v in sorted(dirs.items())[:25]:
        print(f"   {v:>7}  {k}")
    for m in metas[:4]:
        print(f"--- {m}")
        print(open(m, errors="ignore").read()[:1500])
    if yolo_ids:
        print(f"YOLO class ids in first {min(n_txt, 4000)} label files:", dict(sorted(yolo_ids.items())))
    if sample_txt:
        print("sample label", sample_txt[0], sample_txt[1])
    if sample_xml:
        print("sample xml", sample_xml[0], "\n", sample_xml[1])
    if sample_json:
        print("sample json", sample_json[0], "\n", sample_json[1])

for r in find_roots():
    try:
        inspect(r)
    except Exception as exc:
        print(r, "FAILED", exc)
'''


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def main():
    d = Path(__file__).parent / "inspect-datasets"
    d.mkdir(exist_ok=True)
    cells = [md("# Dataset inspection for the Kathmandu traffic vision models\n\nPrints format, folder layout, "
                "class lists and label counts of every candidate dataset, to write the class mapping for training."),
             code("!ls -la /kaggle/input"), code(INSPECT)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (d / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    title = "Kathmandu traffic vision - dataset inspection"
    assert len(title) <= 50
    meta = {"id": f"{USER}/kathmandu-traffic-vision-dataset-inspection", "title": title, "code_file": "notebook.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
            "enable_tpu": False, "enable_internet": False, "dataset_sources": DATASETS,
            "competition_sources": [], "kernel_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {d}")


if __name__ == "__main__":
    main()
