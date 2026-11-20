"""Build the private Kaggle GPU notebook for the plate reader (two models):
  chars:  YOLO11s detector, one class `character`, on cropped Nepali plates
          (inspiring-lab/nepali-vehicles-number-plate-dataset, 2,503 plates)
  glyphs: YOLO11s-cls classifier, one class per character, on single-character crops
          (inspiring-lab Devanagari characters + theciceerguy embossed digits and letters)
reader.py chains them: plate crop -> character boxes -> one label per box -> text.
No dataset has whole-plate text, so only the two stages are scored, not full plates.

    python kaggle/make_reader.py      then  kaggle kernels push -p .  (from kaggle/train-reader)
"""
import json
from pathlib import Path

from make_labelset import READER
from make_train import PREP, USER, code, md

CHARS = r'''
cha = IN / "inspiring-lab/nepali-vehicles-number-plate-dataset/open_source_nepali_plates"
tr, va = split_random(imgs(cha / "images"))
add("plates_dev", "train", tr, ["character"], {"character": 0})
add("plates_dev", "val", va, ["character"], {"character": 0})
write_yaml(["character"])

from ultralytics import YOLO
import glob, shutil
det = YOLO("yolo11s.pt")
det.train(data=str(OUT / "data.yaml"), epochs=80, time=1.5, imgsz=416, batch=32, patience=15,
          project="/kaggle/working/runs", name="chars", workers=4, seed=0, plots=True)
shutil.copy(sorted(glob.glob("/kaggle/working/runs/chars*/weights/best.pt"))[-1], "/kaggle/working/chars_best.pt")
r = YOLO("/kaggle/working/chars_best.pt").val(data=str(OUT / "data.yaml"), imgsz=416, verbose=False, plots=False)
print(f"character detector: mAP50 {r.box.map50:.3f}  mAP50-95 {r.box.map:.3f}  "
      f"precision {r.box.mp:.3f}  recall {r.box.mr:.3f}  ({len(va)} validation plates)")
shutil.rmtree(OUT)
'''

LINK = r'''
import random, shutil, glob
from collections import Counter
from pathlib import Path
from ultralytics import YOLO
CLS = Path("/kaggle/working/glyphs")
VAL = {}     # source -> [(image link, class)]

def link_all(tag, pairs, split):
    for i, (img, c) in enumerate(pairs):
        d = CLS / split / c
        d.mkdir(parents=True, exist_ok=True)
        dst = d / f"{tag}_{i}{img.suffix.lower()}"
        if not dst.exists():
            dst.symlink_to(img)
        if split == "val":
            VAL.setdefault(tag, []).append((dst, c))
    print(f"{tag:<10} {split:<5} {len(pairs):>6} crops, {len(set(c for _, c in pairs))} classes")

# Devanagari: one folder per character, no split given -> random 15% validation
dev = IN / "inspiring-lab/nepali-number-plate-characters-dataset/character_ocr"
pairs = [(p, p.parent.name) for p in imgs(dev)]
random.Random(0).shuffle(pairs)
n = int(len(pairs) * 0.15)
link_all("devanagari", pairs[n:], "train")
link_all("devanagari", pairs[:n], "val")

# Embossed: class folders under train/ (2,670), valid/ (349) and test/ (563); the flat augmented/ folder
# (6,566 files, no class in path or name) is skipped. utsavbaral007's set has the same test folders: a copy, not used.
emb = IN / "theciceerguy/embossed-number-plate-detection-nepal-dataset/digits_and_numbers_dataset"
name = lambda p: "flag" if p.parent.name == "Nepali Flag" else "emb_" + p.parent.name
link_all("embossed", [(p, name(p)) for p in imgs(emb / "train")], "train")
link_all("embossed", [(p, name(p)) for p in imgs(emb / "valid") + imgs(emb / "test")], "val")

'''

TRAIN_GLYPHS = r'''
cls = YOLO("yolo11s-cls.pt")
cls.train(data=str(CLS), epochs=40, time=1.5, imgsz=64, batch=256, patience=10,
          project="/kaggle/working/runs", name="glyphs", workers=4, seed=0, plots=True)
shutil.copy(sorted(glob.glob("/kaggle/working/runs/glyphs*/weights/best.pt"))[-1], "/kaggle/working/glyphs_best.pt")

m = YOLO("/kaggle/working/glyphs_best.pt")
rows = []
for tag, items in VAL.items():
    wrong = Counter()
    ok = 0
    for i in range(0, len(items), 512):
        batch = items[i:i + 512]
        for (img, c), r in zip(batch, m.predict([str(p) for p, _ in batch], imgsz=64, verbose=False)):
            got = r.names[r.probs.top1]
            ok += got == c
            if got != c:
                wrong[f"{c} -> {got}"] += 1
    rows.append({"source": tag, "crops": len(items), "top1": round(ok / len(items), 4),
                 "most_confused": "; ".join(f"{k} ({v})" for k, v in wrong.most_common(5))})
import pandas as pd
res = pd.DataFrame(rows)
res.to_csv("/kaggle/working/glyphs_per_source.csv", index=False)
print(res.to_string(index=False))
shutil.rmtree(CLS)
'''

GLYPHS = LINK + TRAIN_GLYPHS

# Run 2: two more Devanagari character sets with romanised folder names. Before they are used, the run-1
# classifier (never trained on them) labels every crop: that is (a) an honest test on unseen sources and
# (b) a check of the name mapping; a folder whose majority run-1 label disagrees with the mapping is dropped.
NEW_SOURCES = r'''
from collections import Counter
ROMAN = {**{str(i): "०१२३४५६७८९"[i] for i in range(10)},
         "ba": "बा", "pa": "प", "cha": "च", "kha": "ख", "ko": "को", "me": "मे", "pra": "प्र"}
NEW = {"ishwor_chars": IN / "ishworsubedii/nepalese-license-plate-character-dataset/dataset",
       "kshitiz_chars": IN / "kshitizgajurel042/individual-character-images-of-nepali-lp/NepaliLPCharacters"}
run1 = YOLO(sorted(glob.glob("/kaggle/input/**/glyphs_best.pt", recursive=True))[0])
unseen = []
for tag, root in NEW.items():
    ok_n = all_n = 0
    pairs = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        files = imgs(folder)
        want = ROMAN.get(folder.name)
        got = Counter()
        for i in range(0, len(files), 512):
            for r in run1.predict([str(f) for f in files[i:i + 512]], imgsz=64, verbose=False):
                got[r.names[r.probs.top1]] += 1
        top, n = got.most_common(1)[0]
        hit = got.get(want, 0)
        print(f"{tag:<14} {folder.name:<4} -> {want}  {len(files):>4} crops, run-1 says {top} for {n}, {hit / len(files):.1%} agree")
        if want is not None:            # every mapped folder counts, kept or dropped, so the score is not flattered
            ok_n += hit
            all_n += len(files)
        if want is None or top != want:
            print(f"   dropped: mapping not confirmed")
            continue
        pairs += [(f, want) for f in files]
    unseen.append({"source": tag, "crops": all_n, "run1_top1_unseen": round(ok_n / all_n, 4)})
    random.Random(0).shuffle(pairs)
    n = int(len(pairs) * 0.15)
    link_all(tag, pairs[n:], "train")
    link_all(tag, pairs[:n], "val")
import pandas as pd
pd.DataFrame(unseen).to_csv("/kaggle/working/run1_on_unseen_sources.csv", index=False)
print(pd.DataFrame(unseen).to_string(index=False))
'''

GLYPHS2 = LINK + NEW_SOURCES + TRAIN_GLYPHS

# Run 3: real Kathmandu plates. Plate text read by hand-checked AI (labels_claude_train.txt, photos P0151-P0400,
# none of them the 150 test photos or a repeat of a test vehicle) is split into characters and matched to the
# character detector's boxes. A plate is used only if the counts match and the run-2 classifier agrees on at
# least half of the characters it knows (guards against a shifted match).
REAL = r"""
import zipfile, re, cv2, numpy as np
find = lambda name: sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
plates_m = YOLO(find("plates_best.pt")[0])
chars_m = YOLO(find("chars_best.pt")[0])
run2 = YOLO([p for p in find("glyphs_best.pt") if "run-2" in p][0])
PH = Path("/kaggle/working/photos"); zipfile.ZipFile(find("photos.zip")[0]).extractall(PH)
read_tsv = lambda f: {l.split("\t")[0]: l.split("\t")[1] for l in open(f, encoding="utf-8").read().splitlines() if "\t" in l}
LAB = read_tsv(find("labels_claude_train.txt")[0])
MARKS = {chr(c) for c in [*range(0x0900, 0x0904), *range(0x093A, 0x094D), 0x094E, 0x094F, 0x0962, 0x0963]}

def units(text):
    # one unit per written character: consonant (+ virama + consonant)* + vowel signs, e.g. प्रदेश -> प्र, दे, श
    out = []
    for ch in text:
        if out and (ch in MARKS or ch == "्" or out[-1].endswith("्")):
            out[-1] += ch
        else:
            out.append(ch)
    return out
assert units("बागमतीप्रदेश०६") == ["बा", "ग", "म", "ती", "प्र", "दे", "श", "०", "६"]

def plate_crop(img):
    b = plates_m.predict(img, imgsz=1280, conf=0.2, verbose=False)[0].boxes.xyxy.tolist()
    if not b:
        return None
    h, w = img.shape[:2]   # union: the detector splits close-up plates into pieces
    x1, y1, x2, y2 = min(v[0] for v in b), min(v[1] for v in b), max(v[2] for v in b), max(v[3] for v in b)
    px, py = (x2 - x1) * 0.08, (y2 - y1) * 0.15
    return img[int(max(0, y1 - py)):int(min(h, y2 + py)), int(max(0, x1 - px)):int(min(w, x2 + px))]

OUTR = Path("/kaggle/working/real"); OUTR.mkdir(exist_ok=True)
stats, by_photo = Counter(), {}
known = set(run2.names.values())
for pid, text in LAB.items():
    crop = plate_crop(cv2.imread(str(PH / f"{pid}.jpg")))
    if crop is None:
        stats["no plate found"] += 1; continue
    boxes = [(x1, y1, x2, y2, "", 1.0) for x1, y1, x2, y2 in
             chars_m.predict(crop, imgsz=416, conf=0.25, verbose=False)[0].boxes.xyxy.tolist()]
    u = units(text)
    if len(boxes) != len(u):
        stats["character count differs"] += 1; continue
    boxes = reading_order(boxes)
    cs = [crop[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in boxes]
    if any(c.size == 0 for c in cs):
        stats["empty crop"] += 1; continue
    pred = [r.names[r.probs.top1].removeprefix("emb_") for r in run2.predict(cs, imgsz=64, verbose=False)]
    k = [(p, x) for p, x in zip(pred, u) if x in known]
    if k and sum(p == x for p, x in k) < 0.5 * len(k):
        stats["looks shifted"] += 1; continue
    files = []
    for j, (c, x) in enumerate(zip(cs, u)):
        f = OUTR / f"{pid}_{j}.jpg"; cv2.imwrite(str(f), c); files.append((f, x))
    by_photo[pid] = files; stats["used"] += 1
print("real plates:", dict(stats))
ids = sorted(by_photo); random.Random(0).shuffle(ids); nv = max(1, len(ids) * 15 // 100)
tr = [p for k in ids[nv:] for p in by_photo[k]]           # split by photo: no plate on both sides
va = [p for k in ids[:nv] for p in by_photo[k]]
link_all("kathmandu_real", tr * 3, "train")                 # few but closest to real use: weighted x3
link_all("kathmandu_real", va, "val")
print("new classes:", sorted({x for _, x in tr + va} - known))
"""

END2END = r"""
# End-to-end on the test photos (never trained on). Labels are PROVISIONAL: the 88 plates where Gemini and Claude
# read the same text; the owner's hand labels replace them later.
import pandas as pd
TEST = read_tsv(find("test_provisional.tsv")[0])
run3 = YOLO("/kaggle/working/glyphs_best.pt")
digits = lambda s: (re.findall(r"[०-९]+$", s) or [""])[0]
rows = []
for pid, truth in sorted(TEST.items()):
    crop = plate_crop(cv2.imread(str(PH / f"{pid}.jpg")))
    r = {"id": pid, "truth": truth}
    for name, g in [("run2", run2), ("run3", run3)]:
        labels, cf = read_plate(crop, chars_m, g) if crop is not None else ([], 0.0)
        r[name] = "".join(labels); r[name + "_check"] = check_plate(labels, cf)[1]
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv("/kaggle/working/end2end_provisional.csv", index=False)
for name in ("run2", "run3"):
    exact = (df[name] == df.truth).mean(); num = (df[name].map(digits) == df.truth.map(digits)).mean()
    ok = df[name + "_check"] == "ok"; okexact = (df[name] == df.truth)[ok].mean() if ok.any() else float("nan")
    print(f"{name}: whole plate exact {exact:.1%}, number part exact {num:.1%}, passes check {ok.mean():.1%}, "
          f"exact among those passing {okexact:.1%}  (n={len(df)} provisional)")
"""
SELECT_RUN1 = ('run1 = YOLO(sorted(glob.glob("/kaggle/input/**/glyphs_best.pt", recursive=True))[0])',
               'run1 = YOLO([p for p in sorted(glob.glob("/kaggle/input/**/glyphs_best.pt", recursive=True)) if "run-2" not in p][0])')
assert SELECT_RUN1[0] in NEW_SOURCES
GLYPHS3 = LINK + NEW_SOURCES.replace(*SELECT_RUN1) + REAL + TRAIN_GLYPHS


# Run 4: the character DETECTOR on real Kathmandu plates. Boxes come from matching each AI-read plate text
# (labels_claude_train.txt, training photos only) to the run-1 detector's candidates at a low threshold: an
# in-order dynamic-programming match keeps one box per written character, scored by the run-3 classifier.
# Unmatched candidates (bolts, screws, dirt) get no label, so they train as background. Missed province words can
# be recovered when a low-confidence candidate exists for them.
CHARS4 = r"""
import zipfile, re, cv2, numpy as np, pandas as pd, shutil, glob, yaml
from collections import Counter
from ultralytics import YOLO
find = lambda name: sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
plates_m = YOLO(find("plates_best.pt")[0])
old = find("chars_best.pt")[0]; chars_old = YOLO(old)
glyphs = YOLO([p for p in find("glyphs_best.pt") if "run-3" in p][0])
PH = Path("/kaggle/working/photos"); zipfile.ZipFile(find("photos.zip")[0]).extractall(PH)
read_tsv = lambda f: {l.split("\t")[0]: l.split("\t")[1] for l in open(f, encoding="utf-8").read().splitlines() if "\t" in l}
LAB = read_tsv(find("labels_claude_train.txt")[0])
MARKS = {chr(c) for c in [*range(0x0900, 0x0904), *range(0x093A, 0x094D), 0x094E, 0x094F, 0x0962, 0x0963]}

def units(text):
    out = []
    for ch in text:
        if out and (ch in MARKS or ch == "्" or out[-1].endswith("्")):
            out[-1] += ch
        else:
            out.append(ch)
    return out

def plate_crop(img):
    b = plates_m.predict(img, imgsz=1280, conf=0.2, verbose=False)[0].boxes.xyxy.tolist()
    if not b:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = min(v[0] for v in b), min(v[1] for v in b), max(v[2] for v in b), max(v[3] for v in b)
    px, py = (x2 - x1) * 0.08, (y2 - y1) * 0.15
    return img[int(max(0, y1 - py)):int(min(h, y2 + py)), int(max(0, x1 - px)):int(min(w, x2 + px))]

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)

def align(score):
    # pick m of the n boxes, keeping reading order, maximising the summed score (n >= m)
    n, m = score.shape
    best = np.full((n + 1, m + 1), -1e9); best[:, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, min(i, m) + 1):
            best[i, j] = max(best[i - 1, j], best[i - 1, j - 1] + score[i - 1, j - 1])
    pick, i, j = [], n, m
    while j:
        if i - 1 >= j and best[i, j] == best[i - 1, j]:
            i -= 1
        else:
            pick.append(i - 1); i -= 1; j -= 1
    return pick[::-1]
assert align(np.array([[0, 0], [1, 0], [0, 0], [0, 1]], float)) == [1, 3]    # skips the two "bolts"

name_id = {v: k for k, v in glyphs.names.items()}
REAL = Path("/kaggle/working/real"); (REAL / "images").mkdir(parents=True, exist_ok=True)
stats, photos = Counter(), {}
for pid, text in LAB.items():
    crop = plate_crop(cv2.imread(str(PH / f"{pid}.jpg")))
    if crop is None:
        stats["no plate found"] += 1; continue
    cand = [(x1, y1, x2, y2, "", 1.0) for x1, y1, x2, y2 in
            chars_old.predict(crop, imgsz=416, conf=0.05, verbose=False)[0].boxes.xyxy.tolist()]
    cand = [b for b in cand if b[2] - b[0] > 2 and b[3] - b[1] > 2]
    u = units(text)
    if len(cand) < len(u):
        stats["fewer candidates than characters"] += 1; continue
    cand = reading_order(cand)
    probs = [r.probs.data.cpu().numpy() for r in glyphs.predict(
        [crop[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in cand], imgsz=64, verbose=False)]
    score = np.array([[pr[name_id[x]] if x in name_id else 0.3 for x in u] for pr in probs])
    pick = align(score)
    chosen = [cand[i] for i in pick]
    if any(iou(a, b) > 0.5 for k, a in enumerate(chosen) for b in chosen[k + 1:]):
        stats["two characters on one box"] += 1; continue
    known = [(probs[i].argmax(), name_id[x]) for i, x in zip(pick, u) if x in name_id]
    if known and sum(a == b for a, b in known) < 0.6 * len(known):
        stats["classifier disagrees"] += 1; continue
    h, w = crop.shape[:2]
    f = REAL / "images" / f"{pid}.jpg"; cv2.imwrite(str(f), crop)
    f.with_suffix(".txt").write_text("\n".join(
        f"0 {(b[0] + b[2]) / 2 / w:.6f} {(b[1] + b[3]) / 2 / h:.6f} {(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}"
        for b in chosen))
    photos[pid] = f; stats["used"] += 1; stats["background boxes"] += len(cand) - len(chosen)
print("real plates:", dict(stats))

ids = sorted(photos); random.Random(0).shuffle(ids); nv = max(1, len(ids) * 20 // 100)
cha = IN / "inspiring-lab/nepali-vehicles-number-plate-dataset/open_source_nepali_plates"
tr, va = split_random(imgs(cha / "images"))
add("plates_dev", "train", tr, ["character"], {"character": 0})
add("plates_dev", "val", va, ["character"], {"character": 0})
add("kathmandu_real", "train", [photos[k] for k in ids[nv:]], ["character"], {"character": 0}, repeat=4)
add("kathmandu_real", "val", [photos[k] for k in ids[:nv]], ["character"], {"character": 0})
write_yaml(["character"])

det = YOLO(old)                                  # fine-tune run 1, not from scratch
det.train(data=str(OUT / "data.yaml"), epochs=60, time=1.5, imgsz=416, batch=32, patience=15, lr0=0.002,
          project="/kaggle/working/runs", name="chars4", workers=4, seed=0, plots=True)
shutil.copy(sorted(glob.glob("/kaggle/working/runs/chars4*/weights/best.pt"))[-1], "/kaggle/working/chars_best.pt")
new = YOLO("/kaggle/working/chars_best.pt")

rows = []
for tag in ("plates_dev", "kathmandu_real"):
    y = OUT / f"data_{tag}.yaml"
    y.write_text(yaml.safe_dump({"path": str(OUT), "train": "train/images", "val": f"val_{tag}.txt", "names": {0: "character"}}))
    for nm, m in (("run1", chars_old), ("run4", new)):
        r = m.val(data=str(y), imgsz=416, verbose=False, plots=False)
        rows.append({"val_source": tag, "model": nm, "mAP50": round(r.box.map50, 3), "precision": round(r.box.mp, 3),
                     "recall": round(r.box.mr, 3)})
print(pd.DataFrame(rows).to_string(index=False)); pd.DataFrame(rows).to_csv("/kaggle/working/chars_per_source.csv", index=False)

# End to end. The new detector's threshold is chosen on held-out TRAINING photos; the test photos are run once.
digits = lambda s: (re.findall(r"[०-९]+$", s) or [""])[0]
def score(labels_by_id, chars, conf):
    ex = num = ok = okex = 0
    for pid, truth in labels_by_id.items():
        crop = plate_crop(cv2.imread(str(PH / f"{pid}.jpg")))
        labels, cf = read_plate(crop, chars, glyphs, min_conf=conf) if crop is not None else ([], 0.0)
        got = "".join(labels); good = check_plate(labels, cf)[0]
        ex += got == truth; num += digits(got) == digits(truth); ok += good; okex += good and got == truth
    n = len(labels_by_id)
    return {"exact": round(ex / n, 3), "number": round(num / n, 3), "pass_check": round(ok / n, 3),
            "exact_among_passing": round(okex / max(1, ok), 3), "n": n}
held = {k: LAB[k] for k in ids[:nv]}
sweep = {c: score(held, new, c) for c in (0.25, 0.35, 0.45, 0.55, 0.65)}
print("threshold sweep on held-out training photos:", sweep)
pick_c = max(sweep, key=lambda c: (sweep[c]["exact"], c))
TEST = read_tsv(find("test_ai_labelled.tsv")[0])
res = pd.DataFrame([{"detector": "run1 @0.55", **score(TEST, chars_old, 0.55)},
                    {"detector": f"run4 @{pick_c}", **score(TEST, new, pick_c)}])
res.to_csv("/kaggle/working/end2end_test.csv", index=False)
print("TEST (112 AI-labelled plates, run once):"); print(res.to_string(index=False))
shutil.rmtree(OUT); shutil.rmtree(PH); shutil.rmtree(REAL)
"""


BASE_SETS = ["inspiring-lab/nepali-vehicles-number-plate-dataset", "inspiring-lab/nepali-number-plate-characters-dataset",
             "theciceerguy/embossed-number-plate-detection-nepal-dataset"]
RUN1 = "kathmandu-traffic-vision-train-plate-reader"
NOTEBOOKS = {
    "train-reader": dict(
        title="Kathmandu traffic vision - train plate reader", cells=[PREP, CHARS, GLYPHS], sets=BASE_SETS, kernels=[],
        intro="Trains the two models of the **plate reader**. (1) A character detector (YOLO11s, one class) on 2,503 "
              "cropped Nepali plates with character boxes. (2) A character classifier (YOLO11s-cls) on single-character "
              "crops: Devanagari letters and digits, and embossed-plate digits and Latin letters. Each stage is scored on "
              "held-out data, per source. No public dataset gives whole-plate text, so end-to-end reading accuracy "
              "cannot be measured here; that needs hand-labelled plates from Kathmandu footage."),
    "train-reader-2": dict(
        title="Kathmandu traffic vision - plate reader run 2", cells=[PREP, GLYPHS2],
        sets=BASE_SETS[1:] + ["ishworsubedii/nepalese-license-plate-character-dataset",
                              "kshitizgajurel042/individual-character-images-of-nepali-lp"],
        kernels=[f"{USER}/{RUN1}"],
        intro="Run 2 of the plate reader's **character classifier**, with two more Nepali character sets (Apache-2.0 "
              "and MIT; 2,933 and 4,410 crops). First the run-1 classifier, never trained on them, labels every crop: "
              "an honest test on unseen sources, and a check of the romanised folder names (ba, pa, cha ...); folders "
              "whose labels disagree are dropped. Then the classifier is retrained on all sources and scored per source. "
              "The character detector is unchanged (no new plates with character boxes)."),
    "train-reader-3": dict(
        title="Kathmandu traffic vision - plate reader run 3", cells=["READER", PREP, GLYPHS3, END2END],
        sets=BASE_SETS[1:] + ["ishworsubedii/nepalese-license-plate-character-dataset",
                              "kshitizgajurel042/individual-character-images-of-nepali-lp",
                              "aashikkhatri/kathmandu-plate-labels"],
        kernels=[f"{USER}/{RUN1}", f"{USER}/kathmandu-traffic-vision-plate-reader-run-2",
                 f"{USER}/kathmandu-traffic-vision-train-plates", f"{USER}/kathmandu-traffic-vision-plate-label-set"],
        intro="Run 3 of the plate reader's **character classifier**, adding real Kathmandu plates: 188 photos read "
              "by AI (P0151-P0400, none of them a test photo or a repeat of a test vehicle), split into characters "
              "and matched to the character detector's boxes. Then run 2 and run 3 are compared end to end on the "
              "test photos, against provisional labels (88 plates where two AI readers agree)."),
    "train-reader-4": dict(
        title="Kathmandu traffic vision - plate reader run 4", cells=["READER", PREP, CHARS4],
        sets=BASE_SETS[:1] + ["aashikkhatri/kathmandu-plate-labels"],
        kernels=[f"{USER}/{RUN1}", f"{USER}/kathmandu-traffic-vision-plate-reader-run-3",
                 f"{USER}/kathmandu-traffic-vision-train-plates", f"{USER}/kathmandu-traffic-vision-plate-label-set"],
        intro="Run 4 of the plate reader: the **character detector** fine-tuned on real Kathmandu plates. Boxes come "
              "from matching AI-read plate text (training photos only) to the run-1 detector's low-confidence "
              "candidates; unmatched candidates such as bolts become background. Scored per source, then end to "
              "end on 112 AI-labelled test plates against the run-1 detector, run once."),
}


def main():
    for key, c in NOTEBOOKS.items():
        assert len(c["title"]) <= 50, c["title"]
        cells = [md(f"# {c['title']}\n\n{c['intro']}"),
                 code("!pip install -q ultralytics\n!nvidia-smi --query-gpu=name,memory.total --format=csv")]
        cells += [code(READER if s == "READER" else s) for s in c["cells"]]
        d = Path(__file__).parent / key
        d.mkdir(exist_ok=True)
        nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
              "language": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
        (d / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
        slug = c["title"].lower().replace(" - ", "-").replace(" ", "-")
        meta = {"id": f"{USER}/{slug}", "title": c["title"], "code_file": "notebook.ipynb", "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "enable_internet": True, "competition_sources": [], "kernel_sources": c["kernels"],
                "dataset_sources": c["sets"]}
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"wrote {d}  ({slug})")


if __name__ == "__main__":
    main()
