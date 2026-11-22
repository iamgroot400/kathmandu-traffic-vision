# Kathmandu traffic vision

Camera-based traffic counting and red-light violation detection for Kathmandu junctions. Part of a
three-repository project:

| Repository | Contents |
|---|---|
| `kathmandu-traffic-docs` | Project overview, data sources, technology, roadmap |
| `kathmandu-traffic` | Road-network simulation of the valley: signal timing, pricing and other interventions |
| **`kathmandu-traffic-vision`** (this one) | Vehicle detection, counting, red-light violations, number plates |

## What it does

From one fixed camera at a signalised junction:

1. **Detects** vehicles in every frame with a YOLO model: `motorcycle`, `car` (incl. vans, jeeps,
   microbuses), `bus`, `truck`.
2. **Tracks** each vehicle across frames (ByteTrack), so each one is counted once.
3. **Counts** vehicles crossing the stop line, per minute, class and direction, in vehicles and
   passenger-car units (PCU).
4. **Reads the signal** from a box drawn around the signal head (or a fixed timing cycle, or, later,
   a direct feed from the signal controller).
5. **Flags red-light violations**: a vehicle crossing the stop line into the junction while the
   signal has been red for longer than a grace period. Each one is saved as an evidence frame.
6. Finds the vehicle's number plate (detector trained, run 1); reading the characters is the next step.

```
camera frame ─► vehicle detector ─► tracker ─► stop-line crossing ─► counts (vehicles, PCU per minute)
                                                    │
signal state (camera box / timing cycle / controller)┴─► red-light rule ─► violation + evidence frame
                                                                              │
                                                       plate detector ◄───────┘ (on flagged vehicles only)
```

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows; .venv/bin/pip on Linux / macOS
.venv/Scripts/python test_logic.py                  # decision-logic self-checks, no model needed
.venv/Scripts/python watch.py --source junction.mp4 --pick        # once: click stop line, box the signal
.venv/Scripts/python watch.py --source junction.mp4 --save-video  # run
.venv/Scripts/python watch.py --source junction.mp4 --model results/vehicles_run1/vehicles_best.pt --plates results/plates_run1/plates_best.pt  # + read plates
```

`--pick` writes `config.json` for that camera. Outputs go to `out/`:

| File | Contents |
|---|---|
| `counts.csv` | minute, class, direction (`in` = entering the junction), vehicles, PCU |
| `violations.csv` | time, track id, class, signal state, seconds since red began, snapshot path; with `--plates` also plate text, `ok`/`review: why`, and the plate close-up path |
| `snapshots/` | per violation: the frame (`<time>_<id>.jpg`), the vehicle with its rider or driver (`_vehicle.jpg`), 5 vehicle close-ups and 5 whole uncropped frames from the second before the violation (`_before1..5.jpg` / `_before1..5_scene.jpg`, oldest first; `--before N`, `--before-gap S`), the plate (`_plate.jpg`, with `--plates`, read from whichever of these frames gives the most trustworthy result) and an evidence card with all of them and the facts (`_card.jpg`) |
| `report.html` | every violation with its evidence card, plate and plate check, for a person to review. Personal data: keep `out/` private |
| `snapshots/*.jpg` | evidence frame for each violation, vehicle boxed |
| `annotated.mp4` | (with `--save-video`) the video with boxes, stop line and signal state |

`--model` takes any Ultralytics detector: the COCO-pretrained `yolo11n.pt` works out of the box; the
fine-tuned model (`vehicles_best.pt`, run 1, see [docs/training.md](docs/training.md)) is the one to use.

## Code

| File | Role |
|---|---|
| [`logic.py`](logic.py) | Pure decision logic: line crossing, debounced signal state, counts, the violation rule |
| [`light.py`](light.py) | Signal colour from a box in the frame (HSV), or a fixed timing cycle |
| [`watch.py`](watch.py) | Runs detector + tracker on a video or camera and writes the outputs |
| [`test_logic.py`](test_logic.py) | Self-checks with fake vehicles and synthetic frames |
| [`reader.py`](reader.py) | Plate reader: character boxes -> one label each -> text in reading order (one or two rows) |
| [`test_reader.py`](test_reader.py) | Self-checks for the reading order (jumbled, two-row and tilted plates) |
| [`kaggle/make_reader.py`](kaggle/make_reader.py) | Builds the plate-reader training notebook |
| [`kaggle/make_inspect.py`](kaggle/make_inspect.py) | Builds the dataset-inspection notebook |
| [`kaggle/make_train.py`](kaggle/make_train.py) | Builds the two training notebooks (vehicles, plates) |

## The violation rule

```python
def is_violation(direction, forward, light, red_for, grace):
    return direction == forward and light == "red" and red_for >= grace
```

A crossing counts only if it **enters** the junction, the signal is **red** (debounced over 3 frames,
so one misread frame does not flip it), and it has been red for at least `--grace` seconds (default
1 s: a vehicle already at the line when the signal changes cannot always stop). Amber and an unread
signal are never flagged: a false flag fines an innocent driver, a missed one costs little.

## Documentation

- [docs/datasets.md](docs/datasets.md): every dataset considered, what it contains, its licence, and how
  its classes map onto ours
- [docs/training.md](docs/training.md): each training run, its settings, notebook and results
- [docs/violations.md](docs/violations.md): violation types, implemented and planned, and their rules
- [docs/signal-integration.md](docs/signal-integration.md): reading the signal state from the controller
- [docs/privacy.md](docs/privacy.md): handling of plates and footage

## Status

| Part | State |
|---|---|
| Counting and red-light logic | done, self-checked (`test_logic.py`) |
| Signal state from camera box / timing cycle | done, checked on synthetic frames |
| Vehicle detector (South Asian traffic data) | run 1 done: mAP50 0.67-0.73 on crowded street scenes ([docs/training.md](docs/training.md)) |
| Plate detector | run 1 done: mAP50 0.89-0.99; weak on small distant plates (mAP50-95 0.44) |
| Plate reading (Devanagari and embossed) | run 2 done: characters found 98%; named 99% on held-out crops of known sources, but run 1 named only 64-94% on unseen sources; whole plates not yet measured |
| Test on real Kathmandu junction footage | **needs footage**: no open Kathmandu junction video exists |
| Restricted-vehicle rule (trucks on inner roads) | planned |
