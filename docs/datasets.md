# Datasets

Every dataset was inspected on Kaggle before use (notebook `kathmandu-traffic-vision-dataset-inspection`,
built by [`kaggle/make_inspect.py`](../kaggle/make_inspect.py); its report is
[`kaggle/inspect-out/report.txt`](../kaggle/inspect-out/report.txt)). Class mappings below come from that
report, not from dataset descriptions. No dataset is copied into this repository; the training
notebooks attach them on Kaggle.

## Target classes

| Class | Includes | PCU |
|---|---|---|
| `motorcycle` | motorcycles, scooters | 0.5 |
| `car` | cars, jeeps, taxis, vans, microbuses | 1.0 |
| `bus` | buses, minibuses | 3.0 |
| `truck` | trucks, tippers, pickups, trailers | 3.0 |
| `plate` | number plates of all three Nepali formats (separate model) | n/a |

Auto-rickshaws are dropped: they do not run in Kathmandu. Bicycles, people, animals and signs are
dropped too. Vehicles and plates are separate models because no dataset labels both everywhere:
in a single model, every unlabelled plate in a vehicle dataset (and every unlabelled vehicle in a
plate dataset) would be learned as background.

## Used: vehicle model

| Dataset (Kaggle) | Images | Format | Licence | Mapping | Used |
|---|---|---|---|---|---|
| `redzapdos123/indian-driving-dataset-detections-yolov11` (Indian Driving Dataset) | 41,962 (33,569 / 4,196 / 4,197) | YOLO, 15 classes | research use only | car, caravan -> car; motorcycle; bus; truck, trailer -> truck | 12,000 train, 1,500 val (random sample, seed 0) |
| `daudshah/vehicle-detection-dataset` (Roboflow "ai-traffic-system", Indian traffic video frames) | 9,440 (7,556 / 1,072 / 812) | YOLO, 7 classes, 640x640 | CC BY 4.0 | car, van -> car; motorbike -> motorcycle; bus; truck | all train and valid |
| `kshitizgajurel042/nepali-bike-and-car-images-with-annotations` (Nepal) | 1,052 (806 / 246) | YOLO: car, license_plate, bike | as listed on Kaggle | car; bike -> motorcycle | all, train repeated x3 |

## Used: plate model

| Dataset (Kaggle) | Images | Format | Mapping | Used |
|---|---|---|---|---|
| `ishworsubedii/vehicle-number-plate-datasetnepal` (Kathmandu, Bhaktapur, Lalitpur) | 8,078 | YOLO, 1 class | -> plate | random 85 / 15 split (seed 0) |
| `kshitizgajurel042/nepali-bike-and-car-images-with-annotations` | 1,052 | YOLO | license_plate -> plate | given split, train x2 |
| `deepakat002/indian-vehicle-number-plate-yolo-annotation` (Indian traffic video) | 160 | YOLO, number_plate | -> plate | random 85 / 15, train x5 (distant plates, like a junction camera) |

## For plate reading (next step)

| Dataset | Contents |
|---|---|
| `inspiring-lab/nepali-number-plate-characters-dataset` | 26,537 crops in one folder per Devanagari character (zonal and provincial plates) |
| `inspiring-lab/nepali-vehicles-number-plate-dataset` | 2,503 cropped plates with character boxes (one class, `character`) |
| `theciceerguy/embossed-number-plate-detection-nepal-dataset` | 10,158 crops, one folder per digit / letter / flag (embossed plates) |
| `utsavbaral007/nepal-embossed-number-plate-dataset` | 3,592 crops, same layout (likely overlaps the previous) |

## Added for plate reader run 2 (2026-11-20)

| Dataset | Contents | Licence |
|---|---|---|
| `ishworsubedii/nepalese-license-plate-character-dataset` | 2,933 Devanagari character crops in 12 folders with romanised names (0-9, ba, pa) | Apache-2.0 |
| `kshitizgajurel042/individual-character-images-of-nepali-lp` | 4,410 Devanagari character crops in 17 folders (0-9, ba, cha, kha, ko, me, pa, pra) | MIT |

Folder names are mapped to our classes (0 -> ०, ba -> बा, pa -> प ...); a folder is used only if the
run-1 classifier, never trained on it, gives the mapped label to most of its crops.

## Considered and not used

| Dataset | Why not |
|---|---|
| `sdevkota007/vehicles-nepal` | 5,797 cropped vehicles (Kathmandu, 2016), no boxes: classification only |
| `anisha713/vehicle-detection-dataset` | a single 540 KB file |
| `pankajdawadi/nepali-lp-dataset` | character crops in folders numbered 0-12 with no key to the characters; licence "other" |
| `nishapokharel/nepali-vehicles-number-plate-dataset` | 927 raw photos, no labels (CC BY-NC 4.0); could become a test set once hand-labelled |
| `binay07/nepali-vehicle-number-plate-dataset` | 4 GB of raw photos, no labels, licence unknown |
| GTS.ai "Nepali number plate" pages | resale of the Inspiring Lab sets above, behind a sign-up form |
| `kritizsapkota/number-plates`, `kshitizgajurel042/images-of-nepali-license-plate` | images without box labels |
| Kathmandu University set (Roboflow Universe) | could not be inspected (needs a Roboflow account); worth adding |
| DriveIndia | release not confirmed |

## Known biases

- No dataset is Kathmandu junction footage from a raised camera. The Nepali sets are mostly close
  photos of single vehicles; the Indian sets are dashboard or street-level video.
- The Roboflow video-frame set splits frames of the same videos randomly between train and
  validation, so its validation score is optimistic.
- A held-out set of hand-checked frames from Kathmandu junction footage is needed before any
  accuracy claim about Kathmandu.
