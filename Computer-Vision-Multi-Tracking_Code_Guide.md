<div align="center">

# 🎯 Computer-Vision-Multi-Tracking
### A Guided Tour of the Code

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/Detector-YOLOv8-00FFFF)
![DeepSORT](https://img.shields.io/badge/Tracker-DeepSORT-green)
![SORT](https://img.shields.io/badge/Baseline-SORT-red)
![OpenCV](https://img.shields.io/badge/OpenCV-video%20I%2FO-5C3EE8?logo=opencv&logoColor=white)

*Tracking-by-detection: **YOLOv8** finds the objects, **SORT / DeepSORT** keep their identities across frames.*

</div>

---

## 📑 Table of Contents

1. [What this project does](#1--what-this-project-does)
2. [Repository map](#2--repository-map)
3. [Core concepts in 5 minutes](#3--core-concepts-in-5-minutes)
4. [Setup & dependencies](#4--setup--dependencies)
5. [The shared pipeline](#5--the-shared-pipeline-every-script-follows-this)
6. [File-by-file walkthrough](#6--file-by-file-walkthrough)
   - [6.1 `medium_Yolov8_Both.py`](#61-medium_yolov8_bothpy--batch-runner-deepsort-vs-baseline)
   - [6.2 `Large_and_XLarge_Yolov8_DeepSORT.py`](#62-large_and_xlarge_yolov8_deepsortpy--the-stable-deepsort-pipeline)
   - [6.3 `Large_and_XLarge_Yolov8_Sort.py`](#63-large_and_xlarge_yolov8_sortpy--sort-implemented-from-scratch)
   - [6.4 `Comparison.py`](#64-comparisonpy--the-id-count-comparator)
7. [Parameter cheat-sheet](#7--parameter-cheat-sheet)
8. [Output file formats](#8--output-file-formats)
9. [How to run it (recommended workflow)](#9--how-to-run-it-recommended-workflow)
10. [Code observations & gotchas](#10--code-observations--gotchas)
11. [Ideas for extending the project](#11--ideas-for-extending-the-project)
12. [Glossary](#12--glossary)

---

## 1. 🧭 What this project does

**Multi-Object Tracking (MOT)** means following *many* objects through a video while giving each one a **persistent unique ID**: *"car #7 stays car #7 even when it passes behind a truck."*

This repository implements the classic **tracking-by-detection** approach and compares two trackers:

| Tracker | Idea | Strength | Weakness |
|---|---|---|---|
| **SORT** | Kalman filter (motion) + IoU matching (box overlap) | Very fast, simple | Loses identity after occlusion → many ID switches |
| **DeepSORT** | SORT + a **CNN appearance embedding** per object | Can re-identify objects after occlusion | Slower, needs more tuning |

Two sample videos are included: an **urban traffic** scene (`videoplayback.mp4`) and a **football match** (`08fd33_4.mp4`).

> [!NOTE]
> This guide describes **how the code works**, not the experimental results.

---

## 2. 🗂️ Repository map

```text
Computer-Vision-Multi-Tracking/
├── 📜 medium_Yolov8_Both.py                  # Batch runner: YOLOv8-medium + DeepSORT AND "SORT-like" baseline
├── 📜 Large_and_XLarge_Yolov8_DeepSORT.py    # YOLOv8-large/xlarge + DeepSORT (tuned for stability)
├── 📜 Large_and_XLarge_Yolov8_Sort.py        # YOLOv8-large/xlarge + a full SORT implementation (Kalman + Hungarian)
├── 📜 Comparison.py                          # Reads the result .txt files and compares ID counts
├── 🎞️ videoplayback.mp4                      # Sample: urban traffic
└── 🎞️ 08fd33_4.mp4                           # Sample: football match
```

**How the scripts relate to each other**

```text
                ┌──────────────────────────────┐
                │ medium_Yolov8_Both.py        │──► final_results/*.mp4 + *_results.txt
                └──────────────────────────────┘      (self-contained, own comparison pair)

 videoplayback.mp4 ─┬─► Large_and_XLarge_Yolov8_DeepSORT.py ──► my_results_*.txt   ─┐
                    │                                                                ├─► Comparison.py
                    └─► Large_and_XLarge_Yolov8_Sort.py     ──► sort_results_*.txt ─┘
```

---

## 3. 🧠 Core concepts in 5 minutes

### 3.1 Tracking-by-detection
Every frame goes through two independent stages:

1. **Detection** – YOLOv8 answers *"what is where?"* → a list of boxes + confidence + class.
2. **Association** – the tracker answers *"which box is which known object?"* → boxes with **track IDs**.

### 3.2 The Kalman filter (motion model)
Each tracked object owns a small Kalman filter that **predicts** where the box will be in the next frame (assuming roughly constant velocity) and then **corrects** that prediction with the real detection. This lets the tracker survive a few frames of missed detections.

### 3.3 IoU (Intersection over Union)
A geometric similarity between two boxes:

```
IoU = area(A ∩ B) / area(A ∪ B)      # 0 = no overlap, 1 = identical
```

SORT matches predicted boxes to new detections purely by IoU.

### 3.4 The Hungarian algorithm
When several detections compete for several tracks, the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) finds the globally best one-to-one assignment.

### 3.5 Appearance embeddings (DeepSORT's addition)
DeepSORT crops each detection from the frame, passes it through a small CNN (here **MobileNet**) and obtains a feature vector. Two crops of the *same* object have a **small cosine distance**; this is what allows re-identification after an occlusion.

### 3.6 Track lifecycle

```text
 new detection ──► [TENTATIVE] ──(n_init consecutive hits)──► [CONFIRMED] ──(missed > max_age frames)──► [DELETED]
                        │                                          │
                        └── missed early ──► DELETED               └── missed a few frames: "coasting" on Kalman prediction
```

---

## 4. ⚙️ Setup & dependencies

There is no `requirements.txt` in the repo, so here is what the imports need:

```bash
pip install ultralytics opencv-python numpy pandas scipy filterpy deep-sort-realtime
```

| Package | Used by | Purpose |
|---|---|---|
| `ultralytics` | all tracking scripts | YOLOv8 detector (weights `yolov8m/l/x.pt` auto-download on first use) |
| `opencv-python` | all tracking scripts | Read/write video, draw boxes, show window |
| `numpy` | all tracking scripts | Array math |
| `deep-sort-realtime` | `medium_…Both.py`, `…_DeepSORT.py` | Ready-made DeepSORT tracker |
| `filterpy` | `…_Sort.py` | Kalman filter for the hand-built SORT |
| `scipy` | `…_Sort.py` | Hungarian algorithm |
| `pandas` | `Comparison.py` | Load and analyse result files |

> [!TIP]
> A CUDA-capable GPU is strongly recommended. The scripts run YOLOv8-L/X at `imgsz=1088` on every frame, and the DeepSORT embedder is configured for GPU/half-precision.

---

## 5. 🔄 The shared pipeline (every script follows this)

```text
┌──────────┐   ┌───────────┐   ┌────────────────┐   ┌──────────────┐   ┌───────────────────┐
│ cv2      │──►│ YOLOv8    │──►│ Format boxes   │──►│ Tracker      │──►│ Draw + log        │
│ read     │   │ detect    │   │ for the tracker│   │ update       │   │ (video + .txt)    │
│ frame    │   │ (filtered │   │                │   │ (SORT or     │   │                   │
│          │   │  classes) │   │                │   │  DeepSORT)   │   │                   │
└──────────┘   └───────────┘   └────────────────┘   └──────────────┘   └───────────────────┘
```

**The one detail that trips everyone up: bounding-box formats differ.**

| Where | Format | Meaning |
|---|---|---|
| YOLO output (`results.boxes.data`) | `x1, y1, x2, y2, score, class` | corners (left-top / right-bottom) |
| DeepSORT **input** | `([left, top, w, h], score, class)` | top-left + width/height (**LTWH**) |
| SORT **input** | `[x1, y1, x2, y2, score]` | corners |
| Kalman state (inside SORT) | `[cx, cy, s, r, …]` | centre, area, aspect ratio |
| Result `.txt` files | `frame,id,left,top,w,h,…` | MOT format (LTWH) |

Each script contains the conversion glue between these.

---

## 6. 🔍 File-by-file walkthrough

### 6.1 `medium_Yolov8_Both.py` — *batch runner: DeepSORT vs. baseline*

**Purpose:** Process **every `.mp4` in the current folder**, twice per video: once with DeepSORT, once with a "SORT-like" baseline, and save videos + MOT-style text files in `final_results/`.

#### Setup block
```python
input_folder  = "."
output_folder = "final_results"          # created if missing
video_files   = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
model         = YOLO("yolov8m.pt")       # medium: faster, still good enough to catch the football
```
- The model is loaded **once** at module level and reused for all videos and both modes.
- `final_results/` is a subfolder, so output videos are *not* picked up as inputs on a re-run.

#### `run_tracking(video_path, mode_name, use_deep=True)`
This single function runs one full pass over one video.

| Step | What happens |
|---|---|
| 1. Open I/O | `cv2.VideoCapture`, reads `width/height/fps`, creates a `VideoWriter` (`mp4v` codec) and opens `<name>_<mode>_results.txt` |
| 2. Build tracker | `DeepSort(...)` with parameters that depend on `use_deep` (see below) |
| 3. Frame loop | Read frame → detect → convert → `tracker.update_tracks` → draw/log → write frame |
| 4. Cleanup | Release capture, writer, close text file |

**The clever trick: one tracker class, two behaviours**

```python
tracker = DeepSort(
    max_age=30 if use_deep else 5,
    n_init=3,
    nms_max_overlap=0.5,
    max_cosine_distance=0.2 if use_deep else 0.9,   # 0.9 ≈ "ignore appearance"
    embedder="mobilenet",
    half=True,
    embedder_gpu=True,
)
```

| Setting | DeepSORT mode | "SORT_Baseline" mode | Effect |
|---|---|---|---|
| `max_age` | 30 | 5 | Baseline forgets lost objects 6× sooner |
| `max_cosine_distance` | 0.2 (strict) | 0.9 (very permissive) | A 0.9 gate accepts almost any appearance, so matching degenerates to motion/IoU |

> [!IMPORTANT]
> The "baseline" is an **approximation of SORT built from the DeepSORT class**, not the real SORT algorithm. It still computes embeddings and still runs DeepSORT's matching cascade. The real SORT lives in `Large_and_XLarge_Yolov8_Sort.py`.

#### Detection step
```python
# 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck, 32: sports ball
results = model(frame, verbose=False, conf=0.3, classes=[0, 2, 3, 5, 7, 32])[0]
```
- A **low confidence (0.3)** keeps small/blurry objects like the **ball**.
- `classes=[…]` filters at the detector, so everything else is never returned.
- `verbose=False` silences Ultralytics' per-frame console logs.

#### Format conversion
```python
for r in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = r
    detections.append(([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], score, int(class_id)))
```
Corners → `[left, top, width, height]`, exactly what DeepSORT expects.

#### Tracking, logging, drawing
```python
tracks = tracker.update_tracks(detections, frame=frame)   # frame is needed to crop for embeddings
for track in tracks:
    if not track.is_confirmed(): continue                 # skip tentative tracks
    tid  = track.track_id
    ltrb = track.to_ltrb()
    txt_file.write(f"{frame_idx},{tid},{x},{y},{w},{h},1,-1,-1,-1\n")
```
- Colour coding: **green = DeepSORT**, **red = baseline** (BGR tuples), so exported videos are visually distinguishable.
- A progress message prints every 50 frames.

#### Main loop
```python
for v_file in video_files:
    run_tracking(v_file, "DeepSORT",      use_deep=True)
    run_tracking(v_file, "SORT_Baseline", use_deep=False)
```

**Outputs per video:** `final_results/<name>_DeepSORT.mp4`, `<name>_DeepSORT_results.txt`, `<name>_SORT_Baseline.mp4`, `<name>_SORT_Baseline_results.txt`.

> [!NOTE]
> Code comments in this file are written in **Greek** (e.g. *"Εκτελεί tracking και αποθηκεύει βίντεο + TXT αποτελέσματα"* = "Runs tracking and saves video + TXT results").

---

### 6.2 `Large_and_XLarge_Yolov8_DeepSORT.py` — *the stable DeepSORT pipeline*

**Purpose:** A single-video, **higher-accuracy** pipeline: bigger YOLO model, stricter detection threshold, stricter track confirmation, and an explicit **"no ghost boxes"** filter.

#### Configuration (top of file)
```python
VIDEO_PATH  = "videoplayback.mp4"
OUTPUT_PATH = "final_stable_tracking_l_traffic.mp4"
METRIC_FILE = "my_results_l_traffic.txt"
model = YOLO("yolov8l.pt")
```
Commented-out alternatives (`…_x_traffic…`, `yolov8x.pt`) are provided. **To run the Extra-Large variant, swap the three path lines and the model line** by commenting/uncommenting them.

#### Tracker: "the memory, adjusted for stability"
```python
tracker = DeepSort(max_age=40, n_init=5, nms_max_overlap=0.3,
                   max_cosine_distance=0.2, nn_budget=200)
```

| Parameter | Value | Intuition |
|---|---|---|
| `max_age` | 40 | A confirmed track survives 40 missed frames (≈1.3 s @ 30 fps) before deletion |
| `n_init` | 5 | An object needs 5 consecutive detections to become a real track → suppresses flickering false positives |
| `nms_max_overlap` | 0.3 | Aggressive NMS on detections: overlapping boxes are merged more eagerly |
| `max_cosine_distance` | 0.2 | Strict appearance match |
| `nn_budget` | 200 | Max stored appearance samples per track (memory/speed cap) |

The embedder isn't specified, so the library default is used.

#### Output preparation
```python
if os.path.exists(METRIC_FILE): os.remove(METRIC_FILE)
```
Because results are written in **append mode** later, the old file is deleted first to avoid mixing runs.

#### Detection
```python
ALLOWED_CLASSES = [0, 1, 2, 3, 5, 7]     # person, bicycle, car, motorcycle, bus, truck
results = model(frame, imgsz=1088, conf=0.45, classes=ALLOWED_CLASSES)[0]
```
- `imgsz=1088`: a multiple of 32 (YOLO's stride), larger than the 640 default → better for small, distant vehicles.
- `conf=0.45`: stricter than the medium script (0.3), because there is no ball to preserve here and cleaner detections give cleaner tracks.

#### The "walking ghost boxes" filter
```python
if not track.is_confirmed():        # Check 1: must have survived n_init frames
    continue
if track.time_since_update > 1:     # Check 2: must have been matched in (almost) this frame
    continue
```
A confirmed track that lost its detection keeps being **extrapolated** by the Kalman filter, so its box drifts on its own, a "ghost". Check 2 hides those predicted-only boxes while *keeping the ID alive internally*, so it can be re-matched later.

#### Logging, display, exit
```python
with open(METRIC_FILE, "a") as f:
    f.write(f"{frame_count},{track_id},{ltwh[0]},{ltwh[1]},{ltwh[2]},{ltwh[3]},1,-1,-1,-1\n")
cv2.imshow("Final Stable Tracking", frame)
if cv2.waitKey(1) & 0xFF == ord('q'): break
```
Live preview window; press **`q`** to stop early. Boxes are drawn **blue** `(255, 0, 0)` in BGR.

---

### 6.3 `Large_and_XLarge_Yolov8_Sort.py` — *SORT implemented from scratch*

**Purpose:** The *true* baseline. A compact re-implementation of the original **SORT** (Bewley et al.), with no appearance features, only **Kalman + IoU + Hungarian**. It is the longest file (~240 lines), so here is a map of its parts:

```text
Sort                                  ← the manager (owns all trackers, runs each frame's update)
 ├── KalmanBoxTracker                 ← one instance per tracked object (the motion model)
 ├── associate_detections_to_trackers ← IoU matrix + Hungarian matching
 │     └── iou_batch                  ← IoU of one box pair
 └── convert_bbox_to_z / convert_x_to_bbox   ← box ⇄ Kalman-state conversions
```

#### 6.3.1 Coordinate conversions

```python
def convert_bbox_to_z(bbox):          # [x1,y1,x2,y2] → [cx, cy, s, r]
    w = x2-x1;  h = y2-y1
    cx = x1 + w/2;  cy = y1 + h/2
    s  = w*h                          # scale = AREA
    r  = w/h                          # aspect ratio
```
```python
def convert_x_to_bbox(x):             # Kalman state → [x1,y1,x2,y2]
    w = sqrt(s*r);  h = s/w
```
Using **area** and **aspect ratio** (instead of w and h) is a SORT design choice: the aspect ratio is assumed roughly constant, so only the area gets a velocity term.

#### 6.3.2 `KalmanBoxTracker`: one object's motion model

**State vector (7 numbers):** `[cx, cy, s, r, vx, vy, vs]`: position, area, ratio, and the *velocities* of the first three.
**Measurement (4 numbers):** `[cx, cy, s, r]`: what the detector observes.

```python
self.kf = KalmanFilter(dim_x=7, dim_z=4)
self.kf.F = ...   # transition: position += velocity each frame (constant-velocity model)
self.kf.H = ...   # measurement: we observe the first 4 state entries only
self.kf.R[2:, 2:]  *= 10.     # trust area/ratio measurements less
self.kf.P[4:, 4:]  *= 1000.   # initial velocity is unknown → huge uncertainty
self.kf.P          *= 10.
self.kf.Q[-1, -1]  *= 0.01    # process noise: scale velocity changes slowly
self.kf.Q[4:, 4:]  *= 0.01
```

| Matrix | Role |
|---|---|
| `F` | State transition: `x_{t+1} = F·x_t` (position gains velocity) |
| `H` | Maps state → measurement space |
| `R` | Measurement noise covariance |
| `P` | State uncertainty (initial covariance) |
| `Q` | Process noise covariance |

**Bookkeeping fields**

| Field | Meaning |
|---|---|
| `id` | From class-level counter `KalmanBoxTracker.count` (unique per run) |
| `time_since_update` | Frames since last matched detection |
| `hits` / `hit_streak` | Total / consecutive matched frames |
| `age` | Frames since creation |
| `history` | Predicted boxes since last update |

**Methods**
- `update(bbox)` – resets `time_since_update`, increments hit counters, runs the Kalman **correction** with the matched detection.
- `predict()` – Kalman **prediction** step. It first guards against a negative area (`if x[6] + x[2] <= 0: x[6] *= 0`), increments `age`, resets `hit_streak` if the track missed the previous frame, and returns the predicted box.
- `get_state()` – current box estimate.

#### 6.3.3 `associate_detections_to_trackers` — matching

```text
detections (N) × trackers (M)
        │
        ▼
  IoU matrix  [N × M]
        │
        ▼
  Binary mask  IoU > threshold
        │
        ├─ every row/col has ≤ 1 candidate? ──► take them directly (fast path)
        └─ otherwise ───────────────────────► Hungarian on −IoU (linear_sum_assignment)
        │
        ▼
  Reject assigned pairs whose IoU < threshold → they become unmatched
        │
        ▼
  returns: matches, unmatched_detections, unmatched_trackers
```

- If there are **no trackers yet** → every detection is "unmatched" (will spawn a new track).
- Hungarian is run on the **negative** IoU matrix because SciPy *minimises* cost while we want to *maximise* overlap.
- A source comment `# --- FIX IS HERE: Replaced ']' with ')' ---` marks a bracket typo the author fixed.

#### 6.3.4 `Sort.update(dets)` — the per-frame algorithm

```text
1. frame_count += 1
2. PREDICT   : every KalmanBoxTracker.predict() → predicted boxes (drop any with NaN)
3. ASSOCIATE : IoU + Hungarian between detections and predicted boxes
4. UPDATE    : matched trackers  → tracker.update(detection)
5. SPAWN     : unmatched detections → new KalmanBoxTracker (new ID)
6. OUTPUT    : for each tracker, emit [x1,y1,x2,y2,id+1] if
                  time_since_update < 1                    (matched this frame)
              AND (hit_streak >= min_hits  OR  frame_count <= min_hits)
7. PRUNE     : delete trackers with time_since_update > max_age
```

`id + 1` makes IDs start at **1** (MOT convention).

#### 6.3.5 Main script

```python
tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
```

| Parameter | Value | Meaning |
|---|---|---|
| `max_age` | 1 | A track is deleted after **a single** missed frame, so there is almost no occlusion tolerance |
| `min_hits` | 3 | A track must be matched 3 times before it is reported |
| `iou_threshold` | 0.3 | Minimum overlap to accept a match |

The loop mirrors the DeepSORT script (same classes, `imgsz=1088`, `conf=0.45`, same `q`-to-quit window), with these differences:

```python
detections.append([x1, y1, x2, y2, score])      # SORT wants corners + score; class id is dropped
detections = np.array(detections)
track_results = tracker.update(detections)       # → rows of [x1, y1, x2, y2, id]
...
f.write(f"{frame_count},{track_id},{x1},{y1},{x2 - x1},{y2 - y1},1,-1,-1,-1\n")   # converted to LTWH
```
Boxes are drawn **green**; results go to `sort_results_l_traffic.txt`. As in the DeepSORT script, the three path lines and the model line can be swapped for the **X** variants.

---

### 6.4 `Comparison.py` — *the ID-count comparator*

**Purpose:** Turn two result files into one comparison row, using **number of unique track IDs** as a simple fragmentation indicator. Fewer IDs for the same scene suggests fewer identity switches / re-spawned tracks.

```python
def calculate_tracking_metrics(video_label, baseline_path, deepsort_path):
    if not exists(baseline_path) or not exists(deepsort_path):
        return None                                   # graceful skip
    df_baseline = pd.read_csv(baseline_path, header=None)
    df_deepsort = pd.read_csv(deepsort_path, header=None)
    ids_baseline = df_baseline[1].nunique()           # column 1 = track ID
    ids_deepsort = df_deepsort[1].nunique()
    reduction = (ids_baseline - ids_deepsort) / ids_baseline * 100
    return {"Video": ..., "Baseline IDs": ..., "DeepSORT IDs": ..., "Improvement": round(reduction, 2)}
```

**Hard-coded comparison list**
```python
comparisons = [
    ("Urban Traffic (videoplayback)", "sort_results_x_traffic.txt",   "my_results_x_traffic.txt"),
    ("Football Match (08fd33_4)",     "sort_results_x_football.txt",  "my_results_x_football.txt"),
]
```
Each tuple is `(label, baseline_file, deepsort_file)`. Then a formatted table is printed; missing files print a warning (in Greek: *"Τα αρχεία … δεν βρέθηκαν"* = "files … not found").

> [!WARNING]
> The file names expected here follow the **`_x_`** (XLarge) naming of the two Large/XLarge scripts, and a football pair (`…_x_football.txt`) that none of the scripts in the repo produces by default. See [Gotchas](#10--code-observations--gotchas).

---

## 7. 🎛️ Parameter cheat-sheet

### Detector (YOLOv8)

| Script | Weights | `conf` | `imgsz` | Classes |
|---|---|---|---|---|
| `medium_Yolov8_Both.py` | `yolov8m.pt` | 0.30 | default (640) | person, car, motorcycle, bus, truck, **sports ball** |
| `…_DeepSORT.py` | `yolov8l.pt` (or `x`) | 0.45 | 1088 | person, **bicycle**, car, motorcycle, bus, truck |
| `…_Sort.py` | `yolov8l.pt` (or `x`) | 0.45 | 1088 | person, **bicycle**, car, motorcycle, bus, truck |

### Tracker

| Parameter | Medium · DeepSORT | Medium · Baseline | Large · DeepSORT | Large · SORT |
|---|---|---|---|---|
| Implementation | `deep_sort_realtime` | `deep_sort_realtime` (appearance gate relaxed) | `deep_sort_realtime` | hand-written SORT |
| `max_age` | 30 | 5 | 40 | 1 |
| `n_init` / `min_hits` | 3 | 3 | 5 | 3 |
| `max_cosine_distance` | 0.2 | 0.9 | 0.2 | n/a |
| `nms_max_overlap` | 0.5 | 0.5 | 0.3 | n/a |
| `nn_budget` | default | default | 200 | n/a |
| `iou_threshold` | library default | library default | library default | 0.3 |
| Hides "coasting" tracks? | ❌ no | ❌ no | ✅ yes (`time_since_update > 1`) | ✅ yes (by design) |

### COCO class IDs used

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | person | 5 | bus |
| 1 | bicycle | 7 | truck |
| 2 | car | 32 | sports ball |
| 3 | motorcycle | | |

---

## 8. 📄 Output file formats

### Result `.txt` (MOT Challenge style)

One line per confirmed track per frame, comma-separated:

```text
<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
 12,    7,    532,      301,     88,         64,          1,    -1, -1, -1
```

| Column | Value written by these scripts |
|---|---|
| `frame` | 1-based frame number |
| `id` | Track ID |
| `bb_left, bb_top, bb_width, bb_height` | Box in **LTWH** pixels |
| `conf` | Always `1` (placeholder) |
| `x, y, z` | Always `-1` (3-D world coords, unused for 2-D tracking) |

This layout is directly consumable by MOT evaluation tools such as TrackEval or `motmetrics` (the Large scripts' comments mention logging "for MOTA").

### Annotated videos
`mp4v`-encoded copies of the input, with a coloured box and `ID:<n>` label per track:

| Script | Box colour |
|---|---|
| Medium · DeepSORT | 🟩 green |
| Medium · Baseline | 🟥 red |
| Large · DeepSORT | 🟦 blue |
| Large · SORT | 🟩 green |

---

## 9. ▶️ How to run it (recommended workflow)

```bash
# 0. Environment
pip install ultralytics opencv-python numpy pandas scipy filterpy deep-sort-realtime

# 1. Quick end-to-end test on every .mp4 in the folder (outputs → ./final_results/)
python medium_Yolov8_Both.py

# 2. High-accuracy pair on the traffic video (press 'q' in the window to stop early)
python Large_and_XLarge_Yolov8_DeepSORT.py     # → my_results_l_traffic.txt
python Large_and_XLarge_Yolov8_Sort.py         # → sort_results_l_traffic.txt

# 3. Compare unique-ID counts
python Comparison.py
```

**Making the pieces line up for `Comparison.py`:**

1. In both Large scripts, **comment the `_l_` lines and uncomment the `_x_` lines** (paths + `yolov8x.pt`) so outputs are named `my_results_x_traffic.txt` / `sort_results_x_traffic.txt`; or edit the file names in `comparisons` to match the `_l_` files.
2. For the football video, set `VIDEO_PATH = "08fd33_4.mp4"` and rename the output paths to `…_x_football…` before running each script.
3. Run `Comparison.py` from the same folder as the `.txt` files.

> [!TIP]
> If you're on a machine without a display (server / Colab / WSL without GUI), remove the `cv2.imshow` / `cv2.waitKey` lines from the Large scripts. They require a window.

---

## 10. 🔎 Code observations & gotchas

These are things worth knowing before you modify or reuse the code.

| # | Where | Observation | Why it matters |
|---|---|---|---|
| 1 | `medium_…Both.py` | "SORT_Baseline" is **DeepSORT with loosened parameters**, not real SORT | Not a pure motion-only baseline; the true one is in `…_Sort.py` |
| 2 | `medium_…Both.py` | No `time_since_update` filter | Confirmed-but-unmatched ("coasting") tracks are still drawn and logged, which can inflate boxes/IDs vs. the Large DeepSORT script |
| 3 | `Comparison.py` ↔ scripts | File names don't match: `Comparison.py` expects `sort_results_x_*.txt` / `my_results_x_*.txt` in the **working directory**, while the medium script writes `final_results/<video>_<mode>_results.txt`, and the Large scripts default to `_l_` names | You must rename or edit paths for a smooth run |
| 4 | `Comparison.py` | Football files (`…_x_football.txt`) aren't produced by any script as written | Needs a manual run with `VIDEO_PATH` changed |
| 5 | Large scripts | Result file is **opened and closed for every track in every frame** | Works, but slow; open once before the loop instead |
| 6 | Large scripts | `cv2.imshow` is mandatory | Fails on headless machines |
| 7 | Large scripts | `model(frame, …)` is called without `verbose=False` | Console prints a line per frame |
| 8 | All scripts | `fps = int(cap.get(cv2.CAP_PROP_FPS))` truncates fractional fps (e.g. 29.97 → 29) | Slight output speed drift; use `float` |
| 9 | All scripts | Class IDs are not used after detection | A truck and a person are tracked identically; class-aware tracking isn't implemented |
| 10 | `…_Sort.py` | `iou_batch` actually computes IoU for **one** pair (name suggests vectorised) and runs in a Python double loop | Fine for tens of objects; slow for crowds |
| 11 | `…_Sort.py` | `Sort.update` receives `np.array([])` (shape `(0,)`) when nothing is detected | Handled by the loops, but `np.empty((0, 5))` would be the cleaner contract |
| 12 | `Comparison.py` | `reduction` divides by `ids_baseline` | Raises `ZeroDivisionError` if a baseline file has no tracks |
| 13 | Metric design | "Fewer unique IDs" is only a **proxy** for tracking quality | It doesn't verify that IDs are *correct* (no ground truth); real MOT metrics (MOTA, IDF1, HOTA) need annotations |
| 14 | Repo | No README, no `requirements.txt`, no `.gitignore`, and videos are committed to the repo | Reproducibility depends on this guide |

---

## 11. 💡 Ideas for extending the project

- **Factor out the common code**: three scripts repeat the capture/writer/detection loop. A single `run(video, detector, tracker, out_prefix)` function plus a config dict would remove ~70% of the duplication.
- **Command-line interface**: replace hard-coded `VIDEO_PATH` / comment-swapping with `argparse` (`--video`, `--model l|x`, `--tracker sort|deepsort`).
- **Single, shared output convention** (e.g. `results/<video>/<model>_<tracker>.txt`) so `Comparison.py` can auto-discover files instead of using a hard-coded list.
- **Real MOT metrics**: add ground-truth annotations and compute **MOTA / IDF1 / HOTA** with `TrackEval` or `motmetrics`, using the existing MOT-format output.
- **Class-aware tracking**: keep the class ID in the tracker (or run one tracker per class) so a person can never inherit a car's ID.
- **Try newer trackers**: ByteTrack and BoT-SORT are built into Ultralytics (`model.track(..., tracker="bytetrack.yaml")`) and could be dropped in as additional comparisons.
- **Efficiency**: `float` fps, open result files once, vectorise IoU with NumPy broadcasting, optional `--no-display`.

---

## 12. 📚 Glossary

| Term | Meaning |
|---|---|
| **MOT** | Multi-Object Tracking |
| **Tracking-by-detection** | Detect objects in each frame first, then link detections across time |
| **ID switch** | The tracker gives an existing object a new ID (or swaps IDs between two objects) |
| **Fragmentation** | One real trajectory broken into several track IDs |
| **Occlusion** | An object is temporarily hidden by another |
| **Kalman filter** | Recursive estimator that predicts and corrects an object's state |
| **IoU** | Intersection over Union, the overlap score of two boxes |
| **Hungarian algorithm** | Optimal one-to-one assignment solver |
| **Cosine distance** | `1 − cosine similarity` between two embedding vectors; small = similar appearance |
| **NMS** | Non-Maximum Suppression, removes redundant overlapping boxes |
| **LTWH / LTRB** | Box as *left-top-width-height* / *left-top-right-bottom* |
| **Tentative / Confirmed track** | Track not yet / already verified by `n_init` consecutive hits |
| **Coasting** | A track moving on Kalman prediction alone because it got no detection this frame |

---

<div align="center">

*Guide generated from a reading of the repository's four Python scripts.*

</div>
