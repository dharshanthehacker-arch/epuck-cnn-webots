# E-puck CNN Object Classification & Colour-Block Detection in Webots

**Applied Robotics & AI Project — Part 2 (Assignments 3, 4, 5)**
BEng (Hons) Robotics and Artificial Intelligence, University of Hertfordshire

An e-puck robot explores a miniature arena surrounded by CIFAR-10 style objects
(two cars, horse, dog, cat, deer, two birds, a ship) and four coloured blocks.
It detects the blocks with an OpenCV HSV pipeline with shape verification, and
classifies the other objects with a CIFAR-10 CNN running as an INT8 TensorFlow
Lite model in a background thread, so the CNN never blocks the motors.

## Folder layout

```
Webots_CNN_Project/
├── worlds/
│   ├── cnn_object_world.wbt     # main demo world: robot explores (controllerArgs "explore")
│   └── cnn_capture_tour.wbt     # evaluation world: camera tour + CNN benchmark (controllerArgs "tour")
├── controllers/epuck_vision_cnn/
│   ├── epuck_vision_cnn.py      # state machine, explore + tour modes
│   ├── colour_detection.py      # HSV segmentation, shape test, object ROI, annotation
│   ├── cnn_classifier.py        # TFLite/Keras inference + asynchronous worker thread
│   ├── runtime.ini              # tells Webots which Python to use
│   └── models/                  # cifar10_cnn.keras + float32/float16/int8 .tflite
├── notebook/
│   ├── CIFAR10_CNN_Webots_Colab.ipynb   # Google Colab notebook (training, evaluation, optimisation)
│   ├── own_images/              # the three "own" test photos (+ CREDITS.txt)
│   └── webots_images/           # robot-camera crops exported by the tour
├── captures/                    # images written by the controller (tour, scene shots, demo frames)
└── run_webots.sh                # launcher with the macOS fix (see below)
```

## Setup

1. **Webots R2025a**. On this Mac, Webots must be started with `./run_webots.sh` (see *macOS note*).
2. **Python 3.9+** with `tensorflow==2.16.2 opencv-python "numpy<2"`. Edit
   `controllers/epuck_vision_cnn/runtime.ini` so that `COMMAND` points to that Python.
3. Train the CNN in Colab (open `notebook/CIFAR10_CNN_Webots_Colab.ipynb` → *Runtime → Run all*),
   then copy the files of `models/` into `controllers/epuck_vision_cnn/models/`.
   Trained models are already included.
4. Run the demo: `./run_webots.sh worlds/cnn_object_world.wbt`
5. Run the evaluation: `./run_webots.sh worlds/cnn_capture_tour.wbt` (writes `captures/`, then quits).

### macOS note (why `run_webots.sh` exists)
On this macOS version, Qt 6 reports a refused TCP connection as *connected*. Webots probes
ports 1234–1244 before starting its controller server, concludes they are all "in use" and
exits (`failed to open TCP server in the port range`). `~/Applications/webots_fix/libconnect_fix.dylib`
is a 30-line `connect()` shim that checks `SO_ERROR`/`getpeername()` and returns the real
error. Qt's HTTP/2 downloads of textures were also unreliable, so
`~/Applications/webots_fix/prefetch_assets.py` fills the Webots asset cache with `curl`.

## Arena (world file)

| Item | Details |
|---|---|
| Arena | 3 × 3 m `RectangleArena`, light floor, 0.12 m walls |
| Coloured blocks | red, green, blue, yellow — **0.3 × 0.3 × 0.1 m** `Solid` boxes with bounding objects, inner ring r = 0.62 m |
| CIFAR objects (miniature) | Toyota Prius mesh ×2 at 7 % scale (automobile), Horse ×0.11, Dog ×0.32, Cat ×0.5, Deer ×0.12, Rubber ducks ×1.8 / ×1.2 (bird), Paper boat ×0.9 (ship), outer ring r = 1.15 m |
| Robot | e-puck, camera upgraded from 52×39 to **320×240**, anti-aliasing on, tilted up 0.12 rad |

## Hardware notes (e-puck)

* Differential drive, two stepper motors, max wheel speed **6.28 rad/s** (`left/right wheel motor`).
* 8 infrared proximity sensors `ps0`–`ps7`; `ps0`/`ps7` face forward and `ps2`/`ps5` face the sides. A reading > 80 is treated as "obstacle".
* Front colour camera (`camera`), field of view 0.84 rad.

## Controller design (`epuck_vision_cnn.py`)

Finite-state machine: **EXPLORE → APPROACH → REPORT → TURN_AWAY → … → DONE**

* `detect_colour_blocks()`: down-scaled frame (pre-filter), then HSV thresholds, then morphological opening, then the largest
  contour per colour. **Shape test**: solidity ≥ 0.90 and (extent ≥ 0.85 or ≤ 6 polygon corners),
  so a red car or a yellow duck is *not* reported as a block.
* HSV ranges (OpenCV units): red H 0–8 ∪ 172–179, yellow 20–34, green 40–85, blue 100–130; S ≥ 110–150, V ≥ 50–90.
* `request_cnn()`: the CNN runs only when the centre of the image is "salient" (not floor/wall), at most 2× per
  second, on an `object_roi()` square crop. It runs in `AsyncClassifier` (worker thread, queue length 1, frames
  dropped when busy), so `robot.step()` never waits for inference.
* `print_summary()`: lists every block found (with time) and every object class recognised by the CNN.
* Error handling: missing camera frames are skipped; no block in view → keep exploring; low-confidence CNN
  outputs (< 45 %) are not reported.

## Results
See `captures/tour/results.json`, `captures/explore_summary.json` and the notebook's `results/` folder.
