"""Builds slide assets from the Webots captures: result grid, demo GIF, git log."""
import glob
import json
import subprocess
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CAP = ROOT / "captures"
OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)

# ---- 1. grid of robot-camera images with predictions (4 x 2)
res = json.load(open(CAP / "tour/results.json"))["objects"]
tiles = []
for o in res:
    img = cv.imread(str(CAP / "tour" / o["file"]))
    roi = o.get("roi")
    if roi:
        x, y, s, _ = roi
        cv.rectangle(img, (x, y), (x + s, y + s), (255, 255, 255), 2)
    p = o["int8"]
    ok = p["pred"] == o["truth"]
    bar = np.full((58, img.shape[1], 3), (40, 140, 40) if ok else (50, 60, 200), np.uint8)
    cv.putText(bar, f"pred: {p['pred']} ({p['conf'] * 100:.0f}%)", (8, 24), cv.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(bar, f"true: {o['truth']}", (8, 48), cv.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv.LINE_AA)
    tiles.append(np.vstack([img, bar]))
pad = lambda t: cv.copyMakeBorder(t, 6, 6, 6, 6, cv.BORDER_CONSTANT, value=(255, 255, 255))
rows = [np.hstack([pad(t) for t in tiles[i:i + 4]]) for i in range(0, len(tiles), 4)]
cv.imwrite(str(OUT / "webots_grid.jpg"), np.vstack(rows), [cv.IMWRITE_JPEG_QUALITY, 92])

# ---- 2. demo GIF: 3D scene + robot camera side by side
scenes = sorted(glob.glob(str(CAP / "demo_frames/scene_*.jpg")))
# trim: stop ~2 s after the robot comes to rest (mission complete -> scene stops changing)
small = [cv.resize(cv.imread(p, cv.IMREAD_GRAYSCALE), (96, 64)).astype(np.int16) for p in scenes]
diffs = [np.abs(small[i] - small[i - 1]).mean() for i in range(1, len(small))]
end = len(scenes)
for i in range(60, len(diffs) - 40):        # skip the first ~8 s (robot settling at start)
    if max(diffs[i:i + 40]) < 0.15:
        end = min(len(scenes), i + 16)
        break
print("demo frames used:", end, "of", len(scenes))
frames = []
for sp in scenes[:end][::2]:                       # every 2nd saved frame (~0.26 s simulated)
    cp = sp.replace("scene_", "cam_")
    sc, cm = cv.imread(sp), cv.imread(cp)
    if sc is None or cm is None:
        continue
    h = 360
    sc = cv.resize(sc, (int(sc.shape[1] * h / sc.shape[0]), h))
    cm = cv.resize(cm, (int(cm.shape[1] * h / cm.shape[0]), h))
    frame = np.hstack([sc, np.full((h, 8, 3), 255, np.uint8), cm])
    frames.append(Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))
if frames:
    target_w = 720
    frames = [f.resize((target_w, int(f.height * target_w / f.width)), Image.LANCZOS) for f in frames]
    frames = [f.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE) for f in frames]
    frames[0].save(OUT / "demo.gif", save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=True)
    print("gif frames", len(frames), "size MB", round((OUT / "demo.gif").stat().st_size / 1e6, 2))

# ---- 3. git log
log = subprocess.run(["git", "-C", str(ROOT), "log", "--oneline", "-12"], capture_output=True, text=True).stdout
(OUT / "git_log.txt").write_text(log)
print(log)
