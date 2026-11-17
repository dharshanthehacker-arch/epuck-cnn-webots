"""
colour_detection.py - OpenCV colour-block detector for the e-puck camera.

Goes beyond plain RGB thresholding:
  1. Pre-filter: the frame is down-scaled (320x240 -> 160x120) and blurred, which
     cuts the per-frame cost by ~4x and removes rendering noise.
  2. HSV segmentation: hue is robust to the shading on the block faces.
  3. Morphological opening removes speckles.
  4. Shape verification: a real block (0.3 x 0.3 x 0.1 m box) appears as a
     compact, nearly rectangular blob. Its "extent" (blob area / bounding-box
     area) and "solidity" (blob area / convex-hull area) are high, whereas a red
     car or a yellow rubber duck produces irregular blobs. This stops the robot
     from mistaking coloured CIFAR objects for blocks.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np

# HSV ranges in OpenCV units (H: 0-179, S: 0-255, V: 0-255).
# Red wraps around hue 0, so it needs two ranges.
HSV_RANGES: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
    "red":    [((0, 130, 70), (8, 255, 255)), ((172, 130, 70), (179, 255, 255))],
    "yellow": [((20, 130, 90), (34, 255, 255))],
    "green":  [((40, 110, 50), (85, 255, 255))],
    "blue":   [((100, 150, 50), (130, 255, 255))],
}

# BGR colours used when drawing annotations
DRAW_COLOURS = {
    "red": (40, 40, 230), "yellow": (0, 220, 240),
    "green": (40, 190, 40), "blue": (230, 90, 30),
}

DOWNSCALE = 0.5          # pre-filter: process a half-resolution frame
MIN_AREA_FRAC = 0.004    # ignore blobs smaller than 0.4 % of the image
MIN_EXTENT = 0.85        # blob area / bounding-rectangle area (axis-aligned face)
MIN_SOLIDITY = 0.90      # blob area / convex-hull area
MAX_VERTICES = 6         # a box seen from any angle is a polygon with <= 6 corners
KERNEL = np.ones((3, 3), np.uint8)


@dataclass
class Detection:
    colour: str
    area_frac: float               # fraction of the image covered by the blob
    cx: float                      # horizontal centroid in [-1 (left), +1 (right)]
    bbox: Tuple[int, int, int, int]  # x, y, w, h in full-resolution pixels
    is_block: bool                 # passed the shape-verification test
    extent: float
    solidity: float
    vertices: int


def webots_to_bgr(raw: bytes, width: int, height: int) -> np.ndarray:
    """Convert the Webots BGRA byte buffer into an OpenCV BGR image."""
    bgra = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
    return cv.cvtColor(bgra, cv.COLOR_BGRA2BGR)


def _colour_mask(hsv: np.ndarray, colour: str) -> np.ndarray:
    mask = None
    for lo, hi in HSV_RANGES[colour]:
        part = cv.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = part if mask is None else cv.bitwise_or(mask, part)
    return cv.morphologyEx(mask, cv.MORPH_OPEN, KERNEL)


def detect_colour_blocks(bgr: np.ndarray) -> Dict[str, Detection]:
    """Return the largest blob of each colour found in the frame."""
    full_h, full_w = bgr.shape[:2]
    small = cv.resize(bgr, None, fx=DOWNSCALE, fy=DOWNSCALE, interpolation=cv.INTER_AREA)
    small = cv.GaussianBlur(small, (3, 3), 0)
    hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    total = float(h * w)

    detections: Dict[str, Detection] = {}
    for colour in HSV_RANGES:
        mask = _colour_mask(hsv, colour)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        blob = max(contours, key=cv.contourArea)
        area = cv.contourArea(blob)
        if area / total < MIN_AREA_FRAC:
            continue
        x, y, bw, bh = cv.boundingRect(blob)
        hull_area = cv.contourArea(cv.convexHull(blob)) or 1.0
        extent = area / float(bw * bh)
        solidity = area / hull_area
        # polygon approximation: straight-edged boxes give few corners, curved objects many
        vertices = len(cv.approxPolyDP(blob, 0.02 * cv.arcLength(blob, True), True))
        m = cv.moments(blob)
        cx_px = m["m10"] / m["m00"] if m["m00"] else x + bw / 2
        s = 1.0 / DOWNSCALE
        detections[colour] = Detection(
            colour=colour,
            area_frac=area / total,
            cx=(cx_px - w / 2) / (w / 2),
            bbox=(int(x * s), int(y * s), int(bw * s), int(bh * s)),
            is_block=solidity >= MIN_SOLIDITY and (extent >= MIN_EXTENT or vertices <= MAX_VERTICES),
            extent=extent,
            solidity=solidity,
            vertices=vertices,
        )
    return detections


def salient_object_ratio(bgr: np.ndarray) -> float:
    """Cheap pre-filter for the CNN: how 'interesting' is the centre of the frame?

    The arena floor and walls are low-saturation, bright and texture-less.
    Returns the fraction of centre pixels that are NOT floor/wall-like. The CNN
    is only invoked when this is high enough, saving inference on empty views.
    """
    h, w = bgr.shape[:2]
    roi = bgr[h // 6: 5 * h // 6, w // 4: 3 * w // 4]
    roi = cv.resize(roi, (64, 64), interpolation=cv.INTER_AREA)
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    plain = (hsv[..., 1] < 40) & (hsv[..., 2] > 150)
    return 1.0 - float(plain.mean())


def object_roi(bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Find the object in front of the robot and return a square crop (x, y, side, side).

    Object pixels = edges (minus long horizontal floor/wall/sky boundaries) OR
    saturated OR dark pixels. The largest connected region gives the object's
    bounding box; a square around it mimics CIFAR-10 framing (object centred).
    """
    h, w = bgr.shape[:2]
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(cv.GaussianBlur(gray, (5, 5), 0), 50, 140)
    horiz = cv.morphologyEx(edges, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (25, 1)))
    edges = cv.subtract(edges, cv.dilate(horiz, KERNEL))
    mask = cv.dilate(edges, np.ones((5, 5), np.uint8))
    mask |= ((hsv[..., 1] > 60) | (hsv[..., 2] < 90)).astype(np.uint8) * 255
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, stats, _ = cv.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    x, y, bw, bh, _ = stats[i]
    side = int(min(max(bw, bh) * 1.05, h, w))
    cx = x + bw // 2
    bottom = min(h, y + bh + int(0.05 * side))
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    y0 = int(np.clip(bottom - side, 0, h - side))
    return x0, y0, side, side


def annotate(bgr: np.ndarray, detections: Dict[str, Detection],
             cnn_label: Optional[str] = None, cnn_conf: float = 0.0,
             header: Optional[str] = None) -> np.ndarray:
    """Draw detections and the latest CNN prediction on a copy of the frame."""
    out = bgr.copy()
    for d in detections.values():
        x, y, bw, bh = d.bbox
        col = DRAW_COLOURS[d.colour]
        cv.rectangle(out, (x, y), (x + bw, y + bh), col, 2)
        tag = f"{d.colour} {'block' if d.is_block else 'object'}"
        cv.putText(out, tag, (x + 2, max(12, y - 4)), cv.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv.LINE_AA)
    if cnn_label:
        text = f"CNN: {cnn_label} ({cnn_conf * 100:.0f}%)"
        cv.rectangle(out, (0, out.shape[0] - 22), (out.shape[1], out.shape[0]), (25, 25, 25), -1)
        cv.putText(out, text, (6, out.shape[0] - 7), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    if header:
        cv.putText(out, header, (6, 16), cv.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv.LINE_AA)
    return out
