"""
cnn_classifier.py - CIFAR-10 CNN inference for the e-puck controller.

Optimisations so that inference never stalls the robot's control loop:
  * TensorFlow Lite runtime with an INT8-quantised model (about 4x smaller and
    faster than the float Keras model), falling back to float16/float32/Keras.
  * AsyncClassifier runs inference in a background thread with a queue of
    length 1: the controller submits a frame and keeps stepping the simulation;
    if the worker is busy the new frame is simply dropped (no backlog).
  * Only a 32x32 crop is sent to the network (the CNN's native input size).
"""
import os
import queue
import threading
import time
from typing import Optional, Tuple

import cv2 as cv
import numpy as np

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def preprocess(bgr: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Crop (ROI or centre square), resize to 32x32, BGR->RGB, scale to [0, 1]."""
    h, w = bgr.shape[:2]
    if roi is not None:
        x, y, rw, rh = roi
        side = max(rw, rh)
        cx, cy = x + rw // 2, y + rh // 2
    else:
        side = min(h, w)
        cx, cy = w // 2, h // 2
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    crop = bgr[y0:y0 + side, x0:x0 + side]
    crop = cv.resize(crop, (32, 32), interpolation=cv.INTER_AREA)
    rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb[np.newaxis, ...]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class CnnClassifier:
    """Synchronous classifier. backend: 'int8' | 'float16' | 'float32' | 'keras'."""

    def __init__(self, backend: str = "int8", model_dir: str = MODEL_DIR, threads: int = 2):
        import tensorflow as tf  # imported lazily: slow to import
        self.backend = backend
        if backend == "keras":
            self.model = tf.keras.models.load_model(os.path.join(model_dir, "cifar10_cnn.keras"))
        else:
            path = os.path.join(model_dir, f"cifar10_cnn_{backend}.tflite")
            self.interpreter = tf.lite.Interpreter(model_path=path, num_threads=threads)
            self.interpreter.allocate_tensors()
            self.inp = self.interpreter.get_input_details()[0]
            self.out = self.interpreter.get_output_details()[0]

    def _forward(self, x: np.ndarray) -> np.ndarray:
        if self.backend == "keras":
            return np.asarray(self.model(x, training=False))[0]
        if self.inp["dtype"] == np.int8:  # full-integer model: quantise the input
            scale, zero = self.inp["quantization"]
            x = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
        self.interpreter.set_tensor(self.inp["index"], x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.out["index"])[0]
        if self.out["dtype"] == np.int8:
            scale, zero = self.out["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        return y

    def classify(self, bgr: np.ndarray, roi=None):
        """Return (label, confidence, probabilities, latency_ms)."""
        t0 = time.perf_counter()
        logits = self._forward(preprocess(bgr, roi))
        probs = softmax(np.asarray(logits, dtype=np.float32))
        ms = (time.perf_counter() - t0) * 1000.0
        k = int(np.argmax(probs))
        return CLASS_NAMES[k], float(probs[k]), probs, ms


class AsyncClassifier:
    """Runs CnnClassifier in a worker thread so robot.step() is never blocked."""

    def __init__(self, classifier: CnnClassifier):
        self.classifier = classifier
        self._jobs: "queue.Queue" = queue.Queue(maxsize=1)
        self._result = None
        self._lock = threading.Lock()
        self.latencies = []
        self.dropped = 0
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, bgr: np.ndarray, tag=None, roi=None) -> bool:
        """Queue a frame; returns False (frame dropped) if the worker is busy."""
        try:
            self._jobs.put_nowait((bgr.copy(), tag, roi))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def poll(self):
        """Return the newest (tag, label, conf, probs, ms) once, or None."""
        with self._lock:
            res, self._result = self._result, None
        return res

    def busy(self) -> bool:
        return not self._jobs.empty()

    def _worker(self):
        while True:
            bgr, tag, roi = self._jobs.get()
            label, conf, probs, ms = self.classifier.classify(bgr, roi)
            self.latencies.append(ms)
            with self._lock:
                self._result = (tag, label, conf, probs, ms)
