"""
epuck_vision_cnn.py - Applied Robotics & AI Project, Part 2 (Assignments 3-5)

An e-puck explores a miniature arena containing coloured blocks and CIFAR-10
style objects (cars, horse, dog, cat, deer, birds, ship). It

  * detects the RED / GREEN / BLUE / YELLOW blocks with an OpenCV HSV pipeline
    that also verifies block *shape* (so a red car is not mistaken for a block),
  * classifies other objects with a CIFAR-10 CNN running as an INT8 TFLite
    model in a background thread (the control loop never waits for the CNN),
  * avoids obstacles with the IR proximity sensors,
  * prints a summary of every block and object it has encountered.

Modes (Robot.controllerArgs in the world file):
  explore  - autonomous search demo (default world cnn_object_world.wbt)
  tour     - evaluation run (world cnn_capture_tour.wbt): the robot, which is a
             Supervisor, is placed in front of every object, saves the camera
             image, classifies it with each model variant and benchmarks
             latency; then it runs the explore demo while saving frames.

Hardware notes (e-puck): differential drive, max wheel speed 6.28 rad/s,
8 IR proximity sensors ps0..ps7 (ps0/ps7 at the front), front camera upgraded
from 52x39 to 320x240 in the world file for better CNN input.
"""
import json
import math
import os
import sys
import time

import cv2 as cv
import numpy as np
from controller import Supervisor

from colour_detection import annotate, detect_colour_blocks, object_roi, salient_object_ratio, webots_to_bgr
from cnn_classifier import CLASS_NAMES, AsyncClassifier, CnnClassifier

# ------------------------------------------------------------------ constants
MAX_SPEED = 6.28
CRUISE = 0.55 * MAX_SPEED
PROX_THRESHOLD = 80.0          # IR reading meaning "something is close"
BLOCK_MIN_FRAC = 0.01          # block visible enough to approach
BLOCK_REACHED_FRAC = 0.30      # block fills the view -> report it
SALIENCY_TRIGGER = 0.30        # centre of the image is "interesting" -> run CNN
CNN_MIN_CONF = 0.45            # below this, the prediction is reported as uncertain
CNN_PERIOD_S = 0.5             # rate limit: at most 2 CNN requests per second
TURN_AWAY_S = 1.2
BLOCK_COLOURS = ("red", "green", "blue", "yellow")

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "captures"))

# Objects visited in tour mode: (DEF name, CIFAR-10 ground-truth label, viewing distance in m)
TOUR_OBJECTS = [
    ("CAR_RED", "automobile", 0.45), ("CAR_BLUE", "automobile", 0.45),
    ("HORSE", "horse", 0.42), ("DOG", "dog", 0.50), ("CAT", "cat", 0.40),
    ("DEER", "deer", 0.36), ("BIRD_1", "bird", 0.42), ("SHIP", "ship", 0.40),
]


# ---------------------------------------------------------------- utilities
def look_at_rotation(eye, target):
    """Axis-angle rotation so that a Webots Viewpoint at `eye` looks at `target`."""
    f = np.array(target, float) - np.array(eye, float)
    f /= np.linalg.norm(f)
    left = np.cross([0, 0, 1], f)
    left /= np.linalg.norm(left)
    up = np.cross(f, left)
    m = np.column_stack([f, left, up])
    angle = math.acos(max(-1.0, min(1.0, (np.trace(m) - 1) / 2)))
    if abs(angle) < 1e-6:
        return [0, 0, 1, 0]
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    axis /= np.linalg.norm(axis)
    return [*axis.tolist(), angle]


class EpuckVisionController:
    def __init__(self):
        self.robot = Supervisor()
        self.dt = int(self.robot.getBasicTimeStep())

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.dt)
        self.W, self.H = self.camera.getWidth(), self.camera.getHeight()

        self.prox = [self.robot.getDevice(f"ps{i}") for i in range(8)]
        for s in self.prox:
            s.enable(self.dt)

        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        for m in (self.left_motor, self.right_motor):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

        print("Loading CNN (INT8 TFLite)...")
        self.cnn = AsyncClassifier(CnnClassifier("int8"))

        self.blocks_found = {c: None for c in BLOCK_COLOURS}   # colour -> time first reported
        self.objects_seen = {}                                 # CIFAR label -> (time, confidence)
        self.state = "EXPLORE"
        self.state_until = 0.0
        self.turn_dir = 1
        self.last_cnn_request = -1.0
        self.last_cnn = (None, 0.0)
        self.step_times = []
        self.frame_saver = None        # optional callback used in tour mode

    # ------------------------------------------------------------ helpers
    def set_speed(self, left, right):
        self.left_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, left)))
        self.right_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, right)))

    def get_frame(self):
        raw = self.camera.getImage()
        return webots_to_bgr(raw, self.W, self.H) if raw else None

    def read_ir(self):
        return [s.getValue() for s in self.prox]

    def step(self):
        return self.robot.step(self.dt) != -1

    def print_summary(self):
        n = sum(1 for c in BLOCK_COLOURS if self.blocks_found[c] is not None)
        print(f"--- Summary @ t={self.robot.getTime():.1f}s: {n}/{len(BLOCK_COLOURS)} blocks ---")
        for c in BLOCK_COLOURS:
            t = self.blocks_found[c]
            print(f"    {c:<7}: " + (f"found at t={t:.1f}s" if t is not None else "not yet found"))
        if self.objects_seen:
            objs = ", ".join(f"{k} ({v[1] * 100:.0f}%)" for k, v in self.objects_seen.items())
            print(f"    objects classified by CNN: {objs}")

    # ------------------------------------------------------------ behaviours
    def detect_red_block(self, detections):
        """Kept for readability/marking: True if a verified red block is visible."""
        d = detections.get("red")
        return bool(d and d.is_block)

    def choose_target(self, detections):
        """Largest verified block whose colour has not been reported yet."""
        cands = [d for c, d in detections.items()
                 if d.is_block and self.blocks_found[c] is None and d.area_frac >= BLOCK_MIN_FRAC]
        return max(cands, key=lambda d: d.area_frac) if cands else None

    def request_cnn(self, frame, now):
        """Event-triggered + rate-limited CNN call (non-blocking)."""
        if now - self.last_cnn_request < CNN_PERIOD_S:
            return
        if salient_object_ratio(frame) < SALIENCY_TRIGGER:
            return          # pre-filter: empty floor/wall view, don't waste inference
        if self.cnn.submit(frame, tag=now, roi=object_roi(frame)):
            self.last_cnn_request = now

    def handle_cnn_result(self):
        res = self.cnn.poll()
        if not res:
            return
        _, label, conf, _, ms = res
        self.last_cnn = (label, conf)
        if conf >= CNN_MIN_CONF and label not in self.objects_seen:
            self.objects_seen[label] = (self.robot.getTime(), conf)
            print(f"[CNN] I see a {label} ({conf * 100:.0f}% confident, {ms:.1f} ms)")
            self.print_summary()

    def explore_step(self, frame, ir):
        """One control step of the finite-state machine."""
        now = self.robot.getTime()
        detections = detect_colour_blocks(frame)
        front_left, front_right = max(ir[6], ir[7]), max(ir[0], ir[1])
        obstacle = max(ir[0], ir[1], ir[6], ir[7], ir[2] * 0.6, ir[5] * 0.6) > PROX_THRESHOLD

        self.request_cnn(frame, now)
        self.handle_cnn_result()

        if self.state == "DONE":
            self.set_speed(0, 0)
        elif self.state == "TURN_AWAY":
            self.set_speed(-self.turn_dir * 0.45 * MAX_SPEED, self.turn_dir * 0.45 * MAX_SPEED)
            if now >= self.state_until:
                self.state = "EXPLORE"
        else:
            target = self.choose_target(detections)
            if target is not None:
                self.state = "APPROACH"
                if target.area_frac >= BLOCK_REACHED_FRAC or (obstacle and target.area_frac > 0.12):
                    self.report_block(target.colour, frame, detections)
                else:
                    k = 0.6 * target.cx                   # steer towards the block centroid
                    self.set_speed(CRUISE * (1 + k), CRUISE * (1 - k))
            elif obstacle:
                self.turn_dir = 1 if front_right >= front_left else -1
                self.state = "TURN_AWAY"
                self.state_until = now + 0.5 + np.random.rand() * 0.6
            else:
                self.state = "EXPLORE"
                wobble = 0.15 * math.sin(now * 0.7)       # gentle sweeping to scan the arena
                self.set_speed(CRUISE * (1 + wobble), CRUISE * (1 - wobble))
        return detections

    def report_block(self, colour, frame, detections):
        self.blocks_found[colour] = self.robot.getTime()
        print(f"I see {colour}")
        self.print_summary()
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        cv.imwrite(os.path.join(CAPTURE_DIR, f"block_{colour}.png"),
                   annotate(frame, detections, *self.last_cnn, header=f"{colour} block reported"))
        self.turn_dir = -self.turn_dir
        self.state = "TURN_AWAY"
        self.state_until = self.robot.getTime() + TURN_AWAY_S
        if all(v is not None for v in self.blocks_found.values()):
            print("All blocks found - mission complete.")
            self.print_summary()
            self.state = "DONE"

    def run_explore(self, max_time=None):
        print("Explore mode: searching for red, green, blue and yellow blocks; CNN classifies objects.")
        while self.step():
            t0 = time.perf_counter()
            frame = self.get_frame()
            if frame is None:
                continue
            detections = self.explore_step(frame, self.read_ir())
            self.step_times.append((time.perf_counter() - t0) * 1000)
            if self.frame_saver:
                self.frame_saver(frame, detections)
            if max_time and self.robot.getTime() >= max_time:
                break
            if self.state == "DONE" and not self.frame_saver:
                break
        self.set_speed(0, 0)
        lat = self.cnn.latencies
        print(f"Control step: mean {np.mean(self.step_times):.2f} ms | CNN calls {len(lat)}, "
              f"mean {np.mean(lat) if lat else 0:.2f} ms, dropped {self.cnn.dropped}")


# ------------------------------------------------------------------ tour mode
def run_tour(ctl: EpuckVisionController):
    """Evaluation run: capture each object from the robot camera and classify it."""
    robot = ctl.robot
    tour_dir = os.path.join(CAPTURE_DIR, "tour")
    shots_dir = os.path.join(CAPTURE_DIR, "scene")
    demo_dir = os.path.join(CAPTURE_DIR, "demo_frames")
    for d in (tour_dir, shots_dir, demo_dir):
        os.makedirs(d, exist_ok=True)

    me = robot.getSelf()
    tr, rot = me.getField("translation"), me.getField("rotation")
    vp = robot.getFromDef("VIEWPOINT")
    vp_pos, vp_rot = vp.getField("position"), vp.getField("orientation")

    def view(eye, target, name, settle=6):
        """Export a 3D-view screenshot; the camera is disabled so its overlay is hidden."""
        ctl.camera.disable()
        vp_pos.setSFVec3f(list(eye))
        vp_rot.setSFRotation(look_at_rotation(eye, target))
        for _ in range(settle):
            ctl.step()
        robot.exportImage(os.path.join(shots_dir, f"{name}.jpg"), 95)
        for _ in range(3):
            ctl.step()
        ctl.camera.enable(ctl.dt)

    def place_robot(x, y, yaw):
        tr.setSFVec3f([x, y, 0.0])
        rot.setSFRotation([0, 0, 1, yaw])
        me.resetPhysics()

    ctl.step()
    # ---- 1. scene overview shots
    view((0.0, -3.0, 2.3), (0, 0, 0), "overview")
    view((2.4, -2.4, 1.6), (0, 0, 0), "overview_corner")
    view((0.0, -0.05, 4.2), (0, 0, 0), "top_down")
    view((-0.35, -0.55, 0.32), (0.1, 0.4, 0.05), "robot_closeup")
    view((-1.05, -1.35, 0.38), (0.35, 0.45, 0.02), "hero_low")
    view((1.3, -0.2, 0.55), (-0.4, 0.3, 0.0), "hero_side")
    if os.environ.get("SHOTS_ONLY"):
        for def_name, truth, dist in TOUR_OBJECTS:          # close-ups of each object for the slides
            node = robot.getFromDef(def_name)
            ox, oy, _ = node.getPosition()
            yaw = math.atan2(oy, ox)
            view((ox - 0.55 * math.cos(yaw) + 0.2 * math.sin(yaw), oy - 0.55 * math.sin(yaw) - 0.2 * math.cos(yaw), 0.28),
                 (ox, oy, 0.06), f"object_{def_name.lower()}")
        robot.simulationQuit(0)
        return

    # ---- 2. visit every object and classify the camera image with every backend
    backends = {b: CnnClassifier(b) for b in ("keras", "float32", "float16", "int8")}
    results = []
    for def_name, truth, dist in TOUR_OBJECTS:
        node = robot.getFromDef(def_name)
        ox, oy, _ = node.getPosition()
        yaw = math.atan2(oy, ox)
        rx, ry = ox - dist * math.cos(yaw), oy - dist * math.sin(yaw)
        place_robot(rx, ry, yaw)
        ctl.set_speed(0, 0)
        for _ in range(10):
            ctl.step()
        frame = ctl.get_frame()
        name = f"{def_name.lower()}__{truth}"
        cv.imwrite(os.path.join(tour_dir, f"{name}.png"), frame)
        roi = object_roi(frame)
        if roi:  # save the exact square crop the CNN sees (for the Colab notebook, section 8)
            x, y, sd, _ = roi
            os.makedirs(os.path.join(tour_dir, "crops"), exist_ok=True)
            cv.imwrite(os.path.join(tour_dir, "crops", f"{name}.png"), frame[y:y + sd, x:x + sd])
        entry = {"object": def_name, "truth": truth, "file": f"{name}.png", "roi": roi}
        # ablation: the same Keras model on a plain centre crop (no ROI)
        label_c, conf_c, _, _ = backends["keras"].classify(frame)
        entry["keras_centre_crop"] = {"pred": label_c, "conf": conf_c}
        for b, clf in backends.items():
            clf.classify(frame, roi)                 # warm-up
            lat = []
            for _ in range(30):
                label, conf, probs, ms = clf.classify(frame, roi)
                lat.append(ms)
            top3 = np.argsort(probs)[::-1][:3]
            entry[b] = {"pred": label, "conf": conf, "latency_ms": float(np.median(lat)),
                        "top3": [(CLASS_NAMES[i], float(probs[i])) for i in top3]}
        results.append(entry)
        label, conf = entry["int8"]["pred"], entry["int8"]["conf"]
        print(f"[TOUR] {def_name:<9} truth={truth:<10} int8 -> {label:<10} {conf * 100:5.1f}%  "
              f"(keras -> {entry['keras']['pred']})")
        shown = annotate(frame, detect_colour_blocks(frame), label, conf, header=f"robot camera | true: {truth}")
        if roi:
            x, y, sd, _ = roi
            cv.rectangle(shown, (x, y), (x + sd, y + sd), (255, 255, 255), 1)
        cv.imwrite(os.path.join(tour_dir, f"{name}_annotated.png"), shown)
        # 3rd-person shot of the robot looking at the object
        eye = (rx - 0.45 * math.cos(yaw) + 0.25 * math.sin(yaw),
               ry - 0.45 * math.sin(yaw) - 0.25 * math.cos(yaw), 0.30)
        view(eye, ((rx + ox) / 2, (ry + oy) / 2, 0.04), f"view_{def_name.lower()}", settle=3)

    # colour-block camera views for the colour-detection slides
    for colour, def_name in (("red", "RED_BLOCK"), ("green", "GREEN_BLOCK"),
                             ("blue", "BLUE_BLOCK"), ("yellow", "YELLOW_BLOCK")):
        ox, oy, _ = robot.getFromDef(def_name).getPosition()
        yaw = math.atan2(oy, ox)
        place_robot(ox - 0.45 * math.cos(yaw), oy - 0.45 * math.sin(yaw), yaw)
        for _ in range(10):
            ctl.step()
        frame = ctl.get_frame()
        dets = detect_colour_blocks(frame)
        cv.imwrite(os.path.join(tour_dir, f"block_{colour}_raw.png"), frame)
        cv.imwrite(os.path.join(tour_dir, f"block_{colour}_detected.png"),
                   annotate(frame, dets, header=f"HSV detection: {colour}"))
        d = dets.get(colour)
        print(f"[TOUR] block {colour}: detected={bool(d)} verified_block={bool(d and d.is_block)} "
              + (f"extent={d.extent:.2f} solidity={d.solidity:.2f} area={d.area_frac:.3f}" if d else ""))

    # check that coloured *objects* are rejected by the shape test
    shape_check = []
    for def_name, colour in (("CAR_RED", "red"), ("BIRD_1", "yellow"), ("CAR_BLUE", "blue")):
        f = cv.imread(os.path.join(tour_dir, f"{def_name.lower()}__"
                                   f"{dict((a, b) for a, b, _ in TOUR_OBJECTS)[def_name]}.png"))
        d = detect_colour_blocks(f).get(colour)
        shape_check.append({"object": def_name, "colour": colour, "detected": bool(d),
                            "accepted_as_block": bool(d and d.is_block),
                            "extent": d.extent if d else None, "solidity": d.solidity if d else None})
        print(f"[TOUR] shape test {def_name}: {shape_check[-1]}")

    json.dump({"objects": results, "shape_check": shape_check}, open(os.path.join(tour_dir, "results.json"), "w"), indent=1)

    # ---- 3. autonomous explore demo with frames saved for a GIF / video
    place_robot(0, 0, 0.3)
    for _ in range(5):
        ctl.step()
    vp_pos.setSFVec3f([0.0, -2.9, 2.6])
    vp_rot.setSFRotation(look_at_rotation((0.0, -2.9, 2.6), (0, 0, 0)))
    counter = {"n": 0}

    def saver(frame, detections):
        counter["n"] += 1
        if counter["n"] % 4 == 0:      # every 4 steps = 0.128 s simulated
            k = counter["n"] // 4
            robot.exportImage(os.path.join(demo_dir, f"scene_{k:04d}.jpg"), 80)
            cv.imwrite(os.path.join(demo_dir, f"cam_{k:04d}.jpg"),
                       annotate(frame, detections, *ctl.last_cnn, header=f"state: {ctl.state}"))

    ctl.frame_saver = saver
    ctl.run_explore(max_time=robot.getTime() + float(os.environ.get("DEMO_SECONDS", "90")))
    ctl.print_summary()
    json.dump({"blocks_found": ctl.blocks_found,
               "objects_seen": {k: list(v) for k, v in ctl.objects_seen.items()},
               "control_step_ms_mean": float(np.mean(ctl.step_times)),
               "cnn_latency_ms_mean": float(np.mean(ctl.cnn.latencies)) if ctl.cnn.latencies else None,
               "cnn_calls": len(ctl.cnn.latencies), "cnn_dropped": ctl.cnn.dropped},
              open(os.path.join(CAPTURE_DIR, "explore_summary.json"), "w"), indent=1)
    print("Tour finished.")
    robot.simulationQuit(0)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "explore"
    ctl = EpuckVisionController()
    if mode == "tour":
        run_tour(ctl)
    else:
        ctl.run_explore()


if __name__ == "__main__":
    main()
