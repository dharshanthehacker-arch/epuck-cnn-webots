"""Builds the Assignment 3 presentation from the project's real outputs."""
import json
import sys
from pathlib import Path

from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN

sys.path.insert(0, str(Path(__file__).parent))
from deck_lib import *  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent
NB, CAP, ASSETS = ROOT / "notebook", ROOT / "captures", Path(__file__).parent / "assets"
J = lambda p: json.load(open(p))

M = J(NB / "results/metrics.json")
HIST = J(NB / "results/history.json")
LAYERS = J(NB / "results/layers.json")
OWN = J(NB / "results/own_predictions.json")
OPT = J(NB / "results/optimisation.json")
TOUR = J(CAP / "tour/results.json")
EXP = J(CAP / "explore_summary.json")
GIT = (ASSETS / "git_log.txt").read_text().strip().splitlines()

acc = M["test_acc"] * 100
base = M["baseline_test_acc"] * 100
objs = TOUR["objects"]
tour_ok = sum(o["int8"]["pred"] == o["truth"] for o in objs)
tour_ok_centre = sum(o["keras_centre_crop"]["pred"] == o["truth"] for o in objs)
PLAIN = J(CAP / "plain_floor_run/tour_results.json")["objects"]
plain_ok = sum(o["int8"]["pred"] == o["truth"] for o in PLAIN)
NICE = {"CAR_RED": "red car", "CAR_BLUE": "blue car", "HORSE": "horse", "DOG": "dog", "CAT": "cat",
        "DEER": "deer", "BIRD_1": "duck (bird)", "SHIP": "paper boat"}
right_list = ", ".join(NICE[o["object"]] for o in objs if o["int8"]["pred"] == o["truth"])
wrong_list = ", ".join(f"{NICE[o['object']]} → {o['int8']['pred']}" for o in objs if o["int8"]["pred"] != o["truth"])
flipped = [NICE[o["object"]] for o, p in zip(objs, PLAIN) if (o["int8"]["pred"] == o["truth"]) != (p["int8"]["pred"] == p["truth"])]
k, i8 = OPT["Keras (float32)"], OPT["TFLite int8"]
speedup = k["latency_ms"] / i8["latency_ms"]
f32 = OPT["TFLite float32"]
shrink = f32["size_mb"] / i8["size_mb"]
OWN_TRUTH = {"1_car.jpg": "automobile", "2_bird.jpg": "bird", "3_ship.jpg": "ship"}
own_ok = sum(o["pred"] == OWN_TRUTH.get(o["file"]) for o in OWN)
own_naive_ok = sum(o.get("naive_pred") == OWN_TRUTH.get(o["file"]) for o in OWN)
blocks_found = sum(v is not None for v in EXP["blocks_found"].values())

prs = new_deck()
n = 0


def nxt(dark=False):
    global n
    n += 1
    s = blank(prs, dark)
    footer(s, n, dark)
    return s


def section(num, t, sub, img):
    global n
    n += 1
    s = divider(prs, num, t, sub, img)
    footer(s, n, True)
    return s


# ======================================================================= 1. Title
s = nxt(dark=True)
cover_image(s, CAP / "scene/hero_low.jpg", 0, 0, 13.333, 7.5, name="Hero image")
s.shapes.add_picture(str(ASSETS / "hero_fade.png"), 0, 0, W, H).name = "Hero fade"
pill = box(s, 0.6, 1.2, 4.6, 0.42, fill=AMBER, radius=0.5, shadow=False)
shape_text(pill, "APPLIED ROBOTICS & AI  ·  PROJECT PART 2", size=12, bold=True, color=NAVY)
text(s, 0.6, 1.95, 6.4, 2.6, "Teaching an e-puck to see: CNN object classification and colour-block detection in Webots",
     size=36, bold=True, color=WHITE, font=HEAD, line_spacing=1.0)
text(s, 0.6, 4.55, 5.8, 0.9, "CIFAR-10 CNN trained off-robot in Google Colab, deployed as an INT8 TFLite model "
     "inside a Webots e-puck controller", size=16, color=LINE)
text(s, 0.6, 5.9, 5.8, 0.8, [[("[Your name]  ·  [Student ID]", {"bold": True, "color": WHITE})],
                              "BEng (Hons) Robotics & AI  ·  University of Hertfordshire  ·  18 Nov 2026"],
     size=14, color=LINE)
notes(s, "Introduce the project: Part 2 continues the Part 1 colour-block search. Today: the CNN built for "
         "CIFAR-10, how it runs on the robot without slowing it down, and a demo of colour detection in Webots.")

# ======================================================================= 2. At a glance
s = nxt()
title(s, "Project at a glance", "Overview")
stats = [(f"{acc:.1f}%", "CIFAR-10 test accuracy", f"baseline cnn_v3: {base:.1f}%"),
         (f"{tour_ok}/{len(objs)}", "Webots camera images", "classified correctly (INT8 model)"),
         (f"{speedup:.1f}×", "faster inference", f"INT8 TFLite vs Keras, {shrink:.1f}× smaller"),
         (f"{blocks_found}/4", "colour blocks found", f"in {max(EXP['blocks_found'].values()):.0f} s; car & duck rejected")]
for i, (big, lab, sub) in enumerate(stats):
    x = 0.6 + i * 3.08
    box(s, x, 1.75, 2.85, 2.55, fill=TEAL_LT)
    text(s, x + 0.25, 1.95, 2.4, 1.1, big, size=48, bold=True, color=TEAL, font=HEAD)
    text(s, x + 0.25, 3.05, 2.4, 0.45, lab, size=16, bold=True, color=NAVY)
    text(s, x + 0.25, 3.5, 2.45, 0.7, sub, size=13, color=MUTED)
text(s, 0.6, 4.75, 4, 0.4, "What this talk covers", size=18, bold=True, color=NAVY)
agenda = ["Background & aims", "PM & code management", "CNN build & training", "Results: images & Webots",
          "Colour-detection demo", "Optimisation & future work"]
for i, a in enumerate(agenda):
    x = 0.6 + (i % 3) * 4.1
    y = 5.3 + (i // 3) * 0.75
    circle_num(s, x, y, 0.5, str(i + 1), size=13)
    text(s, x + 0.65, y + 0.06, 3.3, 0.4, a, size=16, color=INK)
notes(s, "Headline numbers first, then the agenda.")

section("01", "Background, aims & project management", "Why a small robot needs a CNN, what we set out to measure, and how the work was managed.",
        CAP / "scene/hero_side.jpg")
# ======================================================================= 3. Background & motivation
s = nxt()
title(s, "Why put a CNN on a small robot?", "Background & motivation")
bullets(s, 0.6, 1.65, 6.3, 4.6, [
    ("Part 1 recap: ", "the e-puck searched an arena and reported red, green and blue blocks using only colour thresholds."),
    ("The gap: ", "colour alone cannot tell a red block from a red car. Robots need to know what an object is, not just its colour."),
    ("CNNs are the standard tool ", "for image classification (LeCun et al., 1998; Krizhevsky et al., 2012), but they are compute-hungry."),
    ("Real robot constraint: ", "the e-puck controller steps every 32 ms. A slow network call would freeze the motors and sensors."),
    ("Brief: ", "train the CNN off-robot (Colab), show it working on the robot camera, and propose how to optimise it."),
], size=16, space=10)
image(s, CAP / "scene/robot_closeup.jpg", 7.3, 1.65, 5.45, 4.2)
text(s, 7.3, 5.95, 5.45, 0.5, "The e-puck in the Part 2 arena (Webots R2025a)", size=12, color=MUTED, italic=True,
     align=PP_ALIGN.CENTER)
notes(s, "Motivation: Part 1 used raw colour. Part 2 adds learned object recognition without breaking real-time control.")

# ======================================================================= 4. Aims & objectives
s = nxt()
title(s, "Aims and measurable objectives", "Background & motivation")
rows = [("O1", "Build a CNN for CIFAR-10 and show all its layers", f"{len(LAYERS)} layers, {M['params']:,} parameters", True),
        ("O2", "Reach ≥ 80 % test accuracy (baseline ≈ 70 %)", f"{acc:.1f}% achieved", acc >= 80),
        ("O3", "Predict three images of my own", f"{own_ok}/3 correct", True),
        ("O4", "Predict an image taken by the robot's camera in Webots", f"{tour_ok}/{len(objs)} objects correct", True),
        ("O5", "Optimise so robot control is not affected", f"{i8['latency_ms']:.2f} ms/inference, 0 dropped", True),
        ("O6", "Demonstrate detection of coloured objects in Webots", f"{blocks_found}/4 blocks found", blocks_found >= 3)]
for i, (code, aim, res, ok) in enumerate(rows):
    y = 1.65 + i * 0.82
    box(s, 0.6, y, 12.1, 0.68)
    circle_num(s, 0.75, y + 0.09, 0.5, code, size=12, fill=NAVY)
    text(s, 1.45, y + 0.14, 6.9, 0.45, aim, size=16, color=INK)
    text(s, 8.5, y + 0.14, 3.2, 0.45, res, size=15, bold=True, color=TEAL)
    circle_num(s, 11.95, y + 0.12, 0.44, "✓" if ok else "~", fill=TEAL if ok else AMBER, size=14)
notes(s, "Each objective is measurable, and each result here is taken directly from the notebook or the Webots run.")

# ======================================================================= 5. Project management
s = nxt()
title(s, "Project management: our Trello board", "Methods · project management")
image(s, ASSETS / "trello_board.png", 0.6, 1.55, 8.1, 4.95)
text(s, 0.6, 6.6, 8.1, 0.35, "Live board: trello.com/b/2ao7fOlv  ·  Done 8  ·  In review 2  ·  To do 3", size=12,
     italic=True, color=MUTED, align=PP_ALIGN.CENTER)
text(s, 9.1, 1.65, 3.6, 0.4, "Milestones", size=18, bold=True, color=NAVY)
ms = [("10 Nov", "Part 1 interim report"), ("17 Nov", "CNN trained, Webots tour"), ("18 Nov", "Presentation (A3)"),
      ("21 Nov", "Report + Webots code (A4/A5)")]
for j, (d, t) in enumerate(ms):
    y = 2.2 + j * 1.0
    circle_num(s, 9.1, y, 0.42, "", fill=TEAL if j < 2 else AMBER)
    text(s, 9.7, y - 0.04, 3.0, 0.3, d, size=14, bold=True, color=INK)
    text(s, 9.7, y + 0.27, 3.0, 0.4, t, size=13, color=MUTED)
box(s, 9.1, 6.15, 3.6, 0.65, fill=TEAL_LT, shadow=False)
text(s, 9.25, 6.2, 3.4, 0.55, "Kanban workflow: one card per deliverable, moved when the code is committed.",
     size=12, color=INK)
notes(s, "Our real Trello board (Kanban): Done / In review / To do. Each card is a deliverable; cards move when the matching commit lands on GitHub.")

# ======================================================================= 6. Code management
s = nxt()
title(s, "Code management with Git and GitHub", "Methods · code management")
image(s, ASSETS / "github_repo.png", 0.6, 1.55, 7.9, 3.36)
text(s, 0.6, 5.05, 7.9, 0.35, "github.com/dharshanthehacker-arch/epuck-cnn-webots  ·  public repository",
     size=13, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
short = lambda m: m if len(m) <= 60 else m[:59].rstrip() + "…"
box(s, 0.6, 5.55, 7.9, 1.3, fill=NAVY)
text(s, 0.85, 5.65, 7.5, 1.15, [[(l[:7] + "  ", {"color": AMBER}), (short(l[8:]), {"color": WHITE})] for l in GIT[:4]],
     size=11.5, font="Courier New", line_spacing=1.2)
bullets(s, 8.9, 1.6, 3.85, 5.2, [
    ("One commit per feature: ", "world → detection → CNN → controller → notebook → results."),
    ("Modular repo: ", "worlds/, controllers/, notebook/, captures/, presentation/."),
    ("README ", "covers setup, e-puck hardware notes and the macOS launcher."),
    (".gitignore ", "keeps the 180 MB dataset and generated frames out."),
    ("Models versioned ", "with the code that produced them."),
], size=14, space=10)
notes(s, "Live public repository on GitHub. Show the commit history: each feature landed as its own commit.")

section("02", "Methods: world, pipeline & CNN", "A miniature CIFAR-10 world in Webots, a train-off-robot / infer-on-robot pipeline, and the network itself.",
        CAP / "scene/object_horse.jpg")
# ======================================================================= 7. The Webots arena
s = nxt()
title(s, "The Webots arena: miniature CIFAR-10 world", "Methods · environment")
image(s, CAP / "scene/overview.jpg", 0.6, 1.6, 7.6, 5.15)
items = [("Vehicles", "2 × Toyota Prius mesh at 7 % scale (automobile)"),
         ("Animals", "horse, dog, cat, deer (scaled Webots models)"),
         ("Birds", "2 rubber ducks (yellow, white)"),
         ("Ship", "paper boat"),
         ("Blocks", "red, green, blue, yellow, 0.3 × 0.3 × 0.1 m"),
         ("Robot", "e-puck, camera upgraded 52×39 → 320×240")]
for j, (h1, d) in enumerate(items):
    y = 1.65 + j * 0.85
    text(s, 8.55, y, 4.2, 0.3, h1, size=15, bold=True, color=TEAL)
    text(s, 8.55, y + 0.3, 4.2, 0.5, d, size=14, color=INK)
notes(s, "Every object maps to a CIFAR-10 class, so the network can be tested on what the robot actually sees.")

# ======================================================================= 8. System architecture
s = nxt()
title(s, "System architecture: train off-robot, infer on-robot", "Methods · pipeline")
text(s, 0.6, 1.55, 5, 0.35, "OFF-ROBOT · GOOGLE COLAB (GPU)", size=12, bold=True, color=MUTED)
off = [("CIFAR-10\n50k train / 10k test", PANEL), ("Augment\nflip · shift", PANEL), ("Train CNN\nAdam · BN · Dropout", TEAL_LT),
       ("Convert\nTFLite INT8", TEAL_LT)]
for j, (t, f) in enumerate(off):
    b = box(s, 0.6 + j * 3.1, 1.95, 2.6, 1.0, fill=f)
    shape_text(b, t, size=14, color=NAVY, bold=False)
    if j:
        arrow(s, 0.6 + j * 3.1 - 0.45, 2.45, 0.6 + j * 3.1 - 0.05, 2.45)
arrow(s, 10.2, 2.95, 10.2, 3.7, color=AMBER, width=2.5)
text(s, 10.35, 3.1, 2.5, 0.4, "cifar10_cnn_int8.tflite", size=12, italic=True, color=AMBER)
text(s, 0.6, 3.6, 5, 0.35, "ON-ROBOT · WEBOTS CONTROLLER (32 ms STEP)", size=12, bold=True, color=MUTED)
on = [("Camera\n320×240 BGRA", PANEL), ("Pre-filter\ndownscale · saliency", PANEL), ("HSV + shape test\ncolour blocks", PANEL),
      ("FSM\nexplore · approach · report", NAVY)]
for j, (t, f) in enumerate(on):
    b = box(s, 0.6 + j * 3.1, 4.0, 2.6, 1.0, fill=f)
    shape_text(b, t, size=14, color=WHITE if f == NAVY else NAVY)
    if j:
        arrow(s, 0.6 + j * 3.1 - 0.45, 4.5, 0.6 + j * 3.1 - 0.05, 4.5)
b = box(s, 6.8, 5.55, 3.4, 1.0, fill=TEAL)
shape_text(b, "Async CNN worker thread\nROI crop → INT8 → label", size=14, color=WHITE)
arrow(s, 4.4, 5.0, 6.75, 5.85, color=TEAL, width=2)
arrow(s, 10.25, 5.85, 11.2, 5.05, color=TEAL, width=2)
text(s, 0.6, 5.65, 4.1, 1.0, "Frames are handed over without waiting. If the worker is busy, the frame is dropped "
     "and robot.step() never blocks.", size=13, color=MUTED, italic=True)
notes(s, "Two halves: heavy training in Colab; light, non-blocking inference in the controller.")

# ======================================================================= 9. CNN architecture (all layers)
s = nxt()
title(s, "CNN architecture: every layer", "Methods · CNN build")
groups = [("Block 1", 0, 8, "2× Conv 3×3, 32 filters\nBN · ReLU · MaxPool · Dropout 0.2"),
          ("Block 2", 8, 16, "2× Conv 3×3, 64 filters\nBN · ReLU · MaxPool · Dropout 0.3"),
          ("Block 3", 16, 24, "2× Conv 3×3, 128 filters\nBN · ReLU · MaxPool · Dropout 0.4"),
          ("Head", 24, len(LAYERS), "Flatten · Dense 128 · BN · ReLU\nDropout 0.5 · Dense 10 (logits)")]
widths = [1.0, 0.92, 0.84, 0.76]
for j, (g, a, b_, d) in enumerate(groups):
    y = 1.7 + j * 1.25
    out = LAYERS[b_ - 1]["output"] if b_ - 1 < len(LAYERS) else ""
    p = sum(L["params"] for L in LAYERS[a:b_])
    bw = 2.9 * widths[j] + 0.6
    bb = box(s, 0.6, y, bw, 1.0, fill=[TEAL_LT, TEAL_LT, TEAL_LT, PANEL][j])
    shape_text(bb, [[(g, {"bold": True, "color": NAVY, "size": 14})], [(d, {"size": 12, "color": INK})]], align=PP_ALIGN.LEFT)
    text(s, 0.6 + bw + 0.12, y + 0.2, 1.5, 0.6, [[(f"out {out}", {"bold": True})], f"{p:,} params"], size=12, color=MUTED)
# full layer table, split in two columns so all layers fit
half = (len(LAYERS) + 1) // 2
for part, chunk in enumerate([LAYERS[:half], LAYERS[half:]]):
    rows_t = len(chunk) + 1
    gx = 5.75 + part * 3.55
    tshape = s.shapes.add_table(rows_t, 4, Inches(gx), Inches(1.6), Inches(3.4), Inches(0.3 * rows_t))
    tbl = tshape.table
    for c, (hd, wd) in enumerate([("#", 0.3), ("Layer", 1.3), ("Output", 1.05), ("Params", 0.75)]):
        tbl.columns[c].width = Inches(wd)
        tbl.cell(0, c).text = hd
    for r, L in enumerate(chunk, 1):
        lname = L["layer"].replace("BatchNormalization", "BatchNorm").replace("MaxPooling2D", "MaxPool2D")
        for c, v in enumerate([str(L["#"]), lname, L["output"].replace(" ", ""), f"{L['params']:,}"]):
            tbl.cell(r, c).text = v
    for r in range(rows_t):
        tbl.rows[r].height = Inches(0.3)
        for c in range(4):
            cell = tbl.cell(r, c)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.margin_left = cell.margin_right = Inches(0.05)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if r == 0 else (TEAL_LT if r % 2 else WHITE)
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)
                    run.font.name = BODY
                    run.font.bold = r == 0
                    run.font.color.rgb = WHITE if r == 0 else INK
text(s, 0.6, 6.85, 11, 0.4, f"Total: {M['params']:,} trainable parameters  ·  input 32×32×3  ·  output 10 logits "
     f"(baseline cnn_v3: {M['baseline_params']:,})", size=13, bold=True, color=NAVY)
notes(s, "All layers are shown in the table (from model.summary()). Why: VGG-style 3×3 stacks, BatchNorm for stable "
         "training, dropout increasing with depth against over-fitting.")

# ======================================================================= 10. Training
s = nxt()
title(s, "Training pipeline and learning curves", "Methods · training")
image(s, NB / "figures/training_curves.png", 0.6, 1.6, 8.0, 3.0)
specs = [("Data split", "45k train · 5k validation · 10k test (test used once)"),
         ("Augmentation", "random horizontal flip, ±10 % shift"),
         ("Optimiser", "Adam, lr 1e-3, batch 64, ReduceLROnPlateau ×0.5"),
         ("Stopping", f"early stopping on val. accuracy: {M['epochs']} epochs"),
         ("Loss", "sparse categorical cross-entropy (from logits)")]
for j, (a, b_) in enumerate(specs):
    y = 1.65 + j * 0.62
    text(s, 8.95, y, 3.8, 0.28, a, size=13, bold=True, color=TEAL)
    text(s, 8.95, y + 0.27, 3.8, 0.35, b_, size=13, color=INK)
image(s, NB / "figures/cifar_samples.png", 0.6, 4.85, 8.0, 1.95)
box(s, 8.95, 5.0, 3.8, 1.7, fill=TEAL_LT)
text(s, 9.15, 5.12, 3.45, 1.5, [[("Baseline problem: ", {"bold": True})],
                                "cnn_v3 validated on the test set and over-fitted (train ≫ val). "
                                "The new split, BatchNorm, dropout and augmentation fix both."],
     size=13, color=INK)
notes(s, "Train/validation curves stay close: little over-fitting. Grey dashed line = baseline validation accuracy.")

section("03", "Key findings & live demo", "Accuracy, predictions on my own photos and on the robot's camera, and colour-block detection running in Webots.",
        CAP / "scene/object_car_red.jpg")
# ======================================================================= 11. Accuracy
s = nxt()
title(s, "Accuracy score: test set and per class", "Key findings")
box(s, 0.6, 1.65, 3.4, 2.4, fill=NAVY)
text(s, 0.85, 1.8, 3.0, 1.1, f"{acc:.1f}%", size=54, bold=True, color=WHITE, font=HEAD)
text(s, 0.85, 2.95, 3.0, 1.0, [[("Test accuracy", {"bold": True, "color": AMBER})],
                               f"10,000 unseen images (baseline {base:.1f}%, +{acc - base:.1f} pts)"], size=14, color=LINE)
pc = M["per_class"]
bar_chart(s, 0.45, 4.2, 3.75, 2.9, ["Baseline", "Improved"], [("Test accuracy", [base / 100, acc / 100])],
          [MUTED], number_format='0%', val_min=0, val_max=1, label_size=11)
ch = bar_chart(s, 4.3, 1.55, 4.6, 5.45, list(pc.keys()), [("Per-class accuracy", [v for v in pc.values()])],
               [TEAL], title_text="Per-class accuracy (test set)", number_format='0%', horizontal=True,
               val_min=0, val_max=1, label_size=11, gap=40)
image(s, NB / "figures/confusion_matrix.png", 9.05, 1.6, 3.7, 3.4)
worst = sorted(pc.items(), key=lambda kv: kv[1])[:2]
tc = M["top_confusions"][0]
text(s, 9.05, 5.15, 3.7, 1.8, [[("Analysis: ", {"bold": True})],
     f"weakest classes are {worst[0][0]} ({worst[0][1]*100:.0f}%) and {worst[1][0]} ({worst[1][1]*100:.0f}%). "
     f"Most common error: {tc[0]} → {tc[1]} ({tc[2]}×), similar shapes at 32×32."], size=13, color=INK)
notes(s, "Vehicles score highest; fine-grained animal classes (cat/dog) are hardest, as in the literature.")

# ======================================================================= 12. Own images
s = nxt()
title(s, "Predictions on three images of my own", "Key findings")
image(s, NB / "figures/own_predictions.png", 0.6, 1.55, 8.3, 5.3)
for j, o in enumerate(OWN):
    y = 1.6 + j * 1.3
    box(s, 9.25, y, 3.5, 1.15)
    t3 = ", ".join(f"{c} {p*100:.0f}%" for c, p in o["top3"][1:])
    text(s, 9.45, y + 0.1, 3.2, 1.25, [[(o["file"].split("_", 1)[1].rsplit(".", 1)[0].capitalize() + " photo → ", {"bold": True}),
                                         (f"{o['pred']} ({o['conf']*100:.0f}%)", {"bold": True, "color": TEAL})],
                                        [(f"runners-up: {t3}", {"size": 12, "color": MUTED})]], size=15, color=INK)
box(s, 9.25, 5.55, 3.5, 1.25, fill=NAVY)
text(s, 9.45, 5.65, 3.15, 1.1, [[("Pre-processing matters: ", {"bold": True}), (f"cnn_v3's nearest-neighbour resize got {own_naive_ok}/3; anti-aliased (area) resize gets {own_ok}/3.", {})]], size=13, color=WHITE)
text(s, 0.6, 6.85, 8.3, 0.3, "Photos: Wikimedia Commons (CC0 / public domain / CC BY-SA 4.0), downsampled to 32×32 with area interpolation.",
     size=10, color=MUTED, italic=True)
notes(s, "Full-resolution photos are downsampled to 32×32 before prediction. Top-3 probabilities show how confident the model is.")

# ======================================================================= 13. Webots camera predictions
s = nxt()
title(s, "Prediction on images from the robot's camera", "Key findings")
grid = ASSETS / "webots_grid.jpg"
image(s, grid, 0.6, 1.55, 8.3, 5.35)
box(s, 9.25, 1.65, 3.5, 1.9, fill=NAVY)
text(s, 9.45, 1.75, 3.2, 0.9, f"{tour_ok}/{len(objs)}", size=44, bold=True, color=WHITE, font=HEAD)
text(s, 9.45, 2.65, 3.2, 0.85, "Webots objects classified correctly (INT8 model on ROI crop)", size=13, color=LINE)
text(s, 9.25, 3.7, 3.5, 3.3, [[("Correct: ", {"bold": True, "color": TEAL}), (right_list + ".", {})],
     [("Wrong: ", {"bold": True, "color": RED}), (wrong_list + ".", {})],
     [("Background matters: ", {"bold": True}), (f"plain floor {plain_ok}/{len(PLAIN)} vs chessboard {tour_ok}/{len(objs)}"
      + (f" ({', '.join(flipped)} flipped)." if flipped else "."), {})],
     [("Why: ", {"bold": True}), ("domain gap. Low-angle renders of toy models are not CIFAR photos.", {})]],
     size=12.5, color=INK)
notes(s, "Each tile is a real e-puck camera frame. Green = correct, red = wrong. The white square is the ROI the CNN sees.")

# ======================================================================= 14. Colour detection demo
s = nxt()
title(s, "Demo: Webots detecting coloured objects", "Key findings · demo")
gif = ASSETS / "demo.gif"
image(s, gif, 0.6, 1.45, 12.1, 3.85, rounded=False)
for j, c in enumerate(["red", "green", "blue", "yellow"]):
    image(s, CAP / f"tour/block_{c}_detected.png", 0.6 + j * 2.0, 5.55, 1.85, 1.39)
sc = TOUR["shape_check"]
rej = [x["object"].replace("_", " ").lower() for x in sc if x["detected"] and not x["accepted_as_block"]]
text(s, 8.75, 5.5, 4.0, 1.5, [[("Live run: ", {"bold": True}), ("3D view + robot camera (animated in slideshow). ", {})],
     [("Beyond thresholding: ", {"bold": True}), ("HSV mask → contour → shape test. ", {}),
      (f"Rejected look-alikes: {', '.join(rej)}.", {})]], size=13, color=INK)
notes(s, "Play the slideshow: the GIF animates. Explain the state machine: EXPLORE → APPROACH → REPORT → TURN_AWAY. "
         "Prints 'I see red' and a summary of blocks found.")

section("04", "Optimisation & future work", "Keeping the robot responsive, and how this becomes full object detection.",
        CAP / "scene/object_bird_1.jpg")
# ======================================================================= 15. Optimisation proposal
s = nxt()
title(s, "Optimisation: keeping the robot responsive", "Methods · optimisation proposal")
names = list(OPT.keys())
bar_chart(s, 0.45, 1.5, 4.3, 3.0, [x.replace("TFLite ", "") for x in names],
          [("Latency (ms)", [OPT[x]["latency_ms"] for x in names])], [TEAL],
          title_text="Latency per image (ms)", number_format='0.00', label_size=10)
bar_chart(s, 4.65, 1.5, 4.3, 3.0, [x.replace("TFLite ", "") for x in names],
          [("Size (MB)", [OPT[x]["size_mb"] for x in names])], [NAVY],
          title_text="Model size (MB)", number_format='0.00', label_size=10)
text(s, 0.6, 4.65, 8.3, 0.5, [[("Accuracy kept: ", {"bold": True}),
     (f"Keras {k['acc']*100:.2f}% → INT8 {i8['acc']*100:.2f}% on 10,000 test images", {})]], size=14, color=INK)
steps = [("1", "INT8 quantisation", f"{shrink:.1f}× smaller, {speedup:.1f}× faster"),
         ("2", "Asynchronous worker", "control loop never waits for the CNN"),
         ("3", "Event-triggered", "runs only on salient views, ≤ 2 Hz"),
         ("4", "ROI crop", "only a 32×32 crop of the object")]
for j, (nm, a, b_) in enumerate(steps):
    y = 1.6 + j * 1.3
    circle_num(s, 9.3, y, 0.55, nm, fill=AMBER, color=NAVY)
    text(s, 10.0, y - 0.02, 2.8, 0.35, a, size=15, bold=True, color=NAVY)
    text(s, 10.0, y + 0.33, 2.8, 0.6, b_, size=13, color=MUTED)
box(s, 0.6, 5.3, 8.3, 1.45, fill=TEAL_LT)
text(s, 0.8, 5.42, 7.9, 1.25, [[("Measured in Webots: ", {"bold": True})],
     f"mean control step {EXP['control_step_ms_mean']:.1f} ms (budget 32 ms) while the CNN ran "
     f"{EXP['cnn_calls']} times in the background (mean {EXP['cnn_latency_ms_mean']:.1f} ms each, "
     f"{EXP['cnn_dropped']} frames dropped instead of queued)."], size=14, color=INK)
notes(s, "Proposal = quantise + run asynchronously + only when needed + on a crop. Next step: pruning and an edge TPU.")

# ======================================================================= 16. Proposal: detection in Webots
s = nxt()
title(s, "Proposal: from classification to object detection in Webots", "Future work")
props = [("A", "Region proposals + CNN (implemented as a prototype)",
          "OpenCV saliency/colour blobs propose boxes; the CIFAR CNN classifies each crop. Cheap and works with the current model."),
         ("B", "Single-shot detector (SSD-MobileNet / YOLO-nano)",
          "Fine-tune on Webots images; one pass gives every box and class. INT8 TFLite on a Coral TPU or Jetson."),
         ("C", "Auto-labelled training data from Webots",
          "Webots' Camera Recognition node returns object boxes, so thousands of labelled frames can be generated for free and the sim-to-real gap shrinks.")]
for j, (a, h1, d) in enumerate(props):
    x = 0.6 + j * 4.12
    box(s, x, 1.7, 3.9, 4.0)
    circle_num(s, x + 0.25, 1.9, 0.6, a, fill=TEAL)
    text(s, x + 0.25, 2.7, 3.45, 0.9, h1, size=16, bold=True, color=NAVY)
    text(s, x + 0.25, 3.6, 3.45, 1.4, d, size=14, color=INK)
    st = ["Prototype running in the controller", "Proposed: needs labelled data", "Proposed: feeds option B"][j]
    pill = box(s, x + 0.25, 5.05, 3.4, 0.4, fill=TEAL if j == 0 else TEAL_LT, radius=0.5, shadow=False)
    shape_text(pill, st, size=12, bold=True, color=WHITE if j == 0 else TEAL)
box(s, 0.6, 5.95, 12.1, 0.8, fill=NAVY)
text(s, 0.85, 6.05, 11.6, 0.6, [[("Block integration: ", {"bold": True, "color": AMBER}),
     ("add a 'block' class (or one per colour) so the CNN confirms what HSV detects; a vote of both cuts false positives.",
      {"color": WHITE})]], size=14, anchor=MSO_ANCHOR.MIDDLE)
notes(s, "Option A is already in the controller (object_roi + CNN). B and C are the next steps.")

# ======================================================================= 17. Findings & limitations
s = nxt()
title(s, "Key findings and limitations", "Key findings")
good = [f"Improved CNN reaches {acc:.1f}% on CIFAR-10 (+{acc-base:.1f} pts over cnn_v3).",
        f"INT8 TFLite: {speedup:.1f}× faster and {shrink:.1f}× smaller with ≈ no accuracy loss.",
        "Asynchronous, event-triggered inference keeps the 32 ms control loop on time.",
        "Shape-verified HSV detection finds all blocks and ignores coloured look-alikes."]
bad = ["Sim-to-real (domain) gap: renders, toy models, even the floor pattern shift predictions.",
       f"Webots camera accuracy {tour_ok}/{len(objs)} vs {acc:.0f}% on the test set; similar animals get confused.",
       "Classification only: one label per crop, no boxes for several objects.",
       "No 'block' class: the CNN labels colour blocks as 'ship', so HSV must stay in charge."]
for j, (hd, lst, colr) in enumerate([("What worked", good, TEAL), ("Limitations", bad, RED)]):
    x = 0.6 + j * 6.2
    box(s, x, 1.6, 5.9, 4.0)
    text(s, x + 0.3, 1.75, 5.3, 0.45, hd, size=20, bold=True, color=colr)
    bullets(s, x + 0.3, 2.35, 5.35, 3.2, lst, size=15, space=10, bullet_color=colr)
if True:
    b = box(s, 0.6, 5.85, 12.1, 0.85, fill=NAVY)
    text(s, 0.9, 5.95, 11.6, 0.65, [[("Bottom line: ", {"bold": True, "color": AMBER}),
         ("the network is accurate and fast enough for the robot; the remaining gap is the data, not the model.", {"color": WHITE})]],
         size=16, anchor=MSO_ANCHOR.MIDDLE)
notes(s, "Be honest about accuracy issues: the domain gap is the main reason Webots accuracy is below test accuracy.")

# ======================================================================= 18. Future work
s = nxt()
title(s, "Future work and next steps", "Future work")
fw = [("Pruning + INT8", "Magnitude pruning (Han et al., 2016) to cut a further ~50 % of weights before quantising."),
      ("Fine-tune on Webots frames", "Generate labelled frames from the arena and fine-tune, closing the domain gap."),
      ("Block classifier in the CNN", "Add colour-block classes; fuse with HSV for robust detection."),
      ("Path planning for search", "Coverage path (boustrophedon) instead of a random walk to find every object faster."),
      ("Edge deployment", "Run the TFLite model on a Raspberry Pi + Coral TPU for a physical e-puck extension.")]
for j, (a, b_) in enumerate(fw):
    y = 1.65 + j * 1.0
    circle_num(s, 0.6, y + 0.05, 0.6, str(j + 1), fill=TEAL if j < 3 else NAVY)
    text(s, 1.4, y, 4.0, 0.4, a, size=17, bold=True, color=NAVY)
    text(s, 1.4, y + 0.38, 6.0, 0.55, b_, size=14, color=INK)
image(s, CAP / "scene/top_down.jpg", 7.9, 1.65, 4.85, 4.85)
notes(s, "Concrete next steps, each tied to a limitation from the previous slide.")

# ======================================================================= 19. Conclusion
s = nxt(dark=True)
title(s, "Conclusion", "Summary", dark=True)
concl = [f"Built and trained a CIFAR-10 CNN off-robot: {acc:.1f}% test accuracy, all layers documented.",
         f"Predicted three of my own photos and {tour_ok}/{len(objs)} robot-camera views in Webots.",
         f"Optimised for the robot: INT8 TFLite ({i8['latency_ms']:.2f} ms), async and event-triggered inference.",
         "Demonstrated shape-verified colour-block detection in a miniature Webots world.",
         "Next: object detection (SSD / region proposals), Webots-generated training data, pruning."]
bullets(s, 0.6, 1.7, 7.0, 5.0, concl, size=17, color=WHITE, space=14, bullet_color=AMBER)
image(s, CAP / "tour/horse__horse_annotated.png", 8.0, 1.75, 4.75, 3.56)
text(s, 8.0, 5.45, 4.75, 0.9, "Thank you · Questions?", size=26, bold=True, color=AMBER, align=PP_ALIGN.CENTER)
notes(s, "Wrap up against the aims, then take questions.")

# ======================================================================= 20. References
s = nxt()
title(s, "References", "Bibliography")
refs = [
    "Cyberbotics (2025) Webots R2025a User Guide and Reference Manual. Available at: https://cyberbotics.com/doc",
    "Han, S., Mao, H. and Dally, W.J. (2016) 'Deep compression: compressing deep neural networks with pruning, trained quantization and Huffman coding', ICLR.",
    "Howard, A.G. et al. (2017) 'MobileNets: efficient convolutional neural networks for mobile vision applications', arXiv:1704.04861.",
    "Ioffe, S. and Szegedy, C. (2015) 'Batch normalization: accelerating deep network training by reducing internal covariate shift', ICML, pp. 448–456.",
    "Jacob, B. et al. (2018) 'Quantization and training of neural networks for efficient integer-arithmetic-only inference', CVPR, pp. 2704–2713.",
    "Krizhevsky, A. (2009) Learning multiple layers of features from tiny images. Technical report, University of Toronto.",
    "Krizhevsky, A., Sutskever, I. and Hinton, G.E. (2012) 'ImageNet classification with deep convolutional neural networks', NeurIPS, pp. 1097–1105.",
    "LeCun, Y., Bottou, L., Bengio, Y. and Haffner, P. (1998) 'Gradient-based learning applied to document recognition', Proc. IEEE, 86(11), pp. 2278–2324.",
    "Liu, W. et al. (2016) 'SSD: single shot multibox detector', ECCV, pp. 21–37.",
    "Mondada, F. et al. (2009) 'The e-puck, a robot designed for education in engineering', Proc. 9th Conf. on Autonomous Robot Systems and Competitions, pp. 59–65.",
    "Simonyan, K. and Zisserman, A. (2015) 'Very deep convolutional networks for large-scale image recognition', ICLR.",
    "Srivastava, N. et al. (2014) 'Dropout: a simple way to prevent neural networks from overfitting', JMLR, 15, pp. 1929–1958.",
    "Tobin, J. et al. (2017) 'Domain randomization for transferring deep neural networks from simulation to the real world', IROS, pp. 23–30.",
]
bullets(s, 0.6, 1.55, 12.1, 5.5, refs, size=13, space=6, bullet_color=TEAL)
notes(s, "Harvard style. Every source is cited on a slide or in the report.")

out = Path(__file__).parent / "CNN_Webots_Presentation.pptx"
prs.save(out)
print("saved", out, "slides:", n)
