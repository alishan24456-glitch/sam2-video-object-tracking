#!/usr/bin/env python3
"""
SAM 2-style promptable video object tracking, with a light OpenCV backend.

A point or box prompt selects ONE object in the first frame and the
system follows it through the video with a live mask, a track trail
and an occlusion / reacquisition log -- the video counterpart of the
sam-segment-anything build (image masks). This is the prompt ->
segment -> propagate loop of Meta's SAM 2, re-implemented with
classical tools that run on a CPU.

What it does:
    * Prompt: --prompt X,Y (a point), --box x1,y1,x2,y2 (a box), a
      mouse click in the display window, or --prompt auto (default):
      YOLOv8n proposes the largest person on frame 0, else the
      largest vehicle. The prompt is fixed at frame 0 -- it is never
      re-aimed at a tracked position.
    * Initial mask: GrabCut seeded by the prompt box, refined by an
      HSV colour histogram of the masked pixels.
    * Tracking: each frame, sparse Lucas-Kanade flow on points inside
      the previous mask predicts where the object moved; a histogram
      back-projection re-segments it inside the predicted box, with a
      GrabCut polish every 10th frame. The mask, box and trail are
      read off those pixels only -- no ground truth, no detector
      re-aiming after frame 0.
    * Occlusion: when the mask area collapses below 30% of its
      running median the state flips to OCCLUDED and the search box
      keeps widening; when the colour mask comes back near the
      predicted position the state flips to REACQUIRED and the event
      is logged with a timestamp. Occluded for more than 5 s reads
      LOST; a centre that leaves the frame reads LEFT FRAME.

Modes (one pipeline for all of them):
    Real mode   --source <video file | webcam index | pick>
    Simulation  --simulate (a procedural street scene is rendered
                frame by frame and fed through the SAME tracker;
                the mask, area and occlusion events come only from
                what the tracker finds in those rendered pixels.
                The prompt point is a fixed frame coordinate on the
                blue car at its starting position -- never a tracked
                position. The car drives behind a parked van, so
                the occlusion / reacquisition path is exercised.)

Backends:
    opencv (the only backend, what the demo uses) -- GrabCut +
                HSV histogram + optical flow. It is SAM 2-*style*
                promptable video tracking, not Meta's SAM 2 model:
                the mask follows colour/texture regions, so on real
                footage it can bleed onto same-coloured background,
                shrink on a turning object, or lose a fast mover.
                The code never claims these masks are SAM 2 outputs.
                A true SAM 2 backend would need a SAM 2 checkpoint
                plus torch and a GPU; no SAM 2 ONNX export runs
                usefully on this shared CPU, so the light backend is
                the shipped one -- see Honest notes in the README.

Usage (Windows CMD):
    pip install -r requirements.txt
    python sam2_video_object_tracking.py --source pick
    python sam2_video_object_tracking.py
    python sam2_video_object_tracking.py --source 0
    python sam2_video_object_tracking.py --source video.mp4 --output tracked.mp4
    python sam2_video_object_tracking.py --source video.mp4 --prompt 640,400
    python sam2_video_object_tracking.py --source video.mp4 --box 500,300,700,520
    python sam2_video_object_tracking.py --simulate --output sim-demo.mp4
    python sam2_video_object_tracking.py --simulate --duration 28 --output demo.mp4

Notes:
    - Running with no arguments (or --source pick) opens a file-picker
      dialog to choose a video.
    - In the display window, click anywhere to re-prompt on the current
      frame (the tracker restarts from that point); press Q to quit.
    - Mask area is counted in pixels of the working frame (1280x720),
      so it is a relative size figure, not a real-world area.
    - The event log and summary are written next to the output video
      unless --events / --summary say otherwise.

Author: Abdul Hanan
"""

import argparse
import math
import os
import sys
import time
import urllib.request
from collections import deque

import cv2
import numpy as np

WORK_W = 1280
WORK_H = 720
VIDEO_H = 600          # video area on the canvas; board below it
BOARD_H = 120
CANVAS_H = VIDEO_H + BOARD_H
SIM_FPS = 30.0

# YOLOv8n prompt proposal (frame 0 only).
YOLO_URL = ("https://github.com/ultralytics/assets/releases/"
            "download/v8.3.0/yolov8n.pt")
PERSON_CLASS = 0
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PROMPT_CONF = 0.35
PROMPT_NMS = 0.45

# Tracker tuning.
REFINE_EVERY = 10       # GrabCut polish every Nth frame
OCCLUDE_FRAC = 0.30     # mask area below this x median -> OCCLUDED
REACQUIRE_FRAC = 0.50   # ... and back above this x median -> REACQUIRED
LOST_AFTER_S = 5.0      # occluded this long -> LOST
MIN_MASK_PX = 120       # smallest believable mask
MEDIAN_WINDOW = 45      # running-median window for mask area

# Fixed prompt point for the simulation demo: the blue car's centre
# at t = 0 (car starts centred at x = 80, y = 560). A constant
# coordinate, never a tracked position.
SIM_PROMPT = (80, 560)

PALETTE = [
    (30, 30, 180), (0, 150, 0), (200, 100, 0), (180, 180, 0),
    (180, 0, 180), (0, 180, 180), (0, 100, 220), (130, 0, 200),
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def clamp_box(box, w, h, min_size=20):
    """Clamp [x, y, w, h] inside the frame; None if it collapses."""
    x, y, bw, bh = [float(v) for v in box]
    x = max(0.0, min(x, w - 1))
    y = max(0.0, min(y, h - 1))
    bw = max(min_size, min(bw, w - x))
    bh = max(min_size, min(bh, h - y))
    if bw < min_size or bh < min_size:
        return None
    return [x, y, bw, bh]


def pick_file():
    """Open a file-picker dialog so the user can choose a video file."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            path = filedialog.askopenfilename(
                title="Choose a video file",
                filetypes=[("Video files",
                            "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                           ("All files", "*.*")],
            )
        finally:
            root.destroy()
    except Exception:
        return None
    return path or None


def open_source(source):
    """Return a cv2.VideoCapture for a path, webcam index or picker."""
    if source == "pick":
        path = pick_file()
        if not path:
            print("No file chosen. Pass --source <video file> or a "
                  "webcam index (e.g. --source 0).")
            sys.exit(1)
        source = path
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open video source: {source!r}")
        sys.exit(1)
    return cap


# ---------------------------------------------------------------------------
# YOLOv8n prompt proposal -- frame 0 only (OpenCV DNN, no torch)
# ---------------------------------------------------------------------------
def ensure_weights(path):
    """Return a usable yolov8n.onnx path, exporting it if needed."""
    if os.path.exists(path):
        return path
    print(f"Weights not found at {path}; trying Ultralytics export ...")
    try:
        from ultralytics import YOLO
        pt = os.path.join(os.path.dirname(path) or ".", "yolov8n.pt")
        if not os.path.exists(pt):
            print("Downloading yolov8n.pt ...")
            urllib.request.urlretrieve(YOLO_URL, pt)
        print("Exporting yolov8n.pt -> yolov8n.onnx (one-time) ...")
        YOLO(pt).export(format="onnx", imgsz=640)
        exported = os.path.join(os.path.dirname(pt) or ".", "yolov8n.onnx")
        if os.path.exists(exported) and exported != path:
            os.replace(exported, path)
        return path
    except Exception as exc:  # noqa: BLE001 - report and stop cleanly
        print(f"Could not obtain weights: {exc}")
        print("Download yolov8n.onnx and place it next to this script, or")
        print("install ultralytics (`pip install ultralytics`) and re-run.")
        sys.exit(1)


class YoloProposer:
    """One-shot YOLOv8n detection used ONLY to propose the frame-0
    prompt box. After the prompt, the tracker runs on pixels alone
    and YOLO is never called again."""

    def __init__(self, weights_path):
        self.net = cv2.dnn.readNetFromONNX(ensure_weights(weights_path))

    def detect(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        # YOLOv8 ONNX output is [1, 84, N]: 4 box coords + 80 class
        # scores, NO objectness channel -- confidence is the max
        # class score, class is its argmax. (A v7-style objectness
        # read makes the detector dead; do not repeat that bug.)
        out = self.net.forward()[0].T  # [8400, 84]
        scores = out[:, 4:].max(axis=1)
        classes = out[:, 4:].argmax(axis=1)
        h, w = frame.shape[:2]
        boxes = []
        for (cx, cy, bw, bh), s, c in zip(out[:, :4], scores, classes):
            if s < PROMPT_CONF:
                continue
            if c != PERSON_CLASS and c not in VEHICLE_CLASSES:
                continue
            boxes.append([float((cx - bw / 2) * w / 640),
                          float((cy - bh / 2) * h / 640),
                          float(bw * w / 640), float(bh * h / 640),
                          float(s),
                          "person" if c == PERSON_CLASS
                          else VEHICLE_CLASSES[int(c)]])
        if not boxes:
            return []
        idx = cv2.dnn.NMSBoxes([b[:4] for b in boxes],
                               [b[4] for b in boxes],
                               PROMPT_CONF, PROMPT_NMS)
        return [boxes[int(i)] for i in idx.flatten()] if len(idx) else []

    def propose(self, frame):
        """Largest person box, else largest vehicle box, else None."""
        boxes = self.detect(frame)
        if not boxes:
            return None
        people = [b for b in boxes if b[5] == "person"]
        pool = people if people else boxes
        best = max(pool, key=lambda b: b[2] * b[3])
        return best


# ---------------------------------------------------------------------------
# Prompt tracker -- GrabCut + HSV histogram + optical flow
# ---------------------------------------------------------------------------
class PromptTracker:
    """Tracks one prompted object with a live mask.

    State machine: TRACKING -> OCCLUDED -> TRACKING (REACQUIRED) /
    LOST (occluded > LOST_AFTER_S) / LEFT FRAME (centre out of frame).
    """

    def __init__(self, weights_dir):
        self.weights_dir = weights_dir
        self.box = None            # [x, y, w, h] float
        self.hist = None           # HSV histogram of the object
        self.prev_gray = None
        self.prev_mask = None
        self.state = "TRACKING"
        self.trail = deque(maxlen=60)
        self.areas = deque(maxlen=MEDIAN_WINDOW)
        self.frame_no = 0
        self.events = []           # (t, text)
        self.occlusions = 0
        self.reacquisitions = 0
        self.occluded_frames = 0
        self.tracked_frames = 0
        self.occlusion_start = None
        self.longest_occlusion_s = 0.0
        self.left_frame_logged = False
        self.label = "object"
        # Last measured velocity (px/s) from optical flow, kept so the
        # predicted box keeps moving at the object's speed while it is
        # occluded instead of freezing where it vanished.
        self.vel = (0.0, 0.0)
        # Last full-size box (while the object was fully visible) and
        # last mask area, for the occlusion logic below.
        self.last_good_box = None
        self.prev_area = None
        self._occl_streak = 0
        self._reac_streak = 0

    # -- events ------------------------------------------------------------
    def _log(self, t, text):
        self.events.append((t, text))

    # -- re-prompt (loop wrap / click) ----------------------------------
    def reset_for_reprompt(self):
        """Clear the per-pass state but keep the cumulative event log
        and counters, so a looped clip reads as two clean passes."""
        self.box = None
        self.hist = None
        self.prev_gray = None
        self.prev_mask = None
        self.state = "TRACKING"
        self.trail = deque(maxlen=60)
        self.areas = deque(maxlen=MEDIAN_WINDOW)
        self.occlusion_start = None
        self.left_frame_logged = False
        self.vel = (0.0, 0.0)
        self.last_good_box = None
        self.prev_area = None
        self._occl_streak = 0
        self._reac_streak = 0

    # -- initialisation ----------------------------------------------------
    def init(self, frame, box, label="object", t=0.0):
        h, w = frame.shape[:2]
        box = clamp_box(box, w, h)
        if box is None:
            raise ValueError("prompt box is outside the frame")
        self.box = box
        self.label = label
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_mask = np.zeros((h, w), dtype=np.uint8)
        mask = self._grabcut(frame, box)
        if mask is None or int((mask > 0).sum()) < MIN_MASK_PX:
            # GrabCut found nothing usable: fall back to the box
            # itself as the mask (documented in the README).
            mask = np.zeros((h, w), dtype=np.uint8)
            x, y, bw, bh = [int(v) for v in box]
            mask[y:y + bh, x:x + bw] = 255
            self._log(t, "GrabCut found no mask; using prompt box "
                         "as the initial mask")
        self.prev_mask = mask
        self.hist = self._histogram(frame, mask)
        area = int((mask > 0).sum())
        self.areas.append(area)
        self.tracked_frames += 1
        self.last_good_box = list(box)
        self.prev_area = area
        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2
        self.trail.append((cx, cy))
        self._log(t, f"PROMPT {label} box "
                     f"({int(box[0])},{int(box[1])},"
                     f"{int(box[2])}x{int(box[3])}) mask {area} px")

    def _grabcut(self, frame, box):
        """GrabCut inside a rect; returns a 0/255 mask or None."""
        x, y, bw, bh = [int(v) for v in box]
        roi = frame[y:y + bh, x:x + bw]
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return None
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        rect = (1, 1, roi.shape[1] - 2, roi.shape[0] - 2)
        try:
            # Fixed seed: GrabCut's k-means init is random by
            # default, and an unseeded re-prompt on the same frame
            # gave a 8,311 px mask once and a 19,023 px mask the
            # next time. Seeded, the same prompt gives the same mask.
            cv2.setRNGSeed(42)
            cv2.grabCut(roi, mask, rect, bgd, fgd, 5,
                        cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            return None
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                      255, 0).astype(np.uint8)
        full = np.zeros(frame.shape[:2], dtype=np.uint8)
        full[y:y + bh, x:x + bw] = fg
        return full

    def _histogram(self, frame, mask):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist

    def _median_area(self):
        if len(self.areas) < 10:
            return None
        return float(np.median(np.array(self.areas, dtype=np.float64)))

    # -- per-frame update --------------------------------------------------
    def update(self, frame, t):
        self.frame_no += 1
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, bw, bh = self.box
        cx, cy = x + bw / 2, y + bh / 2

        # 1. Predict motion: median Lucas-Kanade displacement of
        #    points inside the previous mask. While occluded there is
        #    no mask to track points in, so the box keeps moving at
        #    the last measured velocity instead of freezing where the
        #    object vanished (the first build froze there, and a car
        #    that kept driving was 250 px away by the time it
        #    reappeared -- reacquisition never fired).
        dx = dy = 0.0
        if self.state == "TRACKING":
            if self.prev_mask is not None and int((self.prev_mask > 0).sum()) > 50:
                pts = cv2.goodFeaturesToTrack(
                    self.prev_gray, maxCorners=40, qualityLevel=0.01,
                    minDistance=8, mask=self.prev_mask)
                if pts is not None and len(pts) >= 3:
                    nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, pts, None)
                    good = (status.flatten() == 1)
                    if good.sum() >= 3:
                        d = nxt[good].reshape(-1, 2) - pts[good].reshape(-1, 2)
                        dx = float(np.median(d[:, 0]))
                        dy = float(np.median(d[:, 1]))
                        dx = max(-60.0, min(60.0, dx))
                        dy = max(-60.0, min(60.0, dy))
                        # Only learn velocity from a fully-visible
                        # object. While it slides behind an occluder
                        # the visible fragment's centroid moves at
                        # half speed (its far edge is pinned at the
                        # occluder), so learning from it halves the
                        # coast speed and the box lags the real object
                        # by 250 px at reappearance.
                        med_now = self._median_area()
                        if (self.prev_area is None or med_now is None
                                or self.prev_area >= 0.8 * med_now):
                            vx, vy = self.vel
                            self.vel = (0.7 * vx + 0.3 * dx * 30.0,
                                        0.7 * vy + 0.3 * dy * 30.0)
        else:
            # Occluded / lost: coast at the last velocity, one frame.
            dx = self.vel[0] / 30.0
            dy = self.vel[1] / 30.0
        cx += dx
        cy += dy

        # 2. Re-segment by histogram back-projection inside the
        #    predicted box, widened while occluded. Once LOST, the
        #    box may be far off, so search the whole frame for the
        #    target's colours instead of the small ROI.
        occluded_for = (t - self.occlusion_start) if self.occlusion_start else 0.0
        widen = 1.3 if self.state == "TRACKING" else 1.3 + min(
            2.5, occluded_for * 0.5)
        if self.state == "LOST":
            rw, rh = float(w), float(h)
            rx = ry = 0.0
        else:
            rw, rh = bw * widen, bh * widen
            rx = max(0.0, cx - rw / 2)
            ry = max(0.0, cy - rh / 2)
            rw = min(rw, w - rx)
            rh = min(rh, h - ry)
        roi = frame[int(ry):int(ry + rh), int(rx):int(rx + rw)]
        mask = np.zeros((h, w), dtype=np.uint8)
        area = 0
        if roi.size:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            back = cv2.calcBackProject([hsv], [0, 1], self.hist,
                                       [0, 180, 0, 256], 1)
            _, back = cv2.threshold(back, 60, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            back = cv2.morphologyEx(back, cv2.MORPH_OPEN, kernel)
            back = cv2.morphologyEx(back, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(back, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Largest contour near the predicted centre wins.
                # The full-frame (LOST) search takes the largest
                # contour within range outright -- the colour
                # histogram is already specific, and a distance
                # penalty (the first build's `area - 2*dist^2`
                # with best_score starting at -1) rejected every
                # contour more than ~130 px away, which is exactly
                # the case a reacquisition search exists for.
                max_dist = 400.0 if self.state == "LOST" else 1e9
                best = None
                best_score = -float("inf")
                for c in contours:
                    a = cv2.contourArea(c)
                    if a < MIN_MASK_PX:
                        continue
                    mx, my, mw, mh = cv2.boundingRect(c)
                    ccx, ccy = mx + mw / 2, my + mh / 2
                    dist = math.hypot(ccx + rx - cx, ccy + ry - cy)
                    if dist > max_dist:
                        continue
                    score = a if self.state == "LOST" else \
                        a - 0.5 * dist * dist
                    if score > best_score:
                        best_score = score
                        best = c
                if best is not None:
                    m = np.zeros(roi.shape[:2], dtype=np.uint8)
                    cv2.drawContours(m, [best], -1, 255, cv2.FILLED)
                    mask[int(ry):int(ry + rh), int(rx):int(rx + rw)] = m
                    area = int((m > 0).sum())

        # 3. GrabCut polish every REFINE_EVERY frames, while tracking.
        if (area >= MIN_MASK_PX and self.state == "TRACKING"
                and self.frame_no % REFINE_EVERY == 0):
            ys, xs = np.where(mask > 0)
            if len(xs):
                bx = [float(xs.min()), float(ys.min()),
                      float(xs.max() - xs.min()),
                      float(ys.max() - ys.min())]
                refined = self._grabcut(frame, bx)
                if refined is not None:
                    rarea = int((refined > 0).sum())
                    med = self._median_area()
                    # Accept the polish only when GrabCut agrees with
                    # the back-projection mask. GrabCut on a
                    # half-hidden object can return the WRONG part
                    # (the pale window band instead of the blue body),
                    # and the first build then re-learned the colour
                    # model from that fragment -- after which the
                    # tracker followed the window (94x25 px) and never
                    # grew back to the whole car. The colour model
                    # stays frozen from the prompt for the whole run.
                    if (med is not None and 0.3 * med <= rarea <= 3.0 * med
                            and 0.8 * area <= rarea <= 1.25 * area):
                        mask = refined
                        area = rarea

        # 4. State machine on mask area vs running median, with a
        #    3-frame (0.1 s) confirmation on each transition so one
        #    noisy frame cannot flip the state -- the first build
        #    logged REACQUIRED / OCCLUDED / REACQUIRED within 0.2 s
        #    around the van edge.
        med = self._median_area()
        prev_state = self.state
        if self.state == "TRACKING":
            if med is not None and area < OCCLUDE_FRAC * med:
                self._occl_streak += 1
                if self._occl_streak >= 3:
                    self.state = "OCCLUDED"
                    self.occlusion_start = t - (self._occl_streak - 1) / 30.0
                    self.occlusions += 1
                    self._log(t, f"OCCLUDED (mask {area} px, "
                                 f"median {int(med)} px)")
                    self._occl_streak = 0
            else:
                self._occl_streak = 0
        elif self.state in ("OCCLUDED", "LOST"):
            if area >= REACQUIRE_FRAC * (med or area + 1):
                self._reac_streak += 1
                if self._reac_streak >= 3:
                    self.state = "TRACKING"
                    self.reacquisitions += 1
                    if self.occlusion_start is not None:
                        dur = t - self.occlusion_start
                        self.longest_occlusion_s = max(
                            self.longest_occlusion_s, dur)
                        self._log(t, f"REACQUIRED after {dur:.1f} s "
                                     f"(mask {area} px)")
                        self.occlusion_start = None
                    self._reac_streak = 0
            else:
                self._reac_streak = 0
                if (self.occlusion_start is not None
                        and t - self.occlusion_start > LOST_AFTER_S
                        and self.state == "OCCLUDED"):
                    self.state = "LOST"
                    self._log(t, f"LOST (occluded > {LOST_AFTER_S:.0f} s)")

        # 5. Box follows the mask while tracking; holds while occluded.
        if self.state == "TRACKING" and area >= MIN_MASK_PX:
            ys, xs = np.where(mask > 0)
            nx, ny = float(xs.min()), float(ys.min())
            nw, nh = float(xs.max() - xs.min()), float(ys.max() - ys.min())
            # Smooth the box so one noisy frame cannot teleport it.
            self.box = [0.5 * self.box[0] + 0.5 * nx,
                        0.5 * self.box[1] + 0.5 * ny,
                        0.5 * self.box[2] + 0.5 * nw,
                        0.5 * self.box[3] + 0.5 * nh]
            self.areas.append(area)
            self.tracked_frames += 1
            mx, my = float(xs.mean()), float(ys.mean())
            self.trail.append((mx, my))
            if med is None or area >= 0.8 * med:
                self.last_good_box = list(self.box)
        else:
            # Occluded: keep the last FULL box size (the box had
            # shrunk to the visible sliver, 18 px wide, which made
            # the search ROI too small to ever re-find the object).
            if self.last_good_box is not None:
                gw, gh = self.last_good_box[2], self.last_good_box[3]
            else:
                gw, gh = bw, bh
            self.box = [cx - gw / 2, cy - gh / 2, gw, gh]
            self.occluded_frames += 1
            if self.trail:
                self.trail.append(self.trail[-1])
        self.prev_area = area

        # 6. Left the frame?
        cx, cy = self.box[0] + self.box[2] / 2, self.box[1] + self.box[3] / 2
        if ((cx < -50 or cx > w + 50 or cy < -50 or cy > h + 50)
                and not self.left_frame_logged):
            self.state = "LEFT FRAME"
            self.left_frame_logged = True
            self._log(t, "LEFT FRAME (target centre out of view)")

        self.prev_gray = gray
        self.prev_mask = mask if self.state == "TRACKING" else \
            np.zeros((h, w), dtype=np.uint8)
        return {
            "mask": mask,
            "box": list(self.box),
            "state": self.state,
            "area": area,
            "centroid": (cx, cy),
            "median_area": med,
        }

    def summary(self, duration_s, frames):
        return {
            "frames": frames,
            "duration_s": round(duration_s, 1),
            "tracked_frames": self.tracked_frames,
            "occluded_frames": self.occluded_frames,
            "occlusions": self.occlusions,
            "reacquisitions": self.reacquisitions,
            "longest_occlusion_s": round(self.longest_occlusion_s, 1),
            "final_state": self.state,
            "final_mask_area": int(self.areas[-1]) if self.areas else 0,
            "median_mask_area": (int(self._median_area())
                                 if self._median_area() else 0),
            "events": self.events,
        }


# ---------------------------------------------------------------------------
# Simulation -- rendered street scene, same pixel pipeline
# ---------------------------------------------------------------------------
class StreetSim:
    """A street with a blue car driving left->right past a parked grey
    van. The van is drawn OVER the car while the car is behind it, so
    the tracker sees a real occlusion in the rendered pixels. A red
    car drives the other way in the far lane as a decoy (different
    colour, so the blue histogram never matches it)."""

    def __init__(self, w=WORK_W, h=WORK_H, fps=SIM_FPS):
        self.w = w
        self.h = h
        self.fps = fps
        self.t = 0.0
        # Van (occluder) spans x 520-760 in frame coordinates.
        self.van = (520, 470, 240, 250)
        # Blue car: 150x70, centre y 560, 53 px/s from x = 80.
        self.car_speed = 53.0
        self.car_start_x = 80.0
        self.car_y = 560.0
        # Red decoy: 130x60, centre y 470, right->left at 40 px/s.
        self.decoy_speed = -40.0
        self.decoy_start_x = 1250.0
        self.decoy_y = 470.0

    def _draw_car(self, img, cx, cy, w, h, body_bgr, label):
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        x1, y1 = int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(img, (x0, y0), (x1, y1), body_bgr, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 20), 2)
        # Windows: lighter band across the top half.
        wx0, wy0 = int(cx - w * 0.32), int(cy - h * 0.32)
        wx1, wy1 = int(cx + w * 0.32), int(cy + h * 0.05)
        cv2.rectangle(img, (wx0, wy0), (wx1, wy1), (210, 225, 235), -1)
        cv2.rectangle(img, (wx0, wy0), (wx1, wy1), (20, 20, 20), 1)
        # Wheels.
        for wx in (int(cx - w * 0.30), int(cx + w * 0.30)):
            cv2.circle(img, (wx, y1), int(h * 0.22), (15, 15, 15), -1)
            cv2.circle(img, (wx, y1), int(h * 0.10), (150, 150, 150), -1)
        cv2.putText(img, label, (x0 + 6, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2)

    def frame(self):
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Sky, buildings, pavement, road.
        img[0:300] = (235, 220, 200)
        for bx, bw, bh, col in ((40, 180, 210, (150, 160, 175)),
                                (280, 140, 160, (170, 165, 150)),
                                (480, 200, 240, (140, 150, 165)),
                                (740, 160, 180, (160, 155, 145)),
                                (960, 220, 230, (150, 160, 170))):
            cv2.rectangle(img, (bx, 300 - bh), (bx + bw, 300), col, -1)
            for wy in range(300 - bh + 20, 290, 40):
                for wx in range(bx + 15, bx + bw - 15, 35):
                    cv2.rectangle(img, (wx, wy), (wx + 18, wy + 22),
                                  (60, 70, 90), -1)
        cv2.rectangle(img, (0, 300), (self.w, 430), (170, 170, 170), -1)
        cv2.rectangle(img, (0, 430), (self.w, self.h), (60, 60, 60), -1)
        # Lane markings.
        for lx in range(0, self.w, 120):
            cv2.rectangle(img, (lx, 500), (lx + 60, 508), (230, 230, 230), -1)
            cv2.rectangle(img, (lx, 640), (lx + 60, 648), (230, 230, 230), -1)
        # Kerb line.
        cv2.line(img, (0, 430), (self.w, 430), (40, 40, 40), 3)

        # Target car (blue) and decoy (red) positions at time t.
        car_cx = self.car_start_x + self.car_speed * self.t
        dec_cx = self.decoy_start_x + self.decoy_speed * self.t
        # Draw the target first, then the van over it when behind.
        self._draw_car(img, car_cx, self.car_y, 150, 70, (255, 130, 30),
                       "TARGET")
        self._draw_car(img, dec_cx, self.decoy_y, 130, 60, (40, 40, 220),
                       "DECOY")
        vx, vy, vw, vh = self.van
        cv2.rectangle(img, (vx, vy), (vx + vw, vy + vh), (120, 120, 120), -1)
        cv2.rectangle(img, (vx, vy), (vx + vw, vy + vh), (40, 40, 40), 2)
        cv2.rectangle(img, (vx + 20, vy + 25), (vx + 100, vy + 90),
                      (200, 210, 220), -1)
        cv2.putText(img, "PARKED VAN", (vx + 30, vy + 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
        self.t += 1.0 / self.fps
        return img


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def draw_canvas(det_frame, res, elapsed, mode_label, tracker, events_tail):
    """Video (1280x600) on top, board (1280x120) below."""
    video = cv2.resize(det_frame, (WORK_W, VIDEO_H))
    canvas = np.zeros((CANVAS_H, WORK_W, 3), dtype=np.uint8)
    canvas[0:VIDEO_H] = video
    canvas[VIDEO_H:CANVAS_H] = (18, 18, 18)
    cv2.line(canvas, (0, VIDEO_H), (WORK_W, VIDEO_H), (90, 90, 90), 1)

    state = res["state"]
    colour = {"TRACKING": (60, 200, 60),
              "OCCLUDED": (40, 60, 230),
              "LOST": (40, 60, 230),
              "LEFT FRAME": (0, 180, 220)}.get(state, (200, 200, 200))
    med = res["median_area"]
    med_txt = f"{int(med)} px" if med else "warming up"
    line1 = (f"SAM 2-STYLE TRACKING  |  {mode_label}  |  t={elapsed:5.1f}s  |  "
             f"STATE: {state}")
    line2 = (f"target: {tracker.label}  |  mask area: {res['area']} px  "
             f"(median {med_txt})  |  frames tracked: {tracker.tracked_frames}  |  "
             f"occlusions: {tracker.occlusions}  |  reacquired: "
             f"{tracker.reacquisitions}")
    cv2.putText(canvas, line1, (16, VIDEO_H + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2)
    cv2.putText(canvas, line2, (16, VIDEO_H + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 225, 225), 1)
    if events_tail:
        line3 = "  |  ".join(f"{t:4.1f}s {txt}"
                             for t, txt in events_tail[-2:])
        cv2.putText(canvas, line3, (16, VIDEO_H + 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return canvas


def overlay_result(frame, res, tracker, prompt_xy):
    """Mask overlay + box + trail + state label on the working frame."""
    out = frame.copy()
    mask = res["mask"]
    if res["state"] == "TRACKING" and int((mask > 0).sum()) > 0:
        overlay = out.copy()
        overlay[mask > 0] = (60, 200, 60)
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (60, 220, 60), 2)
    x, y, bw, bh = [int(v) for v in res["box"]]
    colour = {"TRACKING": (60, 200, 60),
              "OCCLUDED": (40, 60, 230),
              "LOST": (40, 60, 230),
              "LEFT FRAME": (0, 180, 220)}.get(res["state"], (200, 200, 200))
    cv2.rectangle(out, (x, y), (x + bw, y + bh), colour, 2)
    if len(tracker.trail) > 1:
        pts = np.array([(int(px), int(py)) for px, py in tracker.trail],
                       dtype=np.int32)
        cv2.polylines(out, [pts], False, (0, 200, 255), 2)
    label = f"{tracker.label} | {res['state']} | {res['area']} px"
    lx = max(5, min(x, out.shape[1] - 420))
    ly = max(25, y - 8)
    # Dark backing: the bare label was unreadable where it crossed
    # the bright mask / white shirt.
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                  0.6, 2)
    cv2.rectangle(out, (lx - 4, ly - th - 6), (lx + tw + 4, ly + 6),
                  (15, 15, 15), -1)
    cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, colour, 2)
    if prompt_xy is not None:
        px, py = int(prompt_xy[0]), int(prompt_xy[1])
        cv2.drawMarker(out, (px, py), (0, 200, 255),
                       cv2.MARKER_CROSS, 24, 2)
        cv2.putText(out, "PROMPT", (px + 16, py - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_prompt(value):
    try:
        px, py = (float(v) for v in value.split(","))
        return (px, py)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--prompt needs X,Y (e.g. --prompt 640,400)")


def parse_box(value):
    try:
        x1, y1, x2, y2 = (float(v) for v in value.split(","))
        return [x1, y1, x2 - x1, y2 - y1]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--box needs x1,y1,x2,y2 (e.g. --box 500,300,700,520)")


def main():
    ap = argparse.ArgumentParser(
        description="SAM 2-style promptable video object tracking "
                    "(light OpenCV backend).")
    ap.add_argument("--source", default="pick",
                    help="video file, webcam index, or 'pick' "
                         "(default: pick = file-picker dialog)")
    ap.add_argument("--simulate", action="store_true",
                    help="run on the rendered street scene instead "
                         "of a video file")
    ap.add_argument("--prompt", type=parse_prompt, default=None,
                    help="point prompt X,Y on the first frame "
                         "(e.g. --prompt 640,400)")
    ap.add_argument("--box", type=parse_box, default=None,
                    help="box prompt x1,y1,x2,y2 on the first frame")
    ap.add_argument("--auto", action="store_true",
                    help="propose the prompt from YOLOv8n on frame 0 "
                         "(default when no --prompt/--box is given)")
    ap.add_argument("--max-seconds", type=float, default=30.0,
                    help="stop after this many seconds of video "
                         "(default 30)")
    ap.add_argument("--loop", action="store_true",
                    help="loop the input video until --max-seconds")
    ap.add_argument("--no-display", action="store_true",
                    help="do not open a display window")
    ap.add_argument("--out", default="output.mp4",
                    help="output video path (default output.mp4)")
    ap.add_argument("--events", default=None,
                    help="event log path (default: <out>_events.log)")
    ap.add_argument("--summary", default=None,
                    help="summary path (default: <out>_summary.txt)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    weights = os.path.join(here, "yolov8n.onnx")
    events_path = args.events or os.path.splitext(args.out)[0] + "_events.log"
    summary_path = args.summary or os.path.splitext(args.out)[0] + "_summary.txt"

    sim = StreetSim() if args.simulate else None
    cap = None
    fps = SIM_FPS
    total_frames = None
    if sim is None:
        cap = open_source(args.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = int(args.max_seconds * SIM_FPS)

    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"), SIM_FPS,
        (WORK_W, CANVAS_H))
    if not writer.isOpened():
        print(f"Could not open video writer for {args.out!r}")
        sys.exit(1)

    tracker = PromptTracker(here)
    click_prompt = {"xy": None}
    if not args.no_display:
        cv2.namedWindow("SAM 2-style tracking", cv2.WINDOW_NORMAL)

        def on_mouse(event, mx, my, flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN and my < VIDEO_H:
                click_prompt["xy"] = (float(mx), float(my))

        cv2.setMouseCallback("SAM 2-style tracking", on_mouse)

    frame_no = 0
    t = 0.0
    started = time.time()
    mode_label = "SIMULATION" if args.simulate else "REAL MODE"
    loop_passes = 0
    wrapped = False

    def resolve_prompt(frame):
        """Frame-0 prompt resolution (also used at a loop wrap).

        Returns (box, label, prompt_xy). Auto mode proposes the
        largest person, else the largest vehicle, on this frame."""
        prompt_xy = None
        if args.box is not None:
            box = args.box
            label = "prompted object"
        elif args.prompt is not None:
            prompt_xy = args.prompt
            if sim is not None:
                box = [prompt_xy[0] - 75, prompt_xy[1] - 35, 150, 70]
            else:
                box = [prompt_xy[0] - 60, prompt_xy[1] - 80, 120, 160]
            label = "prompted object"
        elif sim is not None:
            prompt_xy = SIM_PROMPT
            box = [SIM_PROMPT[0] - 75, SIM_PROMPT[1] - 35, 150, 70]
            label = "blue car (sim prompt)"
        else:
            proposer = YoloProposer(weights)
            best = proposer.propose(frame)
            if best is None:
                print("No person or vehicle found on frame 0 -- "
                      "pass --prompt X,Y or --box x1,y1,x2,y2.")
                sys.exit(1)
            box = best[:4]
            label = best[5]
            prompt_xy = (box[0] + box[2] / 2, box[1] + box[3] / 2)
            print(f"Auto prompt: {label} at "
                  f"({int(box[0])},{int(box[1])},"
                  f"{int(box[2])}x{int(box[3])})")
        return box, label, prompt_xy

    while frame_no < max_frames:
        if sim is not None:
            raw = sim.frame()
        else:
            ok, raw = cap.read()
            if not ok:
                if args.loop and total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    wrapped = True
                    loop_passes += 1
                    continue
                break
        frame = cv2.resize(raw, (WORK_W, WORK_H))

        if frame_no == 0 or wrapped:
            # --- (re-)prompt: frame 0, or a loop wrap. A loop wrap is
            # a jump cut back to the start, so the old mask/box is
            # meaningless there; the tracker is re-prompted on the
            # new pass and the event is logged.
            if wrapped:
                tracker._log(t, f"LOOP {loop_passes} -- video wrapped, "
                                "tracker re-prompted")
                tracker.reset_for_reprompt()
                wrapped = False
            box, label, prompt_xy = resolve_prompt(frame)
            tracker.init(frame, box, label=label, t=t)
            res = {"mask": tracker.prev_mask, "box": list(tracker.box),
                   "state": tracker.state, "area": int((tracker.prev_mask > 0).sum()),
                   "centroid": (box[0] + box[2] / 2, box[1] + box[3] / 2),
                   "median_area": None}
        else:
            # --- a click re-prompts on the current frame.
            if click_prompt["xy"] is not None and sim is None:
                px, py = click_prompt["xy"]
                click_prompt["xy"] = None
                tracker = PromptTracker(here)
                tracker.init(frame, [px - 60, py - 80, 120, 160],
                             label="prompted object", t=t)
                prompt_xy = (px, py)
                res = {"mask": tracker.prev_mask,
                       "box": list(tracker.box),
                       "state": tracker.state,
                       "area": int((tracker.prev_mask > 0).sum()),
                       "centroid": (px, py), "median_area": None}
            else:
                res = tracker.update(frame, t)

        shown = overlay_result(frame, res, tracker,
                               prompt_xy if frame_no < 5 else None)
        canvas = draw_canvas(shown, res, t, mode_label, tracker,
                             tracker.events)
        writer.write(canvas)
        if not args.no_display:
            cv2.imshow("SAM 2-style tracking", canvas)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
        frame_no += 1
        t = frame_no / SIM_FPS

    writer.release()
    if cap is not None:
        cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    summ = tracker.summary(t, frame_no)
    summ["loop_passes"] = loop_passes
    with open(events_path, "w", encoding="utf-8") as fh:
        for et, text in tracker.events:
            fh.write(f"{et:7.1f}s  {text}\n")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("SAM 2-style promptable video object tracking\n")
        fh.write(f"mode: {mode_label}\n")
        fh.write(f"duration: {summ['duration_s']} s "
                 f"({summ['frames']} frames)\n")
        fh.write(f"frames tracked: {summ['tracked_frames']} "
                 f"({summ['tracked_frames'] / max(1, summ['frames']) * 100:.1f}%)\n")
        fh.write(f"frames occluded: {summ['occluded_frames']}\n")
        fh.write(f"occlusion events: {summ['occlusions']}\n")
        fh.write(f"reacquisitions: {summ['reacquisitions']}\n")
        fh.write(f"longest occlusion: {summ['longest_occlusion_s']} s\n")
        fh.write(f"final state: {summ['final_state']}\n")
        fh.write(f"loop passes: {summ['loop_passes']} "
                 f"(tracker re-prompted at each wrap)\n")
        fh.write(f"final mask area: {summ['final_mask_area']} px "
                 f"(median {summ['median_mask_area']} px)\n")
        fh.write("events:\n")
        for et, text in tracker.events:
            fh.write(f"  {et:7.1f}s  {text}\n")

    wall = time.time() - started
    print(f"Done: {frame_no} frames in {t:.1f} s of video "
          f"({wall:.0f} s wall time)")
    print(f"  state: {summ['final_state']} | tracked "
          f"{summ['tracked_frames']}/{summ['frames']} frames | "
          f"occlusions {summ['occlusions']} | reacquired "
          f"{summ['reacquisitions']} | longest occlusion "
          f"{summ['longest_occlusion_s']} s")
    print(f"  video: {args.out}")
    print(f"  events: {events_path}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
