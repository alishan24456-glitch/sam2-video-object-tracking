# SAM 2-Style Promptable Video Object Tracking

A point or box prompt selects ONE object in the first frame and the
system follows it through the video with a live mask, a track trail
and an occlusion / reacquisition log — the video counterpart of the
`sam-segment-anything` build (image masks). This is the prompt ->
segment -> propagate loop of Meta's SAM 2, re-implemented with
classical tools that run on a CPU.

## Run it (CMD)

```
pip install -r requirements.txt
python sam2_video_object_tracking.py --source pick
python sam2_video_object_tracking.py
python sam2_video_object_tracking.py --source 0
python sam2_video_object_tracking.py --source video.mp4 --output tracked.mp4
python sam2_video_object_tracking.py --source video.mp4 --prompt 640,400
python sam2_video_object_tracking.py --source video.mp4 --box 500,300,700,520
python sam2_video_object_tracking.py --simulate --output sim-demo.mp4
python sam2_video_object_tracking.py --simulate --duration 28 --output demo.mp4
```

Running with no arguments (or `--source pick`) opens a file-picker
dialog to choose a video. In the display window, click anywhere to
re-prompt on the current frame (the tracker restarts from that
point); press Q to quit.

## How it works

One pipeline serves both modes: a prompt is resolved once on the
first frame, then the tracker runs on pixels alone.

- **Prompt** — `--prompt X,Y` (a point), `--box x1,y1,x2,y2` (a
  box), a mouse click, or auto (default): YOLOv8n proposes the
  largest person on frame 0, else the largest vehicle. The prompt is
  fixed at frame 0 and is never re-aimed at a tracked position.
  YOLO is used ONLY for this first-frame proposal and never again.
- **Initial mask** — GrabCut seeded by the prompt box, plus an HSV
  colour histogram of the masked pixels. The histogram is the
  object's colour model and stays frozen from the prompt for the
  whole run (see the bugs list for why re-learning it was removed).
- **Tracking** — each frame, sparse Lucas-Kanade flow on points
  inside the previous mask predicts where the object moved; a
  histogram back-projection re-segments it inside the predicted
  box, with a GrabCut polish every 10th frame (accepted only when
  GrabCut agrees with the back-projection mask). The mask, box and
  trail are read off those pixels only.
- **Occlusion** — when the mask area collapses below 30% of its
  running median the state flips to OCCLUDED: the predicted box
  keeps coasting at the last measured velocity (learned only from
  fully-visible frames) and the search box widens. When the colour
  mask comes back near the predicted position the state flips to
  REACQUIRED and the event is logged with a timestamp; occluded
  for more than 5 s reads LOST, and a full-frame colour search then
  sweeps for the target. A centre that leaves the frame reads LEFT
  FRAME. State changes need a 3-frame confirmation so one noisy
  frame cannot flip the board.
- **Looping** — a looped clip is a jump cut back to the start, so
  the tracker is re-prompted at each wrap and the event is logged
  (the demo clip is 13.4 s looped to 30 s).

## Demo

- Real-mode demo (30 s, Pexels "Businessman crossing street near
  outdoor restaurant in Halifax", video 5188850, fixed camera,
  13.4 s looped to 30 s, downloaded for this build under the
  14 Nov fresh-video rule):
  `files/project-demos/sam2-video-object-tracking-demo.mp4`
- LinkedIn-ready copy (H.264, same demo):
  `files/project-demos/linkedin-h264/sam2-video-object-tracking-demo.mp4`
- Verification: `py_compile` clean; demo read back (900 frames,
  30 fps, 1280×720, 30.0 s) and frame-inspected at 5 s, 12 s,
  20 s and 28 s — the green mask hugs the crossing man (white
  shirt) through all three loop passes and the second man in the
  background is never picked up.

Real-mode numbers: 900 frames, tracked 900/900 (100%), 0
occlusions, 0 reacquisitions, final state TRACKING, final mask
7,571 px (median 8,042 px), 2 loop re-prompts (deterministic —
every prompt gives exactly the same 8,310 px initial mask).

Simulation (rendered street, same pipeline) verified as an
internal logic check only: 900 frames, tracked 621 (69.0%),
occlusions 2, reacquisitions 1, longest occlusion 3.2 s, final
state LEFT FRAME — the blue car drives behind the parked van
(OCCLUDED at 9.5 s), reappears (REACQUIRED at 12.6 s, mask grown
back to the full car by 15 s) and drives out of the frame
(LEFT FRAME at 24.9 s). The red decoy car is never tracked.

## Honest framing — read this before trusting the mask

**This is SAM 2-*style* promptable video tracking with a light
OpenCV backend (GrabCut + HSV histogram + optical flow), not
Meta's SAM 2 model.** A true SAM 2 backend would need a SAM 2
checkpoint plus torch and a GPU; no SAM 2 ONNX export runs
usefully on this shared CPU, so the light backend is the shipped
one.

- The mask follows colour/texture regions, so on real footage it
  hugs the object's most distinctive part (here the white shirt)
  rather than a full-body silhouette, and it can bleed onto
  same-coloured background or shrink when the object turns.
- Occlusion on real footage is NOT demonstrated — the crossing man
  passes a thin sign pole and the mask dips (9,682 px to 6,335 px)
  but never drops below the occlusion threshold, so the demo shows
  0 occlusions. The occlusion / reacquisition path is verified in
  simulation only.
- The pole crossing, the loop re-prompts and the second man in the
  background are all logged in the event log.
- Mask area is counted in pixels of the 1280×720 working frame, so
  it is a relative size figure, not a real-world area.

## Bugs found and fixed during verification

1. A `CANDVAS_H` typo crashed the first run (NameError).
2. `cv2.drawMarker` rejects float points under OpenCV 5 — the
   prompt marker is cast to int.
3. GrabCut's k-means init is random: the same prompt gave an
   8,311 px mask once and a 19,023 px mask the next time.
   `cv2.setRNGSeed(42)` before GrabCut makes it deterministic.
4. The per-box label was unreadable where it crossed the bright
   mask / white shirt — a dark backing rectangle was added.
5. Velocity was learned from the visible fragment's centroid while
   the car slid behind the van — that centroid moves at half the
   car's speed, so the coasting box lagged the real car by ~250 px
   at reappearance. Velocity is now learned only from
   fully-visible frames.
6. The box shrank to the visible sliver (18 px wide) while
   occluded, so the search ROI was too small to ever re-find the
   object — the occluded box now restores the last full size.
7. The LOST reacquisition search rejected every contour more than
   ~130 px away (`area - 2*dist^2` against a best score starting
   at -1). It now takes the largest contour within 400 px.
8. The GrabCut polish re-learned the colour model from a fragment
   (the pale window band instead of the blue body) and the tracker
   then followed the window (94×25 px) and never grew back to the
   whole car. The colour model is frozen from the prompt, and a
   polish is accepted only when it agrees with the
   back-projection mask.
9. State flicker — REACQUIRED / OCCLUDED / REACQUIRED within
   0.2 s around the van edge — fixed with a 3-frame confirmation
   on state changes.
10. A LEFT FRAME condition mixed `or` and `and` without
    parentheses, so it could fire on the wrong clause — fixed.
