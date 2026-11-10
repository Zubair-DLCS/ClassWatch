"""ClassWatch Live: a classroom demo that runs on your laptop webcam.

    python live/classwatch_live.py                      # webcam 0
    python live/classwatch_live.py --source 1           # another camera
    python live/classwatch_live.py --source lecture.mp4 # a video file
    python live/classwatch_live.py --phones --door 0.85,0.1,0.85,0.95

What it shows, live:
  * people in view, each with an anonymous number (#3), not a name
  * hands raised (and how many times hands went up this session)
  * where heads point: front / turned / down / away, and % facing front
  * standing vs sitting (when legs are visible)
  * phones in view (optional)
  * people crossing a door line, in and out (optional)
  * a 60-second "facing front" trend

Privacy by default: faces are blurred on screen, no video or image is saved,
and IDs are random numbers that are forgotten when the program closes.
`--log` saves only class-level counts per second (no IDs, no images).

Keys:  q / Esc quit   b blur on/off   k skeleton on/off   r reset counters   f fullscreen
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from cues import NOSE, L_EYE, R_EYE, L_EAR, R_EAR, VIS, DoorCounter, Smoother, hand_raised, head_direction, posture, shoulder_width

COLOURS = {  # BGR
    "front": (80, 175, 60),
    "turned": (0, 165, 255),
    "down": (200, 120, 40),
    "away": (60, 60, 220),
    "unknown": (160, 160, 160),
}
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]
PANEL_W = 330
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def blur_face(img, kp, box):
    pts = [kp[i, :2] for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if kp[i, 2] >= VIS]
    x1, y1, x2, y2 = box
    if pts:
        pts = np.array(pts)
        cx, cy = pts.mean(0)
        sw = shoulder_width(kp) or 0
        r = max(0.45 * sw, 1.6 * np.ptp(pts[:, 0]) if len(pts) > 1 else 0, 0.18 * (x2 - x1), 18)
    else:  # no face points: blur the top of the box
        cx, cy, r = (x1 + x2) / 2, y1 + 0.15 * (y2 - y1), max(0.25 * (x2 - x1), 18)
    h, w = img.shape[:2]
    a, b = int(max(cx - r, 0)), int(max(cy - 1.2 * r, 0))
    c, d = int(min(cx + r, w)), int(min(cy + r, h))
    if c - a > 4 and d - b > 4:
        roi = img[b:d, a:c]
        small = cv2.resize(roi, (max(1, (c - a) // 12), max(1, (d - b) // 12)), interpolation=cv2.INTER_LINEAR)
        img[b:d, a:c] = cv2.resize(small, (c - a, d - b), interpolation=cv2.INTER_NEAREST)


def text(img, s, org, scale=0.6, colour=(235, 235, 235), thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def draw_panel(h, stats, trend, fps, blur, door_on, phones_on):
    p = np.full((h, PANEL_W, 3), 32, np.uint8)
    y = 38
    text(p, "ClassWatch Live", (16, y), 0.85, (255, 255, 255), 2); y += 34
    text(p, f"People in view       {stats['people']}", (16, y), 0.65); y += 30
    text(p, f"Hands raised now     {stats['hands']}", (16, y), 0.65, (120, 230, 255)); y += 26
    text(p, f"Hands raised (total) {stats['hand_events']}", (16, y), 0.55, (180, 180, 180)); y += 32

    text(p, "Heads", (16, y), 0.6, (255, 255, 255)); y += 24
    n = max(stats["people"], 1)
    for k in ("front", "turned", "down", "away"):
        cnt = stats[k]
        cv2.rectangle(p, (16, y - 12), (16 + int(160 * cnt / n), y + 2), COLOURS[k], -1)
        text(p, f"{k:7s} {cnt}", (186, y), 0.5)
        y += 22
    y += 6
    pct = 100 * stats["front"] / stats["people"] if stats["people"] else 0
    text(p, f"Facing front  {pct:3.0f}%", (16, y), 0.7, (120, 230, 120), 2); y += 30

    text(p, f"Standing {stats['standing']}   Sitting {stats['sitting']}", (16, y), 0.55); y += 26
    if phones_on:
        text(p, f"Phones visible {stats['phones']}", (16, y), 0.55, (150, 150, 255)); y += 26
    if door_on:
        text(p, f"Door   in {stats['entries']}   out {stats['exits']}", (16, y), 0.55, (255, 220, 150)); y += 26

    # 60-second trend of "% facing front"
    y += 10
    text(p, "Facing front, last 60 s", (16, y), 0.5, (200, 200, 200)); y += 8
    gx, gy, gw, gh = 16, y, PANEL_W - 32, 70
    cv2.rectangle(p, (gx, gy), (gx + gw, gy + gh), (70, 70, 70), 1)
    if len(trend) > 1:
        vals = np.array(trend)
        xs = gx + np.linspace(0, gw, len(vals))
        ys = gy + gh - vals / 100 * gh
        cv2.polylines(p, [np.stack([xs, ys], 1).astype(np.int32)], False, (120, 230, 120), 2, cv2.LINE_AA)
    y += gh + 26

    text(p, f"{fps:4.1f} FPS", (16, h - 52), 0.5, (170, 170, 170))
    text(p, ("Faces blurred" if blur else "Blur OFF") + " | nothing saved", (16, h - 30), 0.5,
         (120, 230, 120) if blur else (60, 60, 220))
    text(p, "q quit  b blur  k skeleton  r reset", (16, h - 10), 0.42, (140, 140, 140))
    return p


def asset(name: str) -> str:
    """Model files bundled inside the Windows app, else the name (Ultralytics downloads it)."""
    for base in (getattr(sys, "_MEIPASS", None), os.path.dirname(os.path.abspath(__file__))):
        if base:
            for p in (os.path.join(base, "models", name), os.path.join(base, name)):
                if os.path.exists(p):
                    return p
    return name


def parse_source(s: str):
    return int(s) if s.isdigit() else s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index (0, 1, ...) or a video/image file")
    ap.add_argument("--model", default="yolo11n-pose.pt", help="pose model (n = fastest, s/m = more accurate)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--phones", action="store_true", help="also detect mobile phones (slower)")
    ap.add_argument("--door", default=None, help="door line as x1,y1,x2,y2 in 0..1 image fractions")
    ap.add_argument("--no-blur", action="store_true", help="show faces (only with everyone's consent)")
    ap.add_argument("--log", default=None, help="CSV of class-level counts per second (no IDs, no images)")
    ap.add_argument("--save", default=None, help="save the annotated output (video or image) - for testing")
    ap.add_argument("--no-show", action="store_true", help="run without a window")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args(argv)

    from ultralytics import YOLO
    pose = YOLO(asset(a.model))
    det = YOLO(asset("yolo11n.pt")) if a.phones else None

    src = parse_source(a.source)
    single_image = isinstance(src, str) and Path(src).suffix.lower() in IMAGE_EXT
    cap = None
    if not single_image:
        if isinstance(src, int) and sys.platform == "win32":
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)  # opens laptop cameras much faster on Windows
            if not cap.isOpened():
                cap = cv2.VideoCapture(src)
        else:
            cap = cv2.VideoCapture(src)
        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera or file {a.source!r}. "
                             "Check that no other app (Zoom, Teams, Camera) is using it.")

    smooth = Smoother()
    door = None
    blur, skeleton = not a.no_blur, True
    hand_up_before: dict[int, bool] = {}
    hand_events = 0
    trend: deque = deque(maxlen=60)
    last_second, sec_vals = time.time(), []
    phone_boxes, frame_i = [], 0
    fps, t_prev = 0.0, time.time()
    writer = None
    log_f = log_w = None
    if a.log:
        log_f = open(a.log, "w", newline="")
        log_w = csv.writer(log_f)
        log_w.writerow(["time", "people", "hands_raised", "facing_front_pct", "turned", "down", "away",
                        "standing", "sitting", "phones", "door_in", "door_out"])
    win = "ClassWatch Live"
    if not a.no_show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        if single_image:
            if frame_i:
                break
            frame = cv2.imread(src)
            ok = frame is not None
        else:
            ok, frame = cap.read()
        if not ok:
            break
        if isinstance(src, int):
            frame = cv2.flip(frame, 1)  # mirror view feels natural on a laptop
        h, w = frame.shape[:2]
        if a.door and door is None:
            x1, y1, x2, y2 = (float(v) for v in a.door.split(","))
            door = DoorCounter((x1 * w, y1 * h), (x2 * w, y2 * h))

        res = (pose.predict if single_image else pose.track)(
            frame, conf=a.conf, imgsz=a.imgsz, verbose=False,
            **({} if single_image else {"persist": True, "tracker": "bytetrack.yaml"}))[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        ids = (res.boxes.id.cpu().numpy().astype(int) if res.boxes is not None and res.boxes.id is not None
               else -1 - np.arange(len(boxes)))
        kps = res.keypoints.data.cpu().numpy() if res.keypoints is not None else np.zeros((0, 17, 3))

        if det is not None and frame_i % 3 == 0:
            r = det.predict(frame, conf=0.35, classes=[67], imgsz=a.imgsz, verbose=False)[0]
            phone_boxes = r.boxes.xyxy.cpu().numpy().astype(int).tolist()

        stats = {k: 0 for k in ("people", "hands", "front", "turned", "down", "away", "standing", "sitting")}
        stats.update(people=len(boxes), phones=len(phone_boxes), hand_events=hand_events,
                     entries=door.entries if door else 0, exits=door.exits if door else 0)
        view = frame.copy()
        people = []
        for box, tid, kp in zip(boxes, ids, kps):
            hd = smooth(tid, "head", head_direction(kp))
            up = smooth(tid, "hand", hand_raised(kp))
            ps = smooth(tid, "posture", posture(kp))
            stats[hd] = stats.get(hd, 0) + 1
            stats["hands"] += up
            if ps in ("standing", "sitting"):
                stats[ps] += 1
            if up and not hand_up_before.get(tid, False):
                hand_events += 1
            hand_up_before[tid] = up
            if door is not None:
                door.update(tid, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            people.append((box, tid, kp, hd, up, ps))
        stats["hand_events"] = hand_events
        if door:
            stats["entries"], stats["exits"] = door.entries, door.exits
        smooth.forget(set(int(t) for t in ids))

        # draw: blur first, so nothing is drawn on top of a visible face
        if blur:
            for box, tid, kp, *_ in people:
                blur_face(view, kp, box)
        for box, tid, kp, hd, up, ps in people:
            x1, y1, x2, y2 = box.astype(int)
            c = COLOURS.get(hd, COLOURS["unknown"])
            cv2.rectangle(view, (x1, y1), (x2, y2), c, 2)
            if skeleton:
                for i, j in SKELETON:
                    if kp[i, 2] >= VIS and kp[j, 2] >= VIS:
                        cv2.line(view, tuple(kp[i, :2].astype(int)), tuple(kp[j, :2].astype(int)), c, 2, cv2.LINE_AA)
            label = f"#{tid if tid >= 0 else '?'} {hd}" + ("  HAND UP" if up else "") + (f"  {ps}" if ps != "unknown" else "")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(view, (x1, max(y1 - th - 10, 0)), (x1 + tw + 8, max(y1, th + 10)), c, -1)
            text(view, label, (x1 + 4, max(y1 - 6, th + 4)), 0.55, (255, 255, 255))
        for (px1, py1, px2, py2) in phone_boxes:
            cv2.rectangle(view, (px1, py1), (px2, py2), (150, 150, 255), 2)
            text(view, "phone", (px1, max(py1 - 6, 12)), 0.5, (150, 150, 255))
        if door is not None:
            cv2.line(view, tuple(door.p1.astype(int)), tuple(door.p2.astype(int)), (255, 220, 150), 2)
            text(view, "door", tuple((door.p1 + 6).astype(int)), 0.5, (255, 220, 150))

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if fps else 1 / max(now - t_prev, 1e-6)
        t_prev = now
        sec_vals.append(100 * stats["front"] / stats["people"] if stats["people"] else 0)
        if now - last_second >= 1 or single_image:
            trend.append(float(np.mean(sec_vals)))
            if log_w:
                log_w.writerow([time.strftime("%H:%M:%S"), stats["people"], stats["hands"], round(trend[-1]),
                                stats["turned"], stats["down"], stats["away"], stats["standing"],
                                stats["sitting"], stats["phones"], stats["entries"], stats["exits"]])
            last_second, sec_vals = now, []

        out = np.hstack([view, draw_panel(h, stats, trend, fps, blur, door is not None, det is not None)])
        if a.save:
            if single_image:
                cv2.imwrite(a.save, out)
            else:
                if writer is None:
                    writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                             cap.get(cv2.CAP_PROP_FPS) or 15, (out.shape[1], out.shape[0]))
                writer.write(out)
        if not a.no_show:
            cv2.imshow(win, out)
            k = cv2.waitKey(0 if single_image else 1) & 0xFF
            if k in (ord("q"), 27):
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if k == ord("b"):
                blur = not blur
            if k == ord("k"):
                skeleton = not skeleton
            if k == ord("r"):
                hand_events = 0
                hand_up_before.clear()
                trend.clear()
                if door:
                    door.entries = door.exits = 0
            if k == ord("f"):
                full = cv2.getWindowProperty(win, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL if full else cv2.WINDOW_FULLSCREEN)
        frame_i += 1
        if a.max_frames and frame_i >= a.max_frames:
            break

    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()
    if log_f:
        log_f.close()
    if not a.no_show:
        cv2.destroyAllWindows()
    print(f"done: {frame_i} frames, hands raised {hand_events} times"
          + (f", door in {door.entries} / out {door.exits}" if door else ""))


if __name__ == "__main__":
    main()
