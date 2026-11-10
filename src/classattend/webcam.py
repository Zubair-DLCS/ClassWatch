"""On-device feature extraction from a webcam or a video file.

Privacy by design: each frame is turned into six numbers (gaze, head pose,
eye openness, face found) and then dropped. No image is written to disk or
kept in memory beyond the current frame. Only the numbers leave this module.

Uses MediaPipe Face Landmarker (478 face landmarks incl. irises). The model
file (~3.7 MB) is downloaded once to ~/.cache/classattend/.
"""
from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import numpy as np

from .simulate import HZ, T

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = Path.home() / ".cache" / "classattend" / "face_landmarker.task"

# MediaPipe face-mesh indices
EYES = {  # outer corner, inner corner, upper lid, lower lid, iris centre
    "left": (33, 133, 159, 145, 468),
    "right": (263, 362, 386, 374, 473),
}
NOSE, LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD, CHIN = 1, 234, 454, 10, 152


def _eye(pts: np.ndarray, idx) -> tuple[float, float, float]:
    outer, inner, top, bottom, iris = (pts[i] for i in idx)
    centre = (outer + inner) / 2
    width = np.linalg.norm(outer - inner) + 1e-6
    height = np.linalg.norm(top - bottom)
    # horizontal: -1 = iris at the image-left corner, +1 = image-right corner
    left_x, right_x = min(outer[0], inner[0]), max(outer[0], inner[0])
    gx = 2 * (iris[0] - left_x) / (right_x - left_x + 1e-6) - 1
    mid_y = (top[1] + bottom[1]) / 2
    gy = 2 * (iris[1] - mid_y) / (height + 0.3 * width)  # stable when the eye is half closed
    return float(gx), float(gy), float(height / width)


def head_angles(matrix: np.ndarray | None, pts: np.ndarray) -> tuple[float, float]:
    """Yaw and pitch in degrees, from MediaPipe's face transform (or a landmark proxy)."""
    if matrix is not None:
        R = np.asarray(matrix)[:3, :3]
        yaw = math.degrees(math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0])))
        pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        return yaw, pitch
    face_w = np.linalg.norm(pts[RIGHT_CHEEK] - pts[LEFT_CHEEK]) + 1e-6
    face_h = np.linalg.norm(pts[CHIN] - pts[FOREHEAD]) + 1e-6
    mid = (pts[RIGHT_CHEEK] + pts[LEFT_CHEEK]) / 2
    yaw = math.degrees(math.asin(np.clip(2 * (pts[NOSE][0] - mid[0]) / face_w, -1, 1)))
    pitch = math.degrees(math.asin(np.clip(2 * (pts[NOSE][1] - (pts[FOREHEAD][1] + pts[CHIN][1]) / 2) / face_h, -1, 1)))
    return yaw, pitch


def frame_features(pts: np.ndarray | None, matrix: np.ndarray | None = None) -> np.ndarray:
    """pts: (478, 2) landmark x, y in image-normalised units, or None if no face.
    Returns [gaze_x, gaze_y, head_yaw, head_pitch, eye_open, face_present]."""
    if pts is None:
        return np.zeros(6, np.float32)
    l, r = _eye(pts, EYES["left"]), _eye(pts, EYES["right"])
    yaw, pitch = head_angles(matrix, pts)
    return np.array([(l[0] + r[0]) / 2, (l[1] + r[1]) / 2, yaw, pitch, (l[2] + r[2]) / 2, 1.0], np.float32)


def to_windows(frames: np.ndarray) -> np.ndarray:
    """(N, 6) per-frame features at HZ -> (W, T, 6) non-overlapping 10-s windows."""
    w = len(frames) // T
    return frames[: w * T].reshape(w, T, frames.shape[1])


def _landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading face model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    opts = vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    return mp, vision.FaceLandmarker.create_from_options(opts)


def extract(source: str | int = 0, max_seconds: float | None = None) -> np.ndarray:
    """Read a video file (path) or webcam (index), sample at HZ, return (N, 6) features.
    Frames are processed one at a time and never stored."""
    import cv2

    mp, lm = _landmarker()
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = 1.0 / HZ
    next_t, out, i = 0.0, [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            i += 1
            if t + 1e-9 < next_t:
                continue
            next_t += step
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
            del frame, rgb  # the image is gone from here on
            if res.face_landmarks:
                pts = np.array([[p.x, p.y] for p in res.face_landmarks[0]], np.float32)
                mat = res.facial_transformation_matrixes[0] if res.facial_transformation_matrixes else None
                out.append(frame_features(pts, mat))
            else:
                out.append(frame_features(None))
            if max_seconds and t >= max_seconds:
                break
    finally:
        cap.release()
        lm.close()
    return np.stack(out) if out else np.zeros((0, 6), np.float32)
