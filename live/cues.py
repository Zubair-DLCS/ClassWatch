"""Simple, explainable cues from body keypoints (COCO-17, as returned by YOLO pose).

These are posture rules, not mind reading. "Head down" can mean writing,
reading or sleeping; "turned away" can mean talking to a neighbour or looking
at the board on the side wall. The demo shows them as hints for a teacher.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

VIS = 0.5  # minimum keypoint confidence to trust a point


def _ok(kp: np.ndarray, i: int) -> bool:
    return bool(kp.shape[1] < 3 or kp[i, 2] >= VIS)


def _mid(kp, a, b):
    if _ok(kp, a) and _ok(kp, b):
        return (kp[a, :2] + kp[b, :2]) / 2
    if _ok(kp, a):
        return kp[a, :2]
    if _ok(kp, b):
        return kp[b, :2]
    return None


def shoulder_width(kp) -> float | None:
    if _ok(kp, L_SH) and _ok(kp, R_SH):
        return float(np.linalg.norm(kp[L_SH, :2] - kp[R_SH, :2]))
    return None


def hand_raised(kp: np.ndarray) -> bool:
    """A wrist clearly above the head (or above the shoulders by half a shoulder width)."""
    sw = shoulder_width(kp) or 0.0
    for wr, sh in ((L_WR, L_SH), (R_WR, R_SH)):
        if not _ok(kp, wr):
            continue
        y = kp[wr, 1]  # image y grows downwards
        if _ok(kp, NOSE) and y < kp[NOSE, 1] - 0.1 * sw:
            return True
        if _ok(kp, sh) and sw > 0 and y < kp[sh, 1] - 0.9 * sw:
            return True
    return False


def head_direction(kp: np.ndarray) -> str:
    """'front', 'turned', 'down', 'away' or 'unknown', from face and shoulder points."""
    eyes = int(_ok(kp, L_EYE)) + int(_ok(kp, R_EYE))
    ears = int(_ok(kp, L_EAR)) + int(_ok(kp, R_EAR))
    nose = _ok(kp, NOSE)
    sh = _mid(kp, L_SH, R_SH)
    sw = shoulder_width(kp)
    if not nose and eyes == 0:
        return "away" if ears or sh is not None else "unknown"
    if nose and sh is not None and sw:
        # head bent forward: nose sinks towards the shoulder line
        if (sh[1] - kp[NOSE, 1]) < 0.35 * sw:
            return "down"
    if eyes == 2:
        ex = sorted([kp[L_EYE, 0], kp[R_EYE, 0]])
        span = ex[1] - ex[0] + 1e-6
        rel = (kp[NOSE, 0] - ex[0]) / span if nose else 0.5
        if 0.25 <= rel <= 0.75:  # nose between the eyes: facing the camera
            return "front"
        return "turned"
    return "turned"


def posture(kp: np.ndarray) -> str:
    """'standing', 'sitting' or 'unknown' (legs are often hidden behind desks)."""
    hip, knee = _mid(kp, L_HIP, R_HIP), _mid(kp, L_KNEE, R_KNEE)
    sh = _mid(kp, L_SH, R_SH)
    if hip is None or knee is None or sh is None:
        return "unknown"
    torso = abs(hip[1] - sh[1]) + 1e-6
    thigh_drop = knee[1] - hip[1]  # vertical drop from hip to knee
    ank = _mid(kp, L_ANK, R_ANK)
    if thigh_drop > 0.6 * torso and (ank is None or ank[1] > knee[1] + 0.3 * torso):
        return "standing"
    if thigh_drop < 0.35 * torso:
        return "sitting"
    return "unknown"


@dataclass
class Smoother:
    """Majority vote over the last few frames, so a label does not flicker."""
    n: int = 8
    hist: dict = field(default_factory=dict)

    def __call__(self, track_id: int, key: str, value):
        d = self.hist.setdefault((track_id, key), deque(maxlen=self.n))
        d.append(value)
        vals, counts = np.unique(np.array(list(d), dtype=object).astype(str), return_counts=True)
        top = vals[counts.argmax()]
        return top == "True" if isinstance(value, bool) else str(top)

    def forget(self, alive: set[int]):
        for k in [k for k in self.hist if k[0] not in alive]:
            del self.hist[k]


class DoorCounter:
    """Counts anonymous track IDs crossing a line (e.g. the classroom door)."""

    def __init__(self, p1, p2):
        self.p1, self.p2 = np.array(p1, float), np.array(p2, float)
        self.side: dict[int, float] = {}
        self.entries = self.exits = 0

    def _side(self, pt) -> float:
        d = self.p2 - self.p1
        return float(np.sign(d[0] * (pt[1] - self.p1[1]) - d[1] * (pt[0] - self.p1[0])))

    def _within(self, pt) -> bool:
        d = self.p2 - self.p1
        t = np.dot(np.asarray(pt) - self.p1, d) / (np.dot(d, d) + 1e-9)
        return -0.1 <= t <= 1.1

    def update(self, track_id: int, pt):
        s = self._side(pt)
        prev = self.side.get(track_id)
        if prev is not None and s != 0 and prev != 0 and s != prev and self._within(pt):
            if s > 0:
                self.entries += 1
            else:
                self.exits += 1
        if s != 0:
            self.side[track_id] = s
