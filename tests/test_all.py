import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classattend.data import per_learner_normalise, learner_split  # noqa: E402
from classattend.evaluate import alert_report, choose_alert_threshold, ece  # noqa: E402
from classattend.model import FusionModel  # noqa: E402
from classattend.simulate import T, simulate  # noqa: E402
from classattend.webcam import EYES, frame_features, head_angles, to_windows  # noqa: E402


def test_simulate_shapes_and_privacy_zeroing():
    d = simulate(n_learners=4, windows_per_learner=10, seed=1)
    assert d["vision"].shape == (40, T, 6) and d["logs"].shape == (40, 6)
    assert set(np.unique(d["label"])) <= {0, 1, 2}
    lost = d["vision"][..., 5] == 0
    assert np.all(d["vision"][lost][:, :5] == 0)


def test_learner_split_has_no_overlap():
    learner = np.repeat(np.arange(10), 5)
    tr, va, te = learner_split(learner)
    sets = [set(learner[m]) for m in (tr, va, te)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


def test_per_learner_normalisation_removes_personal_offset():
    d = simulate(n_learners=3, windows_per_learner=30, seed=2)
    v, _ = per_learner_normalise(d["vision"], d["logs"], d["learner"])
    for who in range(3):
        yaw = v[d["learner"] == who][..., 2][d["vision"][d["learner"] == who][..., 5] > 0]
        assert abs(yaw.mean()) < 1e-3


def _face(iris_shift=0.0, eye_h=0.03):
    pts = np.full((478, 2), 0.5, np.float32)
    for side, cx in (("left", 0.4), ("right", 0.6)):
        o, i, top, bot, iris = EYES[side]
        pts[o], pts[i] = [cx - 0.05, 0.4], [cx + 0.05, 0.4]
        pts[top], pts[bot] = [cx, 0.4 - eye_h / 2], [cx, 0.4 + eye_h / 2]
        pts[iris] = [cx + iris_shift, 0.4]
    return pts


def test_gaze_and_eye_openness():
    centre, right = frame_features(_face(0.0)), frame_features(_face(0.04))
    assert abs(centre[0]) < 1e-5 and right[0] > 0.7
    closed = frame_features(_face(eye_h=0.004))
    assert closed[4] < centre[4] and centre[5] == 1.0
    assert np.all(frame_features(None) == 0)


def test_head_angles_from_rotation():
    a = math.radians(20)
    R = np.eye(4)
    R[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    yaw, pitch = head_angles(R, None)
    assert abs(abs(yaw) - 20) < 1e-6 and abs(pitch) < 1e-6


def test_model_handles_missing_modality():
    m = FusionModel()
    v, l = np.random.randn(8, T, 6).astype(np.float32), np.random.randn(8, 6).astype(np.float32)
    p = m.predict_proba(v, l, has_vision=np.ones(8, np.float32), has_logs=np.zeros(8, np.float32))
    assert p.shape == (8, 3) and np.allclose(p.sum(1), 1, atol=1e-5)
    p2 = m.predict_proba(v, l * 100, has_vision=np.ones(8, np.float32), has_logs=np.zeros(8, np.float32))
    assert np.allclose(p, p2, atol=1e-6)  # a missing modality has no effect, whatever it holds


def test_webcam_windows_feed_the_model():
    frames = np.random.randn(T * 4 + 7, 6).astype(np.float32)
    w = to_windows(frames)
    assert w.shape == (4, T, 6)
    with torch.no_grad():
        assert FusionModel().predict_proba(w, np.zeros((4, 6), np.float32)).shape == (4, 3)


def test_alert_threshold_respects_budget_for_thinkers():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 3000)
    p = rng.dirichlet([1, 1, 1], 3000)
    p[y == 2, 2] += 0.5
    p /= p.sum(1, keepdims=True)
    thr = choose_alert_threshold(y, p, max_false_alarm=0.1)
    r = alert_report(y, p, thr)
    assert r["thinking_wrongly_flagged"] <= 0.11 and r["engaged_wrongly_flagged"] <= 0.11
    assert 0 <= ece(y, p) <= 1
