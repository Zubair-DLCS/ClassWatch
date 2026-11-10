"""Preprocessing: learner-wise splits and per-learner normalisation.

Per-learner normalisation needs no labels. Each learner's features are compared
with that learner's own baseline, so a student who simply blinks a lot, or sits
with the head turned, is not mistaken for a wandering one. It can run on the
student's own device, so raw numbers never have to be pooled across students.
"""
from __future__ import annotations

import numpy as np

COUNT_LOGS = [0, 1, 2, 4]  # keystrokes, mouse_px, clicks, scroll_backs -> log1p


def transform_logs(logs: np.ndarray) -> np.ndarray:
    out = logs.astype(np.float32).copy()
    out[:, COUNT_LOGS] = np.log1p(out[:, COUNT_LOGS])
    return out


def per_learner_normalise(vision: np.ndarray, logs: np.ndarray, learner: np.ndarray):
    """Z-score each learner's features against that learner's own windows (label-free)."""
    v, l = vision.copy(), transform_logs(logs)
    for who in np.unique(learner):
        m = learner == who
        vv = v[m]
        present = vv[..., 5] > 0.5
        for c in range(5):
            vals = vv[..., c][present]
            mu, sd = (vals.mean(), vals.std() + 1e-6) if vals.size else (0.0, 1.0)
            ch = vv[..., c]
            ch[present] = (ch[present] - mu) / sd
            ch[~present] = 0.0
            vv[..., c] = ch
        v[m] = vv
        mu, sd = l[m].mean(0), l[m].std(0) + 1e-6
        l[m] = (l[m] - mu) / sd
    return v.astype(np.float32), l.astype(np.float32)


class GlobalNormaliser:
    """Baseline: one mean/std for everyone, fitted on the training learners."""

    def fit(self, vision, logs):
        present = vision[..., 5] > 0.5
        self.v_mu = np.array([vision[..., c][present].mean() for c in range(5)])
        self.v_sd = np.array([vision[..., c][present].std() + 1e-6 for c in range(5)])
        l = transform_logs(logs)
        self.l_mu, self.l_sd = l.mean(0), l.std(0) + 1e-6
        return self

    def __call__(self, vision, logs):
        v = vision.copy()
        present = v[..., 5] > 0.5
        v[..., :5] = (v[..., :5] - self.v_mu) / self.v_sd
        v[..., :5][~present] = 0.0
        l = (transform_logs(logs) - self.l_mu) / self.l_sd
        return v.astype(np.float32), l.astype(np.float32)


def learner_split(learner: np.ndarray, seed: int = 0, frac=(0.6, 0.2, 0.2)):
    """Split by learner, so no student appears in both training and testing."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.unique(learner))
    n_tr = int(frac[0] * len(ids))
    n_va = int(frac[1] * len(ids))
    groups = ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]
    return [np.isin(learner, g) for g in groups]
