"""Metrics that matter in a classroom, not only accuracy."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

WANDER = 2


def macro_f1(y, p):
    return float(f1_score(y, p.argmax(1), average="macro"))


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error: gap between confidence and accuracy."""
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def reliability(y, p, bins: int = 10):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    xs, accs, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 5:
            xs.append(conf[m].mean()); accs.append((pred[m] == y[m]).mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(accs), np.array(ns)


def risk_coverage(y, p, points=np.linspace(0.1, 1.0, 19)):
    """Accuracy when the model only answers on its most confident fraction of windows."""
    order = np.argsort(-p.max(1))
    correct = (p.argmax(1) == y)[order]
    out = []
    for c in points:
        k = max(1, int(round(c * len(y))))
        out.append(float(correct[:k].mean()))
    return points, np.array(out)


def choose_alert_threshold(y_val, p_val, max_false_alarm: float = 0.10) -> float:
    """Smallest P(wandering) threshold that wrongly flags at most `max_false_alarm`
    of the engaged learners AND at most that share of the thinking learners
    (checked separately on validation data). Thinkers are the group most at
    risk of being wrongly flagged, so they get their own budget."""
    thr = []
    for c in (0, 1):
        s = p_val[y_val == c, WANDER]
        if len(s):
            thr.append(np.quantile(s, 1 - max_false_alarm))
    return float(max(thr)) if thr else 0.5


def alert_report(y, p, threshold: float) -> dict:
    alert = p[:, WANDER] >= threshold
    rate = lambda m: float(alert[m].mean()) if m.any() else float("nan")
    return {
        "wandering_caught": rate(y == WANDER),
        "thinking_wrongly_flagged": rate(y == 1),
        "engaged_wrongly_flagged": rate(y == 0),
        "alert_rate": float(alert.mean()),
    }


def confusion(y, p):
    return confusion_matrix(y, p.argmax(1), labels=[0, 1, 2], normalize="true")
