"""Run every experiment in the README and write results/ (metrics.json + figures).

    python scripts/run_experiments.py            # 3 seeds, ~10 min on a laptop CPU
    python scripts/run_experiments.py --quick    # 1 seed, fewer epochs, for a smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classattend import evaluate as E  # noqa: E402
from classattend.data import GlobalNormaliser, learner_split, per_learner_normalise  # noqa: E402
from classattend.model import FusionModel, fit_temperature, train_model  # noqa: E402
from classattend.simulate import CLASSROOM, LAB, STATES, simulate  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def run_seed(seed: int, epochs: int) -> dict:
    lab = simulate(n_learners=60, windows_per_learner=120, setting=LAB, seed=seed)
    room = simulate(n_learners=20, windows_per_learner=120, setting=CLASSROOM, seed=1000 + seed)
    tr, va, te = learner_split(lab["learner"], seed=seed)
    y = lab["label"]

    # Two ways to normalise: per learner (label-free, on-device) vs one global scale.
    v_pl, l_pl = per_learner_normalise(lab["vision"], lab["logs"], lab["learner"])
    rv_pl, rl_pl = per_learner_normalise(room["vision"], room["logs"], room["learner"])
    g = GlobalNormaliser().fit(lab["vision"][tr], lab["logs"][tr])
    v_g, l_g = g(lab["vision"], lab["logs"])
    rv_g, rl_g = g(room["vision"], room["logs"])

    def fit(use_v, use_l, v, l, p_drop=0.5):
        m = FusionModel(use_v, use_l, modality_dropout=p_drop)
        m = train_model(m, (v[tr], l[tr], y[tr]), (v[va], l[va], y[va]), epochs=epochs, seed=seed)
        return m

    res: dict = {}
    t0 = time.time()
    models = {
        "camera only": fit(True, False, v_pl, l_pl),
        "activity logs only": fit(False, True, v_pl, l_pl),
        "fusion": fit(True, True, v_pl, l_pl),
        "fusion, no modality dropout": fit(True, True, v_pl, l_pl, p_drop=0.0),
        "fusion, global normalisation": fit(True, True, v_g, l_g),
    }
    print(f"  seed {seed}: trained 5 models in {time.time() - t0:.0f}s")

    # 1) which signals matter; 2) lab -> classroom shift
    for name, m in models.items():
        lv, ll, rv, rl = (v_g, l_g, rv_g, rl_g) if "global" in name else (v_pl, l_pl, rv_pl, rl_pl)
        res[f"f1_lab/{name}"] = E.macro_f1(y[te], m.predict_proba(lv[te], ll[te]))
        res[f"f1_classroom/{name}"] = E.macro_f1(room["label"], m.predict_proba(rv, rl))

    # 3) a modality goes missing at test time (camera off / no logs)
    n = te.sum()
    zeros, ones = np.zeros(n, np.float32), np.ones(n, np.float32)
    for name in ("fusion", "fusion, no modality dropout"):
        m = models[name]
        res[f"f1_missing_camera/{name}"] = E.macro_f1(y[te], m.predict_proba(
            np.zeros_like(v_pl[te]), l_pl[te], has_vision=zeros, has_logs=ones))
        res[f"f1_missing_logs/{name}"] = E.macro_f1(y[te], m.predict_proba(
            v_pl[te], np.zeros_like(l_pl[te]), has_vision=ones, has_logs=zeros))
    res["f1_missing_camera/activity logs only"] = res["f1_lab/activity logs only"]
    res["f1_missing_logs/camera only"] = res["f1_lab/camera only"]

    # 4) calibration, abstention and teacher alerts (fusion model, classroom data)
    m = models["fusion"]
    p_raw = m.predict_proba(rv_pl, rl_pl)
    res["ece_before"] = E.ece(room["label"], p_raw)
    res["temperature"] = fit_temperature(m, v_pl[va], l_pl[va], y[va])
    p = m.predict_proba(rv_pl, rl_pl)
    res["ece_after"] = E.ece(room["label"], p)
    cov, acc = E.risk_coverage(room["label"], p)
    res["risk_coverage"] = {"coverage": cov.tolist(), "accuracy": acc.tolist()}
    thr = E.choose_alert_threshold(y[va], m.predict_proba(v_pl[va], l_pl[va]), max_false_alarm=0.10)
    res["alert_threshold"] = thr
    res["alerts"] = E.alert_report(room["label"], p, thr)
    res["confusion_classroom"] = E.confusion(room["label"], p).tolist()
    res["reliability_before"] = [a.tolist() for a in E.reliability(room["label"], p_raw)]
    res["reliability_after"] = [a.tolist() for a in E.reliability(room["label"], p)]
    if seed == 0:
        OUT.mkdir(exist_ok=True)
        torch.save(m.state_dict(), OUT / "fusion_seed0.pt")
    return res


def summarise(runs: list[dict]) -> dict:
    out = {}
    for k, v in runs[0].items():
        if isinstance(v, float):
            vals = np.array([r[k] for r in runs])
            out[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
        elif k == "alerts":
            out[k] = {kk: {"mean": float(np.mean([r[k][kk] for r in runs])),
                           "std": float(np.std([r[k][kk] for r in runs]))} for kk in v}
        elif k == "risk_coverage":
            out[k] = {"coverage": v["coverage"],
                      "accuracy": np.mean([r[k]["accuracy"] for r in runs], 0).tolist()}
        elif k == "confusion_classroom":
            out[k] = np.mean([r[k] for r in runs], 0).tolist()
    return out


def figures(summary: dict, runs: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    blue, orange, grey = "#2f6db5", "#e07b39", "#9aa0a6"

    # Figure 1: modalities + lab vs classroom
    names = ["camera only", "activity logs only", "fusion", "fusion, global normalisation"]
    labels = ["Camera\nonly", "Activity logs\nonly", "Fusion\n(per-learner norm.)", "Fusion\n(global norm.)"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, (where, c) in enumerate([("lab", blue), ("classroom", orange)]):
        m = [summary[f"f1_{where}/{n}"]["mean"] for n in names]
        s = [summary[f"f1_{where}/{n}"]["std"] for n in names]
        ax.bar(x + (i - 0.5) * 0.38, m, 0.36, yerr=s, color=c, label=f"{where} test", capsize=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Macro F1 (unseen learners)")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(1 / 3, color=grey, lw=0.8, ls=":")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Which signals help, and what survives a cheap classroom webcam")
    fig.tight_layout(); fig.savefig(OUT / "fig1_modalities_and_shift.png", dpi=160); plt.close(fig)

    # Figure 2: missing modality
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    groups = [("Camera off", "f1_missing_camera", "activity logs only"),
              ("No activity logs", "f1_missing_logs", "camera only")]
    for gi, (title, key, single) in enumerate(groups):
        vals = [(f"{single}\n(trained for it)", summary[f"{key}/{single}"], grey),
                ("fusion, no\nmodality dropout", summary[f"{key}/fusion, no modality dropout"], orange),
                ("fusion with\nmodality dropout", summary[f"{key}/fusion"], blue)]
        for j, (lab, v, c) in enumerate(vals):
            ax.bar(gi * 4 + j, v["mean"], yerr=v["std"], color=c, capsize=3)
            ax.text(gi * 4 + j, 0.42, lab, ha="center", va="bottom", fontsize=7, rotation=90, color="white")
        ax.text(gi * 4 + 1, 1.0, title, ha="center", fontsize=10)
    ax.set_xticks([])
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Macro F1")
    ax.set_title("When one signal is missing at test time", pad=18)
    fig.tight_layout(); fig.savefig(OUT / "fig2_missing_modality.png", dpi=160); plt.close(fig)

    # Figure 3: calibration + abstention
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.4))
    r0 = runs[0]
    for key, c, lab in [("reliability_before", orange, "before"), ("reliability_after", blue, "after temperature scaling")]:
        xs, acc, _ = (np.array(a) for a in r0[key])
        axs[0].plot(xs, acc, "o-", color=c, label=lab, ms=4)
    axs[0].plot([0, 1], [0, 1], color=grey, ls=":")
    axs[0].set_xlabel("Model confidence"); axs[0].set_ylabel("Actual accuracy")
    axs[0].set_title("Is '80% sure' really 80%?"); axs[0].legend(frameon=False, fontsize=8)
    rc = summary["risk_coverage"]
    axs[1].plot(np.array(rc["coverage"]) * 100, rc["accuracy"], "o-", color=blue, ms=4)
    axs[1].set_xlabel("% of windows the model answers"); axs[1].set_ylabel("Accuracy on those")
    axs[1].set_title("Saying 'not sure' buys accuracy")
    axs[1].invert_xaxis()
    fig.tight_layout(); fig.savefig(OUT / "fig3_calibration_abstention.png", dpi=160); plt.close(fig)

    # Figure 4: confusion matrix
    cm = np.array(summary["confusion_classroom"])
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xticks(range(3), STATES); ax.set_yticks(range(3), STATES)
    ax.set_xlabel("predicted"); ax.set_ylabel("reported (thought probe)")
    ax.set_title("Fusion model, classroom test")
    fig.tight_layout(); fig.savefig(OUT / "fig4_confusion.png", dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs = 1, 4
    torch.set_num_threads(max(1, torch.get_num_threads()))
    OUT.mkdir(exist_ok=True)
    runs = []
    for s in range(a.seeds):
        runs.append(run_seed(s, a.epochs))
    summary = summarise(runs)
    (OUT / "metrics.json").write_text(json.dumps({"summary": summary, "runs": runs}, indent=1))
    figures(summary, runs)
    print(json.dumps({k: v for k, v in summary.items() if k.startswith(("f1", "ece", "alerts"))}, indent=1))


if __name__ == "__main__":
    main()
