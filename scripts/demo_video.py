"""Run the trained fusion model on your own webcam recording.

    python scripts/demo_video.py my_lecture_clip.mp4
    python scripts/demo_video.py --webcam --seconds 120

Only six numbers per frame are kept (see classattend/webcam.py); the video
itself is never saved. There are no activity logs here, so the model runs
with the camera branch alone and says so. It was trained on SIMULATED data,
so read its output as a demonstration of the pipeline, not as a judgement
about the person in the video.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from classattend.data import per_learner_normalise  # noqa: E402
from classattend.model import FusionModel  # noqa: E402
from classattend.simulate import STATES, WINDOW_S  # noqa: E402
from classattend.webcam import extract, to_windows  # noqa: E402

UNSURE = 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", help="path to a video file")
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "demo"))
    a = ap.parse_args()
    if not a.video and not a.webcam:
        ap.error("give a video file or --webcam")

    frames = extract(0 if a.webcam else a.video, a.seconds)
    win = to_windows(frames)
    if len(win) < 3:
        sys.exit(f"need at least {3 * WINDOW_S} s of video, got {len(frames) / 5:.0f} s")
    print(f"{len(frames)} frames -> {len(win)} windows; face found in {frames[:, 5].mean():.0%} of frames")

    logs = np.zeros((len(win), 6), np.float32)
    v, _ = per_learner_normalise(win, logs, np.zeros(len(win), int))  # baseline = this person only
    model = FusionModel()
    model.load_state_dict(torch.load(ROOT / "results" / "fusion_seed0.pt"))
    p = model.predict_proba(v, logs, has_vision=np.ones(len(win), np.float32),
                            has_logs=np.zeros(len(win), np.float32))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "features_per_frame.csv", frames, delimiter=",", fmt="%.4f",
               header="gaze_x,gaze_y,head_yaw,head_pitch,eye_open,face_present", comments="")
    rows = ["start_s,engaged,thinking,wandering,decision"]
    for i, pr in enumerate(p):
        dec = STATES[pr.argmax()] if pr.max() >= UNSURE else "not sure"
        rows.append(f"{i * WINDOW_S},{pr[0]:.3f},{pr[1]:.3f},{pr[2]:.3f},{dec}")
        print(f"{i * WINDOW_S:5d}s  {dec:10s}  " + "  ".join(f"{s}={x:.2f}" for s, x in zip(STATES, pr)))
    (out / "timeline.csv").write_text("\n".join(rows) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(p)) * WINDOW_S
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.stackplot(t, p.T, labels=STATES, colors=["#2f6db5", "#8fb8e8", "#e07b39"], step="post")
    ax.set_xlabel("seconds"); ax.set_ylabel("probability"); ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title("Camera-only estimate (demo; model trained on simulated data)")
    fig.tight_layout(); fig.savefig(out / "timeline.png", dpi=150)
    print(f"saved {out}/timeline.csv, timeline.png and features_per_frame.csv (no images)")


if __name__ == "__main__":
    main()
