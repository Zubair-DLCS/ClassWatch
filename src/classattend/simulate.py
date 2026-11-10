"""Simulated classroom sessions with three learner states.

Everything produced here is SIMULATED. It is a stand-in for real thought-probe
data, so that the full pipeline (features -> fusion model -> calibrated,
abstaining alerts) can be built and tested before any real study is run.

Each sample is one 10-second window of one learner:

* vision  : (T=50, 6) sequence at 5 Hz, in the same units the webcam extractor
            (``classattend.webcam``) produces:
            gaze_x, gaze_y   iris position inside the eye, roughly in [-1, 1]
            head_yaw, head_pitch   degrees
            eye_open         eye aspect ratio (about 0.28 open, < 0.12 closed)
            face_present     1 if a face was found in that frame, else 0
* logs    : (6,) activity summary for the same window:
            keystrokes, mouse_px, clicks, idle_fraction, scroll_backs, tab_away_fraction
* label   : 0 = engaged (reading / working), 1 = thinking hard, 2 = mind wandering

"Thinking hard" and "mind wandering" are deliberately similar: both are quiet,
still and low in activity. Telling them apart is the point of the project.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STATES = ("engaged", "thinking", "wandering")
VISION_CHANNELS = ("gaze_x", "gaze_y", "head_yaw", "head_pitch", "eye_open", "face_present")
LOG_FEATURES = ("keystrokes", "mouse_px", "clicks", "idle_fraction", "scroll_backs", "tab_away_fraction")

HZ = 5
WINDOW_S = 10
T = HZ * WINDOW_S

# Markov chain over states between consecutive windows.
TRANSITIONS = np.array([
    [0.80, 0.10, 0.10],   # engaged ->
    [0.30, 0.55, 0.15],   # thinking ->
    [0.30, 0.05, 0.65],   # wandering ->
])


@dataclass
class Setting:
    """Recording conditions. 'lab' = good camera; 'classroom' = cheap webcam, poor light."""
    name: str = "lab"
    gaze_noise: float = 0.03
    drop_rate: float = 0.02        # chance per frame of starting a tracking-loss burst
    eye_open_bias: float = 0.0     # poor light makes eyes look more closed
    yaw_jitter: float = 2.0


LAB = Setting("lab")
CLASSROOM = Setting("classroom", gaze_noise=0.07, drop_rate=0.05, eye_open_bias=-0.03, yaw_jitter=4.0)


@dataclass
class Learner:
    """Individual differences: people blink, move and type differently."""
    blink_mult: float
    activity_mult: float
    yaw_offset: float
    pitch_offset: float
    eye_open_base: float

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "Learner":
        return cls(
            blink_mult=float(rng.lognormal(0, 0.3)),
            activity_mult=float(rng.lognormal(0, 0.4)),
            yaw_offset=float(rng.normal(0, 5)),
            pitch_offset=float(rng.normal(-8, 4)),
            eye_open_base=float(rng.normal(0.29, 0.025)),
        )


def _fixations(rng, dur_lo, dur_hi, drift):
    """Long fixations, often looking up or aside. Used both for thinking and for
    'zoning out' - from the outside the two look alike."""
    gx, gy, t = np.zeros(T), np.zeros(T), 0
    while t < T:
        dur = rng.integers(dur_lo, dur_hi)
        if rng.random() < 0.45:
            px, py = rng.uniform(-0.7, 0.7), rng.uniform(-0.7, -0.2)
        else:
            px, py = rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)
        n = min(dur, T - t)
        gx[t:t + n] = px + np.cumsum(rng.normal(0, drift, n)) + rng.normal(0, 0.012, n)
        gy[t:t + n] = py + np.cumsum(rng.normal(0, drift, n)) + rng.normal(0, 0.012, n)
        t += n
    return gx, gy


def _gaze_track(state: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    gx, gy = np.zeros(T), np.zeros(T)
    if state == 0:  # reading: left-to-right sweeps, line by line
        x, y, t = -0.4, -0.3, 0
        while t < T:
            line_len = rng.integers(7, 13)
            xs = np.linspace(-0.4, 0.4, line_len) + rng.normal(0, 0.02, line_len)
            n = min(line_len, T - t)
            gx[t:t + n], gy[t:t + n] = xs[:n], y
            t += n
            y = y + 0.08 if y < 0.4 else -0.3
        if rng.random() < 0.3:  # sometimes looking at the keyboard / notes
            s = rng.integers(0, T - 10)
            gy[s:s + 10] = rng.uniform(0.6, 0.9)
    elif state == 1:  # thinking: few long fixations, often looking slightly away
        gx, gy = _fixations(rng, 12, 26, drift=0.002)
    else:  # wandering: either a blank stare or a slow, unstructured drift
        if rng.random() < 0.5:  # "zoning out": empty fixations that look like thinking
            gx, gy = _fixations(rng, 15, 35, drift=0.006)
        else:
            x, y = rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)
            for t in range(T):
                x += rng.normal(0, 0.05) - 0.08 * x
                y += rng.normal(0, 0.05) - 0.08 * y
                gx[t], gy[t] = x, y
        if rng.random() < 0.25:
            s = rng.integers(0, T - 8)
            gx[s:s + 8] = rng.choice([-1, 1]) * rng.uniform(0.8, 1.0)
    return gx, gy


def _vision(state: int, who: Learner, where: Setting, rng: np.random.Generator) -> np.ndarray:
    gx, gy = _gaze_track(state, rng)
    gx = gx + rng.normal(0, where.gaze_noise, T)
    gy = gy + rng.normal(0, where.gaze_noise, T)

    head_sd = {0: 1.5, 1: 1.0, 2: 1.8}[state]
    yaw = who.yaw_offset + np.cumsum(rng.normal(0, head_sd * 0.3, T)) + rng.normal(0, where.yaw_jitter * 0.3, T)
    pitch = who.pitch_offset + np.cumsum(rng.normal(0, head_sd * 0.3, T))
    if state in (1, 2):
        yaw = yaw + 6 * gx  # people turn the head slightly when looking away to think
    if state == 2:
        yaw = yaw + 25 * np.sign(gx) * (np.abs(gx) > 0.75)  # head follows an off-screen glance

    base_blinks_per_min = {0: 15, 1: 11, 2: 17}[state]
    p_blink = base_blinks_per_min * who.blink_mult / 60 / HZ
    eye = np.full(T, who.eye_open_base - (0.012 if state == 2 else 0.0)) + rng.normal(0, 0.01, T)
    eye[rng.random(T) < p_blink] = rng.uniform(0.04, 0.10)
    eye = eye + where.eye_open_bias

    present = np.ones(T)
    t = 0
    while t < T:
        if rng.random() < where.drop_rate:
            n = rng.integers(2, 8)
            present[t:t + n] = 0
            t += n
        t += 1
    x = np.stack([gx, gy, yaw, pitch, eye, present], axis=1)
    x[present == 0, :5] = 0.0  # nothing is known when the face is lost
    return x.astype(np.float32)


def _logs(state: int, who: Learner, rng: np.random.Generator) -> np.ndarray:
    a = who.activity_mult
    if state == 0:
        keys, mouse, clicks = rng.poisson(12 * a), rng.gamma(2, 150 * a), rng.poisson(2 * a)
        idle, back = rng.beta(2, 6), rng.poisson(0.8)
        away = 0.0
    elif state == 1:
        keys, mouse, clicks = rng.poisson(3 * a), rng.gamma(2, 40 * a), rng.poisson(0.5 * a)
        idle, back = rng.beta(6, 3), rng.poisson(1.2)
        away = 0.0
    else:
        keys, mouse, clicks = rng.poisson(2 * a), rng.gamma(2, 60 * a), rng.poisson(0.5 * a)
        idle, back = rng.beta(7, 2.5), rng.poisson(0.3)
        away = rng.beta(2, 5) if rng.random() < 0.25 else 0.0
    return np.array([keys, mouse, clicks, idle, back, away], dtype=np.float32)


def _other(state: int) -> int | None:
    """The state this one is most easily confused with."""
    return {0: 1, 1: 2, 2: 1}[state]


def _vision_of(state, who, where, rng, ambiguity):
    v = _vision(state, who, where, rng)
    p = ambiguity if state else ambiguity / 3
    if rng.random() < p:  # part of the window shows the look-alike state
        cut = rng.integers(int(0.3 * T), int(0.7 * T))
        o = _vision(_other(state), who, where, rng)
        if rng.random() < 0.5:
            v[cut:] = o[cut:]
        else:
            v[:cut] = o[:cut]
    return v


def _logs_of(state, who, rng, ambiguity):
    l = _logs(state, who, rng)
    p = ambiguity if state else ambiguity / 3
    if rng.random() < p:
        w = rng.uniform(0.4, 0.7)
        l = (1 - w) * l + w * _logs(_other(state), who, rng)
    return l


def simulate(n_learners: int = 60, windows_per_learner: int = 120, setting: Setting = LAB,
             label_noise: float = 0.08, ambiguity: float = 0.4, seed: int = 0) -> dict[str, np.ndarray]:
    """Simulate sessions. Returns arrays: vision (N,T,6), logs (N,6), label, learner."""
    rng = np.random.default_rng(seed)
    V, L, Y, ID = [], [], [], []
    for i in range(n_learners):
        who = Learner.sample(rng)
        state = rng.choice(3, p=[0.5, 0.25, 0.25])
        for _ in range(windows_per_learner):
            state = rng.choice(3, p=TRANSITIONS[state])
            # Each modality is sometimes misleading, independently of the other:
            # a thinker can look like a day-dreamer on camera, a day-dreamer can
            # click around like a thinker. This is why one signal is not enough.
            V.append(_vision_of(state, who, setting, rng, ambiguity))
            L.append(_logs_of(state, who, rng, ambiguity))
            # thought-probe answers are imperfect: learners mis-report sometimes
            y = state if rng.random() > label_noise else rng.choice([s for s in range(3) if s != state])
            Y.append(y)
            ID.append(i)
    return {"vision": np.stack(V), "logs": np.stack(L), "label": np.array(Y), "learner": np.array(ID)}
