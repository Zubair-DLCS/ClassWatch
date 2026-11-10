import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live"))

from cues import DoorCounter, Smoother, hand_raised, head_direction, posture  # noqa: E402


def person(moves=None):
    """A front-facing standing person (image y grows downwards); override points by index."""
    kp = np.zeros((17, 3))
    pts = {0: (100, 50), 1: (106, 44), 2: (94, 44), 3: (112, 48), 4: (88, 48),
           5: (125, 100), 6: (75, 100), 7: (130, 150), 8: (70, 150), 9: (130, 195), 10: (70, 195),
           11: (115, 200), 12: (85, 200), 13: (115, 280), 14: (85, 280), 15: (115, 360), 16: (85, 360)}
    pts.update(moves or {})
    for i, (x, y) in pts.items():
        kp[i] = (x, y, 0.9)
    return kp


def test_front_and_turned_and_away():
    assert head_direction(person()) == "front"
    assert head_direction(person({0: (93, 50)})) == "turned"     # nose beside the eyes
    kp = person(); kp[[0, 1, 2], 2] = 0.1                           # face hidden, back of head
    assert head_direction(kp) == "away"


def test_head_down():
    assert head_direction(person({0: (100, 92), 1: (106, 86), 2: (94, 86)})) == "down"


def test_hand_raised():
    assert not hand_raised(person())
    assert hand_raised(person({10: (60, 20), 8: (65, 60)}))


def test_sitting_vs_standing():
    assert posture(person()) == "standing"
    assert posture(person({13: (140, 210), 14: (110, 210), 15: (140, 280), 16: (110, 280)})) == "sitting"
    kp = person(); kp[11:, 2] = 0.0                                  # legs behind a desk
    assert posture(kp) == "unknown"


def test_door_counter_counts_crossings_once():
    d = DoorCounter((50, 0), (50, 100))
    for x in (10, 30, 60, 80):          # left -> right
        d.update(1, (x, 50))
    for x in (80, 40):                  # right -> left
        d.update(2, (x, 50))
    assert d.entries + d.exits == 2 and d.entries == 1 and d.exits == 1


def test_smoother_stops_flicker():
    s = Smoother(n=5)
    out = [s(1, "head", v) for v in ["front", "front", "turned", "front", "front"]]
    assert out[-1] == "front" and out[2] == "front"
