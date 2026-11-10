"""A small two-branch fusion model with modality dropout, calibration and abstention."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    """1-D convolutions over time, then a GRU: gaze, head and eye signals -> 64-d."""

    def __init__(self, in_ch: int = 6, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.GELU(),
        )
        self.gru = nn.GRU(32, hidden, batch_first=True)

    def forward(self, x):                      # x: (B, T, C)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        _, last = self.gru(h)
        return last[-1]


class LogEncoder(nn.Module):
    def __init__(self, in_dim: int = 6, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, hidden), nn.GELU())

    def forward(self, x):
        return self.net(x)


class FusionModel(nn.Module):
    """Late fusion. Each modality also passes an 'available' flag, so the model knows
    when a modality is missing (camera off, no activity log) instead of guessing."""

    def __init__(self, use_vision: bool = True, use_logs: bool = True, n_classes: int = 3,
                 modality_dropout: float = 0.5):
        super().__init__()
        self.use_vision, self.use_logs, self.p_drop = use_vision, use_logs, modality_dropout
        self.vision = VisionEncoder() if use_vision else None
        self.logs = LogEncoder() if use_logs else None
        d = (64 + 1 if use_vision else 0) + (32 + 1 if use_logs else 0)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, n_classes))
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, vision, logs, has_vision=None, has_logs=None):
        B = vision.shape[0]
        dev = vision.device
        hv = torch.ones(B, 1, device=dev) if has_vision is None else has_vision.float().view(B, 1)
        hl = torch.ones(B, 1, device=dev) if has_logs is None else has_logs.float().view(B, 1)
        if self.training and self.use_vision and self.use_logs and self.p_drop > 0:
            # randomly hide one modality so the model learns to cope without it
            r = torch.rand(B, 1, device=dev)
            hv = hv * (r >= self.p_drop / 2).float()
            hl = hl * ((r < self.p_drop / 2) | (r >= self.p_drop)).float()
        parts = []
        if self.use_vision:
            parts += [self.vision(vision) * hv, hv]
        if self.use_logs:
            parts += [self.logs(logs) * hl, hl]
        return self.head(torch.cat(parts, dim=1)) / self.temperature

    @torch.no_grad()
    def predict_proba(self, vision, logs, has_vision=None, has_logs=None, batch: int = 1024) -> np.ndarray:
        self.eval()
        out = []
        for i in range(0, len(vision), batch):
            sl = slice(i, i + batch)
            hv = None if has_vision is None else torch.as_tensor(has_vision[sl])
            hl = None if has_logs is None else torch.as_tensor(has_logs[sl])
            logits = self(torch.as_tensor(vision[sl]), torch.as_tensor(logs[sl]), hv, hl)
            out.append(F.softmax(logits, 1).numpy())
        return np.concatenate(out)


def train_model(model: FusionModel, tr, va, epochs: int = 25, lr: float = 2e-3, seed: int = 0,
                batch: int = 128, verbose: bool = False) -> FusionModel:
    """tr/va: tuples (vision, logs, label). Keeps the weights with the best validation loss."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Vt, Lt, Yt = (torch.as_tensor(a) for a in tr)
    Vv, Lv, Yv = (torch.as_tensor(a) for a in va)
    counts = np.bincount(tr[2], minlength=3)
    weight = torch.as_tensor(len(tr[2]) / (3 * np.maximum(counts, 1)), dtype=torch.float32)
    best, best_state = float("inf"), None
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Yt), generator=g)
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(model(Vt[idx], Lt[idx]), Yt[idx], weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = F.cross_entropy(model(Vv, Lv), Yv, weight=weight).item()
        if verbose:
            print(f"epoch {ep:2d}  val loss {vl:.4f}")
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def fit_temperature(model: FusionModel, vision, logs, labels) -> float:
    """Temperature scaling on validation data, so that '80% sure' means right ~80% of the time."""
    model.eval()
    model.temperature.fill_(1.0)
    logits = model(torch.as_tensor(vision), torch.as_tensor(logs))
    y = torch.as_tensor(labels)
    grid = torch.linspace(0.5, 4.0, 71)
    nll = torch.stack([F.cross_entropy(logits / t, y) for t in grid])
    t = float(grid[nll.argmin()])
    model.temperature.fill_(t)
    return t
