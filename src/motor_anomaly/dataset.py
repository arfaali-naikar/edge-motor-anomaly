"""
Turn simulated segments into the arrays the autoencoder wants.

The important thing this module gets right: **the split is by segment, not by
window**. Windows overlap by 50%, so if you split at window level, adjacent
windows from the same 4-second recording end up in both train and test and
your validation score becomes fiction. Window-level splitting produces
inflated validation scores, roughly 0.99 AUC, because adjacent overlapping
windows land in both splits.

Also note the autoencoder trains on healthy windows *only*. Faults are never
shown during training -- they exist purely to set and evaluate the threshold.
That's the point of the semi-supervised framing: in a real factory you have
years of healthy data and almost no labelled failures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import extract_segment


@dataclass
class Scaler:
    """Z-score, stored explicitly so the exact same numbers can be baked into
    the firmware header. sklearn's StandardScaler would work fine here but I'd
    then have to pickle it and re-implement it in C anyway."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        # A constant feature would divide by zero. Clamp rather than drop the
        # column, so the vector length stays fixed at 46 for the firmware.
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        return cls(np.asarray(d["mean"], np.float32), np.asarray(d["scale"], np.float32))


def windows_from_segments(segments: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (X [n, 46], labels [n] str, severity [n] float)"""
    xs, labels, sevs = [], [], []
    for seg in segments:
        w = extract_segment(seg["accel"], seg["rpm"], seg["fs"])
        xs.append(w)
        labels += [seg["fault"]] * len(w)
        sevs += [seg["severity"]] * len(w)
    return np.concatenate(xs), np.asarray(labels), np.asarray(sevs, dtype=np.float32)


def split_segments(
    segments: list[dict], val_frac: float = 0.15, test_frac: float = 0.20, seed: int = 11
) -> tuple[list, list, list]:
    """Segment-level split, stratified by fault so a rare class can't land
    entirely in test."""
    rng = np.random.default_rng(seed)
    by_fault: dict[str, list] = {}
    for seg in segments:
        by_fault.setdefault(seg["fault"], []).append(seg)

    train, val, test = [], [], []
    for fault, segs in by_fault.items():
        idx = rng.permutation(len(segs))
        n_test = max(1, int(round(len(segs) * test_frac)))
        n_val = max(1, int(round(len(segs) * val_frac)))
        for j, i in enumerate(idx):
            if j < n_test:
                test.append(segs[i])
            elif j < n_test + n_val:
                val.append(segs[i])
            else:
                train.append(segs[i])
    return train, val, test


def build(segments: list[dict], seed: int = 11) -> dict:
    """Everything train.py needs, in one dict."""
    tr_segs, va_segs, te_segs = split_segments(segments, seed=seed)

    x_tr, y_tr, _ = windows_from_segments(tr_segs)
    x_va, y_va, s_va = windows_from_segments(va_segs)
    x_te, y_te, s_te = windows_from_segments(te_segs)

    # Fit the scaler on healthy training windows only. Fitting on everything
    # would leak fault statistics into the normalisation -- a subtle version
    # of the same mistake as training on the test set.
    healthy_tr = x_tr[y_tr == "healthy"]
    scaler = Scaler.fit(healthy_tr)

    return {
        "scaler": scaler,
        "x_train": scaler.transform(healthy_tr),  # healthy only -- see module docstring
        "x_val_healthy": scaler.transform(x_va[y_va == "healthy"]),
        "x_val": scaler.transform(x_va),
        "y_val": y_va,
        "sev_val": s_va,
        "x_test": scaler.transform(x_te),
        "y_test": y_te,
        "sev_test": s_te,
        "n_segments": {"train": len(tr_segs), "val": len(va_segs), "test": len(te_segs)},
    }
