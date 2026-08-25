"""
Score the model on the held-out test split.

Run: python -m motor_anomaly.evaluate

Headline AUC is the least interesting number here. What actually decides
whether this is deployable is the two breakdowns:

  * per fault type   -- is there a failure mode it's blind to?
  * per severity     -- how early does it catch things? A detector that only
                        fires at severity 0.9 is a very expensive way of
                        telling you the motor is already broken.

Writes artifacts/report.json and prints a table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from .model import reconstruction_error


def binary_metrics(scores: np.ndarray, is_anomaly: np.ndarray, threshold: float) -> dict:
    pred = scores > threshold
    tp = int(np.sum(pred & is_anomaly))
    fp = int(np.sum(pred & ~is_anomaly))
    fn = int(np.sum(~pred & is_anomaly))
    tn = int(np.sum(~pred & ~is_anomaly))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC. Avoids pulling in sklearn for one number, and handles
    ties correctly via average ranks."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(ranks[order[i : j + 1]])
        i = j + 1

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels.astype(bool)].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args(argv)
    art = Path(args.artifacts)

    meta = json.loads((art / "threshold.json").read_text())
    cache = np.load(art / "test_split.npz", allow_pickle=True)
    x, y, sev = cache["x"], cache["y"], cache["sev"]
    thr = meta["threshold"]

    model = keras.models.load_model(art / "model.keras")
    scores = reconstruction_error(model, x)
    is_anom = (y != "healthy")

    overall = binary_metrics(scores, is_anom, thr)
    auc = roc_auc(scores, is_anom.astype(int))

    # --- per fault ---------------------------------------------------------
    healthy_scores = scores[~is_anom]
    per_fault = {}
    for fault in sorted(set(y.tolist())):
        if fault == "healthy":
            continue
        m = y == fault
        detected = float(np.mean(scores[m] > thr))
        # One-vs-healthy AUC, so a class that's merely *harder* than the others
        # doesn't get hidden by the overall figure.
        sub_scores = np.concatenate([healthy_scores, scores[m]])
        sub_labels = np.concatenate([np.zeros(len(healthy_scores)), np.ones(int(m.sum()))])
        per_fault[fault] = {
            "n_windows": int(m.sum()),
            "detection_rate": detected,
            "auc_vs_healthy": roc_auc(sub_scores, sub_labels),
            "median_score": float(np.median(scores[m])),
        }

    # --- per severity ------------------------------------------------------
    bins = [(0.15, 0.35), (0.35, 0.55), (0.55, 0.75), (0.75, 1.01)]
    per_sev = {}
    for lo, hi in bins:
        m = is_anom & (sev >= lo) & (sev < hi)
        if not m.any():
            continue
        per_sev[f"{lo:.2f}-{hi:.2f}"] = {
            "n_windows": int(m.sum()),
            "detection_rate": float(np.mean(scores[m] > thr)),
        }

    report = {
        "threshold": thr,
        "auc": auc,
        "overall": overall,
        "per_fault": per_fault,
        "per_severity": per_sev,
        "healthy_score_p50": float(np.percentile(healthy_scores, 50)),
        "healthy_score_p99": float(np.percentile(healthy_scores, 99)),
    }
    (art / "report.json").write_text(json.dumps(report, indent=2))

    # --- print -------------------------------------------------------------
    print(f"\nAUC {auc:.4f}   threshold {thr:.5f}")
    print(
        f"precision {overall['precision']:.3f}  recall {overall['recall']:.3f}  "
        f"f1 {overall['f1']:.3f}  fpr {overall['fpr']:.3%}"
    )
    print(f"\n{'fault':<20}{'n':>7}{'detected':>11}{'auc':>9}")
    print("-" * 47)
    for k, v in per_fault.items():
        print(
            f"{k:<20}{v['n_windows']:>7}{v['detection_rate']:>10.1%}"
            f"{v['auc_vs_healthy']:>9.3f}"
        )
    print(f"\n{'severity':<20}{'n':>7}{'detected':>11}")
    print("-" * 38)
    for k, v in per_sev.items():
        print(f"{k:<20}{v['n_windows']:>7}{v['detection_rate']:>10.1%}")
    print(f"\nwrote {art / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
