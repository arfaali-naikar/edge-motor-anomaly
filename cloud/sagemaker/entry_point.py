"""
Runs *inside* the SageMaker training container.

SageMaker's contract, so I stop looking it up every time:
  SM_CHANNEL_TRAINING  -> where the S3 input got downloaded to
  SM_MODEL_DIR         -> anything written here is tarred into model.tar.gz
  SM_OUTPUT_DATA_DIR   -> metrics/plots, kept separately from the model

The retrain deliberately reuses motor_anomaly.model and .train so the cloud
model is architecturally identical to the one that was quantised. If this file
ever starts defining its own layers, that's drift and it will bite.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from motor_anomaly.dataset import Scaler
from motor_anomaly.model import build_autoencoder, compile_model, reconstruction_error
from motor_anomaly.train import calibrate_threshold


def load_spool(channel: Path, feature_version: int) -> tuple[np.ndarray, np.ndarray]:
    """Read every .jsonl(.gz) under the channel. Returns (features, scores)."""
    import gzip

    feats, scores = [], []
    files = sorted(list(channel.rglob("*.jsonl")) + list(channel.rglob("*.jsonl.gz")))
    if not files:
        raise SystemExit(f"no spool files under {channel}")

    for fp in files:
        opener = gzip.open if fp.suffix == ".gz" else open
        with opener(fp, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Hard skip, not a warning. Mixing feature versions is silent
                # corruption -- the vectors have the same length but different
                # meaning.
                if rec.get("feature_version") != feature_version:
                    continue
                feats.append(rec["features"])
                scores.append(rec["score"])

    print(f"loaded {len(feats)} records from {len(files)} file(s)")
    return np.asarray(feats, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--feature-version", type=int, default=3)
    ap.add_argument(
        "--score-percentile",
        type=float,
        default=60.0,
        help="only spooled windows BELOW this score percentile are treated as pseudo-healthy",
    )
    ap.add_argument("--train-channel", default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args = ap.parse_args()

    x, scores = load_spool(Path(args.train_channel), args.feature_version)

    # The spool only ever contains windows the *edge* flagged as anomalous.
    # Retraining an autoencoder directly on those would teach it to reconstruct
    # faults -- exactly backwards.
    #
    # What this data is actually good for is concept drift: the low-scoring
    # tail of the spool is mostly borderline-normal operation the deployed
    # model hasn't adapted to (seasonal temperature, a rebalanced load, new
    # bearings bedding in). Taking the bottom 60% by score and folding it into
    # the healthy set is how the model tracks the machine as it ages.
    #
    # This is the single most dangerous knob in the repo. Set it too high and
    # you train on real faults and go blind to them. There is a guard rail
    # below; do not remove it without thinking hard.
    if not 0.0 < args.score_percentile <= 75.0:
        raise SystemExit("--score-percentile must be in (0, 75]; higher risks training on faults")

    cutoff = float(np.percentile(scores, args.score_percentile))
    pseudo_healthy = x[scores <= cutoff]
    print(f"score cutoff {cutoff:.4f} -> {len(pseudo_healthy)}/{len(x)} windows kept as pseudo-healthy")

    if len(pseudo_healthy) < 500:
        raise SystemExit("fewer than 500 usable windows; wait for more telemetry before retraining")

    scaler = Scaler.fit(pseudo_healthy)
    xs = scaler.transform(pseudo_healthy)

    n_val = max(200, int(0.2 * len(xs)))
    x_val, x_tr = xs[:n_val], xs[n_val:]

    model = compile_model(build_autoencoder(latent_dim=args.latent_dim), lr=args.lr)
    model.fit(
        x_tr, x_tr,
        validation_data=(x_val, x_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )

    threshold = calibrate_threshold(reconstruction_error(model, x_val), args.target_fpr)

    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.keras")
    (out / "threshold.json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "target_fpr": args.target_fpr,
                "scaler": scaler.to_dict(),
                "n_windows": int(len(pseudo_healthy)),
                "feature_version": args.feature_version,
                "score_percentile": args.score_percentile,
            },
            indent=2,
        )
    )
    print(f"saved model + threshold {threshold:.5f} to {out}")
    print("NOTE: still needs quantising + int8 threshold recalibration before it ships")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
