"""
Train the autoencoder and calibrate the alarm threshold.

Run:  python -m motor_anomaly.train --config config/default.yaml

Two outputs that matter downstream:
  artifacts/model.keras   -- float model, input to the TFLite converter
  artifacts/threshold.json -- scaler stats + threshold + metadata

The threshold is set from *healthy validation* reconstruction error, at a
percentile chosen to hit a target false-alarm rate. It is deliberately NOT
chosen by maximising F1 on the fault data: doing that tunes the detector to
the specific faults I happened to simulate, which defeats the purpose of an
open-set anomaly detector. Target FPR is a business decision -- how many
nuisance alarms will the maintenance team tolerate before they mute it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import load_config
from .dataset import build
from .model import build_autoencoder, compile_model, reconstruction_error
from .simulate import RigConfig, generate_dataset


def calibrate_threshold(errors_healthy: np.ndarray, target_fpr: float) -> float:
    """Percentile of the healthy error distribution.

    target_fpr=0.01 -> 99th percentile -> roughly 1 window in 100 false-alarms.
    At 0.32 s windows with 50% overlap that's a false positive every ~16 s,
    which sounds appalling until you remember the N-of-M debounce in the edge
    runner turns it into roughly one every few hours.
    """
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0, 1)")
    return float(np.percentile(errors_healthy, 100.0 * (1.0 - target_fpr)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the motor anomaly autoencoder")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--epochs", type=int, default=None, help="override config")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    seed = args.seed if args.seed is not None else cfg["seed"]

    keras.utils.set_random_seed(seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[1/4] simulating {cfg['data']['n_healthy']} healthy + fault segments ...")
    rig = RigConfig(fs=cfg["data"]["fs"], nominal_rpm=cfg["data"]["nominal_rpm"])
    segments = generate_dataset(
        n_healthy=cfg["data"]["n_healthy"],
        n_per_fault=cfg["data"]["n_per_fault"],
        duration_s=cfg["data"]["duration_s"],
        cfg=rig,
        seed=seed,
    )

    print("[2/4] extracting features ...")
    data = build(segments, seed=seed)
    print(
        f"      train {data['x_train'].shape}  val {data['x_val'].shape}  "
        f"test {data['x_test'].shape}   (segments: {data['n_segments']})"
    )

    print("[3/4] training ...")
    model = compile_model(
        build_autoencoder(
            latent_dim=cfg["model"]["latent_dim"],
            hidden=tuple(cfg["model"]["hidden"]),
            dropout=cfg["model"]["dropout"],
            l2=cfg["model"]["l2"],
        ),
        lr=cfg["train"]["lr"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg["train"]["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(3, cfg["train"]["patience"] // 3), verbose=0
        ),
    ]

    model.fit(
        data["x_train"],
        data["x_train"],  # autoencoder: target is the input
        validation_data=(data["x_val_healthy"], data["x_val_healthy"]),
        epochs=cfg["train"]["epochs"],
        batch_size=cfg["train"]["batch_size"],
        callbacks=callbacks,
        verbose=2,
    )

    print("[4/4] calibrating threshold ...")
    err_val_healthy = reconstruction_error(model, data["x_val_healthy"])
    threshold = calibrate_threshold(err_val_healthy, cfg["train"]["target_fpr"])

    err_val_all = reconstruction_error(model, data["x_val"])
    achieved_fpr = float(
        np.mean(err_val_healthy > threshold)
    )
    recall_val = float(np.mean(err_val_all[data["y_val"] != "healthy"] > threshold))

    model.save(out / "model.keras")
    meta = {
        "threshold": threshold,
        "target_fpr": cfg["train"]["target_fpr"],
        "achieved_fpr_val": achieved_fpr,
        "recall_val": recall_val,
        "scaler": data["scaler"].to_dict(),
        "n_features": int(data["x_train"].shape[1]),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "config": cfg,
    }
    (out / "threshold.json").write_text(json.dumps(meta, indent=2))

    # Cache the test split so evaluate.py scores the same data the model was
    # never shown, without re-simulating and hoping the RNG lines up.
    np.savez_compressed(
        out / "test_split.npz",
        x=data["x_test"],
        y=data["y_test"],
        sev=data["sev_test"],
        x_val_healthy=data["x_val_healthy"],
    )

    print(
        f"\ndone in {time.time() - t0:.1f}s\n"
        f"  threshold        {threshold:.5f}\n"
        f"  val FPR          {achieved_fpr:.3%} (target {cfg['train']['target_fpr']:.1%})\n"
        f"  val recall       {recall_val:.1%}\n"
        f"  artifacts        {out.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
