"""
The thing that would actually run on the gateway (or, ported, on the MCU).

Run against a simulated stream:
    python -m motor_anomaly.edge.runner --demo --minutes 2

Design notes, because this is the part people get wrong:

1. **Debounce.** A 1% per-window false alarm rate at 6.25 windows/sec is an
   alarm every 16 seconds. Nobody will keep that switched on. N-of-M debounce
   (default 3 of the last 5) turns that into roughly one nuisance alarm per
   several hours while barely touching detection latency for a real fault --
   a real fault trips consecutive windows, noise doesn't.

2. **Spooling, not streaming.** The device does not phone home per window.
   It writes anomalous windows to a local JSONL spool and the uploader ships
   them in batches. That survives a dropped link, which a streaming design
   does not, and it's the difference between 40 MB/day and 4 KB/day of cell
   data.

3. **What gets spooled is the 46-float feature vector, not raw waveform.**
   46 floats is 184 bytes; 1024x3 int16 samples is 6 KB. Retraining on
   features is enough because the feature extractor is frozen -- if I ever
   change features.py, the whole spool history becomes invalid and needs a
   version bump. Hence "feature_version" in every record.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from ..config import load_config
from ..features import WINDOW, extract_window

FEATURE_VERSION = 3  # bump whenever features.py changes shape or semantics


def _load_interpreter(model_path: Path):
    """Prefer the standalone tflite_runtime (what you'd actually install on a
    Pi -- ~2 MB vs TensorFlow's 600 MB), fall back to full TF for dev."""
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        return Interpreter(model_path=str(model_path))
    except ImportError:
        import tensorflow as tf

        return tf.lite.Interpreter(model_path=str(model_path))


class AnomalyDetector:
    def __init__(self, artifacts: Path, debounce_n: int = 3, debounce_m: int = 5):
        meta = json.loads((artifacts / "threshold.json").read_text())
        # threshold_int8 is written by convert_tflite.py and is calibrated
        # against the quantised model. Falling back to the float threshold
        # would be wrong (different error scale) -- fail loudly instead.
        if "threshold_int8" not in meta:
            raise RuntimeError(
                "threshold.json has no threshold_int8 -- run "
                "`python -m motor_anomaly.convert_tflite` first"
            )
        self.threshold = float(meta["threshold_int8"])
        self.mean = np.asarray(meta["scaler"]["mean"], dtype=np.float32)
        self.scale = np.asarray(meta["scaler"]["scale"], dtype=np.float32)

        self.interp = _load_interpreter(artifacts / "model_int8.tflite")
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.in_scale, self.in_zp = self.inp["quantization"]
        self.out_scale, self.out_zp = self.out["quantization"]

        self.n, self.m = debounce_n, debounce_m
        self.history: deque[bool] = deque(maxlen=debounce_m)
        self.alarm_active = False

    def score(self, features: np.ndarray) -> float:
        z = (features - self.mean) / self.scale
        q = np.clip(np.round(z / self.in_scale + self.in_zp), -128, 127).astype(np.int8)
        self.interp.set_tensor(self.inp["index"], q.reshape(self.inp["shape"]))
        self.interp.invoke()
        recon = (self.interp.get_tensor(self.out["index"]).astype(np.float32) - self.out_zp) * self.out_scale
        z_q = (q.astype(np.float32) - self.in_zp) * self.in_scale
        return float(np.mean((z_q - recon.ravel()) ** 2))

    def step(self, features: np.ndarray) -> dict:
        s = self.score(features)
        flagged = s > self.threshold
        self.history.append(flagged)

        hits = sum(self.history)
        should_alarm = hits >= self.n

        # Edge-triggered: only report state *changes*, so downstream doesn't
        # get spammed with "still broken" every 160 ms.
        transition = None
        if should_alarm and not self.alarm_active:
            transition = "raised"
        elif not should_alarm and self.alarm_active and hits == 0:
            # Require a fully clean window buffer to clear. Hysteresis --
            # clearing at hits < n makes it chatter around the boundary.
            transition = "cleared"

        if transition == "raised":
            self.alarm_active = True
        elif transition == "cleared":
            self.alarm_active = False

        return {
            "score": s,
            "flagged": flagged,
            "alarm": self.alarm_active,
            "transition": transition,
            "margin": s / self.threshold,
        }


class Spool:
    """Append-only JSONL. Deliberately not sqlite -- a half-written JSON line
    is recoverable by skipping it; a half-written sqlite page is not, and this
    box can lose power at any moment."""

    def __init__(self, path: Path, max_records: int = 20000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records

    def append(self, features: np.ndarray, score: float, alarm: bool) -> None:
        rec = {
            "ts": time.time(),
            "feature_version": FEATURE_VERSION,
            "score": round(score, 6),
            "alarm": alarm,
            "features": [round(float(v), 5) for v in features],
        }
        with self.path.open("a") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open() as f:
            return sum(1 for _ in f)


def run_stream(
    accel: np.ndarray,
    rpm: np.ndarray,
    fs: int,
    detector: AnomalyDetector,
    spool: Spool | None = None,
    hop: int = 512,
    verbose: bool = True,
) -> list[dict]:
    """Walk a signal as if it were arriving live."""
    results = []
    for start in range(0, len(accel) - WINDOW + 1, hop):
        feats = extract_window(accel[start : start + WINDOW], rpm[start : start + WINDOW], fs)
        r = detector.step(feats)
        r["t"] = start / fs
        results.append(r)

        if spool is not None and r["flagged"]:
            spool.append(feats, r["score"], r["alarm"])

        if verbose and r["transition"]:
            sym = "ALARM RAISED " if r["transition"] == "raised" else "alarm cleared"
            print(f"  t={r['t']:7.2f}s  {sym}  score={r['score']:.4f} ({r['margin']:.1f}x thr)")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the edge detector")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--demo", action="store_true", help="synthesise a healthy->fault stream")
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--fault", default="outer_race")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    art = Path(args.artifacts)
    det = AnomalyDetector(art, cfg["edge"]["debounce_n"], cfg["edge"]["debounce_m"])
    spool = Spool(Path(cfg["edge"]["spool_path"]))

    if not args.demo:
        print("no live sensor driver in this repo yet -- use --demo")
        return 1

    from ..simulate import RigConfig, generate_segment

    rig = RigConfig(fs=cfg["data"]["fs"], nominal_rpm=cfg["data"]["nominal_rpm"])
    half = args.minutes * 60.0 / 2.0

    print(f"threshold {det.threshold:.5f}, debounce {det.n}-of-{det.m}\n")
    print(f"[{half:.0f}s healthy]")
    h = generate_segment("healthy", half, 0.0, rig, seed=101)
    r1 = run_stream(h["accel"], h["rpm"], rig.fs, det, spool)

    # Ramp severity so we can see *when* it trips, not just that it does.
    print(f"[{half:.0f}s {args.fault}, severity ramping 0.2 -> 0.9]")
    chunks_a, chunks_r = [], []
    n_chunks = 8
    for i in range(n_chunks):
        sev = 0.2 + (0.7 * i / (n_chunks - 1))
        seg = generate_segment(args.fault, half / n_chunks, sev, rig, seed=200 + i)
        chunks_a.append(seg["accel"])
        chunks_r.append(seg["rpm"])
    r2 = run_stream(np.concatenate(chunks_a), np.concatenate(chunks_r), rig.fs, det, spool)

    fp = sum(1 for r in r1 if r["flagged"])
    tp = sum(1 for r in r2 if r["flagged"])
    print(
        f"\nhealthy phase: {fp}/{len(r1)} windows flagged ({fp / len(r1):.1%} -- "
        f"target {cfg['train']['target_fpr']:.0%})"
    )
    print(f"fault phase:   {tp}/{len(r2)} windows flagged ({tp / len(r2):.1%})")
    print(f"spool now holds {spool.count()} records at {spool.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
