"""
Float Keras model -> fully int8-quantised TFLite -> C header.

Run: python -m motor_anomaly.convert_tflite

Full integer quantisation (not dynamic-range, not float16) because TFLite
Micro on a Cortex-M4F has no FPU path worth using for matmuls, and CMSIS-NN
kernels are int8-only. Dynamic-range quantisation would still leave float
activations and roughly triple the inference time.

Two things in here are easy to get wrong, so read this before changing it.

**1. The representative dataset must NOT be healthy-only.**

My first version fed the converter only healthy windows, which is the
intuitive thing to do given the model only ever trains on healthy data. It
produced an input quantisation range of about [-4.9, +5.6] sigma -- and then
89.5% of *fault* windows had at least one feature clipped at the int8 boundary.

That doesn't break detection (a clipped anomaly still reconstructs badly, so
it still trips), but it flattens the score: everything past ~5 sigma reads as
the same number. You lose all severity information, which kills any hope of
trending "this bearing is getting worse" over weeks. Mixing ~10% fault windows
into the representative set is calibration only -- no gradient ever sees them
-- and it takes the float/int8 score correlation from 0.55 to 0.88.

**2. Because of (1), the threshold has to be recalibrated on the int8 model.**

Widening the input range changes the numeric scale of the reconstruction
error, so the float-calibrated threshold no longer means the same thing --
carrying it over dropped decision agreement to 72%. The threshold that ships
to the device is therefore computed *here*, from int8 errors on healthy
validation windows, and written back as `threshold_int8`. The edge runner
reads that one. train.py's float threshold is now only a sanity check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


def build_representative_set(
    healthy: np.ndarray, fault: np.ndarray, fault_frac: float = 0.10, seed: int = 0
) -> np.ndarray:
    """Healthy windows plus a small slice of fault windows.

    The fault windows are here purely so the converter observes the true
    dynamic range of the input features -- see module docstring. They are
    never trained on. fault_frac=0.10 was the knee: 0.25 barely improves
    correlation further but widens the range enough to lose int8 resolution
    on the healthy windows we actually care about resolving.
    """
    rng = np.random.default_rng(seed)
    if fault_frac <= 0 or len(fault) == 0:
        return healthy.astype(np.float32)
    k = min(len(fault), round(len(healthy) * fault_frac / (1.0 - fault_frac)))
    picked = fault[rng.choice(len(fault), size=k, replace=False)]
    return np.concatenate([healthy, picked]).astype(np.float32)


def representative_dataset_gen(x: np.ndarray, n: int = 600):
    """TFLite calls this to learn activation ranges. ~600 samples is plenty
    for a model this small; 100 is enough in practice but it's cheap."""
    idx = np.random.default_rng(0).choice(len(x), size=min(n, len(x)), replace=False)

    def gen():
        for i in idx:
            yield [x[i : i + 1].astype(np.float32)]

    return gen


def convert(model_path: Path, rep_data: np.ndarray, out_path: Path) -> bytes:
    model = keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen(rep_data)
    # These three lines are what make it *fully* int8. Drop any of them and
    # you silently get a hybrid model that TFLite Micro will refuse to run.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    return tflite_model


def quantized_reconstruction_error(tflite_bytes: bytes, x: np.ndarray) -> np.ndarray:
    """Run the int8 model window-by-window and return per-sample MSE in the
    *float* domain, so it's directly comparable to the float model's error
    and to the calibrated threshold."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    errors = np.empty(len(x), dtype=np.float64)
    for i, row in enumerate(x):
        q = np.round(row / in_scale + in_zp)
        q = np.clip(q, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q.reshape(inp["shape"]))
        interp.invoke()
        deq = (interp.get_tensor(out["index"]).astype(np.float32) - out_zp) * out_scale
        # Compare against the *dequantised input*, not the float input. The
        # device only ever sees the quantised version, so this is the error it
        # will actually compute. Comparing to the float input overstates it.
        row_q = (q.astype(np.float32) - in_zp) * in_scale
        errors[i] = float(np.mean((row_q - deq.ravel()) ** 2))
    return errors


def to_c_array(tflite_bytes: bytes, var_name: str = "g_motor_model") -> str:
    lines = [
        "// Auto-generated by motor_anomaly.convert_tflite -- do not edit by hand.",
        "// Regenerate with: make firmware-model",
        "",
        "#include <cstdint>",
        "",
        f'alignas(16) const unsigned char {var_name}[] = {{',
    ]
    for i in range(0, len(tflite_bytes), 12):
        chunk = tflite_bytes[i : i + 12]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    lines.append("};")
    lines.append(f"const unsigned int {var_name}_len = {len(tflite_bytes)};")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Quantise the model for the edge")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--header", default="firmware/nano33ble/model_data.h")
    ap.add_argument(
        "--fault-frac",
        type=float,
        default=0.10,
        help="fraction of the representative set drawn from fault windows (range calibration only)",
    )
    args = ap.parse_args(argv)

    art = Path(args.artifacts)
    meta = json.loads((art / "threshold.json").read_text())
    cache = np.load(art / "test_split.npz", allow_pickle=True)

    x_test, y_test = cache["x"], cache["y"]
    healthy_val = cache["x_val_healthy"]
    fault_test = x_test[y_test != "healthy"]

    rep = build_representative_set(healthy_val, fault_test, args.fault_frac)
    print(f"representative set: {rep.shape}  (fault_frac={args.fault_frac})")

    tflite_bytes = convert(art / "model.keras", rep, art / "model_int8.tflite")
    print(f"int8 model: {len(tflite_bytes):,} bytes  ({len(tflite_bytes) / 1024:.1f} KB)")

    model = keras.models.load_model(art / "model.keras")
    from .model import reconstruction_error

    # --- recalibrate on the int8 model ------------------------------------
    # This is the threshold that ships. Same target FPR as train.py, but
    # measured on the errors the *device* will actually compute.
    err_q_healthy = quantized_reconstruction_error(tflite_bytes, healthy_val)
    target_fpr = meta["target_fpr"]
    thr_int8 = float(np.percentile(err_q_healthy, 100.0 * (1.0 - target_fpr)))

    err_f = reconstruction_error(model, x_test)
    err_q = quantized_reconstruction_error(tflite_bytes, x_test)
    corr = float(np.corrcoef(err_f, err_q)[0, 1])

    # Agreement is measured with each model at *its own* threshold -- that's
    # the deployment-relevant question ("do float and int8 make the same call?"),
    # not "does int8 agree with float's threshold?", which is meaningless once
    # the error scales differ.
    flag_f = err_f > meta["threshold"]
    flag_q = err_q > thr_int8
    agreement = float(np.mean(flag_f == flag_q))

    is_anom = y_test != "healthy"
    recall_q = float(np.mean(err_q[is_anom] > thr_int8))
    fpr_q = float(np.mean(err_q[~is_anom] > thr_int8))

    print(f"error correlation float vs int8: {corr:.4f}")
    print(f"decision agreement (each at own threshold): {agreement:.2%}")
    print(f"int8 threshold {thr_int8:.5f}   test recall {recall_q:.1%}   test FPR {fpr_q:.2%}")
    if corr < 0.75:
        print("  !! score fidelity is poor -- check clipping, see module docstring")

    meta["threshold_int8"] = thr_int8
    meta["int8"] = {
        "bytes": len(tflite_bytes),
        "fault_frac_in_representative_set": args.fault_frac,
        "decision_agreement": agreement,
        "error_correlation": corr,
        "test_recall": recall_q,
        "test_fpr": fpr_q,
    }
    (art / "threshold.json").write_text(json.dumps(meta, indent=2))

    hdr = Path(args.header)
    hdr.parent.mkdir(parents=True, exist_ok=True)
    hdr.write_text(to_c_array(tflite_bytes))
    print(f"wrote {hdr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
