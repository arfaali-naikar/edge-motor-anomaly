"""
The model. It's small, and it's small on purpose.

46 -> 32 -> 16 -> 8 -> 16 -> 32 -> 46, ~3.4k parameters. Fully int8-quantised
that's under 10 KB of flash, which leaves room on a Nano 33 BLE (1 MB flash,
256 KB RAM) for the ring buffer, the FFT scratch space and TFLite Micro's
arena.

Why an autoencoder rather than a classifier: I have no faith that a hand-built
list of five fault classes covers what a real motor will do to itself. An
autoencoder trained only on healthy data flags *anything* it hasn't seen,
including failure modes nobody simulated. The cost is that it tells you
"something is wrong" rather than "the outer race is pitted" -- which is the
right trade for a first-line alarm.

Why not an LSTM/CNN over the raw waveform: it would probably score better, but
a 1024-sample conv stack blows the RAM budget and the feature extractor is
doing most of the heavy lifting anyway. Revisit if I ever move to a Portenta.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .features import N_FEATURES


def build_autoencoder(
    n_features: int = N_FEATURES,
    latent_dim: int = 8,
    hidden: tuple[int, ...] = (32, 16),
    dropout: float = 0.05,
    l2: float = 1e-5,
) -> keras.Model:
    reg = keras.regularizers.l2(l2) if l2 else None

    inp = keras.Input(shape=(n_features,), name="features")
    x = inp
    for i, units in enumerate(hidden):
        x = layers.Dense(units, activation="relu", kernel_regularizer=reg, name=f"enc_{i}")(x)
        if dropout:
            x = layers.Dropout(dropout, name=f"enc_drop_{i}")(x)

    # No activation on the bottleneck. ReLU here would clip half the latent
    # space to zero and measurably hurt reconstruction -- tried it, lost about
    # 0.03 AUC.
    z = layers.Dense(latent_dim, activation=None, name="latent")(x)

    x = z
    for i, units in enumerate(reversed(hidden)):
        x = layers.Dense(units, activation="relu", kernel_regularizer=reg, name=f"dec_{i}")(x)

    # Linear output: inputs are z-scored, so they're signed and unbounded.
    # sigmoid/tanh here would be a bug.
    out = layers.Dense(n_features, activation=None, name="reconstruction")(x)

    return keras.Model(inp, out, name="motor_autoencoder")


def compile_model(model: keras.Model, lr: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def reconstruction_error(model_or_fn, x, batch_size: int = 512) -> tf.Tensor:
    """Per-sample MSE. This is the anomaly score, full stop -- everything
    downstream (threshold, debounce, alarm) is a function of this number."""
    if hasattr(model_or_fn, "predict"):
        recon = model_or_fn.predict(x, batch_size=batch_size, verbose=0)
    else:
        recon = model_or_fn(x)
    import numpy as np

    return np.mean((np.asarray(x) - np.asarray(recon)) ** 2, axis=1)
