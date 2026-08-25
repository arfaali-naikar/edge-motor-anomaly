"""Feature extractor tests. These are the ones that actually catch bugs --
the model can be retrained, but a wrong feature silently corrupts everything
downstream including the firmware."""

import numpy as np
import pytest

from motor_anomaly.features import (
    BANDS_HZ,
    N_FEATURES,
    _time_features,
    extract_segment,
    extract_window,
    feature_names,
)


def test_vector_length_matches_names():
    # The firmware indexes this vector positionally. If these ever disagree,
    # the C side reads garbage and nothing warns you.
    assert len(feature_names()) == N_FEATURES == 46


def test_time_features_on_known_signal():
    fs = 3200
    t = np.arange(fs) / fs
    x = np.sin(2 * np.pi * 50 * t)
    rms, peak, crest, kurt, skew, zcr = _time_features(x)
    assert rms == pytest.approx(1 / np.sqrt(2), abs=0.01)
    assert peak == pytest.approx(1.0, abs=0.01)
    assert crest == pytest.approx(np.sqrt(2), abs=0.02)
    # A pure sine has excess kurtosis of -1.5. This is the check that told me
    # I'd forgotten to subtract 3.
    assert kurt == pytest.approx(-1.5, abs=0.05)
    assert abs(skew) < 0.05


def test_crest_factor_survives_silence():
    # Regression: a stopped motor gave rms=0 -> inf -> NaN in the scaler,
    # which then made every subsequent window score as anomalous.
    _, _, crest, kurt, _, _ = _time_features(np.zeros(1024))
    assert np.isfinite(crest) and np.isfinite(kurt)


def test_band_fractions_sum_to_one():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1024)
    accel = np.column_stack([x, x, x])
    rpm = np.full(1024, 1455.0)
    f = extract_window(accel, rpm, 3200)
    bands = f[6 : 6 + len(BANDS_HZ)]
    # Bands tile 0..Nyquist with no gaps, so they must be a partition of unity.
    assert bands.sum() == pytest.approx(1.0, abs=1e-3)


def test_bands_track_where_the_energy_is():
    fs = 3200
    t = np.arange(1024) / fs
    lo = np.column_stack([np.sin(2 * np.pi * 30 * t)] * 3)
    hi = np.column_stack([np.sin(2 * np.pi * 1400 * t)] * 3)
    rpm = np.full(1024, 1455.0)
    f_lo = extract_window(lo, rpm, fs)
    f_hi = extract_window(hi, rpm, fs)
    assert f_lo[6] > 0.9          # band 0 = 0-50 Hz
    assert f_hi[6 + 6] > 0.9      # band 6 = 1250-1600 Hz
    assert f_hi[13] > f_lo[13]    # spectral centroid


def test_order_amplitude_finds_the_shaft_peak():
    fs = 3200
    rpm_val = 1500.0  # -> fr = 25 Hz exactly
    t = np.arange(1024) / fs
    accel = np.column_stack([np.sin(2 * np.pi * 25 * t)] * 3)
    f = extract_window(accel, np.full(1024, rpm_val), fs)
    order_1x, order_2x = f[44], f[45]
    assert order_1x > 10 * max(order_2x, 1e-6)


def test_segment_windowing_shape():
    rng = np.random.default_rng(1)
    accel = rng.normal(0, 0.05, (3200 * 2, 3))
    rpm = np.full(3200 * 2, 1455.0)
    w = extract_segment(accel, rpm, 3200)
    # (6400 - 1024) / 512 + 1 = 11
    assert w.shape == (11, N_FEATURES)
    assert np.isfinite(w).all()


def test_short_segment_raises():
    with pytest.raises(ValueError, match="shorter than one"):
        extract_segment(np.zeros((100, 3)), np.zeros(100), 3200)
