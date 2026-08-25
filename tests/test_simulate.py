"""The simulator is the ground truth for everything else, so it needs to
actually produce distinguishable fault signatures. If these fail, a good AUC
downstream means nothing."""

import numpy as np
import pytest

from motor_anomaly.features import extract_segment
from motor_anomaly.simulate import FAULTS, generate_dataset, generate_segment


def _mean_features(fault, severity=0.9, seed=42):
    seg = generate_segment(fault, 4.0, severity, seed=seed)
    return extract_segment(seg["accel"], seg["rpm"], seg["fs"]).mean(axis=0)


def test_all_faults_generate():
    for fault in FAULTS:
        seg = generate_segment(fault, 1.0, 0.5, seed=3)
        assert seg["accel"].shape == (3200, 3)
        assert np.isfinite(seg["accel"]).all()
        assert np.isfinite(seg["rpm"]).all()


def test_unknown_fault_rejected():
    with pytest.raises(ValueError, match="unknown fault"):
        generate_segment("gremlins")


def test_seed_is_reproducible():
    a = generate_segment("outer_race", 1.0, 0.7, seed=99)["accel"]
    b = generate_segment("outer_race", 1.0, 0.7, seed=99)["accel"]
    np.testing.assert_array_equal(a, b)


def test_imbalance_raises_1x_order():
    healthy, imb = _mean_features("healthy"), _mean_features("imbalance")
    assert imb[44] > 3 * healthy[44]


def test_bearing_faults_are_impulsive():
    # Kurtosis is *the* classic bearing indicator: impulse trains have heavy
    # tails, sinusoidal imbalance does not.
    healthy = _mean_features("healthy")
    for fault in ("outer_race", "inner_race"):
        assert _mean_features(fault)[3] > healthy[3] + 0.5


def test_misalignment_shows_axial_energy():
    # z is the axial axis. Imbalance is radial; misalignment is not. This is
    # the feature that separates the two.
    healthy_z_rms = _mean_features("healthy")[28]
    assert _mean_features("misalignment")[28] > 2 * healthy_z_rms


def test_speed_instability_shows_in_rpm_std_not_vibration():
    healthy, unstable = _mean_features("healthy"), _mean_features("speed_instability")
    assert unstable[43] > 5 * healthy[43]      # rpm std
    assert unstable[0] < 2 * healthy[0]        # x rms barely moves


def test_dataset_is_healthy_dominant():
    segs = generate_dataset(n_healthy=20, n_per_fault=4, duration_s=1.0, seed=5)
    n_healthy = sum(1 for s in segs if s["fault"] == "healthy")
    assert n_healthy == 20
    assert len(segs) == 20 + 4 * (len(FAULTS) - 1)
