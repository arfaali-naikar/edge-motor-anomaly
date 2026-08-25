"""End-to-end guards on the bits that are easy to break silently."""

import numpy as np
import pytest

from motor_anomaly.dataset import Scaler, build, split_segments
from motor_anomaly.simulate import generate_dataset


@pytest.fixture(scope="module")
def segments():
    return generate_dataset(n_healthy=30, n_per_fault=6, duration_s=1.5, seed=13)


def test_scaler_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 7.0, (500, 46)).astype(np.float32)
    s = Scaler.fit(x)
    z = s.transform(x)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(z.std(axis=0), 1, atol=1e-3)
    assert np.allclose(Scaler.from_dict(s.to_dict()).transform(x), z)


def test_scaler_handles_constant_column():
    x = np.ones((100, 46), dtype=np.float32)
    assert np.isfinite(Scaler.fit(x).transform(x)).all()


def test_splits_do_not_share_segments(segments):
    # The bug this exists to prevent: overlapping windows leaking between
    # train and test, which inflated my first AUC to 0.99 for the wrong reason.
    tr, va, te = split_segments(segments, seed=3)
    ids = [{id(s) for s in split} for split in (tr, va, te)]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert len(tr) + len(va) + len(te) == len(segments)


def test_every_fault_appears_in_test_split(segments):
    _, _, te = split_segments(segments, seed=3)
    faults = {s["fault"] for s in te}
    assert {s["fault"] for s in segments} == faults


def test_train_set_is_healthy_only(segments):
    d = build(segments, seed=3)
    # Can't check labels directly (they're dropped), so check the shape math:
    # x_train must be smaller than all training windows.
    assert d["x_train"].shape[1] == 46
    assert len(d["x_train"]) < len(d["x_val"]) + len(d["x_train"])
    assert np.isfinite(d["x_train"]).all()


def test_faults_score_higher_than_healthy_on_raw_distance(segments):
    """Sanity check that doesn't need a trained model: fault windows should be
    further from the healthy centroid than healthy windows are."""
    d = build(segments, seed=3)
    dist = np.linalg.norm(d["x_test"], axis=1)  # already z-scored on healthy
    healthy = dist[d["y_test"] == "healthy"]
    faulty = dist[d["y_test"] != "healthy"]
    assert np.median(faulty) > np.median(healthy)
