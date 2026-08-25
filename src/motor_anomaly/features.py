"""
Feature extraction. 46 floats per window.

Hard constraint driving every choice in here: this has to run on a Cortex-M4F
at 64 MHz inside a 0.32 s budget, in ~40 KB of RAM, using nothing but CMSIS-DSP.
So:

  * no scipy -- everything is hand-rolled numpy that maps 1:1 to C
  * one FFT per axis, reused for every spectral feature
  * no sliding-window statistics that need the whole segment in memory

If you're tempted to add a feature, ask whether you'd enjoy writing it in C.

Layout of the 46-vector (order matters -- the firmware indexes it positionally,
see firmware/nano33ble/features.h):

    [0:14]   axis x   time (6) + bands (7) + centroid (1)
    [14:28]  axis y   same
    [28:42]  axis z   same
    [42:46]  speed    mean_rpm, rpm_std, order_1x, order_2x
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9

# Fixed absolute bands, Hz. Chosen against a 3200 Hz fs / 1600 Hz Nyquist:
# band 0 catches 1x/2x shaft (24/48 Hz), band 1 catches line hum and low
# harmonics, bands 2-3 catch bearing defect fundamentals, bands 4-6 straddle
# the structural resonances where bearing impulses ring.
BANDS_HZ = ((0, 50), (50, 120), (120, 260), (260, 520), (520, 900), (900, 1250), (1250, 1600))

TIME_FEATURES = ("rms", "peak", "crest", "kurtosis", "skew", "zcr")
FEATURES_PER_AXIS = len(TIME_FEATURES) + len(BANDS_HZ) + 1
SPEED_FEATURES = ("mean_rpm", "rpm_std", "order_1x", "order_2x")
N_FEATURES = 3 * FEATURES_PER_AXIS + len(SPEED_FEATURES)  # 46

WINDOW = 1024  # 0.32 s at 3200 Hz. Power of two so the MCU FFT is happy.
HOP = 512  # 50% overlap


def feature_names() -> list[str]:
    names = []
    for ax in ("x", "y", "z"):
        names += [f"{ax}_{f}" for f in TIME_FEATURES]
        names += [f"{ax}_band{i}" for i in range(len(BANDS_HZ))]
        names.append(f"{ax}_centroid")
    names += list(SPEED_FEATURES)
    return names


def _time_features(x: np.ndarray) -> np.ndarray:
    """Six scalars. All single-pass, all trivially portable."""
    x = x - x.mean()  # DC removal; gravity on a real accelerometer swamps everything
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    # EPS in the denominator matters more than it looks: on a stopped motor
    # rms goes to ~0 and crest factor goes to infinity, which then poisons the
    # scaler statistics.
    crest = peak / (rms + EPS)

    var = float(np.mean(x * x))
    std = np.sqrt(var) + EPS
    kurt = float(np.mean((x / std) ** 4)) - 3.0  # excess kurtosis; 0 for Gaussian
    skew = float(np.mean((x / std) ** 3))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.int8)))))

    return np.array([rms, peak, crest, kurt, skew, zcr], dtype=np.float64)


def _spectral_features(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (band_fractions, centroid, magnitude_spectrum).

    Bands are returned as *fractions of total power*, not absolute. That makes
    them robust to accelerometer mounting differences and to overall load,
    which is what we want -- we care about where the energy moved to, not how
    loud the room got. Absolute level is already captured by rms.
    """
    x = x - x.mean()
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    power = spec * spec
    total = float(power.sum()) + EPS

    bands = np.empty(len(BANDS_HZ))
    for i, (lo, hi) in enumerate(BANDS_HZ):
        mask = (freqs >= lo) & (freqs < hi)
        bands[i] = float(power[mask].sum()) / total

    centroid = float((freqs * power).sum() / total)
    return bands, np.array([centroid]), spec


def _order_amplitude(spec: np.ndarray, freqs: np.ndarray, target_hz: float, tol_hz: float = 3.0):
    """Peak magnitude within +/- tol of a target frequency.

    This is cheap order tracking. Proper order tracking resamples the signal
    against shaft angle; we can't afford that on the MCU, so instead we take
    the measured rpm, work out where 1x *should* be, and look in a small
    neighbourhood. tol=3 Hz covers the smearing from speed wander at 1024-point
    resolution (3.125 Hz/bin) without letting the 2x search wander into 3x.
    """
    if target_hz <= 0 or target_hz >= freqs[-1]:
        return 0.0
    mask = (freqs >= target_hz - tol_hz) & (freqs <= target_hz + tol_hz)
    if not mask.any():
        return 0.0
    return float(spec[mask].max())


def extract_window(accel: np.ndarray, rpm: np.ndarray, fs: int) -> np.ndarray:
    """One window (n, 3) + rpm (n,) -> one 46-vector."""
    if accel.shape[0] != rpm.shape[0]:
        raise ValueError("accel and rpm must be the same length")

    feats = []
    specs = []
    for ax in range(3):
        col = accel[:, ax].astype(np.float64)
        tf = _time_features(col)
        bands, centroid, spec = _spectral_features(col, fs)
        specs.append(spec)
        feats.append(np.concatenate([tf, bands, centroid]))

    freqs = np.fft.rfftfreq(accel.shape[0], d=1.0 / fs)
    mean_rpm = float(rpm.mean())
    rpm_std = float(rpm.std())
    fr = mean_rpm / 60.0

    # Order amplitudes come off the radial axes (x and y), summed. Misalignment
    # and imbalance are radial phenomena; including z here just adds noise.
    order_1x = _order_amplitude(specs[0], freqs, fr) + _order_amplitude(specs[1], freqs, fr)
    order_2x = _order_amplitude(specs[0], freqs, 2 * fr) + _order_amplitude(specs[1], freqs, 2 * fr)

    speed = np.array([mean_rpm, rpm_std, order_1x, order_2x], dtype=np.float64)
    out = np.concatenate(feats + [speed])

    assert out.shape[0] == N_FEATURES, f"expected {N_FEATURES}, built {out.shape[0]}"
    return out.astype(np.float32)


def extract_segment(
    accel: np.ndarray, rpm: np.ndarray, fs: int, window: int = WINDOW, hop: int = HOP
) -> np.ndarray:
    """Slide over a segment -> (n_windows, 46)."""
    n = accel.shape[0]
    if n < window:
        raise ValueError(f"segment of {n} samples is shorter than one {window}-sample window")
    starts = range(0, n - window + 1, hop)
    return np.stack([extract_window(accel[s : s + window], rpm[s : s + window], fs) for s in starts])
