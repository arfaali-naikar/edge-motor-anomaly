"""
Synthetic tri-axial vibration + shaft speed for a 3-phase induction motor.

I don't have a real test rig, so this stands in for one until I can get a
Nano 33 BLE strapped to an actual motor. It's not trying to be a physics
engine -- it's trying to produce signals whose *statistical fingerprint*
matches what the literature says each fault looks like, so that the feature
extractor and the autoencoder are being asked to solve a realistic problem.

Model per axis:

    healthy = 1x imbalance + 2x line hum + bearing-race broadband + sensor noise

Fault modes layer extra structure on top of that. References for the
characteristic frequencies are in docs/ARCHITECTURE.md.

Everything here is pure numpy on purpose. The exact same maths has to be
portable to a Cortex-M4F later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- rig constants -----------------------------------------------------------
# Modelled on a 4-pole 1.5 kW motor on 50 Hz UK mains, running a fan load.
LINE_HZ = 50.0
POLE_PAIRS = 2
SYNC_RPM = 60.0 * LINE_HZ / POLE_PAIRS  # 1500 rpm
NOMINAL_RPM = 1455.0  # ~3% slip under load

# SKF 6205, the bearing on basically every teaching rig and the CWRU dataset.
# These are multiples of shaft rotation frequency, not absolute Hz -- they
# scale with speed, which is exactly why order tracking matters.
BPFO = 3.585  # outer race
BPFI = 5.415  # inner race
BSF = 2.357  # rolling element
FTF = 0.398  # cage

FAULTS = ("healthy", "imbalance", "misalignment", "outer_race", "inner_race", "speed_instability")


@dataclass
class RigConfig:
    fs: int = 3200  # Hz. Nyquist 1600 Hz covers up to ~65x shaft order.
    nominal_rpm: float = NOMINAL_RPM
    load_pct: float = 70.0
    # Per-axis sensitivity. Radial axes see imbalance; the axial axis (z here)
    # is the one that lights up under misalignment.
    axis_gain: tuple = (1.0, 0.85, 0.55)
    noise_g: float = 0.012  # LSM9DS1 noise density, roughly, in g RMS
    seed: int | None = None
    resonances_hz: tuple = field(default=(820.0, 1310.0))


def _rpm_profile(n: int, fs: int, base_rpm: float, rng, unstable: bool = False) -> np.ndarray:
    """Shaft speed over time. Real motors wander; constant rpm is a giveaway
    that you're looking at simulated data, and it also makes order tracking
    look better than it deserves to."""
    t = np.arange(n) / fs
    # Slow load-driven drift, ~0.5% peak.
    drift = 0.005 * base_rpm * np.sin(2 * np.pi * 0.11 * t + rng.uniform(0, 2 * np.pi))
    jitter = rng.normal(0.0, 0.0008 * base_rpm, n)
    # 3-sample moving average so jitter isn't white -- a tacho can't change
    # that fast, and leaving it white leaks into the speed-std feature.
    jitter = np.convolve(jitter, np.ones(3) / 3, mode="same")
    rpm = base_rpm + drift + jitter

    if unstable:
        # VFD hunting / supply sag: a couple of hard dips per window.
        for _ in range(rng.integers(2, 5)):
            start = rng.integers(0, max(1, n - fs // 4))
            width = rng.integers(fs // 20, fs // 5)
            depth = rng.uniform(0.04, 0.11) * base_rpm
            env = np.hanning(width)
            rpm[start : start + width] -= depth * env[: len(rpm[start : start + width])]
    return rpm


def _phase_from_rpm(rpm: np.ndarray, fs: int) -> np.ndarray:
    """Integrate instantaneous shaft frequency into phase. Doing it this way
    (rather than 2*pi*f*t) is what makes the speed wander actually smear the
    spectral peaks, which is the whole point of simulating the wander."""
    fr = rpm / 60.0
    return 2.0 * np.pi * np.cumsum(fr) / fs


def _impulse_train(n: int, fs: int, rate_hz: np.ndarray, rng, resonance_hz: float, decay: float):
    """Bearing defect impulses ringing down a structural resonance.

    Each time a ball rolls over a spall you get a broadband hit that excites
    the housing; what the accelerometer sees is a decaying sinusoid at the
    resonance, repeating at the defect frequency. Getting this right is the
    difference between the model learning 'bearing fault' and it learning
    'slightly louder'.
    """
    out = np.zeros(n)
    phase = 2.0 * np.pi * np.cumsum(rate_hz) / fs
    # Fire an impulse each time the defect phase crosses a multiple of 2*pi.
    fire_idx = np.where(np.diff(np.floor(phase / (2 * np.pi))) > 0)[0]

    ring_len = int(fs * 0.006)  # 6 ms of ringdown
    tt = np.arange(ring_len) / fs
    ring = np.exp(-decay * tt) * np.sin(2 * np.pi * resonance_hz * tt)

    for idx in fire_idx:
        end = min(n, idx + ring_len)
        # Amplitude modulation: the load zone means impulses are stronger on
        # one side of the race than the other.
        amp = rng.uniform(0.65, 1.35)
        out[idx:end] += amp * ring[: end - idx]
    return out


def generate_segment(
    fault: str = "healthy",
    duration_s: float = 4.0,
    severity: float = 0.5,
    cfg: RigConfig | None = None,
    seed: int | None = None,
) -> dict:
    """Produce one labelled segment.

    Returns a dict with 'accel' (n, 3), 'rpm' (n,), 'fs', 'fault', 'severity'.
    severity is 0..1 and scales the fault contribution -- 0.15 is 'you'd never
    hear it', 1.0 is 'the maintenance guy already knows'.
    """
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}, expected one of {FAULTS}")

    cfg = cfg or RigConfig()
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    n = int(cfg.fs * duration_s)

    rpm = _rpm_profile(n, cfg.fs, cfg.nominal_rpm, rng, unstable=(fault == "speed_instability"))
    theta = _phase_from_rpm(rpm, cfg.fs)
    fr = rpm / 60.0
    t = np.arange(n) / cfg.fs

    accel = np.zeros((n, 3))

    for ax in range(3):
        g = cfg.axis_gain[ax]
        sig = np.zeros(n)

        # --- always present -------------------------------------------------
        # Residual imbalance. Every real machine has some.
        sig += 0.055 * g * np.sin(theta + rng.uniform(0, 2 * np.pi))
        # Electrical: 2x line frequency, strongest radially.
        sig += 0.030 * g * np.sin(2 * np.pi * 2 * LINE_HZ * t + rng.uniform(0, 2 * np.pi))
        # Structural resonances excited by broadband bearing/flow noise.
        for f0 in cfg.resonances_hz:
            band = rng.normal(0, 1, n)
            band = np.convolve(band, np.hanning(31), mode="same")
            sig += 0.010 * g * band * np.sin(2 * np.pi * f0 * t)
        # Sensor + quantisation noise.
        sig += rng.normal(0.0, cfg.noise_g, n)

        # --- fault-specific --------------------------------------------------
        if fault == "imbalance":
            # Pure 1x growth, radial only. Classic.
            radial = g if ax < 2 else 0.15 * g
            sig += severity * 0.42 * radial * np.sin(theta + rng.uniform(0, 2 * np.pi))

        elif fault == "misalignment":
            # 2x dominant, plus a real axial component -- that axial energy is
            # the thing that distinguishes it from imbalance.
            axial_boost = 2.2 if ax == 2 else 1.0
            sig += severity * 0.30 * g * axial_boost * np.sin(2 * theta + rng.uniform(0, 2 * np.pi))
            sig += severity * 0.11 * g * axial_boost * np.sin(3 * theta)

        elif fault in ("outer_race", "inner_race"):
            mult = BPFO if fault == "outer_race" else BPFI
            res = cfg.resonances_hz[0] if ax != 2 else cfg.resonances_hz[1]
            train = _impulse_train(n, cfg.fs, fr * mult, rng, res, decay=380.0)
            if fault == "inner_race":
                # Inner race defect passes through the load zone once per
                # revolution, so the impulse train gets 1x AM sidebands.
                train *= 1.0 + 0.6 * np.sin(theta)
            sig += severity * 0.24 * g * train

        elif fault == "speed_instability":
            # The vibration itself is near-healthy; the tell is in the tacho.
            # This is deliberately the hardest class for a vibration-only
            # model, which is exactly why speed features are in the vector.
            sig += severity * 0.05 * g * np.sin(theta)

        accel[:, ax] = sig

    return {
        "accel": accel.astype(np.float32),
        "rpm": rpm.astype(np.float32),
        "fs": cfg.fs,
        "fault": fault,
        "severity": float(severity),
    }


def generate_dataset(
    n_healthy: int = 220,
    n_per_fault: int = 45,
    duration_s: float = 4.0,
    cfg: RigConfig | None = None,
    seed: int = 7,
) -> list[dict]:
    """A run's worth of segments. Deliberately imbalanced towards healthy --
    that's the real-world ratio, and the autoencoder only trains on healthy
    anyway."""
    cfg = cfg or RigConfig()
    rng = np.random.default_rng(seed)
    segments = []

    for i in range(n_healthy):
        segments.append(
            generate_segment("healthy", duration_s, 0.0, cfg, seed=int(rng.integers(1 << 31)))
        )

    for fault in FAULTS:
        if fault == "healthy":
            continue
        for i in range(n_per_fault):
            # Spread severity so the eval tells us where detection breaks down,
            # instead of only reporting on obvious faults.
            sev = float(rng.uniform(0.15, 1.0))
            segments.append(
                generate_segment(fault, duration_s, sev, cfg, seed=int(rng.integers(1 << 31)))
            )

    rng.shuffle(segments)
    return segments
