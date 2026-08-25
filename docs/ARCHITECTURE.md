# Architecture & decision log

> Written during a build session with Claude Code, in Arfaali's voice and
> from Arfaali's decisions -- the reasoning below is real, the prose is
> AI-assisted.

Written as I went. The value here is mostly in the things that went wrong.

---

## 1. Framing: semi-supervised, not classification

In a real factory you have years of healthy data and almost no labelled
failures. Any approach that needs balanced fault labels is solving a problem
you don't have.

So: train an autoencoder on healthy windows only, score by reconstruction
error, threshold it. Faults are used *exclusively* for setting the operating
point and for evaluation. They never enter a gradient.

The trade-off is real. This tells you "abnormal", not "outer race". If you need
diagnosis rather than detection, you want a classifier on top, and you want
labelled data to build it, which is where most predictive-maintenance projects
actually die.

## 2. Characteristic frequencies

Bearing defect frequencies are multiples of *shaft rotation rate*, not fixed
Hz, which is why they smear when speed wanders and why order tracking matters.
For an SKF 6205 (9 balls, the bearing on the CWRU rig and most teaching rigs):

| Defect | Multiple of shaft rate | At 1455 rpm (24.25 Hz) |
|---|---|---|
| Cage (FTF) | 0.398× | 9.7 Hz |
| Rolling element (BSF) | 2.357× | 57.2 Hz |
| Outer race (BPFO) | 3.585× | 86.9 Hz |
| Inner race (BPFI) | 5.415× | 131.3 Hz |

Other signatures the simulator reproduces:

- **Imbalance** — 1× radial, nothing axial. The cleanest fault there is.
- **Misalignment** — 2× dominant *plus a real axial component*. That axial
  energy is the only thing that separates it from imbalance, which is why the
  z-axis features earn their place in the vector.
- **Rotor bar / electrical** — sidebands around 2× line frequency (100 Hz on
  UK mains). Not simulated yet.

## 3. Feature design

46 features, all chosen under one constraint: could I write this in C against
CMSIS-DSP and run it in 160 ms on a 64 MHz M4F?

Per axis (14): rms, peak, crest factor, excess kurtosis, skew, zero-crossing
rate, 7 spectral band fractions, spectral centroid. Plus 4 speed features:
mean rpm, rpm std, and 1×/2× order amplitudes.

Two decisions worth explaining:

**Bands are fractions of total power, not absolute.** We care where the energy
moved to, not how loud the room got — that makes the features robust to
mounting differences and load changes. Absolute level is already captured by
rms, so nothing is lost.

**Order amplitudes use measured rpm, not nominal.** Proper order tracking
resamples against shaft angle, which is unaffordable on the MCU. Instead we
take the measured rpm, compute where 1× should be, and take the peak within
±3 Hz. At 1024-point resolution that's ±1 bin, which covers speed-wander
smearing without letting the 2× search wander into 3×.

---

## 4. Things that went wrong

### 4.1 The 0.99 AUC that meant nothing

First run scored 0.99 AUC and I was delighted for about ten minutes. The split
was at *window* level, and windows overlap by 50% — so adjacent windows from
the same 4-second recording were landing in both train and test. The model was
being scored on data it had effectively memorised.

Fixed by splitting at segment level (`dataset.split_segments`), stratified by
fault so a class can't land entirely in one split. `test_splits_do_not_share_segments`
guards it. AUC after the fix was 0.96 and climbed back to 0.989 through actual
improvements.

Generalisable lesson: whenever the preprocessing creates overlapping or
correlated samples, the split has to happen at the level of the *original
recording*, not the derived sample.

### 4.2 Crest factor went to infinity

A window with the motor stopped has rms ≈ 0, so `peak / rms` → inf → the
scaler's mean and std became NaN → every subsequent window scored as anomalous.
Silent, and it took a while to trace because the failure surfaced three
modules downstream.

Fixed with an epsilon in the denominator and the same for kurtosis
normalisation. `test_crest_factor_survives_silence` guards it.

### 4.3 Representative dataset clipping

This is the interesting one.

Building the TFLite representative dataset from healthy windows only is the
intuitive choice, since the model only ever trains on healthy data. The
converter then learns an input range of about **[-4.9, +5.6] sigma**.

Measured against that range: **89.5% of fault windows had at least one
feature clipped** at the int8 boundary.

Detection still worked — a clipped anomaly still reconstructs badly, so it
still trips the threshold, and decision agreement with the float model was
99.9%. Which is exactly why this is hard to spot. But the *score* was
flattened: everything past ~5 sigma read as roughly the same number. Float/int8
score correlation was **0.55**. That kills any hope of trending "this bearing
is getting worse over six weeks", which is most of the value of condition
monitoring.

Fix, part one: mix ~10% fault windows into the representative set. They are
never trained on — the converter uses them purely to observe the true dynamic
range. Correlation went 0.55 → **0.88**. 25% barely improved it further while
widening the range enough to lose int8 resolution on the healthy windows that
actually need resolving, so 10% is the knee.

Fix, part two, less obvious than the first: widening the input range
**changes the numeric scale of the reconstruction error**, so the
float-calibrated threshold no longer means the same thing. Carrying it over
dropped decision agreement to **72%**.

The right design, and what's now in the repo: `convert_tflite.py` recalibrates
the threshold against the int8 model's errors on healthy validation windows,
and writes it as `threshold_int8`. That's what ships to the device.
`edge/runner.py` raises a hard error if it's missing rather than falling back
to the float threshold, because the fallback would be silently wrong.

Final: correlation 0.88, agreement 97.9%, recall 92.6%, FPR 1.33%.

**The generalisable lesson:** the calibration data for quantisation defines the
representable range, and if your anomalies live outside the range your
calibration set covers, they get clipped into indistinguishability. Any
threshold must be calibrated on the artifact that actually ships, not on the
model it was derived from.

---

## 5. Debounce

At 1% per-window FPR and 6.25 windows/sec, that's a false alarm every 16
seconds. Nobody keeps that switched on.

N-of-M debounce (3 of the last 5) exploits the fact that a real fault trips
*consecutive* windows while noise doesn't. Assuming rough independence, 3+ of 5
at p=0.01 is on the order of 10⁻⁵ per window — roughly one nuisance alarm per
several hours — while a real fault, tripping most windows, still trips in under
a second of detection latency.

Observed in the demo: 7 healthy windows flagged over 60 seconds, **zero**
alarms raised.

Clearing uses hysteresis — the buffer must be *fully* clean, not merely below
N. Clearing at `hits < N` makes the alarm chatter around the boundary.

---

## 6. Cloud loop

Device spools flagged windows locally as JSONL, uploader ships them gzipped to
S3 in batches, SageMaker retrains periodically.

**Spool, don't stream.** Batching survives a dropped link, which a per-window
streaming design does not, and it's 4 KB/day instead of 40 MB/day of cell data.

**Spool features, not waveform.** 46 floats is 184 bytes; 1024×3 int16 samples
is 6 KB. This is only safe because the feature extractor is frozen — so every
record carries `feature_version`, S3 keys are partitioned on it, and the
retraining job hard-skips mismatches. If `features.py` changes, old records
describe a different 46-dimensional space and mixing them is silent corruption.

**The retraining pseudo-label problem.** The spool contains *only* windows the
edge flagged as anomalous. Retraining an autoencoder directly on those teaches
it to reconstruct faults — exactly backwards, and a genuinely dangerous
mistake because the model would quietly go blind to the very thing it was
built to catch.

What the spool is actually good for is **concept drift**: the low-scoring tail
is mostly borderline-normal operation the deployed model hasn't adapted to
(seasonal temperature, rebalanced load, new bearings bedding in). Taking the
bottom 60% by score as pseudo-healthy tracks the machine as it ages.

That percentile is the most dangerous knob in the repo. There's a hard guard
rejecting anything above 75%.

---

## 7. What I'd do next, in order

1. **Envelope analysis** for bearing faults. Hilbert transform, then FFT of the
   envelope — this is the textbook technique for exactly the case where BPFO
   detection is currently weakest, and it's affordable on an M4F.
2. **Validate against CWRU.** Real accelerometer data with seeded faults. Until
   that's done, every number in the README is a statement about my simulator.
3. **Real tacho.** Hall sensor on an interrupt pin. Until then the order
   features are decorative on-device.
4. **Rotor bar fault mode.** Sidebands around 2× line frequency — a common
   real failure the simulator doesn't cover.
5. **Model-drift monitoring in the cloud.** Track the healthy score
   distribution over time; a rising median means the model needs retraining
   *before* the FPR gets bad enough for anyone to complain.
