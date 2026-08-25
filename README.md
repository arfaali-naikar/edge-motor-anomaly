# Edge AI Anomaly Detection for Motor Control

This is a TinyML pipeline for detecting bearing and shaft faults in an
induction motor from vibration and speed data. It trains an autoencoder on
healthy data only, quantises it to a 12 KB int8 TFLite model, and runs it at
the edge with debounced alarming so a single noisy window doesn't trip a
false alarm. Flagged windows get spooled and shipped to S3 in batches for
periodic retraining on SageMaker.

I built this because I wanted a full sensor-to-cloud loop I could reason
about end to end, not just a notebook that trains a model on a fixed
dataset. The interesting constraints are all at the edges: what fits in
40 KB of RAM on a Cortex-M4F, and what a maintenance team will actually
tolerate before they switch the alarm off.

**Status:** the Python side (simulate, train, evaluate, quantise, demo) runs
end to end and is what the numbers below come from. The Arduino sketch
compiles and runs the inference loop on the bench, but it has not been run
against a real motor. See [Known limitations](#known-limitations).

---

## Results

From `artifacts/report.json` and `artifacts/threshold.json`, held-out test
split (1080 fault windows, 1056 healthy windows).

Float model, threshold set at the 99th percentile of healthy validation
error (target 1% false-positive rate):

| Metric | Value |
|---|---|
| Threshold | 0.9265 |
| ROC AUC | 0.989 |
| Precision | 0.988 |
| Recall | 0.941 |
| False positive rate | 1.14% |

Float model vs int8 after quantisation:

| Metric | Float | int8 |
|---|---|---|
| Recall | 94.1% | 92.6% |
| False positive rate | 1.14% | 1.33% |
| Flash size | n/a | 12.1 KB (12,360 bytes) |

The int8 model agrees with the float model's alarm decision on 97.9% of
test windows, and their per-window reconstruction errors correlate at 0.88.
The model itself is small on purpose: 46 to 32 to 16 to 8 to 16 to 32 to 46,
about 3.4k parameters.

Per fault type (float model):

| Fault | Detection rate | AUC vs healthy |
|---|---|---|
| imbalance | 100.0% | 1.000 |
| misalignment | 100.0% | 1.000 |
| speed_instability | 100.0% | 1.000 |
| inner_race | 94.0% | 0.990 |
| outer_race | 76.4% | 0.955 |

Per severity (float model), which matters more than the headline AUC for a
predictive maintenance use case, since catching a fault at severity 0.9 just
means finding out the motor is already broken:

| Severity | Detection rate |
|---|---|
| 0.15-0.35 | 55.6% |
| 0.35-0.55 | 100% |
| 0.55-0.75 | 100% |
| 0.75-1.00 | 100% |

So the detector is reliable from roughly 35% severity upward, and closer to
a coin flip below that. Outer race is the weak fault type. Both are
discussed in [Known limitations](#known-limitations) and in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Pipeline

```
 accelerometer (3.2 kHz)      device / gateway
 + tacho             |
                      v
             +----------------+
             |  features.py   |   1024-sample window, 50% overlap
             |  46 features   |   time + spectral + order-tracking
             +----------------+
                      |
                      v
              z-score  ->  int8 autoencoder (12 KB)
                      |
                      v
              reconstruction error
                      |
              above threshold? ---no---> discard
                      |
                     yes
                      |
                      v
              3-of-5 debounce
                |            |
             not yet        yes
                |            |
                v            v
          spool.jsonl     ALARM
                |
                | batched, gzipped
                v
   s3://bucket/raw/feature_version=3/dt=YYYY-MM-DD/
                |
                v
   SageMaker retrain -> new model -> requantise
```

---

## Quick start

Inside WSL2 Ubuntu, not PowerShell (see [Why WSL](#why-wsl)):

```bash
git clone git@github.com:<you>/edge-motor-anomaly.git ~/projects/edge-motor-anomaly
cd ~/projects/edge-motor-anomaly
./scripts/bootstrap_wsl.sh
source .venv/bin/activate
make all
```

`make all` runs simulate, train, evaluate, quantise and demo in about 90
seconds on a laptop CPU. No GPU needed, the model has 3.4k parameters.

Individual stages:

```bash
make train      # simulate + train the autoencoder
make eval       # per-fault / per-severity breakdown
make quantize   # int8 TFLite + recalibrated threshold + C header
make demo       # healthy -> fault stream through the edge detector
make test       # unit tests
make upload     # dry-run the S3 spool upload
```

Demo output looks like this:

```
threshold 2.04368, debounce 3-of-5
[60s healthy]
[60s outer_race, severity ramping 0.2 -> 0.9]
  t=  10.56s  ALARM RAISED   score=2.3639 (1.2x thr)
healthy phase: 7/374 windows flagged (1.9%)
fault phase:   295/374 windows flagged (78.9%)
```

Seven healthy windows tripped the per-window threshold and none of them
became an alarm. That gap is the debounce doing its job.

---

## How it works

`src/motor_anomaly/simulate.py`
Generates tri-axial vibration and shaft speed for a healthy or faulty
motor. There is no real rig behind this, see
[What the data is](#what-the-data-is). Pure numpy so the same maths can be
ported to the MCU later.

`src/motor_anomaly/features.py`
Turns a 1024-sample window into 46 floats: six time-domain stats, seven
band-energy features and a spectral centroid, per axis, plus four
speed-derived features. No scipy, one FFT per axis, nothing that needs the
whole segment in memory, because this has to compile against CMSIS-DSP on a
Cortex-M4F.

`src/motor_anomaly/dataset.py`
Splits data by segment, not by window. Windows overlap 50%, so splitting at
window level lets adjacent windows from the same recording leak between
train and test, which inflates validation scores. The autoencoder trains on
healthy windows only; faults are used only to set and evaluate the
threshold.

`src/motor_anomaly/model.py`
The autoencoder: 46-32-16-8-16-32-46. An autoencoder rather than a
classifier because a fixed list of fault classes will not cover every way a
real motor can fail. Trained only on healthy data, it flags anything it
hasn't seen. The trade-off is that it reports "something is wrong" rather
than naming the fault.

`src/motor_anomaly/train.py`
Trains the model and calibrates the threshold from healthy validation
error at a percentile chosen to hit a target false-positive rate, not by
maximising F1 against the simulated faults. Tuning to the specific faults
you happened to simulate defeats the point of an open-set detector.

`src/motor_anomaly/evaluate.py`
Scores the held-out split and writes `artifacts/report.json`, broken down
by fault type and by severity. The severity breakdown matters more than
the headline AUC for this use case.

`src/motor_anomaly/convert_tflite.py`
Converts the float Keras model to full int8 TFLite. Full integer
quantisation, not dynamic-range, because TFLite Micro on a Cortex-M4F has
no FPU path worth using and CMSIS-NN kernels are int8-only. The
representative dataset used for calibration is 90% healthy, 10% fault
windows, deliberately not healthy-only: calibrating on healthy data alone
produced a quantisation range that clipped most fault windows at the int8
boundary, which is not a correctness bug (a clipped anomaly still
reconstructs badly) but it does distort the recalibrated threshold if left
uncorrected.

`src/motor_anomaly/edge/runner.py`
The inference loop: extract features, score, debounce, spool. Debounce is
3-of-5 by default, because a 1% per-window false-alarm rate at roughly
6 windows/second works out to a nuisance alarm every 16 seconds on raw
per-window output, which nobody would leave switched on. A real fault trips
consecutive windows, so N-of-M debounce removes most nuisance alarms while
costing little detection latency.

`src/motor_anomaly/edge/uploader.py`
Ships the spool to S3 in batches, Hive-partitioned by feature version and
date, defaulting to a dry run. Sends only the 46-float feature vector per
window (184 bytes), not raw waveform (6 KB), since the feature extractor is
frozen and retraining only needs features. The feature version is in the
partition path so a change to `features.py` can't silently mix incompatible
data into a retrain.

```
cloud/sagemaker/       retraining job + launcher
cloud/terraform/       S3 bucket, lifecycle rules, least-privilege role
firmware/nano33ble/    Arduino sketch + generated model/scaler headers
tests/                 unit tests, mostly guarding the feature extractor
```

---

## What the data is

There is no real motor behind any of these numbers. `simulate.py` generates
synthetic tri-axial vibration with bearing defect frequencies for an SKF
6205 (ball pass frequency outer race at 3.585x shaft rate, inner race at
5.415x), impulse trains ringing down structural resonances, load-zone
amplitude modulation, and speed wander folded into the phase so spectral
peaks smear the way real ones do.

That gives the pipeline a realistic shape of problem, and the design
choices (feature set, debounce, quantisation strategy) would carry over to
real data. It does not mean the accuracy numbers above would hold on a real
motor. Swapping in real accelerometer data means changing `fs` and
`nominal_rpm` in `config/default.yaml` and re-checking the band edges in
`features.py`. Nothing else should need to move.

The obvious next step, if this ever gets a real rig, is the
[CWRU bearing dataset](https://engineering.case.edu/bearingdatacenter),
which has real accelerometer data with seeded faults.

---

## Known limitations

- **Outer race detection is 76.4%.** Outer race impulses are low-energy and
  land in the same frequency bands that healthy operation also excites, so
  they don't stand out from the noise floor as cleanly as the other fault
  types. Envelope analysis (Hilbert transform, then FFT of the envelope)
  is the standard technique for this and would likely close most of the
  gap. It's affordable on a Cortex-M4F and is the next thing to add.
- **Detection below 35% severity is unreliable (55.6%).** Good enough to
  catch a fault before it takes the motor down, not good enough for
  long-horizon trend monitoring.
- **Firmware has not been validated on hardware.** The inference loop and
  timings are bench-measured on the Nano 33 BLE. Fault detection on an
  actual motor has not been demonstrated, and I am not claiming it works
  until it has been.
- **The tacho input is faked in firmware.** The sketch currently feeds
  nameplate RPM, so the four speed-derived features carry no real
  information on-device until a tacho is wired to an interrupt pin.
- **Retraining uses pseudo-labels.** The spool only contains windows the
  edge device already flagged, so the retraining job takes the bottom 60%
  by score as pseudo-healthy to track drift. Set that percentile too high
  and the job trains on real faults and goes blind to them, so there's a
  hard guard at 75%.

## Why WSL

Development happens inside WSL2 Ubuntu rather than Windows directly. The
toolchain is Unix-shaped, and keeping the repo on the ext4 side avoids the
9p filesystem bridge: running out of `/mnt/c/` makes every file operation
several times slower, which becomes obvious as soon as pytest starts
walking the tree. `scripts/bootstrap_wsl.sh` warns if you're under `/mnt/`.

## Licence

MIT.
