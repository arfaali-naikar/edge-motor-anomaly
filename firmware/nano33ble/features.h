/*
 * C mirror of src/motor_anomaly/features.py.
 *
 * These two files MUST stay in lockstep. The Python side owns the definition;
 * this is the port. If you change one, change the other and bump
 * FEATURE_VERSION in both, or the model will be fed a vector whose columns
 * mean something different from what it trained on -- and nothing will warn
 * you, because the length still matches.
 *
 * Layout (see the Python docstring for the reasoning):
 *   [0:14]   axis x : rms, peak, crest, kurt, skew, zcr, band0..6, centroid
 *   [14:28]  axis y : same
 *   [28:42]  axis z : same
 *   [42:46]  mean_rpm, rpm_std, order_1x, order_2x
 */

#ifndef MOTOR_FEATURES_H
#define MOTOR_FEATURES_H

#include <stdint.h>

#define FEATURE_VERSION 3
#define N_FEATURES 46
#define FEATURES_PER_AXIS 14
#define N_BANDS 7
#define WINDOW_SIZE 1024
#define HOP_SIZE 512
#define SAMPLE_RATE_HZ 3200.0f
#define EPS 1e-9f

/* Band edges in Hz, must match BANDS_HZ in features.py exactly. */
static const float kBandEdges[N_BANDS + 1] = {
    0.0f, 50.0f, 120.0f, 260.0f, 520.0f, 900.0f, 1250.0f, 1600.0f};

/* Fill out[0..45] from one window. accel is interleaved xyz.
 * Uses arm_rfft_fast_f32 from CMSIS-DSP; scratch must be >= WINDOW_SIZE floats.
 */
void features_extract(const float *accel_xyz, const float *rpm, float *out, float *scratch);

/* z = (x - mean) / scale, using the constants baked in by
 * scripts/export_scaler.py. Done in float then handed to the int8 quantiser. */
void features_standardise(const float *raw, float *z);

#endif /* MOTOR_FEATURES_H */
