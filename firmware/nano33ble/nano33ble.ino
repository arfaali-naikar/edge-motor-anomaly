/*
 * Motor anomaly detector -- Arduino Nano 33 BLE Sense Rev2.
 *
 * STATUS: this compiles and runs the inference loop against the quantised
 * model, but I have not yet had it on a real motor -- I'm still waiting on the
 * mounting bracket. Treat the timing numbers below as measured-on-bench, and
 * the fault detection as unvalidated on real hardware. The Python edge runner
 * is the validated path today.
 *
 * Budget on a 64 MHz Cortex-M4F:
 *   FFT (3x 1024-pt, CMSIS-DSP)   ~28 ms
 *   time-domain stats             ~ 4 ms
 *   TFLite Micro invoke           ~ 3 ms
 *   ------------------------------------
 *   total                         ~35 ms  against a 160 ms hop. Comfortable.
 *
 * RAM: the two 1024x3 float ring buffers are 24 KB each and are the thing that
 * will bite you, not the model. 256 KB total on this board.
 */

#include <Arduino_BMI270_BMM150.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "features.h"
#include "model_data.h"      // generated: make firmware-model
#include "scaler_data.h"     // generated: make firmware-model

// Set by convert_tflite.py -- this is threshold_int8, NOT the float one.
// Getting this wrong is the single easiest way to ship a detector that either
// never fires or never stops firing.
static const float kThreshold = THRESHOLD_INT8;

static const int kDebounceN = 3;
static const int kDebounceM = 5;

// TFLite Micro arena. 16 KB is measured with headroom; if you enlarge the
// model, watch for AllocateTensors() returning kTfLiteError rather than
// guessing.
constexpr int kArenaSize = 16 * 1024;
alignas(16) static uint8_t g_arena[kArenaSize];

static tflite::MicroInterpreter *g_interpreter = nullptr;
static TfLiteTensor *g_input = nullptr;
static TfLiteTensor *g_output = nullptr;

static float g_ring[WINDOW_SIZE * 3];
static float g_rpm_ring[WINDOW_SIZE];
static int g_write_idx = 0;
static bool g_debounce[kDebounceM] = {false};
static int g_debounce_idx = 0;
static bool g_alarm = false;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}

  if (!IMU.begin()) {
    Serial.println("IMU init failed");
    while (1) { digitalWrite(LED_BUILTIN, HIGH); delay(200); digitalWrite(LED_BUILTIN, LOW); delay(200); }
  }
  pinMode(LED_BUILTIN, OUTPUT);

  const tflite::Model *model = tflite::GetModel(g_motor_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("model schema mismatch -- regenerate model_data.h");
    while (1) {}
  }

  // Only the ops this model actually uses. AllOpsResolver would work but
  // costs ~40 KB of flash for ops we never call.
  static tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddQuantize();

  static tflite::MicroInterpreter interpreter(model, resolver, g_arena, kArenaSize);
  g_interpreter = &interpreter;
  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed -- raise kArenaSize");
    while (1) {}
  }
  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);

  Serial.print("ready. features="); Serial.print(N_FEATURES);
  Serial.print(" threshold="); Serial.println(kThreshold, 5);
}

// Reconstruction error in the dequantised domain -- must match
// AnomalyDetector.score() in the Python runner exactly, or the threshold
// calibrated in the cloud is meaningless on the device.
static float score_window(const float *z) {
  const float in_scale = g_input->params.scale;
  const int in_zp = g_input->params.zero_point;
  const float out_scale = g_output->params.scale;
  const int out_zp = g_output->params.zero_point;

  for (int i = 0; i < N_FEATURES; i++) {
    int32_t q = (int32_t)lroundf(z[i] / in_scale) + in_zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    g_input->data.int8[i] = (int8_t)q;
  }

  if (g_interpreter->Invoke() != kTfLiteOk) return -1.0f;

  float sse = 0.0f;
  for (int i = 0; i < N_FEATURES; i++) {
    float in_deq = ((float)g_input->data.int8[i] - in_zp) * in_scale;
    float out_deq = ((float)g_output->data.int8[i] - out_zp) * out_scale;
    float d = in_deq - out_deq;
    sse += d * d;
  }
  return sse / (float)N_FEATURES;
}

void loop() {
  float ax, ay, az;
  if (!IMU.accelerationAvailable()) return;
  IMU.readAcceleration(ax, ay, az);

  g_ring[g_write_idx * 3 + 0] = ax;
  g_ring[g_write_idx * 3 + 1] = ay;
  g_ring[g_write_idx * 3 + 2] = az;
  // TODO: real tacho on a hardware interrupt pin. Until that's wired, feed the
  // nameplate speed so the order-tracking features are at least consistent --
  // they just won't carry any information.
  g_rpm_ring[g_write_idx] = 1455.0f;
  g_write_idx++;

  if (g_write_idx < WINDOW_SIZE) return;

  static float raw[N_FEATURES];
  static float z[N_FEATURES];
  static float scratch[WINDOW_SIZE];

  features_extract(g_ring, g_rpm_ring, raw, scratch);
  features_standardise(raw, z);
  float score = score_window(z);

  bool flagged = score > kThreshold;
  g_debounce[g_debounce_idx] = flagged;
  g_debounce_idx = (g_debounce_idx + 1) % kDebounceM;

  int hits = 0;
  for (int i = 0; i < kDebounceM; i++) if (g_debounce[i]) hits++;

  if (hits >= kDebounceN && !g_alarm) {
    g_alarm = true;
    Serial.print("ALARM score="); Serial.println(score, 5);
  } else if (hits == 0 && g_alarm) {
    g_alarm = false;
    Serial.println("clear");
  }
  digitalWrite(LED_BUILTIN, g_alarm ? HIGH : LOW);

  // Slide by HOP_SIZE, keeping the 50% overlap the Python side uses.
  memmove(g_ring, g_ring + HOP_SIZE * 3, HOP_SIZE * 3 * sizeof(float));
  memmove(g_rpm_ring, g_rpm_ring + HOP_SIZE, HOP_SIZE * sizeof(float));
  g_write_idx = HOP_SIZE;
}
