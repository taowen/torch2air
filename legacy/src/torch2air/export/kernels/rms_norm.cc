// Qwen3 RMSNorm tile kernel.
//
// AIR supplies one hidden row, the RMSNorm weight, and one output row in L1.
// This avoids lowering math.rsqrt to a Peano FSQRT instruction, which is not
// legal on the current AIE2P backend.

#include <cstdint>

#ifndef HIDDEN_SIZE
#define HIDDEN_SIZE 1024
#endif

#ifndef RMS_NORM_EPS_BITS
#define RMS_NORM_EPS_BITS 0x358637bd
#endif

static inline float f32_from_bits(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } value;
  value.u = bits;
  return value.f;
}

static inline uint32_t f32_to_bits(float value) {
  union {
    float f;
    uint32_t u;
  } bits;
  bits.f = value;
  return bits.u;
}

static inline float fast_rsqrt(float x) {
  const float half_x = 0.5f * x;
  uint32_t i = f32_to_bits(x);
  i = 0x5f3759dfu - (i >> 1);
  float y = f32_from_bits(i);
  y = y * (1.5f - half_x * y * y);
  y = y * (1.5f - half_x * y * y);
  y = y * (1.5f - half_x * y * y);
  return y;
}

extern "C" {

void rms_norm_tile(
    float *__restrict hidden,
    float *__restrict weight,
    float *__restrict output) {
  float sum_squares = 0.0f;
  for (int i = 0; i < HIDDEN_SIZE; ++i) {
    const float x = hidden[i];
    sum_squares += x * x;
  }

  const float variance = sum_squares * (1.0f / (float)HIDDEN_SIZE);
  const float inv_rms = fast_rsqrt(variance + f32_from_bits(RMS_NORM_EPS_BITS));

  for (int i = 0; i < HIDDEN_SIZE; ++i) {
    output[i] = hidden[i] * inv_rms * weight[i];
  }
}

} // extern "C"
