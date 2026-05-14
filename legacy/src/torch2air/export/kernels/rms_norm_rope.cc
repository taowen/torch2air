// Qwen3 q/k RMSNorm + RoPE tile kernel.
//
// AIR supplies one projected head row, the 128-wide q_norm/k_norm weight, and
// one RoPE cos/sin row in L1. This mirrors torch2vk's fused norm+rope shader
// while keeping launch/herd/DMA ownership in MLIR-AIR.

#include <cstdint>

#ifndef HEAD_DIM
#define HEAD_DIM 128
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

void rms_norm_rope_tile(
    float *__restrict input,
    float *__restrict weight,
    float *__restrict cos_row,
    float *__restrict sin_row,
    float *__restrict output) {
  float sum_squares = 0.0f;
  for (int d = 0; d < HEAD_DIM; ++d) {
    const float x = input[d];
    sum_squares += x * x;
  }

  const float variance = sum_squares * (1.0f / (float)HEAD_DIM);
  const float inv_rms = fast_rsqrt(variance + f32_from_bits(RMS_NORM_EPS_BITS));

  for (int d = 0; d < HEAD_DIM; ++d) {
    const int rotated_d = d < (HEAD_DIM / 2) ? d + (HEAD_DIM / 2) : d - (HEAD_DIM / 2);
    const float sign = d < (HEAD_DIM / 2) ? -1.0f : 1.0f;
    const float norm_value = input[d] * inv_rms * weight[d];
    const float norm_rotated = sign * input[rotated_d] * inv_rms * weight[rotated_d];
    output[d] = norm_value * cos_row[d] + norm_rotated * sin_row[d];
  }
}

} // extern "C"
