// Qwen3 RoPE cos/sin table tile kernel.
//
// AIR supplies one token position and one head_dim-wide output row in L1. The
// current Peano external kernel environment does not provide libm, so this
// first Qwen3 path uses the model's RoPE frequency ratio as a compile-time
// constant and local polynomial sin/cos approximations.

#include <cstdint>

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

#ifndef ROPE_INV_FREQ_RATIO
#define ROPE_INV_FREQ_RATIO 0.8058421877614819f
#endif

static inline float wrap_pi(float x) {
  const float pi = 3.14159265358979323846f;
  const float two_pi = 6.28318530717958647692f;
  while (x > pi) {
    x -= two_pi;
  }
  while (x < -pi) {
    x += two_pi;
  }
  return x;
}

static inline void reduce_half_pi(float x, float &reduced, float &sin_sign, float &cos_sign) {
  x = wrap_pi(x);
  const float half_pi = 1.57079632679489661923f;
  const float pi = 3.14159265358979323846f;
  sin_sign = 1.0f;
  cos_sign = 1.0f;
  if (x > half_pi) {
    reduced = pi - x;
    cos_sign = -1.0f;
    return;
  }
  if (x < -half_pi) {
    reduced = x + pi;
    sin_sign = -1.0f;
    cos_sign = -1.0f;
    return;
  }
  reduced = x;
}

static inline float sin_approx(float x) {
  float reduced = 0.0f;
  float sin_sign = 1.0f;
  float cos_sign = 1.0f;
  reduce_half_pi(x, reduced, sin_sign, cos_sign);
  x = reduced;
  const float x2 = x * x;
  const float x4 = x2 * x2;
  const float x6 = x4 * x2;
  const float x8 = x4 * x4;
  return sin_sign * x *
         (1.0f - x2 * (1.0f / 6.0f) + x4 * (1.0f / 120.0f) -
          x6 * (1.0f / 5040.0f) + x8 * (1.0f / 362880.0f));
}

static inline float cos_approx(float x) {
  float reduced = 0.0f;
  float sin_sign = 1.0f;
  float cos_sign = 1.0f;
  reduce_half_pi(x, reduced, sin_sign, cos_sign);
  x = reduced;
  const float x2 = x * x;
  const float x4 = x2 * x2;
  const float x6 = x4 * x2;
  const float x8 = x4 * x4;
  return cos_sign *
         (1.0f - x2 * 0.5f + x4 * (1.0f / 24.0f) -
          x6 * (1.0f / 720.0f) + x8 * (1.0f / 40320.0f));
}

static inline float qwen3_inv_freq(int freq_idx) {
  float value = 1.0f;
  for (int i = 0; i < freq_idx; ++i) {
    value *= ROPE_INV_FREQ_RATIO;
  }
  return value;
}

extern "C" {

void rope_table_tile(
    int32_t *__restrict position,
    float *__restrict cos_out,
    float *__restrict sin_out) {
  const float token_position = (float)position[0];
  for (int d = 0; d < HEAD_DIM; ++d) {
    const int freq_idx = d < (HEAD_DIM / 2) ? d : d - (HEAD_DIM / 2);
    const float angle = token_position * qwen3_inv_freq(freq_idx);
    cos_out[d] = cos_approx(angle);
    sin_out[d] = sin_approx(angle);
  }
}

} // extern "C"
