// Qwen3 single-head causal attention tile step.
//
// AIR owns q-block/kv-block tiling and L3<->L1 DMA. This external body updates
// one row of an output tile with one K/V tile using online softmax:
//   O = softmax(QK^T / sqrt(HEAD_DIM)) @ V

#include <cstdint>

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

#ifndef SEQUENCE_LENGTH
#define SEQUENCE_LENGTH 8
#endif

#ifndef QUERY_TILE_ROWS
#define QUERY_TILE_ROWS 4
#endif

#ifndef KEY_TILE_ROWS
#define KEY_TILE_ROWS 4
#endif

static_assert(HEAD_DIM == 128, "attention_core currently targets Qwen3 head_dim=128");
static_assert(KEY_TILE_ROWS == 4, "attention_core currently uses explicit four-row K/V tiles");
static_assert(SEQUENCE_LENGTH % QUERY_TILE_ROWS == 0, "sequence length must divide Q tile rows");
static_assert(SEQUENCE_LENGTH % KEY_TILE_ROWS == 0, "sequence length must divide K/V tile rows");

static inline float f32_from_bits(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } value;
  value.u = bits;
  return value.f;
}

static inline float fast_exp(float x) {
  if (x < -20.0f) {
    return 0.0f;
  }
  if (x > 0.0f) {
    x = 0.0f;
  }

  const float inv_ln2 = 1.4426950408889634f;
  const float ln2 = 0.6931471805599453f;
  float scaled = x * inv_ln2;
  int n = (int)scaled;
  if ((float)n > scaled) {
    --n;
  }
  const float r = x - (float)n * ln2;
  const float r2 = r * r;
  const float r3 = r2 * r;
  const float r4 = r2 * r2;
  const float r5 = r4 * r;
  const float exp_r =
      1.0f + r + 0.5f * r2 + (1.0f / 6.0f) * r3 + (1.0f / 24.0f) * r4 + (1.0f / 120.0f) * r5;
  const int exponent = n + 127;
  if (exponent <= 0) {
    return 0.0f;
  }
  return f32_from_bits((uint32_t)exponent << 23) * exp_r;
}

extern "C" {

void attention_core_tile(
    int32_t *__restrict meta,
    float *__restrict q,
    float *__restrict k,
    float *__restrict v,
    float *__restrict row_max,
    float *__restrict row_sum,
    float *__restrict output) {
  const int q_base = meta[0];
  const int kv_base = meta[1];
  const int q_row = meta[2];
  const bool first_kv_tile = kv_base == 0;
  const bool last_kv_tile = (kv_base + KEY_TILE_ROWS) >= SEQUENCE_LENGTH;
  const float neg_inf = -3.4028234663852886e38f;
  const float scale = 0.08838834764831845f; // 1 / sqrt(128)

  if (first_kv_tile) {
    row_max[q_row] = neg_inf;
    row_sum[q_row] = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
      output[q_row * HEAD_DIM + d] = 0.0f;
    }
  }

  const int global_q = q_base + q_row;
  float score0 = neg_inf;
  float score1 = neg_inf;
  float score2 = neg_inf;
  float score3 = neg_inf;

  if (kv_base <= global_q) {
    float dot = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
      dot += q[q_row * HEAD_DIM + d] * k[d];
    }
    score0 = dot * scale;
  }
  if (kv_base + 1 <= global_q) {
    float dot = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
      dot += q[q_row * HEAD_DIM + d] * k[HEAD_DIM + d];
    }
    score1 = dot * scale;
  }
  if (kv_base + 2 <= global_q) {
    float dot = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
      dot += q[q_row * HEAD_DIM + d] * k[2 * HEAD_DIM + d];
    }
    score2 = dot * scale;
  }
  if (kv_base + 3 <= global_q) {
    float dot = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
      dot += q[q_row * HEAD_DIM + d] * k[3 * HEAD_DIM + d];
    }
    score3 = dot * scale;
  }

  float tile_max = score0;
  if (score1 > tile_max) {
    tile_max = score1;
  }
  if (score2 > tile_max) {
    tile_max = score2;
  }
  if (score3 > tile_max) {
    tile_max = score3;
  }

  const float old_max = row_max[q_row];
  const float old_sum = row_sum[q_row];
  const float new_max = old_max > tile_max ? old_max : tile_max;
  const float old_scale = old_sum > 0.0f ? fast_exp(old_max - new_max) : 0.0f;
  const float weight0 = score0 == neg_inf ? 0.0f : fast_exp(score0 - new_max);
  const float weight1 = score1 == neg_inf ? 0.0f : fast_exp(score1 - new_max);
  const float weight2 = score2 == neg_inf ? 0.0f : fast_exp(score2 - new_max);
  const float weight3 = score3 == neg_inf ? 0.0f : fast_exp(score3 - new_max);
  const float tile_sum = weight0 + weight1 + weight2 + weight3;
  const float new_sum = old_sum * old_scale + tile_sum;

  for (int d = 0; d < HEAD_DIM; ++d) {
    float acc = output[q_row * HEAD_DIM + d] * old_scale;
    acc += weight0 * v[d];
    acc += weight1 * v[HEAD_DIM + d];
    acc += weight2 * v[2 * HEAD_DIM + d];
    acc += weight3 * v[3 * HEAD_DIM + d];
    output[q_row * HEAD_DIM + d] = acc;
  }

  row_max[q_row] = new_max;
  row_sum[q_row] = new_sum;

  if (last_kv_tile) {
    const float inv_sum = 1.0f / new_sum;
    for (int d = 0; d < HEAD_DIM; ++d) {
      output[q_row * HEAD_DIM + d] *= inv_sum;
    }
  }
}

} // extern "C"
