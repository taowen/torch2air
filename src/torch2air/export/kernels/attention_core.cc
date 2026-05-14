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
static_assert(KEY_TILE_ROWS > 0, "attention_core needs at least one K/V row per tile");
static_assert(KEY_TILE_ROWS <= 8, "attention_core currently validates K/V tiles up to 8 rows");
static_assert(SEQUENCE_LENGTH % QUERY_TILE_ROWS == 0, "sequence length must divide Q tile rows");
static_assert(SEQUENCE_LENGTH % KEY_TILE_ROWS == 0, "sequence length must divide K/V tile rows");

static inline float fast_exp(float x) {
  if (x < -20.0f) {
    return 0.0f;
  }
  if (x > 0.0f) {
    return 1.0f;
  }

  float y = 1.0f + x * 0.00390625f;
  if (y <= 0.0f) {
    return 0.0f;
  }
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  return y;
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
  float old_max = neg_inf;
  float old_sum = 0.0f;

  if (first_kv_tile) {
    row_max[q_row] = neg_inf;
    row_sum[q_row] = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
      output[q_row * HEAD_DIM + d] = 0.0f;
    }
  } else {
    old_max = row_max[q_row];
    old_sum = row_sum[q_row];
  }

  const int global_q = q_base + q_row;
  float scores[KEY_TILE_ROWS];
  int tile_max_index = 0;
  float tile_max = neg_inf;
  float second_max = neg_inf;
  for (int row = 0; row < KEY_TILE_ROWS; ++row) {
    float score = neg_inf;
    if (kv_base + row <= global_q) {
      float dot = 0.0f;
      for (int d = 0; d < HEAD_DIM; ++d) {
        dot += q[q_row * HEAD_DIM + d] * k[row * HEAD_DIM + d];
      }
      score = dot * scale;
    }
    scores[row] = score;
    if (score > tile_max) {
      second_max = tile_max;
      tile_max = score;
      tile_max_index = row;
    } else if (score > second_max) {
      second_max = score;
    }
  }

  const float new_max = old_max > tile_max ? old_max : tile_max;
  float old_scale = 0.0f;
  float weights[KEY_TILE_ROWS];
  for (int row = 0; row < KEY_TILE_ROWS; ++row) {
    weights[row] = 0.0f;
  }
  if (old_sum == 0.0f && tile_max - second_max > 20.0f) {
    weights[tile_max_index] = 1.0f;
  } else {
    old_scale = old_sum > 0.0f ? fast_exp(old_max - new_max) : 0.0f;
    for (int row = 0; row < KEY_TILE_ROWS; ++row) {
      weights[row] = scores[row] == neg_inf ? 0.0f : fast_exp(scores[row] - new_max);
    }
  }
  float tile_sum = 0.0f;
  for (int row = 0; row < KEY_TILE_ROWS; ++row) {
    tile_sum += weights[row];
  }

#if SEQUENCE_LENGTH == KEY_TILE_ROWS
  const float inv_tile_sum = tile_sum > 0.0f ? 1.0f / tile_sum : 0.0f;
  for (int d = 0; d < HEAD_DIM; ++d) {
    float acc = 0.0f;
    for (int row = 0; row < KEY_TILE_ROWS; ++row) {
      acc += weights[row] * v[row * HEAD_DIM + d];
    }
    output[q_row * HEAD_DIM + d] = acc * inv_tile_sum;
  }
  row_max[q_row] = tile_max;
  row_sum[q_row] = tile_sum;
  return;
#else
  const float new_sum = old_sum * old_scale + tile_sum;

  for (int d = 0; d < HEAD_DIM; ++d) {
    float acc = output[q_row * HEAD_DIM + d] * old_scale;
    for (int row = 0; row < KEY_TILE_ROWS; ++row) {
      acc += weights[row] * v[row * HEAD_DIM + d];
    }
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
#endif
}

} // extern "C"
