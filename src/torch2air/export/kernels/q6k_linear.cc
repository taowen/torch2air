// Q6_K linear tile kernel for quantized_qwen3 v_proj.
//
// Each packed row stores widened GGUF Q6_K halfwords followed by f32 bit
// patterns for the per-block super-block scale d:
//
//   words[0:BLOCKS_PER_ROW*105]             original uint16 Q6_K block words
//   words[Q6K_ROW_WORDS + block]            d as f32 bits

#include <cstdint>

#include <aie_api/aie.hpp>

#ifndef OUTPUT_TILE_ROWS
#define OUTPUT_TILE_ROWS 16
#endif

#ifndef BLOCKS_PER_ROW
#define BLOCKS_PER_ROW 4
#endif

#ifndef HIDDEN_SIZE
#define HIDDEN_SIZE 1024
#endif

#define Q6K_WORDS_PER_BLOCK 105
#define Q6K_ROW_WORDS (BLOCKS_PER_ROW * Q6K_WORDS_PER_BLOCK)
#define Q6K_WEIGHT_WORDS (Q6K_ROW_WORDS + BLOCKS_PER_ROW)
#define DOT_VECTOR_LANES 16

static inline uint8_t q6k_byte(const uint32_t *__restrict block, int byte_offset) {
  const uint32_t halfword = block[byte_offset >> 1];
  return (uint8_t)((halfword >> ((byte_offset & 1) * 8)) & 0xffu);
}

static inline float f32_from_bits(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } value;
  value.u = bits;
  return value.f;
}

static inline void q6k_fill_weights_16(
    const uint32_t *__restrict block,
    int half_index,
    int group,
    int output_part,
    float d,
    float *__restrict weights) {
  const int l_base = group * DOT_VECTOR_LANES;
  const int ql_base = half_index * 64;
  const int qh_base = half_index * 32;
  const int sc_base = half_index * 8;
  const int scale_offset = sc_base + group + output_part * 2;
  const float scale = d * (float)(int8_t)q6k_byte(block, 192 + scale_offset);

  for (int lane = 0; lane < DOT_VECTOR_LANES; ++lane) {
    const int l = l_base + lane;
    const uint8_t ql0 = q6k_byte(block, ql_base + l);
    const uint8_t ql1 = q6k_byte(block, ql_base + l + 32);
    const uint8_t qh = q6k_byte(block, 128 + qh_base + l);
    int q = 0;
    if (output_part == 0) {
      q = (int)((ql0 & 0x0fu) | (((qh >> 0) & 3u) << 4)) - 32;
    } else if (output_part == 1) {
      q = (int)((ql1 & 0x0fu) | (((qh >> 2) & 3u) << 4)) - 32;
    } else if (output_part == 2) {
      q = (int)((ql0 >> 4) | (((qh >> 4) & 3u) << 4)) - 32;
    } else {
      q = (int)((ql1 >> 4) | (((qh >> 6) & 3u) << 4)) - 32;
    }
    weights[lane] = scale * (float)q;
  }
}

static inline float q6k_dot_block(
    const uint32_t *__restrict block,
    const float *__restrict hidden,
    float d) {
  ::aie::accum<accfloat, DOT_VECTOR_LANES> acc =
      ::aie::zeros<accfloat, DOT_VECTOR_LANES>();
  alignas(64) float weights[DOT_VECTOR_LANES];
  for (int half_index = 0; half_index < 2; ++half_index) {
    const int half_base = half_index * 128;
    for (int group = 0; group < 2; ++group) {
      const int group_base = group * DOT_VECTOR_LANES;
      for (int output_part = 0; output_part < 4; ++output_part) {
        q6k_fill_weights_16(block, half_index, group, output_part, d, weights);
        const int hidden_base = half_base + output_part * 32 + group_base;
        const ::aie::vector<float, DOT_VECTOR_LANES> hidden_v =
            ::aie::load_v<DOT_VECTOR_LANES>(hidden + hidden_base);
        const ::aie::vector<float, DOT_VECTOR_LANES> weight_v =
            ::aie::load_v<DOT_VECTOR_LANES>(weights);
        acc = ::aie::add(acc, ::aie::mul(hidden_v, weight_v));
      }
    }
  }
  return ::aie::reduce_add(acc.to_vector<float>());
}

extern "C" {

void q6k_linear_tile(
    float *__restrict hidden,
    uint32_t *__restrict packed_weights,
    float *__restrict output) {
  for (int row = 0; row < OUTPUT_TILE_ROWS; ++row) {
    const uint32_t *__restrict row_weight = packed_weights + row * Q6K_WEIGHT_WORDS;
    float acc = 0.0f;
    for (int block_index = 0; block_index < BLOCKS_PER_ROW; ++block_index) {
      const uint32_t *__restrict block = row_weight + block_index * Q6K_WORDS_PER_BLOCK;
      const uint32_t d_bits = row_weight[Q6K_ROW_WORDS + block_index];
      acc += q6k_dot_block(block, hidden + block_index * 256, f32_from_bits(d_bits));
    }
    output[row] = acc;
  }
}

} // extern "C"
