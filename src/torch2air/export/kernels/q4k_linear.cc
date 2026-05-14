// Q4_K linear tile kernel for quantized_qwen3 projections.
//
// AIR supplies one normalized hidden row, OUTPUT_TILE_ROWS packed Q4_K rows,
// and one output tile in L1. Each packed row stores the original GGUF Q4_K
// words followed by f32 bit patterns for per-block d/dmin values:
//
//   words[0:BLOCKS_PER_ROW*36]              original Q4_K blocks
//   words[Q4K_ROW_WORDS + block*2 + 0]      d as f32 bits
//   words[Q4K_ROW_WORDS + block*2 + 1]      dmin as f32 bits

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

#define Q4K_WORDS_PER_BLOCK 36
#define Q4K_ROW_WORDS (BLOCKS_PER_ROW * Q4K_WORDS_PER_BLOCK)
#define Q4K_WEIGHT_WORDS (Q4K_ROW_WORDS + BLOCKS_PER_ROW * 2)
#define DOT_VECTOR_LANES 16

static inline uint8_t q4k_byte(const uint32_t *__restrict block, int byte_offset) {
  const uint32_t word = block[byte_offset >> 2];
  return (uint8_t)((word >> ((byte_offset & 3) * 8)) & 0xffu);
}

static inline void q4k_scale_min(
    const uint32_t *__restrict block,
    int subblock,
    uint32_t &scale,
    uint32_t &minimum) {
  if (subblock < 4) {
    scale = q4k_byte(block, 4 + subblock) & 0x3fu;
    minimum = q4k_byte(block, 8 + subblock) & 0x3fu;
    return;
  }

  const uint32_t d_byte = q4k_byte(block, subblock);
  const uint32_t m_byte = q4k_byte(block, 4 + subblock);
  const uint32_t md_byte = q4k_byte(block, 8 + subblock);
  scale = (md_byte & 0x0fu) | ((d_byte >> 2) & 0x30u);
  minimum = (md_byte >> 4) | ((m_byte >> 2) & 0x30u);
}

static inline float f32_from_bits(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } value;
  value.u = bits;
  return value.f;
}

static inline void q4k_fill_weights_16(
    const uint32_t *__restrict block,
    int subblock,
    int lane_base,
    float d,
    float dmin,
    float *__restrict weights) {
  uint32_t scale = 0;
  uint32_t minimum = 0;
  q4k_scale_min(block, subblock, scale, minimum);
  const float scaled_d = d * (float)scale;
  const float scaled_min = dmin * (float)minimum;
  const int q_base = 16 + (subblock >> 1) * 32 + lane_base;
  for (int lane = 0; lane < DOT_VECTOR_LANES; ++lane) {
    const uint32_t q_byte = q4k_byte(block, q_base + lane);
    const uint32_t q = (subblock & 1) == 0 ? (q_byte & 15u) : (q_byte >> 4);
    weights[lane] = scaled_d * (float)q - scaled_min;
  }
}

static inline float q4k_dot_block(
    const uint32_t *__restrict block,
    const float *__restrict hidden,
    float d,
    float dmin) {
  ::aie::accum<accfloat, DOT_VECTOR_LANES> acc =
      ::aie::zeros<accfloat, DOT_VECTOR_LANES>();
  alignas(64) float weights[DOT_VECTOR_LANES];
  for (int subblock = 0; subblock < 8; ++subblock) {
    const int hidden_base = subblock * 32;
    for (int lane_base = 0; lane_base < 32; lane_base += DOT_VECTOR_LANES) {
      q4k_fill_weights_16(block, subblock, lane_base, d, dmin, weights);
      const ::aie::vector<float, DOT_VECTOR_LANES> hidden_v =
          ::aie::load_v<DOT_VECTOR_LANES>(hidden + hidden_base + lane_base);
      const ::aie::vector<float, DOT_VECTOR_LANES> weight_v =
          ::aie::load_v<DOT_VECTOR_LANES>(weights);
      acc = ::aie::add(acc, ::aie::mul(hidden_v, weight_v));
    }
  }
  return ::aie::reduce_add(acc.to_vector<float>());
}

extern "C" {

void q4k_linear_tile(
    float *__restrict hidden,
    uint32_t *__restrict packed_weights,
    float *__restrict output) {
  for (int row = 0; row < OUTPUT_TILE_ROWS; ++row) {
    const uint32_t *__restrict row_weight = packed_weights + row * Q4K_WEIGHT_WORDS;
    float acc = 0.0f;
    for (int block_index = 0; block_index < BLOCKS_PER_ROW; ++block_index) {
      const uint32_t *__restrict block = row_weight + block_index * Q4K_WORDS_PER_BLOCK;
      const uint32_t d_bits = row_weight[Q4K_ROW_WORDS + block_index * 2];
      const uint32_t dmin_bits = row_weight[Q4K_ROW_WORDS + block_index * 2 + 1];
      acc += q4k_dot_block(
          block,
          hidden + block_index * 256,
          f32_from_bits(d_bits),
          f32_from_bits(dmin_bits));
    }
    output[row] = acc;
  }
}

} // extern "C"
