// Real GGUF Q4_K matvec kernel used by torch2air Spike 5.
//
// Computes output[row] = dot(dequant_q4_k(weight[row, :]), x).
// Weight layout is GGUF Q4_K: each 256-value block is 36 uint32 words
// containing fp16 d/dmin, 12 scale/min bytes, and 128 packed nibble bytes.

#include <cstdint>

#ifndef ROWS
#define ROWS 64
#endif

#ifndef BLOCKS_PER_ROW
#define BLOCKS_PER_ROW 4
#endif

static inline uint8_t q4k_byte(const uint32_t *__restrict block, int byte_offset) {
  const uint32_t word = block[byte_offset >> 2];
  return (uint8_t)((word >> ((byte_offset & 3) * 8)) & 0xffu);
}

static float fp16_to_f32(uint16_t h) {
  const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
  int exp = ((uint32_t)h >> 10) & 0x1f;
  uint32_t mant = (uint32_t)h & 0x03ffu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        exp -= 1;
      }
      exp += 1;
      mant &= ~0x0400u;
      bits = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
  }
  union {
    uint32_t u;
    float f;
  } value;
  value.u = bits;
  return value.f;
}

static inline void q4k_scale_min(
    const uint32_t *__restrict block,
    int subblock,
    uint32_t &scale,
    uint32_t &minimum) {
  const uint32_t scidx0 = subblock < 4 ? subblock : subblock + 4;
  const uint32_t scidx1 = subblock < 4 ? subblock : subblock - 4;
  const uint32_t scidxmask1 = subblock < 4 ? 0x30u : 0xc0u;
  const uint32_t scidxshift1 = subblock < 4 ? 0u : 2u;
  const uint32_t mbidx0 = subblock + 4;
  const uint32_t mbidx1 = subblock < 4 ? subblock + 4 : subblock;
  const uint32_t mbidxmask0 = subblock < 4 ? 0x0fu : 0xf0u;
  const uint32_t mbidxshift0 = subblock < 4 ? 0u : 4u;
  const uint32_t mbidxmask1 = subblock < 4 ? 0x30u : 0xc0u;
  const uint32_t mbidxshift1 = subblock < 4 ? 0u : 2u;
  scale = (q4k_byte(block, 4 + scidx0) & 0x0fu) |
          ((q4k_byte(block, 4 + scidx1) & scidxmask1) >> scidxshift1);
  minimum = ((q4k_byte(block, 4 + mbidx0) & mbidxmask0) >> mbidxshift0) |
            ((q4k_byte(block, 4 + mbidx1) & mbidxmask1) >> mbidxshift1);
}

static inline float q4k_value(const uint32_t *__restrict block, int local) {
  uint32_t scale = 0;
  uint32_t minimum = 0;
  q4k_scale_min(block, local >> 5, scale, minimum);
  const float d = fp16_to_f32((uint16_t)(block[0] & 0xffffu));
  const float dmin = fp16_to_f32((uint16_t)(block[0] >> 16));
  const int q_byte_offset = ((local >> 6) * 32) + (local & 31);
  const uint32_t q_byte = q4k_byte(block, 16 + q_byte_offset);
  const uint32_t q = (local & 32) == 0 ? (q_byte & 15u) : (q_byte >> 4);
  return d * (float)scale * (float)q - dmin * (float)minimum;
}

extern "C" {

void q4k_matvec_f32(
    uint32_t *__restrict weight,
    float *__restrict x,
    float *__restrict output) {
  for (int row = 0; row < ROWS; ++row) {
    float acc = 0.0f;
    const uint32_t *__restrict row_weight = weight + row * BLOCKS_PER_ROW * 36;
    for (int block_index = 0; block_index < BLOCKS_PER_ROW; ++block_index) {
      const uint32_t *__restrict block = row_weight + block_index * 36;
      const float *__restrict x_block = x + block_index * 256;
      for (int local = 0; local < 256; ++local) {
        acc += q4k_value(block, local) * x_block[local];
      }
    }
    output[row] = acc;
  }
}

} // extern "C"
