// Small bf16 elementwise kernels used by exported aten op lowering.

#include <aie_api/aie.hpp>

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

#ifndef HEAD_COUNT
#define HEAD_COUNT 1
#endif

#ifndef RMS_NORM_EPS
#define RMS_NORM_EPS 0.000001f
#endif

#define VECTOR_LANES 16

extern "C" {

void bf16_rms_norm_128_tile(
    bfloat16 *__restrict input,
    bfloat16 *__restrict weight,
    bfloat16 *__restrict output) {
  ::aie::vector<float, VECTOR_LANES> sum_v =
      ::aie::zeros<float, VECTOR_LANES>();
  for (int i = 0; i < HEAD_DIM; i += VECTOR_LANES) {
    const ::aie::vector<bfloat16, VECTOR_LANES> input_v =
        ::aie::load_v<VECTOR_LANES>(input + i);
    const ::aie::vector<float, VECTOR_LANES> square_v = ::aie::mul_square(input_v);
    sum_v = ::aie::add(sum_v, square_v);
  }

  const float sum = ::aie::reduce_add(sum_v);
  const float inv_rms = ::aie::invsqrt(sum / (float)HEAD_DIM + RMS_NORM_EPS);
  const ::aie::vector<bfloat16, VECTOR_LANES> inv_rms_v =
      ::aie::broadcast<bfloat16, VECTOR_LANES>((bfloat16)inv_rms);

  for (int i = 0; i < HEAD_DIM; i += VECTOR_LANES) {
    const ::aie::vector<bfloat16, VECTOR_LANES> input_v =
        ::aie::load_v<VECTOR_LANES>(input + i);
    const ::aie::vector<bfloat16, VECTOR_LANES> weight_v =
        ::aie::load_v<VECTOR_LANES>(weight + i);
    const ::aie::vector<bfloat16, VECTOR_LANES> norm_v =
        ::aie::mul(input_v, inv_rms_v);
    const ::aie::vector<bfloat16, VECTOR_LANES> output_v =
        ::aie::mul(norm_v, weight_v);
    ::aie::store_v(output + i, output_v);
  }
}

void bf16_rope_128_tile(
    bfloat16 *__restrict input,
    bfloat16 *__restrict lut,
    bfloat16 *__restrict output) {
  constexpr int half_dim = HEAD_DIM / 2;
  for (int i = 0; i < half_dim; i += VECTOR_LANES) {
    const ::aie::vector<bfloat16, VECTOR_LANES> lo_v =
        ::aie::load_v<VECTOR_LANES>(input + i);
    const ::aie::vector<bfloat16, VECTOR_LANES> hi_v =
        ::aie::load_v<VECTOR_LANES>(input + half_dim + i);
    const ::aie::vector<bfloat16, VECTOR_LANES> cos_v =
        ::aie::load_v<VECTOR_LANES>(lut + i);
    const ::aie::vector<bfloat16, VECTOR_LANES> sin_v =
        ::aie::load_v<VECTOR_LANES>(lut + HEAD_DIM + i);
    const ::aie::vector<bfloat16, VECTOR_LANES> out_lo =
        ::aie::sub(::aie::mul(lo_v, cos_v), ::aie::mul(hi_v, sin_v));
    const ::aie::vector<bfloat16, VECTOR_LANES> out_hi =
        ::aie::add(::aie::mul(hi_v, cos_v), ::aie::mul(lo_v, sin_v));
    ::aie::store_v(output + i, out_lo);
    ::aie::store_v(output + half_dim + i, out_hi);
  }
}

void bf16_rms_norm_heads_tile(
    bfloat16 *__restrict input,
    bfloat16 *__restrict weight,
    bfloat16 *__restrict output) {
  for (int head_i = 0; head_i < HEAD_COUNT; ++head_i) {
    const int base = head_i * HEAD_DIM;
    bf16_rms_norm_128_tile(input + base, weight, output + base);
  }
}

void bf16_rope_heads_tile(
    bfloat16 *__restrict input,
    bfloat16 *__restrict lut,
    bfloat16 *__restrict output) {
  for (int head_i = 0; head_i < HEAD_COUNT; ++head_i) {
    const int base = head_i * HEAD_DIM;
    bf16_rope_128_tile(input + base, lut, output + base);
  }
}

} // extern "C"
