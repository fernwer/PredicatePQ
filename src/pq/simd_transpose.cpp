#include "predicatepq/simd_transpose.hpp"
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace ppq {
namespace {

static inline void transpose_scalar(const uint8_t* row_codes, uint32_t batch, uint32_t code_size, uint8_t* soa_out) {
    for (uint32_t m = 0; m < code_size; ++m) {
        uint8_t* dst = soa_out + static_cast<size_t>(m) * batch;
        for (uint32_t i = 0; i < batch; ++i) {
            dst[i] = row_codes[static_cast<size_t>(i) * code_size + m];
        }
    }
}

#if defined(__AVX512F__)
static inline void transpose_avx512(const uint8_t* row_codes, uint32_t batch, uint32_t code_size, uint8_t* soa_out) {
    constexpr uint32_t V = 64;
    alignas(64) uint8_t tmp[V];

    for (uint32_t m = 0; m < code_size; ++m) {
        uint8_t* dst = soa_out + static_cast<size_t>(m) * batch;
        uint32_t i = 0;
        for (; i + V <= batch; i += V) {
            for (uint32_t k = 0; k < V; ++k) {
                tmp[k] = row_codes[static_cast<size_t>(i + k) * code_size + m];
            }
            __m512i v = _mm512_load_si512(reinterpret_cast<const void*>(tmp));
            _mm512_storeu_si512(reinterpret_cast<void*>(dst + i), v);
        }
        for (; i < batch; ++i) {
            dst[i] = row_codes[static_cast<size_t>(i) * code_size + m];
        }
    }
}
#endif

#if defined(__AVX2__)
static inline void transpose_avx2(const uint8_t* row_codes, uint32_t batch, uint32_t code_size, uint8_t* soa_out) {
    constexpr uint32_t V = 32;
    alignas(32) uint8_t tmp[V];

    for (uint32_t m = 0; m < code_size; ++m) {
        uint8_t* dst = soa_out + static_cast<size_t>(m) * batch;
        uint32_t i = 0;
        for (; i + V <= batch; i += V) {
            for (uint32_t k = 0; k < V; ++k) {
                tmp[k] = row_codes[static_cast<size_t>(i + k) * code_size + m];
            }
            __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), v);
        }
        for (; i < batch; ++i) {
            dst[i] = row_codes[static_cast<size_t>(i) * code_size + m];
        }
    }
}
#endif

} // namespace

void transpose_codes_on_the_fly(const uint8_t* row_codes, uint32_t batch, uint32_t code_size, uint8_t* soa_out) {
#if defined(__AVX512F__)
    transpose_avx512(row_codes, batch, code_size, soa_out);
#elif defined(__AVX2__)
    transpose_avx2(row_codes, batch, code_size, soa_out);
#else
    transpose_scalar(row_codes, batch, code_size, soa_out);
#endif
}

} // namespace ppq