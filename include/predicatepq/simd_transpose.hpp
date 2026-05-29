#pragma once
#include <cstdint>
#include <vector>

namespace ppq {

// row_codes: [batch * code_size], each row contiguous
// soa_out  : [code_size * batch], layout soa_out[m * batch + i]
void transpose_codes_on_the_fly(const uint8_t* row_codes, uint32_t batch, uint32_t code_size, uint8_t* soa_out);

} // namespace ppq