#pragma once

#include "utils.h"

namespace flash_attn_lab::paged::k02 {

// Adapted from https://leimao.github.io/blog/CuTe-Swizzle/.
template <int BBits = 3, int MBase = 0, int SShift = 3>
struct CuteSwizzle {
    static constexpr int mbase = MBase;
    static constexpr int mask_bits = BBits;
    static constexpr int mask_shift = SShift;

    static constexpr int bit_mask = (1 << mask_bits) - 1;
    static constexpr int yy_mask = bit_mask << (mbase + mask_shift);
    static constexpr int yy_mask_lowest_bit = yy_mask & -yy_mask;

    DEVICE_INLINE constexpr static auto apply(int const &offset) {
        const int row_shifted = (offset & yy_mask) >> mask_shift;
        return offset ^ row_shifted;
    }
};

struct NoSwizzle {
    DEVICE_INLINE constexpr static auto apply(int const &offset) {
        return offset;
    }
};

}
