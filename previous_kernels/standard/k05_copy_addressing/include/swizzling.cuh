#pragma once
#include "utils.h"

namespace flash_attn_lab::standard::k05 {

struct SwizzleStride {
    int s0;
    int s1;
    int s2;
    int s3;

    DEVICE_INLINE constexpr int offset_s2rmem(int iter) const {
        const int i0 = (iter >> 2) & 1;
        const int i1 = (iter >> 1) & 1;
        const int i2 = iter & 1;
        return i0 * s0 + i1 * s1 + i2 * s2;
    }

    DEVICE_INLINE constexpr int offset_r2smem(int iter) const {
        const int i0 = (iter >> 3) & 1;
        const int i1 = (iter >> 2) & 1;
        const int i2 = (iter >> 1) & 1;
        const int i3 = iter & 1;
        return i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3;
    }
};

template <int col_fragments>
DEVICE_INLINE int swizzled_col_fragment(int row, int col_fragment) {
    static_assert(col_fragments % BANKS_PER_VEC4_ACCESS == 0,
                  "# col tiles is a multiple of # BANKS");

    return (row % ROWS_PER_FRAGMENT) ^ col_fragment;
}

template <int col_fragments, bool swizzle>
DEVICE_INLINE int get_smem_col_fragment(const int row, const int col_fragment) {
    return swizzle ? swizzled_col_fragment<col_fragments>(row, col_fragment)
                   : col_fragment;
}

}
