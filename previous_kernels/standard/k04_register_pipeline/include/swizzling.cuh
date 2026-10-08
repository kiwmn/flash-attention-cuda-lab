#pragma once
#include "utils.h"

namespace flash_attn_lab::standard::k04 {

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
