#pragma once
#include "utils.h"

namespace flash_attn_lab::standard::k03 {

template <int col_fragments>
DEVICE_INLINE int swizzled_col_fragment(int row, int col_fragment) {
    static_assert(col_fragments % BANKS_PER_VEC4_ACCESS == 0,
                  "# col tiles is a multiple of # BANKS");

    return (row % ROWS_PER_FRAGMENT) ^ col_fragment;
}

}
