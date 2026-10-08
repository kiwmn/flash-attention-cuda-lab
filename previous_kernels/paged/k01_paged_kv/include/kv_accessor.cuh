#pragma once

// Logical KV rows map through the request block table into physical cache
// pages.

#include <cstdint>

#include "utils.h"

namespace flash_attn_lab::paged::k01 {

template <typename T>
struct ContiguousRowAccessor {
    T *base;
    int64_t token_stride;

    DEVICE_INLINE T *row_ptr(int logical_row, bool valid) const {
        return valid ? base + int64_t(logical_row) * token_stride : base;
    }
};

template <typename T>
struct PagedKVAccessor {
    T *base;
    int64_t token_stride;
    const int *page_table;
    int page_size;
    int64_t page_stride;

    DEVICE_INLINE T *row_ptr(int logical_row, bool valid) const {
        if (!valid) {
            return base;
        }
        const int logical_page = logical_row / page_size;
        const int offset_in_page = logical_row % page_size;
        const int physical_page = page_table[logical_page];
        return base + int64_t(physical_page) * page_stride +
               int64_t(offset_in_page) * token_stride;
    }
};

}
