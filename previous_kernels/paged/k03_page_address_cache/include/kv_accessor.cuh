#pragma once

// Logical KV rows map through the request block table into physical cache
// pages.

#include <cstdint>

#include "utils.h"

namespace flash_attn_lab::paged::k03 {

template <typename T>
struct ContiguousRowAccessor {
    T *base;
    int64_t token_stride;

    struct RowCursor {
        T *base;
        T *current;
        int64_t token_stride;

        DEVICE_INLINE T *row_ptr(bool valid) const {
            return valid ? current : base;
        }

        DEVICE_INLINE void advance(int rows, bool next_valid) {
            if (next_valid) {
                current += int64_t(rows) * token_stride;
            }
        }
    };

    template <int kThreadsPerRow>
    DEVICE_INLINE RowCursor make_row_cursor(int logical_row, bool valid) const {
        return RowCursor{
            base,
            valid ? base + int64_t(logical_row) * token_stride : base,
            token_stride,
        };
    }
};

template <typename T, int PageSize, int Bc>
struct KVPageCache {
    static_assert(PageSize > 0 && (PageSize & (PageSize - 1)) == 0);
    static_assert(PageSize % Bc == 0 || Bc % PageSize == 0);
    static constexpr int page_size = PageSize;
    static constexpr int pages_per_tile = Bc > PageSize ? Bc / PageSize : 1;
    static constexpr int rows_per_page_chunk = Bc < PageSize ? Bc : PageSize;

    T *k_base, *v_base;
    int64_t k_page_stride, v_page_stride;
    int64_t k_token_stride, v_token_stride;
    const int *table;
    int cached_first_page = -1;
    T *k_pages[pages_per_tile];
    T *v_pages[pages_per_tile];

    DEVICE_INLINE KVPageCache(T *k, T *v, int64_t kp, int64_t vp, int64_t kt,
                              int64_t vt, const int *page_table)
        : k_base(k), v_base(v), k_page_stride(kp), v_page_stride(vp),
          k_token_stride(kt), v_token_stride(vt), table(page_table) {}

    DEVICE_INLINE void prepare(int tile_begin, int valid_rows) {
        const int first_page = tile_begin / PageSize;
        if (first_page == cached_first_page) {
            return;
        }
        cached_first_page = first_page;
#pragma unroll
        for (int p = 0; p < pages_per_tile; ++p) {
            const bool valid_page = (first_page + p) * PageSize < valid_rows;
            int physical_page = 0;
            if ((threadIdx.x % WARP_SIZE) == 0 && valid_page) {
                physical_page = table[first_page + p];
            }
            physical_page = __shfl_sync(0xffffffffu, physical_page, 0);
            k_pages[p] = valid_page
                             ? k_base + int64_t(physical_page) * k_page_stride
                             : k_base;
            v_pages[p] = valid_page
                             ? v_base + int64_t(physical_page) * v_page_stride
                             : v_base;
        }
    }

    template <bool IsV>
    DEVICE_INLINE ContiguousRowAccessor<T> accessor(int page) const {
        if constexpr (IsV) {
            return {v_pages[page], v_token_stride};
        } else {
            return {k_pages[page], k_token_stride};
        }
    }
};

}
