#pragma once

// Logical KV rows map through the request block table into physical cache
// pages.

#include <cstdint>

#include "utils.h"

namespace flash_attn_lab::paged::k06 {

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

template <typename T>
struct ThreadRowAccessor {
    T *base;
    T *current;
    int64_t token_stride;

    template <int kThreadsPerRow>
    DEVICE_INLINE typename ContiguousRowAccessor<T>::RowCursor
    make_row_cursor(int, bool valid) const {
        return {base, valid ? current : base, token_stride};
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
    int64_t k_tile_step, v_tile_step;
    const int *table;
    T *k_pages[pages_per_tile];
    T *v_pages[pages_per_tile];

    DEVICE_INLINE KVPageCache(T *k, T *v, int64_t kp, int64_t vp, int64_t kt,
                              int64_t vt, const int *page_table)
        : k_base(k), v_base(v), k_page_stride(kp), v_page_stride(vp),
          k_token_stride(kt), v_token_stride(vt), k_tile_step(Bc * kt),
          v_tile_step(Bc * vt), table(page_table) {}

    template <int TileInPage>
    DEVICE_INLINE void prepare(int tile_begin, int valid_rows, int thread_row) {
        if constexpr (TileInPage != 0) {
            if (tile_begin + thread_row < valid_rows) {
                k_pages[0] += k_tile_step;
                v_pages[0] += v_tile_step;
            }
            return;
        }
        const int first_page = tile_begin / PageSize;
#pragma unroll
        for (int p = 0; p < pages_per_tile; ++p) {
            const bool valid_page = (first_page + p) * PageSize < valid_rows;
            int physical_page = 0;
            if ((threadIdx.x % WARP_SIZE) == 0 && valid_page) {
                physical_page = table[first_page + p];
            }
            physical_page = __shfl_sync(0xffffffffu, physical_page, 0);
            const int row_in_page = thread_row;
            const bool valid_thread_row =
                tile_begin + p * PageSize + thread_row < valid_rows;
            k_pages[p] = valid_thread_row
                             ? k_base + int64_t(physical_page) * k_page_stride +
                                   int64_t(row_in_page) * k_token_stride
                             : k_base;
            v_pages[p] = valid_thread_row
                             ? v_base + int64_t(physical_page) * v_page_stride +
                                   int64_t(row_in_page) * v_token_stride
                             : v_base;
        }
    }

    template <bool IsV>
    DEVICE_INLINE ThreadRowAccessor<T> accessor(int page) const {
        if constexpr (IsV) {
            return {v_base, v_pages[page], v_token_stride};
        } else {
            return {k_base, k_pages[page], k_token_stride};
        }
    }
};

}
