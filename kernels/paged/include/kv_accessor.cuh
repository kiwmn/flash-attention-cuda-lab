#pragma once

// Logical KV rows map through the request block table into physical cache
// pages.

#include <cstdint>

#include "utils.h"

namespace flash_attn_lab::paged::k07 {

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
    uintptr_t *shared_pages;

    static constexpr int ids_per_warp = (pages_per_tile + 3) / 4;
    int pending_page_ids[ids_per_warp];
    T *k_pages[pages_per_tile];
    T *v_pages[pages_per_tile];

    DEVICE_INLINE KVPageCache(T *k, T *v, int64_t kp, int64_t vp, int64_t kt,
                              int64_t vt, const int *page_table,
                              uintptr_t *shared_metadata)
        : k_base(k), v_base(v), k_page_stride(kp), v_page_stride(vp),
          k_token_stride(kt), v_token_stride(vt), k_tile_step(Bc * kt),
          v_tile_step(Bc * vt), table(page_table),
          shared_pages(shared_metadata) {}

    template <int WarpNum>
    DEVICE_INLINE void issue_block_page_ids(int tile_begin, int valid_rows) {
        static_assert(PageSize <= Bc && PageSize % 16 == 0);
        static_assert(WarpNum == 4);
        static_assert(pages_per_tile == 1 || pages_per_tile == 2 ||
                      pages_per_tile == 4 || pages_per_tile == 8);
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        const int first_page = tile_begin / PageSize;
#pragma unroll
        for (int slot = 0; slot < ids_per_warp; ++slot) {
            const int page = warp_id + slot * WarpNum;
            if (lane_id == 0 && page < pages_per_tile &&
                (first_page + page) * PageSize < valid_rows) {
                asm volatile("ld.global.u32 %0, [%1];"
                             : "=r"(pending_page_ids[slot])
                             : "l"(table + first_page + page)
                             : "memory");
            }
        }
    }

    template <int WarpNum>
    DEVICE_INLINE void write_block_page_bases(int tile_begin, int valid_rows) {
        static_assert(PageSize <= Bc && WarpNum == 4);
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        const int buffer = (tile_begin / Bc) & 1;
        auto *shared_k = shared_pages + buffer * 2 * pages_per_tile;
        auto *shared_v = shared_k + pages_per_tile;
#pragma unroll
        for (int slot = 0; slot < ids_per_warp; ++slot) {
            const int page = warp_id + slot * WarpNum;
            if (lane_id == 0 && page < pages_per_tile) {
                T *k_ptr = k_base;
                T *v_ptr = v_base;
                if (tile_begin + page * PageSize < valid_rows) {
                    const int physical_page = pending_page_ids[slot];
                    k_ptr += int64_t(physical_page) * k_page_stride;
                    v_ptr += int64_t(physical_page) * v_page_stride;
                }
                shared_k[page] = reinterpret_cast<uintptr_t>(k_ptr);
                shared_v[page] = reinterpret_cast<uintptr_t>(v_ptr);
            }
        }
    }

    DEVICE_INLINE void prepare_block_pages(int tile_begin, int valid_rows,
                                           int thread_row) {
        static_assert(PageSize <= Bc);
        const int buffer = (tile_begin / Bc) & 1;
        auto *shared_k = shared_pages + buffer * 2 * pages_per_tile;
        auto *shared_v = shared_k + pages_per_tile;
        const int64_t k_row_offset = int64_t(thread_row) * k_token_stride;
        const int64_t v_row_offset = int64_t(thread_row) * v_token_stride;
#pragma unroll
        for (int page = 0; page < pages_per_tile; ++page) {
            const bool valid_thread_row =
                tile_begin + page * PageSize + thread_row < valid_rows;
            k_pages[page] =
                valid_thread_row
                    ? reinterpret_cast<T *>(shared_k[page]) + k_row_offset
                    : k_base;
            v_pages[page] =
                valid_thread_row
                    ? reinterpret_cast<T *>(shared_v[page]) + v_row_offset
                    : v_base;
        }
    }

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

            const int lane_id = threadIdx.x % WARP_SIZE;
            uintptr_t k_page_addr = 0;
            uintptr_t v_page_addr = 0;
            if (lane_id == p && valid_page) {
                const int physical_page = table[first_page + p];
                k_page_addr = reinterpret_cast<uintptr_t>(
                    k_base + int64_t(physical_page) * k_page_stride);
                v_page_addr = reinterpret_cast<uintptr_t>(
                    v_base + int64_t(physical_page) * v_page_stride);
            }
            const uint32_t k_lo =
                __shfl_sync(0xffffffffu, static_cast<uint32_t>(k_page_addr), p);
            const uint32_t k_hi = __shfl_sync(
                0xffffffffu, static_cast<uint32_t>(k_page_addr >> 32), p);
            const uint32_t v_lo =
                __shfl_sync(0xffffffffu, static_cast<uint32_t>(v_page_addr), p);
            const uint32_t v_hi = __shfl_sync(
                0xffffffffu, static_cast<uint32_t>(v_page_addr >> 32), p);
            const uintptr_t k_page_base =
                (static_cast<uintptr_t>(k_hi) << 32) | k_lo;
            const uintptr_t v_page_base =
                (static_cast<uintptr_t>(v_hi) << 32) | v_lo;
            const int row_in_page = thread_row;
            const bool valid_thread_row =
                tile_begin + p * PageSize + thread_row < valid_rows;
            k_pages[p] = valid_thread_row
                             ? reinterpret_cast<T *>(k_page_base) +
                                   int64_t(row_in_page) * k_token_stride
                             : k_base;
            v_pages[p] = valid_thread_row
                             ? reinterpret_cast<T *>(v_page_base) +
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
