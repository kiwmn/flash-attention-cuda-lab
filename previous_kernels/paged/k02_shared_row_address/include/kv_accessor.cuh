#pragma once

// Logical KV rows map through the request block table into physical cache
// pages.

#include <cstdint>

#include "utils.h"

namespace flash_attn_lab::paged::k02 {

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
struct PagedKVAccessor {
    T *base;
    int64_t token_stride;
    const int *page_table;
    int page_size;
    int64_t page_stride;

    template <int kThreadsPerRow>
    struct RowCursor {
        T *base;
        T *current;
        int64_t token_stride;
        const int *page_table;
        int page_size;
        int64_t page_stride;
        int logical_page;
        int offset_in_page;
        int group_lane;
        int source_lane;
        unsigned group_mask;

        DEVICE_INLINE T *row_ptr(bool valid) const {
            return valid ? current : base;
        }

        DEVICE_INLINE void advance(int rows, bool next_valid) {
            if (!next_valid) {
                return;
            }
            offset_in_page += rows;
            if (offset_in_page < page_size) {
                current += int64_t(rows) * token_stride;
                return;
            }

            offset_in_page -= page_size;
            ++logical_page;
            unsigned long long raw_ptr =
                reinterpret_cast<unsigned long long>(current);
            if (group_lane == 0) {
                const int physical_page = page_table[logical_page];
                raw_ptr = reinterpret_cast<unsigned long long>(
                    base + int64_t(physical_page) * page_stride +
                    int64_t(offset_in_page) * token_stride);
            }
            raw_ptr = __shfl_sync(group_mask, raw_ptr, source_lane);
            current = reinterpret_cast<T *>(raw_ptr);
        }
    };

    template <int kThreadsPerRow>
    DEVICE_INLINE RowCursor<kThreadsPerRow> make_row_cursor(int logical_row,
                                                            bool valid) const {
        static_assert(kThreadsPerRow > 0 && kThreadsPerRow <= WARP_SIZE);
        static_assert(WARP_SIZE % kThreadsPerRow == 0);

        const int lane = threadIdx.x % WARP_SIZE;
        const int group_lane = lane % kThreadsPerRow;
        const int source_lane = lane - group_lane;
        constexpr unsigned kBaseMask = kThreadsPerRow == WARP_SIZE
                                           ? 0xffffffffu
                                           : (1u << kThreadsPerRow) - 1u;
        const unsigned group_mask = kBaseMask << source_lane;
        int logical_page = 0;
        int offset_in_page = 0;
        unsigned long long raw_ptr = reinterpret_cast<unsigned long long>(base);
        if (group_lane == 0 && valid) {
            logical_page = logical_row / page_size;
            offset_in_page = logical_row % page_size;
            const int physical_page = page_table[logical_page];
            raw_ptr = reinterpret_cast<unsigned long long>(
                base + int64_t(physical_page) * page_stride +
                int64_t(offset_in_page) * token_stride);
        }
        logical_page = __shfl_sync(group_mask, logical_page, source_lane);
        offset_in_page = __shfl_sync(group_mask, offset_in_page, source_lane);
        raw_ptr = __shfl_sync(group_mask, raw_ptr, source_lane);
        return RowCursor<kThreadsPerRow>{
            base,         reinterpret_cast<T *>(raw_ptr),
            token_stride, page_table,
            page_size,    page_stride,
            logical_page, offset_in_page,
            group_lane,   source_lane,
            group_mask,
        };
    }
};

}
