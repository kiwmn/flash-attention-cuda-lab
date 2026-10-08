#pragma once

// A query can read keys up to its right-aligned position within the request.

#include <cuda/std/limits>

#include "utils.h"

namespace flash_attn_lab::paged::k05 {

template <int kRowsPerWarp, typename ScoreView>
DEVICE_INLINE void apply_causal_mask(ScoreView &scores, int q_begin,
                                     int kv_begin, int q_len, int kv_len,
                                     int context_len) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int q_base = q_begin + warp * kRowsPerWarp + lane / 4;

    const int k_base = kv_begin + 2 * (lane % 4);
    constexpr float neg_inf = -cuda::std::numeric_limits<float>::infinity();

#pragma unroll
    for (int row = 0; row < ScoreView::Shape::rows(); ++row) {
        const int q_idx = q_base + 8 * row;
#pragma unroll
        for (int col = 0; col < ScoreView::Shape::cols(); ++col) {
            const int k_idx = k_base + 8 * (col / 2) + col % 2;

            if (q_idx >= q_len || k_idx >= kv_len ||
                k_idx > context_len + q_idx) {
                scores(row, col) = neg_inf;
            }
        }
    }
}

}
