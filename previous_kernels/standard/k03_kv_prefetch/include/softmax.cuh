#pragma once

// m stores each row maximum; l and O_accum hold partial exponential sums and
// weighted values. Four lanes share a row. Row maxima are reduced per tile; l
// is reduced before final normalization.

#include "utils.h"

namespace flash_attn_lab::standard::k03 {

template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
DEVICE_INLINE void
scale_S_accum(accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
              const accum_t &softmax_scale) {
#pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
#pragma unroll
        for (int k = 0; k < KV_accum_fragments; ++k) {
            S_accum[q][k] *= softmax_scale;
        }
    }
}

template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
DEVICE_INLINE void
calc_row_max(accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
             accum_t (&m_next)[QO_fragments], accum_t (&m_cur)[QO_fragments]) {
#pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        m_next[q] = m_cur[q];

#pragma unroll
        for (int k = 0; k < KV_accum_fragments; ++k) {
            m_next[q] = max(m_next[q], S_accum[q][k]);
        }

        m_next[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 2),
                        m_next[q]);
        m_next[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 1),
                        m_next[q]);
    }
}

template <int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
DEVICE_INLINE void
scale_l_O(accum_t (&m_next)[QO_fragments], accum_t (&m_cur)[QO_fragments],
          accum_t (&l)[QO_fragments],
          accum_t (&O_accum)[QO_fragments][d_head_accum_fragments]) {
#pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        const accum_t scale = expf(m_cur[q] - m_next[q]);
        m_cur[q] = m_next[q];
        l[q] *= scale;
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            O_accum[q][d_head] *= scale;
        }
    }
}

template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
DEVICE_INLINE void
exponentiate_tensor(accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
                    accum_t (&m)[QO_fragments]) {
#pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
#pragma unroll
        for (int k = 0; k < KV_accum_fragments; ++k) {
            S_accum[q][k] = expf(S_accum[q][k] - m[q]);
        }
    }
}

template <int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
DEVICE_INLINE void
update_row_exp_sum(accum_t (&P_accum)[QO_fragments][d_head_accum_fragments],
                   accum_t (&l)[QO_fragments]) {
#pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
#pragma unroll
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            l[q] += P_accum[q][d_head];
        }
    }
}

template <int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
DEVICE_INLINE void final_softmax_normalization(
    accum_t (&O_accum)[QO_fragments][d_head_accum_fragments],
    accum_t (&l)[QO_fragments]) {
#pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 2);
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 1);
    }

#pragma unroll
    for (int q = 0; q < QO_fragments; ++q) {
#pragma unroll
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            O_accum[q][d_head] /= l[q];
        }
    }
}

}
