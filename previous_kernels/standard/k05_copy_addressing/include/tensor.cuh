#pragma once

// Tensor wrappers bind global/shared-memory tiles to per-thread register
// fragments.

#include "load_store.cuh"
#include "utils.h"

namespace flash_attn_lab::standard::k05 {

template <typename value_t, int stages, int row_fragments, int col_fragments>
struct RFMatrix;

template <TensorLDSTConfig ldst, typename value_t, typename index_t = int64_t>
struct MatrixLDST {
    using matrix_storage_t = RFMatrix<value_t, ldst.mma_load_stages,
                                      ldst.register_layout.row_fragments,
                                      ldst.register_layout.col_fragments>;

    using GM2SM_op = GM2SMAsync<value_t>;

    using SM2GM_op = SM2GM<value_t>;
    static constexpr int mma_load_stages = ldst.mma_load_stages;
    static constexpr bool load_entire_block_into_rf =
        ldst.load_entire_block_into_rf;
    static constexpr bool transposed = ldst.transposed;

    value_t *gmem_ptr;
    index_t gmem_seq_stride;

    value_t *smem_s2r_ptr;
    value_t *smem_r2s_ptr;

    value_t *smem_gsm_ptr;

    const int lane_id;
    SwizzleStride s2rmem_swizzle_stride;
    SwizzleStride r2smem_swizzle_stride;
    matrix_storage_t storage;

    DEVICE_INLINE MatrixLDST(value_t *gmem_block_ptr, index_t _gmem_seq_stride,
                             value_t *_smem_ptr)
        : lane_id(threadIdx.x % WARP_SIZE) {
        constexpr int threads_per_row = 8;
        const int tid = threadIdx.x;
        const int warp_idx = tid / WARP_SIZE;
        const index_t warp_seq = ldst.warp_ldst_rows * warp_idx;

        const int gsm_thread_row = tid / threads_per_row;
        const int gsm_thread_col =
            (tid % threads_per_row) * ELEMS_PER_VEC4_ACCESS;

        gmem_seq_stride = _gmem_seq_stride;
        gmem_ptr =
            gmem_block_ptr + gsm_thread_row * gmem_seq_stride + gsm_thread_col;

        smem_gsm_ptr =
            _smem_ptr + get_smem_offset<ldst>(gsm_thread_row, gsm_thread_col);

        value_t *smem_sr_base =
            _smem_ptr +
            (ldst.compute_over_entire_block ? 0 : warp_seq * ldst.smem_cols);

        s2rmem_swizzle_stride =
            lane_to_thr_swizzle_stride_s2rmem<ldst>(lane_id);
        r2smem_swizzle_stride =
            lane_to_thr_swizzle_stride_r2smem<ldst>(lane_id);
        smem_s2r_ptr = smem_sr_base + lane_to_thr_offset_s2rmem<ldst>(lane_id);
        smem_r2s_ptr = smem_sr_base + lane_to_thr_offset_r2smem<ldst>(lane_id);
    }

    DEVICE_INLINE void zero() { storage.zero(); }

    DEVICE_INLINE auto data(const int stage = 0) ->
        typename matrix_storage_t::storage_t (&)[matrix_storage_t::rows]
                                                [matrix_storage_t::cols] {
        return storage.data(stage);
    }

    DEVICE_INLINE void advance_gmem_block() {
        gmem_ptr += ldst.block_size * gmem_seq_stride;
    }

    DEVICE_INLINE void copy_GM2SM() {
        copy_block_GSM<GM2SM_op, ldst>(gmem_ptr, smem_gsm_ptr, gmem_seq_stride);
    }

    DEVICE_INLINE void copy_SM2GM() {
        copy_block_GSM<SM2GM_op, ldst>(gmem_ptr, smem_gsm_ptr, gmem_seq_stride);
    }

    DEVICE_INLINE void copy_SM2RF(int stage = 0, int tile_offset = 0) {
        if constexpr (!transposed) {
            copy_warp_fragment_SM2RF<ldst, value_t>(
                storage.data(stage), smem_s2r_ptr, s2rmem_swizzle_stride,
                tile_offset);
        } else {
            copy_warp_fragment_transposed_SM2RF<ldst, value_t>(
                storage.data(stage), smem_s2r_ptr, s2rmem_swizzle_stride,
                tile_offset);
        }
    }

    DEVICE_INLINE void copy_RF2SM() {
        copy_warp_fragment_RF2SM<ldst, value_t>(data(), smem_r2s_ptr,
                                                r2smem_swizzle_stride);
    }
};

template <typename value_t, int N>
struct RFVector {
    static constexpr int size = N;
    value_t regs[N];

    DEVICE_INLINE value_t &operator[](int idx) { return regs[idx]; }
};

template <typename value_t, int stages, int row_fragments, int col_fragments>
struct RFMatrix {
    using storage_t = std::conditional_t<sizeof(value_t) == 4, float, uint32_t>;
    static constexpr int regs_per_fragment = sizeof(value_t) / 2;
    static constexpr int rows = row_fragments;
    static constexpr int cols = col_fragments * regs_per_fragment;

    storage_t regs[stages][rows][cols];

    DEVICE_INLINE storage_t (&data(const int stage = 0)) [rows][cols] {
        return reinterpret_cast<storage_t(&)[rows][cols]>(regs[stage]);
    }

    DEVICE_INLINE void zero() {
#pragma unroll
        for (int s = 0; s < stages; ++s) {
#pragma unroll
            for (int j = 0; j < rows; ++j) {
#pragma unroll
                for (int k = 0; k < cols; ++k) {
                    regs[s][j][k] = 0;
                }
            }
        }
    }
};

}
