#pragma once

// Tensor wrappers bind global/shared-memory tiles to per-thread register
// fragments.

#include "load_store.cuh"
#include "utils.h"

namespace flash_attn_lab::standard::k01 {

template <typename value_t, int row_fragments, int col_fragments>
struct RFMatrix;

template <TensorLDSTConfig ldst, typename value_t, typename index_t = int64_t>
struct MatrixLDST {
    using matrix_storage_t =
        RFMatrix<value_t, ldst.register_layout.row_fragments,
                 ldst.register_layout.col_fragments>;
    using GM2SM_op = GM2SMAsync<value_t>;

    using SM2GM_op = SM2GM<value_t>;
    static constexpr bool transposed = ldst.transposed;

    value_t *gmem_ptr;
    index_t gmem_seq_stride;

    value_t *smem_srm_ptr;

    value_t *smem_gsm_ptr;

    const int lane_id;
    matrix_storage_t storage;

    DEVICE_INLINE MatrixLDST(value_t *gmem_block_ptr, index_t _gmem_seq_stride,
                             value_t *_smem_ptr)
        : lane_id(threadIdx.x % WARP_SIZE) {
        const int warp_idx = threadIdx.x / WARP_SIZE;
        const index_t warp_seq = ldst.warp_ldst_rows * warp_idx;

        gmem_seq_stride = _gmem_seq_stride;
        gmem_ptr = gmem_block_ptr + warp_seq * gmem_seq_stride;

        smem_gsm_ptr = _smem_ptr + warp_seq * ldst.smem_cols;

        smem_srm_ptr =
            ldst.compute_over_entire_block ? _smem_ptr : smem_gsm_ptr;
    }

    DEVICE_INLINE void zero() { storage.zero(); }

    DEVICE_INLINE auto data() -> typename matrix_storage_t::storage_t (
            &)[matrix_storage_t::rows][matrix_storage_t::cols] {
        return storage.data();
    }

    DEVICE_INLINE void advance_gmem_block() {
        gmem_ptr += ldst.block_size * gmem_seq_stride;
    }

    DEVICE_INLINE void copy_GM2SM() {
        copy_block_GSM<GM2SM_op, ldst>(gmem_ptr, smem_gsm_ptr, gmem_seq_stride,
                                       lane_id);
    }

    DEVICE_INLINE void copy_SM2GM() {
        copy_block_GSM<SM2GM_op, ldst>(gmem_ptr, smem_gsm_ptr, gmem_seq_stride,
                                       lane_id);
    }

    DEVICE_INLINE void copy_SM2RF() {
        if constexpr (!transposed) {
            copy_warp_fragment_SM2RF<ldst, value_t>(storage.data(),
                                                    smem_srm_ptr, lane_id);
        } else {
            copy_warp_fragment_transposed_SM2RF<ldst, value_t>(
                storage.data(), smem_srm_ptr, lane_id);
        }
    }

    DEVICE_INLINE void copy_RF2SM() {
        copy_warp_fragment_RF2SM<ldst, value_t>(data(), smem_srm_ptr, lane_id);
    }
};

template <typename value_t, int row_fragments, int col_fragments>
struct RFMatrix {
    using storage_t = std::conditional_t<sizeof(value_t) == 4, float, uint32_t>;
    static constexpr int regs_per_fragment = sizeof(value_t) / 2;
    static constexpr int rows = row_fragments;
    static constexpr int cols = col_fragments * regs_per_fragment;

    storage_t regs[rows][cols];

    DEVICE_INLINE storage_t (&data()) [rows][cols] { return regs; }

    DEVICE_INLINE void zero() {
#pragma unroll
        for (int j = 0; j < rows; ++j) {
#pragma unroll
            for (int k = 0; k < cols; ++k) {
                regs[j][k] = 0;
            }
        }
    }
};

}
