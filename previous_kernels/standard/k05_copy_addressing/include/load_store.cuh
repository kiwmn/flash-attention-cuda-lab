#pragma once

// Copies preserve the shared-memory layout used by ldmatrix and MMA operand
// fragments.
#include "utils.h"
#include "ptx_functions.cuh"
#include "swizzling.cuh"

namespace flash_attn_lab::standard::k05 {

struct LDSTCommon {
    const bool swizzled;
    const bool async_copy;
};

struct TileLayout {
    const int row_fragments;
    const int col_fragments;
};

struct TensorLDSTConfig {
    const TileLayout gmem_smem_layout;

    const TileLayout register_layout;

    const LDSTCommon Common;

    const bool transposed;

    const int block_size;

    const int smem_cols;

    const int warp_ldst_rows;

    const bool compute_over_entire_block;
    const bool load_entire_block_into_rf;

    const int mma_load_stages;
};

template <TensorLDSTConfig ldst>
DEVICE_INLINE int get_smem_offset(const int row, const int col) {
    constexpr int col_fragments = ldst.smem_cols / COLS_PER_FRAGMENT;
    const int col_fragment = col / COLS_PER_FRAGMENT;
    const int col_in_fragment = col % COLS_PER_FRAGMENT;
    const int swizzled_col_fragment =
        get_smem_col_fragment<col_fragments, ldst.Common.swizzled>(
            row, col_fragment);
    return row * ldst.smem_cols + swizzled_col_fragment * COLS_PER_FRAGMENT +
           col_in_fragment;
}

template <typename T>
struct GM2SMAsync {
    DEVICE_INLINE constexpr void operator()(T *gmem, T *smem) {
        cp_async<BYTES_PER_VEC4_ACCESS>(smem, gmem);
    }
};

template <typename T>
struct SM2GM {
    DEVICE_INLINE constexpr void operator()(T *gmem, T *smem) {
        LDST128BITS(gmem[0]) = LDST128BITS(smem[0]);
    }
};

template <typename op, TensorLDSTConfig ldst, typename value_t,
          typename index_t = int64_t>
DEVICE_INLINE void copy_block_GSM(value_t *gmem, value_t *smem,
                                  index_t gmem_row_stride) {
    static_assert(ldst.warp_ldst_rows > 0,
                  "GMEM <-> SMEM tensors must have rows assigned per warp");
    static_assert(ldst.block_size % ldst.warp_ldst_rows == 0,
                  "block rows must divide evenly across warps");
    static_assert(sizeof(value_t) == 2,
                  "GMEM <-> SMEM vector copies require 16-bit elements");
    static_assert(ldst.gmem_smem_layout.row_fragments * ROWS_PER_FRAGMENT ==
                      ldst.warp_ldst_rows,
                  "GMEM <-> SMEM row layout must match warp_ldst_rows");
    static_assert(ldst.gmem_smem_layout.col_fragments * COLS_PER_FRAGMENT ==
                      ldst.smem_cols,
                  "GMEM <-> SMEM column layout must match smem_cols");

    constexpr int warp_num = ldst.block_size / ldst.warp_ldst_rows;
    constexpr int thread_num = warp_num * WARP_SIZE;
    constexpr int threads_per_row = 8;
    constexpr int rows_per_iter = thread_num / threads_per_row;
    constexpr int cols_per_iter = threads_per_row * ELEMS_PER_VEC4_ACCESS;

#pragma unroll
    for (int ir = 0; ir < ldst.block_size / rows_per_iter; ++ir) {
#pragma unroll
        for (int ic = 0; ic < ldst.smem_cols / cols_per_iter; ++ic) {
            const int row_offset = ir * rows_per_iter;
            const int col_offset = ic * cols_per_iter;
            op{}(&gmem[row_offset * gmem_row_stride + col_offset],
                 &smem[row_offset * ldst.smem_cols + col_offset]);
        }
    }
}

template <TensorLDSTConfig ldst>
DEVICE_INLINE int lane_to_thr_offset_s2rmem(const int lane_id) {
    const int thread_row = lane_id % 16;
    const int thread_col = (lane_id / 16) * COLS_PER_FRAGMENT;
    return get_smem_offset<ldst>(thread_row, thread_col);
}

template <TensorLDSTConfig ldst>
DEVICE_INLINE int lane_to_thr_offset_r2smem(const int lane_id) {
    constexpr int threads_per_row = 4;
    constexpr int elems_per_thread = 2;
    const int thread_row = lane_id / threads_per_row;
    const int thread_col = (lane_id % threads_per_row) * elems_per_thread;
    return get_smem_offset<ldst>(thread_row, thread_col);
}

template <TensorLDSTConfig ldst>
DEVICE_INLINE SwizzleStride
lane_to_thr_swizzle_stride_s2rmem(const int lane_id) {
    if constexpr (!ldst.Common.swizzled) {
        return SwizzleStride{64, 32, 16, 8};
    } else {
        const int base_swizzle_offset =
            lane_to_thr_offset_s2rmem<ldst>(lane_id);
        const int base_offset_cmp = ldst.smem_cols << 1;
        const int s1 =
            (base_swizzle_offset & (base_offset_cmp << 1)) == 0 ? 32 : -32;
        const int s2 = (base_swizzle_offset & base_offset_cmp) == 0 ? 16 : -16;
        return SwizzleStride{64, s1, s2, 8};
    }
}

template <TensorLDSTConfig ldst>
DEVICE_INLINE SwizzleStride
lane_to_thr_swizzle_stride_r2smem(const int lane_id) {
    if constexpr (!ldst.Common.swizzled) {
        return SwizzleStride{64, 32, 16, 8};
    } else {
        const int base_swizzle_offset =
            lane_to_thr_offset_r2smem<ldst>(lane_id);
        const int base_offset_cmp = ldst.smem_cols;
        const int s1 =
            (base_swizzle_offset & (base_offset_cmp << 2)) == 0 ? 32 : -32;
        const int s2 =
            (base_swizzle_offset & (base_offset_cmp << 1)) == 0 ? 16 : -16;
        const int s3 = (base_swizzle_offset & base_offset_cmp) == 0 ? 8 : -8;
        return SwizzleStride{64, s1, s2, s3};
    }
}

template <TensorLDSTConfig ldst, typename value_t>
DEVICE_INLINE void
copy_warp_fragment_SM2RF(uint32_t (&regs)[ldst.register_layout.row_fragments]
                                         [ldst.register_layout.col_fragments],
                         value_t *smem, const SwizzleStride &swizzle_stride,
                         const int col_fragment_offset = 0) {
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = row_fragments_per_iter * ROWS_PER_FRAGMENT;

    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;

#pragma unroll
    for (int i = 0; i < ldst.register_layout.row_fragments;
         i += row_fragments_per_iter) {
#pragma unroll
        for (int j = 0; j < ldst.register_layout.col_fragments;
             j += col_fragments_per_iter) {
            const int tile = (j + col_fragment_offset) / col_fragments_per_iter;
            const int smem_offset = i * ROWS_PER_FRAGMENT * ldst.smem_cols +
                                    swizzle_stride.offset_s2rmem(tile);

            ldmatrix_x4(regs[i][j], regs[i + 1][j], regs[i][j + 1],
                        regs[i + 1][j + 1], &smem[smem_offset]);
        }
    }
}

template <TensorLDSTConfig ldst, typename value_t>
DEVICE_INLINE void copy_warp_fragment_transposed_SM2RF(
    uint32_t (&regs)[ldst.register_layout.row_fragments]
                    [ldst.register_layout.col_fragments],
    value_t *smem, const SwizzleStride &swizzle_stride,
    const int row_fragment_offset = 0) {
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = row_fragments_per_iter * ROWS_PER_FRAGMENT;

    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;

#pragma unroll
    for (int i = 0; i < ldst.register_layout.col_fragments;
         i += row_fragments_per_iter) {
#pragma unroll
        for (int j = 0; j < ldst.register_layout.row_fragments;
             j += col_fragments_per_iter) {
            const int tile = j / col_fragments_per_iter;
            const int smem_offset =
                (i + row_fragment_offset) * ROWS_PER_FRAGMENT * ldst.smem_cols +
                swizzle_stride.offset_s2rmem(tile);
            ldmatrix_x4_transpose(regs[j][i], regs[j][i + 1], regs[j + 1][i],
                                  regs[j + 1][i + 1], &smem[smem_offset]);
        }
    }
}

template <TensorLDSTConfig ldst, typename value_t>
DEVICE_INLINE void
copy_warp_fragment_RF2SM(uint32_t (&regs)[ldst.register_layout.row_fragments]
                                         [ldst.register_layout.col_fragments],
                         value_t *smem, const SwizzleStride &swizzle_stride) {
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT;

#pragma unroll
    for (int i = 0; i < ldst.register_layout.row_fragments; ++i) {
#pragma unroll
        for (int j = 0; j < ldst.register_layout.col_fragments; ++j) {
            const int smem_offset = i * rows_per_iter * ldst.smem_cols +
                                    swizzle_stride.offset_r2smem(j);
            reinterpret_cast<uint32_t *>(&smem[smem_offset])[0] = regs[i][j];
        }
    }
}

template <typename value_t, int M_fragments, int N_fragments>
DEVICE_INLINE void
convert_to_16_bit_dtype(float (&src_float)[M_fragments][N_fragments * 2],
                        uint32_t (&dest_uint)[M_fragments][N_fragments]) {
    using value2_t =
        std::conditional_t<std::is_same_v<value_t, half>, half2, nv_bfloat162>;

    float2(&src)[M_fragments][N_fragments] =
        reinterpret_cast<float2(&)[M_fragments][N_fragments]>(src_float);
    value2_t(&dest)[M_fragments][N_fragments] =
        reinterpret_cast<value2_t(&)[M_fragments][N_fragments]>(dest_uint);
#pragma unroll
    for (int m = 0; m < M_fragments; ++m) {
#pragma unroll
        for (int n = 0; n < N_fragments; ++n) {
            if constexpr (std::is_same_v<value_t, half>) {
                dest[m][n] = __float22half2_rn(src[m][n]);
            } else {
                dest[m][n] = __float22bfloat162_rn(src[m][n]);
            }
        }
    }
}

}
