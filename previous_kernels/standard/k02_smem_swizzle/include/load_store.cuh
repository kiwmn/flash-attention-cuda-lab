#pragma once

// Copies preserve the shared-memory layout used by ldmatrix and MMA operand
// fragments.
#include "utils.h"
#include "ptx_functions.cuh"
#include "swizzling.cuh"

namespace flash_attn_lab::standard::k02 {

struct TileLayout {
    const int row_fragments;
    const int col_fragments;
};

struct TensorLDSTConfig {
    const TileLayout gmem_smem_layout;

    const TileLayout register_layout;

    const bool transposed;

    const int block_size;

    const int smem_cols;

    const int warp_ldst_rows;

    const bool compute_over_entire_block;
};

template <typename T>
struct GM2SMAsync {
    __device__ void operator()(T *gmem, T *smem) {
        cp_async<BYTES_PER_VEC4_ACCESS>(smem, gmem);
    }
};

template <typename T>
struct SM2GM {
    __device__ void operator()(T *gmem, T *smem) {
        LDST128BITS(gmem[0]) = LDST128BITS(smem[0]);
    }
};

template <typename op, TensorLDSTConfig Config, typename value_t,
          typename index_t = int64_t>
DEVICE_INLINE void copy_block_GSM(value_t *gmem, value_t *smem,
                                  index_t gmem_seq_stride, const int lane_id) {
    constexpr int n_row_iters = Config.gmem_smem_layout.row_fragments *
                                ROWS_PER_FRAGMENT / GSM_LDST_ROWS_PER_ITER;

    constexpr int col_fragments_per_iter = WARP_SIZE / GSM_LDST_ROWS_PER_ITER;

    constexpr int col_fragments_per_row = Config.smem_cols / COLS_PER_FRAGMENT;

    const int thread_row = lane_id / col_fragments_per_iter;

    const int thread_col_fragment = lane_id % col_fragments_per_iter;

#pragma unroll
    for (int i = 0; i < n_row_iters; ++i) {
        const int cur_row = i * GSM_LDST_ROWS_PER_ITER + thread_row;
#pragma unroll
        for (int j = 0; j < col_fragments_per_row;
             j += col_fragments_per_iter) {
            const int col_fragment = j + thread_col_fragment;
            const int col_fragment_swizzled =
                swizzled_col_fragment<col_fragments_per_row>(cur_row,
                                                             col_fragment);
            op{}(&gmem[cur_row * gmem_seq_stride +
                       col_fragment * COLS_PER_FRAGMENT],
                 &smem[cur_row * Config.smem_cols +
                       col_fragment_swizzled * COLS_PER_FRAGMENT]);
        }
    }
}

template <TensorLDSTConfig Config, typename value_t>
DEVICE_INLINE void
copy_warp_fragment_SM2RF(uint32_t (&regs)[Config.register_layout.row_fragments]
                                         [Config.register_layout.col_fragments],
                         value_t *smem, const int lane_id) {
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = row_fragments_per_iter * ROWS_PER_FRAGMENT;

    constexpr int col_fragments = Config.smem_cols / ELEMS_PER_VEC4_ACCESS;
    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;

    const int thread_row = lane_id % rows_per_iter;
    const int thread_col_fragment = lane_id / rows_per_iter;

#pragma unroll
    for (int i = 0; i < Config.register_layout.row_fragments;
         i += row_fragments_per_iter) {
        const int cur_row = i * ROWS_PER_FRAGMENT + thread_row;
#pragma unroll
        for (int j = 0; j < col_fragments; j += col_fragments_per_iter) {
            const int cur_col_fragment = j + thread_col_fragment;
            const int cur_col_fragment_swizzled =
                swizzled_col_fragment<col_fragments>(cur_row, cur_col_fragment);
            ldmatrix_x4(
                regs[i][j], regs[i + 1][j], regs[i][j + 1], regs[i + 1][j + 1],
                &(smem[cur_row * Config.smem_cols +
                       cur_col_fragment_swizzled * ELEMS_PER_VEC4_ACCESS]));
        }
    }
}

template <TensorLDSTConfig Config, typename value_t>
DEVICE_INLINE void copy_warp_fragment_transposed_SM2RF(
    uint32_t (&regs)[Config.register_layout.row_fragments]
                    [Config.register_layout.col_fragments],
    value_t *smem, const int lane_id) {
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = row_fragments_per_iter * ROWS_PER_FRAGMENT;

    constexpr int col_fragments = Config.smem_cols / ELEMS_PER_VEC4_ACCESS;
    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;

    const int thread_row = lane_id % rows_per_iter;
    const int thread_col_fragment = lane_id / rows_per_iter;

#pragma unroll
    for (int i = 0; i < Config.register_layout.col_fragments;
         i += row_fragments_per_iter) {
        const int cur_row = i * ROWS_PER_FRAGMENT + thread_row;
#pragma unroll
        for (int j = 0; j < Config.register_layout.row_fragments;
             j += col_fragments_per_iter) {
            const int cur_col_fragment = j + thread_col_fragment;
            const int cur_col_fragment_swizzled =
                swizzled_col_fragment<col_fragments>(cur_row, cur_col_fragment);
            ldmatrix_x4_transpose(
                regs[j][i], regs[j][i + 1], regs[j + 1][i], regs[j + 1][i + 1],
                &(smem[cur_row * Config.smem_cols +
                       cur_col_fragment_swizzled * ELEMS_PER_VEC4_ACCESS]));
        }
    }
}

template <TensorLDSTConfig Config, typename value_t>
DEVICE_INLINE void
copy_warp_fragment_RF2SM(uint32_t (&regs)[Config.register_layout.row_fragments]
                                         [Config.register_layout.col_fragments],
                         value_t *smem, const int lane_id) {
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT;
    constexpr int elems_per_store = 2;

    const int thread_row = lane_id / 4;
    const int thread_inner_col = (lane_id % 4) * elems_per_store;

#pragma unroll
    for (int i = 0; i < Config.register_layout.row_fragments; ++i) {
        const int cur_row = thread_row + i * rows_per_iter;

#pragma unroll
        for (int j = 0; j < Config.register_layout.col_fragments; ++j) {
            const int col_fragment_swizzled =
                swizzled_col_fragment<Config.smem_cols / ELEMS_PER_VEC4_ACCESS>(
                    cur_row, j);
            reinterpret_cast<uint32_t *>(
                &smem[cur_row * Config.smem_cols +
                      col_fragment_swizzled * ELEMS_PER_VEC4_ACCESS +
                      thread_inner_col])[0] = regs[i][j];
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
