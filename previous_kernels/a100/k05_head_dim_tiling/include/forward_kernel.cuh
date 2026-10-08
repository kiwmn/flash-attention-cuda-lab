#pragma once

#include <cuda/std/limits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "utils.h"
#include "kernel_traits.cuh"
#include "gemm.cuh"
#include "ptx_functions.cuh"
#include "softmax.cuh"
#include "static_kernel_configuration.cuh"
#include "tensor.cuh"

namespace flash_attn_lab::a100::k05 {

struct ForwardKernelArgs {
    using index_t = int64_t;

    void *__restrict__ q;
    void *__restrict__ k;
    void *__restrict__ v;
    void *__restrict__ o;

    const index_t batch_stride;
    const index_t seq_stride;
    const index_t head_stride;

    const index_t seq_len;
    const index_t n_heads;

    const int n_q_blocks;
    const int n_kv_blocks;
};

template <bool is_first, bool optimized_softmax, bool prefetch_kv_tiles,
          typename Q_t, typename K_t, typename V_t, typename S_accum_t,
          typename P_value_t, typename O_accum_t, typename row_statistics_t,
          typename S_QK_GEMM, typename O_PV_GEMM>
DEVICE_INLINE void process_kv_block(Q_t &Q, K_t &K, V_t &V, O_accum_t &O_accum,
                                    row_statistics_t &m, row_statistics_t &l,
                                    const float &softmax_scale, int block) {
    S_accum_t S_accum;

    S_accum.zero();

    if constexpr (!prefetch_kv_tiles) {
        K.copy_GM2SM(block);
        cp_async_commit();
    }
    // Wait for asynchronous copies before any thread consumes the shared tile.
    cp_async_wait<0>();

    __syncthreads();

    if constexpr (prefetch_kv_tiles) {
        V.copy_GM2SM(block);
        cp_async_commit();
    }

    if constexpr (K_t::load_entire_block_into_rf) {
        K.copy_SM2RF_all_tiles();
    }

    matmul<S_QK_GEMM>(Q, K, S_accum);

    cp_async_wait<0>();

    __syncthreads();

    if constexpr (is_first) {
        O_accum.zero();
    }

    if constexpr (prefetch_kv_tiles) {
        if (block > 0) {
            K.copy_GM2SM(block - 1);
            cp_async_commit();
        }
    }

    auto S_accum_untiled = S_accum.view_with_op_tiling_removed();
    auto O_accum_no_op_tiling = O_accum.view_with_op_tiling_removed();
    local_softmax<is_first, optimized_softmax>(
        S_accum_untiled, O_accum_no_op_tiling, m, l, softmax_scale);

    P_value_t P_b16;
    auto S_accum_view = S_accum.view();
    auto P_b16_view = P_b16.view();

    convert_to_16_bit_dtype(S_accum_view, P_b16_view);

    if constexpr (!prefetch_kv_tiles) {
        V.copy_GM2SM(block);
        cp_async_commit();
        cp_async_wait<0>();
        __syncthreads();
    }

    if constexpr (V_t::load_entire_block_into_rf) {
        V.copy_SM2RF_all_tiles();
    }

    matmul<O_PV_GEMM>(P_b16, V, O_accum);
}

template <typename Kernel>
__global__ void
flash_attention_forward_kernel(__grid_constant__ const ForwardKernelArgs args) {
    using accum_t = float;
    using index_t = int64_t;
    using N = typename Kernel::N;

    using value_t = typename Kernel::value_t;
    using Q_t = typename Kernel::Q_t;
    using K_t = typename Kernel::K_t;
    using V_t = typename Kernel::V_t;
    using P_value_t = typename Kernel::P_value_t;
    using O_value_t = typename Kernel::O_value_t;
    using S_accum_t = typename Kernel::S_accum_t;
    using O_accum_t = typename Kernel::O_accum_t;
    using S_QK_GEMM = typename Kernel::S_QK_GEMM;
    using O_PV_GEMM = typename Kernel::O_PV_GEMM;
    using row_statistics_t = typename Kernel::row_statistics_t;

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;

    const index_t gmem_seq_stride = args.seq_stride;

    const index_t batch_head_offset =
        batch_idx * args.batch_stride + head_idx * args.head_stride;

    const index_t QO_gmem_block_offset =
        batch_head_offset +
        static_cast<index_t>(q_block_idx) * Kernel::Br * gmem_seq_stride;

    const index_t KV_gmem_block_offset = batch_head_offset;

    value_t *gmem_Q = &static_cast<value_t *>(args.q)[QO_gmem_block_offset];
    value_t *gmem_O = &static_cast<value_t *>(args.o)[QO_gmem_block_offset];
    value_t *gmem_K = &static_cast<value_t *>(args.k)[KV_gmem_block_offset];
    value_t *gmem_V = &static_cast<value_t *>(args.v)[KV_gmem_block_offset];

    extern __shared__ __align__(16) char smem[];
    value_t *smem_Q = reinterpret_cast<value_t *>(smem);
    // Q storage is reused for O after the last QK product.
    value_t *smem_O = smem_Q;
    value_t *smem_K = &smem_Q[Kernel::Br * Kernel::head_dim];
    value_t *smem_V = &smem_K[Kernel::Bc * Kernel::head_dim];

    Q_t Q(gmem_Q, gmem_seq_stride, smem_Q);
    K_t K(gmem_K, gmem_seq_stride, smem_K);
    V_t V(gmem_V, gmem_seq_stride, smem_V);

    const int last_kv_block = args.n_kv_blocks - 1;

    Q.copy_GM2SM(0);
    cp_async_commit();
    if constexpr (Kernel::prefetch_kv_tiles) {
        K.copy_GM2SM(last_kv_block);
        cp_async_commit();
    }

    const accum_t softmax_scale =
        rsqrt(static_cast<accum_t>(Kernel::head_dim)) * M_LOG2E;
    constexpr accum_t neg_inf = -cuda::std::numeric_limits<float>::infinity();

    row_statistics_t m;
    row_statistics_t l;

    if constexpr (!Kernel::optimized_softmax) {
        m.fill(neg_inf);
        l.fill(0.0);
    }

    if constexpr (Q_t::load_entire_block_into_rf) {
        if constexpr (Kernel::prefetch_kv_tiles) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }

        __syncthreads();
        Q.copy_SM2RF_all_tiles();
    }

    O_accum_t O_accum;
    auto O_accum_no_op_tiling = O_accum.view_with_op_tiling_removed();

    process_kv_block<true, Kernel::optimized_softmax, Kernel::prefetch_kv_tiles,
                     Q_t, K_t, V_t, S_accum_t, P_value_t, O_accum_t,
                     row_statistics_t, S_QK_GEMM, O_PV_GEMM>(
        Q, K, V, O_accum, m, l, softmax_scale, last_kv_block);

    for (int block = last_kv_block - 1; block >= 0; --block) {
        process_kv_block<false, Kernel::optimized_softmax,
                         Kernel::prefetch_kv_tiles, Q_t, K_t, V_t, S_accum_t,
                         P_value_t, O_accum_t, row_statistics_t, S_QK_GEMM,
                         O_PV_GEMM>(Q, K, V, O_accum, m, l, softmax_scale,
                                    block);
    }

    final_softmax_normalization(O_accum_no_op_tiling, l);

    O_value_t O_b16(gmem_O, gmem_seq_stride, smem_O);
    auto O_b16_view = O_b16.view();
    convert_to_16_bit_dtype(O_accum_no_op_tiling, O_b16_view);

    O_b16.copy_RF2SM();

    __syncthreads();

    O_b16.copy_SM2GM();
}

}
