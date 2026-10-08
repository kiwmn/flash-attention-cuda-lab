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

namespace flash_attn_lab::standard::k06 {

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

    constexpr int async = Kernel::async_copy;

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;

    const index_t gmem_seq_stride = args.seq_stride;

    const index_t batch_head_offset =
        batch_idx * args.batch_stride + head_idx * args.head_stride;

    const index_t QO_gmem_block_offset =
        batch_head_offset + q_block_idx * Kernel::Br * gmem_seq_stride;

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

    typename Kernel::O_accum_t O_accum;
    auto O_accum_no_op_tiling = O_accum.view().with_op_tiling_removed();
    using O_accum_no_op_tiling_shape =
        decltype(O_accum_no_op_tiling)::Layout::Shape;

    Q.copy_GM2SM();
    cp_async_commit<async>();
    if constexpr (Kernel::prefetch_kv_tiles) {
        K.copy_GM2SM();
        K.advance_gmem_block();
        cp_async_commit<async>();
    }

    O_accum.zero();

    const accum_t softmax_scale =
        rsqrt(static_cast<accum_t>(Kernel::head_dim)) * M_LOG2E;
    constexpr accum_t neg_inf = -cuda::std::numeric_limits<float>::infinity();

    ArrayAligned<N::QO_fragments_per_warp, accum_t> m;
    ArrayAligned<N::QO_fragments_per_warp, accum_t> l;

    m.fill(neg_inf);
    l.fill(0.0);

    if constexpr (Q_t::load_entire_block_into_rf) {
        if constexpr (Kernel::prefetch_kv_tiles) {
            cp_async_wait<1, async>();
        } else {
            cp_async_wait<0, async>();
        }

        __syncthreads();
        Q.copy_SM2RF_all_tiles();
    }

    for (int j = 0; j < args.n_kv_blocks; ++j) {
        if constexpr (!Kernel::prefetch_kv_tiles) {
            K.copy_GM2SM();
            K.advance_gmem_block();
            cp_async_commit<async>();
        }

        typename Kernel::S_accum_t S_accum;

        S_accum.zero();

        cp_async_wait<0, async>();

        __syncthreads();

        if constexpr (Kernel::prefetch_kv_tiles) {
            V.copy_GM2SM();
            V.advance_gmem_block();
            cp_async_commit<async>();
        }
        if constexpr (K_t::load_entire_block_into_rf) {
            K.copy_SM2RF_all_tiles();
        }

        matmul<Kernel::S_QK_GEMM>(Q, K, S_accum);

        cp_async_wait<0, async>();

        __syncthreads();

        if constexpr (Kernel::prefetch_kv_tiles) {
            if (j < args.n_kv_blocks - 1) {
                K.copy_GM2SM();
                K.advance_gmem_block();
                cp_async_commit<async>();
            }
        }

        auto S_accum_untiled = S_accum.view().with_op_tiling_removed();
        ArrayAligned<N::QO_fragments_per_warp, accum_t> m_next;
        calc_row_max(S_accum_untiled, m_next, m);
        scale_l_O_and_update_rowmax(m_next, m, l, O_accum_no_op_tiling,
                                    softmax_scale);
        exponentiate_tensor<Kernel::optimized_softmax>(S_accum_untiled, m,
                                                       softmax_scale);
        update_row_exp_sum(S_accum_untiled, l);

        typename Kernel::P_value_t P_b16;

        auto S_accum_view = S_accum.view();
        auto P_b16_view = P_b16.view();
        convert_to_16_bit_dtype(S_accum_view, P_b16_view);

        if constexpr (!Kernel::prefetch_kv_tiles) {
            V.copy_GM2SM();
            V.advance_gmem_block();
            cp_async_commit<async>();
            cp_async_wait<0, async>();
            __syncthreads();
        }

        if constexpr (V_t::load_entire_block_into_rf) {
            V.copy_SM2RF_all_tiles();
        }

        matmul<typename Kernel::O_PV_GEMM>(P_b16, V, O_accum);
    }

    final_softmax_normalization(O_accum_no_op_tiling, l);

    typename Kernel::O_value_t O_b16(gmem_O, gmem_seq_stride, smem_O);
    auto O_b16_view = O_b16.view();
    convert_to_16_bit_dtype(O_accum_no_op_tiling, O_b16_view);

    O_b16.copy_RF2SM();

    __syncthreads();

    O_b16.copy_SM2GM();
}

}
