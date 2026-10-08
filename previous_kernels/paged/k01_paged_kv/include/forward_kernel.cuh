#pragma once

#include <cuda/std/limits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "utils.h"
#include "kernel_traits.cuh"
#include "gemm.cuh"
#include "mask.cuh"
#include "ptx_functions.cuh"
#include "softmax.cuh"
#include "static_kernel_configuration.cuh"
#include "tensor.cuh"

namespace flash_attn_lab::paged::k01 {

struct ForwardKernelArgs {
    using index_t = int64_t;

    void *__restrict__ q;
    void *__restrict__ k;
    void *__restrict__ v;
    void *__restrict__ o;

    struct Strides {
        index_t head;
        index_t token;
    };

    struct KVStrides {
        index_t page;
        index_t head;
        index_t token;
    };
    Strides q_stride, o_stride;
    KVStrides k_stride, v_stride;

    const int *query_start_loc;
    const int *seq_lens;
    int heads_per_kv;
    const int *block_table;
    index_t table_stride;
    int page_size;
};

struct RequestInfo {
    int q_start, q_len, kv_len;

    __device__ RequestInfo(const ForwardKernelArgs &args, int request)
        : q_start(args.query_start_loc[request]),
          q_len(args.query_start_loc[request + 1] - q_start),
          kv_len(args.seq_lens[request]) {}
};

template <bool is_first, bool kApplyCausalMask, bool optimized_softmax,
          bool prefetch_kv_tiles, typename Kernel, typename Q_t, typename K_t,
          typename V_t, typename S_accum_t, typename P_value_t,
          typename O_accum_t, typename row_statistics_t, typename S_QK_GEMM,
          typename O_PV_GEMM>
DEVICE_INLINE void process_kv_block(Q_t &Q, K_t &K, V_t &V, O_accum_t &O_accum,
                                    row_statistics_t &m, row_statistics_t &l,
                                    const float &softmax_scale,
                                    const bool &is_last_block, int q_block_idx,
                                    int kv_block, const RequestInfo &request) {
    S_accum_t S_accum;

    S_accum.zero();

    if constexpr (!prefetch_kv_tiles) {
        K.copy_GM2SM();
        K.advance_gmem_block();
        cp_async_commit();
    }
    // Wait for asynchronous copies before any thread consumes the shared tile.
    cp_async_wait<0>();

    __syncthreads();

    if constexpr (prefetch_kv_tiles) {
        V.copy_GM2SM();
        V.advance_gmem_block();
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
        if (!is_last_block) {
            K.copy_GM2SM();
            K.advance_gmem_block();
            cp_async_commit();
        }
    }

    auto S_accum_untiled = S_accum.view_with_op_tiling_removed();
    auto O_accum_no_op_tiling = O_accum.view_with_op_tiling_removed();
    if constexpr (kApplyCausalMask) {
        static_assert(Kernel::causal);
        const int q_begin = q_block_idx * Kernel::Br;
        const int kv_begin = kv_block * Kernel::Bc;
        apply_causal_mask<Kernel::N::QO_rows_per_warp>(
            S_accum_untiled, q_begin, kv_begin, request.q_len, request.kv_len,
            request.kv_len - request.q_len);
    }
    local_softmax<is_first, optimized_softmax>(
        S_accum_untiled, O_accum_no_op_tiling, m, l, softmax_scale);

    P_value_t P_b16;
    auto S_accum_view = S_accum.view();
    auto P_b16_view = P_b16.view();

    convert_to_16_bit_dtype(S_accum_view, P_b16_view);

    if constexpr (!prefetch_kv_tiles) {
        V.copy_GM2SM();
        V.advance_gmem_block();
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
    const RequestInfo request(args, batch_idx);
    const int q0 = q_block_idx * Kernel::Br;

    if (q0 >= request.q_len) {
        return;
    }
    const int q1 = min(q0 + Kernel::Br, request.q_len);
    const int context_len = request.kv_len - request.q_len;
    const int kv_end = min(request.kv_len, context_len + q1);
    const int n_kv_blocks = (kv_end + Kernel::Bc - 1) / Kernel::Bc;

    const int n_full_kv_blocks =
        min(request.kv_len, context_len + q0 + 1) / Kernel::Bc;

    const int kv_head = head_idx / args.heads_per_kv;
    auto offset = [=] __device__(ForwardKernelArgs::Strides stride,
                                 int token_start, int tensor_head) {
        return index_t(token_start) * stride.token +
               index_t(tensor_head) * stride.head;
    };
    value_t *gmem_Q = static_cast<value_t *>(args.q) +
                      offset(args.q_stride, request.q_start, head_idx);
    value_t *gmem_O = static_cast<value_t *>(args.o) +
                      offset(args.o_stride, request.q_start, head_idx);

    value_t *gmem_K =
        static_cast<value_t *>(args.k) + index_t(kv_head) * args.k_stride.head;
    value_t *gmem_V =
        static_cast<value_t *>(args.v) + index_t(kv_head) * args.v_stride.head;

    extern __shared__ __align__(16) char smem[];
    value_t *smem_Q = reinterpret_cast<value_t *>(smem);
    // Q storage is reused for O after the last QK product.
    value_t *smem_O = smem_Q;
    value_t *smem_K = &smem_Q[Kernel::Br * Kernel::head_dim];
    value_t *smem_V = &smem_K[Kernel::Bc * Kernel::head_dim];

    Q_t Q(gmem_Q, args.q_stride.token, smem_Q, request.q_len, q0);
    K_t K(gmem_K, args.k_stride.token, smem_K, request.kv_len);
    V_t V(gmem_V, args.v_stride.token, smem_V, request.kv_len);

    const int *table =
        args.block_table + index_t(batch_idx) * args.table_stride;
    K.set_page_table(table, args.page_size, args.k_stride.page);
    V.set_page_table(table, args.page_size, args.v_stride.page);

    Q.copy_GM2SM();
    cp_async_commit();
    if constexpr (Kernel::prefetch_kv_tiles) {
        K.copy_GM2SM();
        K.advance_gmem_block();
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

    if (n_full_kv_blocks > 0) {
        process_kv_block<true, false, Kernel::optimized_softmax,
                         Kernel::prefetch_kv_tiles, Kernel, Q_t, K_t, V_t,
                         S_accum_t, P_value_t, O_accum_t, row_statistics_t,
                         S_QK_GEMM, O_PV_GEMM>(Q, K, V, O_accum, m, l,
                                               softmax_scale, n_kv_blocks == 1,
                                               q_block_idx, 0, request);
    } else {
        process_kv_block<true, Kernel::causal, Kernel::optimized_softmax,
                         Kernel::prefetch_kv_tiles, Kernel, Q_t, K_t, V_t,
                         S_accum_t, P_value_t, O_accum_t, row_statistics_t,
                         S_QK_GEMM, O_PV_GEMM>(Q, K, V, O_accum, m, l,
                                               softmax_scale, n_kv_blocks == 1,
                                               q_block_idx, 0, request);
    }

    for (int block = 1; block < n_full_kv_blocks; ++block) {
        process_kv_block<false, false, Kernel::optimized_softmax,
                         Kernel::prefetch_kv_tiles, Kernel, Q_t, K_t, V_t,
                         S_accum_t, P_value_t, O_accum_t, row_statistics_t,
                         S_QK_GEMM, O_PV_GEMM>(
            Q, K, V, O_accum, m, l, softmax_scale, block == n_kv_blocks - 1,
            q_block_idx, block, request);
    }

    for (int block = max(1, n_full_kv_blocks); block < n_kv_blocks; ++block) {
        process_kv_block<false, true, Kernel::optimized_softmax,
                         Kernel::prefetch_kv_tiles, Kernel, Q_t, K_t, V_t,
                         S_accum_t, P_value_t, O_accum_t, row_statistics_t,
                         S_QK_GEMM, O_PV_GEMM>(
            Q, K, V, O_accum, m, l, softmax_scale, block == n_kv_blocks - 1,
            q_block_idx, block, request);
    }

    final_softmax_normalization(O_accum_no_op_tiling, l);

    O_value_t O_b16(gmem_O, args.o_stride.token, smem_O, request.q_len, q0);
    auto O_b16_view = O_b16.view();
    convert_to_16_bit_dtype(O_accum_no_op_tiling, O_b16_view);

    O_b16.copy_RF2SM();

    __syncthreads();

    O_b16.copy_SM2GM();
}

}
