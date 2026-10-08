#pragma once

// Warp-level Tensor Core products accumulate into FP32 output fragments.
#include "utils.h"
#include "ptx_functions.cuh"

namespace flash_attn_lab::standard::k01 {

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define MMA_M_FRAGMENTS_PER_ITER 2
#define MMA_N_FRAGMENTS_PER_ITER 1
#define MMA_K_FRAGMENTS_PER_ITER 2

template <typename _A_t, typename _B_t, typename _C_t, typename value_t_>
struct GEMM {
    using A_t = _A_t;
    using B_t = _B_t;
    using C_t = _C_t;
    using value_t = value_t_;
};

template <typename value_t, const int M_fragments, const int N_fragments,
          const int K_fragments_A, const int K_fragments_B,
          typename accum_t = float>
DEVICE_INLINE void warp_fragment_mma_f32_accum(
    uint32_t (&regs_A)[M_fragments][K_fragments_A],
    uint32_t (&regs_B)[N_fragments][K_fragments_B],
    accum_t (
        &regs_C)[M_fragments][N_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT]) {
    static_assert(K_fragments_A == K_fragments_B);
#pragma unroll
    for (int k = 0; k < K_fragments_A; k += MMA_K_FRAGMENTS_PER_ITER) {
#pragma unroll
        for (int m = 0; m < M_fragments; m += MMA_M_FRAGMENTS_PER_ITER) {
#pragma unroll
            for (int n = 0; n < N_fragments; n += MMA_N_FRAGMENTS_PER_ITER) {
                mma_m16n8k16_f32_accum<value_t>(
                    regs_C[m][n * 2], regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2], regs_C[m + 1][n * 2 + 1],
                    regs_A[m][k], regs_A[m + 1][k], regs_A[m][k + 1],
                    regs_A[m + 1][k + 1], regs_B[n][k], regs_B[n][k + 1],
                    regs_C[m][n * 2], regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2], regs_C[m + 1][n * 2 + 1]);
            }
        }
    }
}

template <typename GEMM>
DEVICE_INLINE void matmul(typename GEMM::A_t &A, typename GEMM::B_t &B,
                          typename GEMM::C_t &C) {
    using value_t = typename GEMM::value_t;
    warp_fragment_mma_f32_accum<value_t>(A.data(), B.data(), C.data());
}

}
