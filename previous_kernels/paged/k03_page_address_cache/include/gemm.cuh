#pragma once

// Warp-level Tensor Core products accumulate into FP32 output fragments.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

#include "utils.h"
#include "layout.cuh"
#include "ptx_functions.cuh"

namespace flash_attn_lab::paged::k03 {

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define MMA_M_FRAGMENTS_PER_ITER 2
#define MMA_N_FRAGMENTS_PER_ITER 1
#define MMA_K_FRAGMENTS_PER_ITER 2

template <typename _A_t, typename _B_t, typename _C_t, int tiles,
          typename value_t_>
struct GEMM {
    using A_t = _A_t;
    using B_t = _B_t;
    using C_t = _C_t;
    using value_t = value_t_;

    static constexpr int Tiles = tiles;

    static constexpr bool DoubleBufferA =
        !A_t::load_entire_block_into_rf && A_t::rmem_tile_buffer_size > 1;
    static constexpr bool DoubleBufferB =
        !B_t::load_entire_block_into_rf && B_t::rmem_tile_buffer_size > 1;
    static constexpr bool DoubleBuffer = DoubleBufferA || DoubleBufferB;
};

template <typename value_t, typename A_t, typename B_t, typename C_t>
DEVICE_INLINE constexpr void warp_fragment_mma_f32_accum(A_t &A, B_t &B, C_t &C,
                                                         const int &tile) {
    static_assert(is_supported_mma_input_type<typename A_t::value_t>(),
                  "A must be a half or bfloat16 tensor");
    static_assert(is_supported_mma_input_type<typename B_t::value_t>(),
                  "B must be a half or bfloat16 tensor");
    static_assert(std::is_same_v<typename C_t::value_t, float>,
                  "C must be a float tensor");
    static_assert(A_t::Shape::tiles() == B_t::Shape::tiles(),
                  "A and B must have the same number of tiles");
    static_assert(A_t::Shape::cols() == 1 && B_t::Shape::cols() == 1,
                  "A and B must have a tile size of 1 op tile");
    static_assert(A_t::Shape::rows() == C_t::Shape::rows(),
                  "A and C must have the same M shape");

    static_assert(B_t::Shape::rows() / 2 ==
                      C_t::Shape::cols() / THR_COLS_PER_ACCUM_FRAGMENT,
                  "B and C must have the same N shape");
    auto A_uint = A.view();
    auto B_uint = B.view();
    auto C_view = C.view();
    constexpr int M = decltype(A_uint)::Shape::rows();
    constexpr int N = decltype(B_uint)::Shape::rows();

#pragma unroll
    for (int n = 0; n < N; ++n) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
            int ms = (n & 1) ? M - m - 1 : m;
            mma_m16n8k16_f32_accum<value_t>(
                C_view(ms, n, 0, 0, 0), C_view(ms, n, 0, 0, 1),
                C_view(ms, n, 0, 1, 0), C_view(ms, n, 0, 1, 1),
                A_uint(ms, 0, tile, 0, 0), A_uint(ms, 0, tile, 1, 0),
                A_uint(ms, 0, tile, 0, 1), A_uint(ms, 0, tile, 1, 1),
                B_uint(n, 0, tile, 0, 0), B_uint(n, 0, tile, 0, 1),
                C_view(ms, n, 0, 0, 0), C_view(ms, n, 0, 0, 1),
                C_view(ms, n, 0, 1, 0), C_view(ms, n, 0, 1, 1));
        }
    }
}

template <typename GEMM>
DEVICE_INLINE constexpr void
matmul(typename GEMM::A_t &A, typename GEMM::B_t &B, typename GEMM::C_t &C) {
    using A_t = typename GEMM::A_t;
    using B_t = typename GEMM::B_t;
    using value_t = typename GEMM::value_t;

    if constexpr (GEMM::DoubleBuffer) {
        if constexpr (GEMM::DoubleBufferA) {
            A.copy_SM2RF(0);
        }
        if constexpr (GEMM::DoubleBufferB) {
            B.copy_SM2RF(0);
        }
    }

#pragma unroll
    for (int tile = 0; tile < GEMM::Tiles; ++tile) {
        if constexpr (!A_t::load_entire_block_into_rf ||
                      !B_t::load_entire_block_into_rf) {
            int load_tile = tile + (GEMM::DoubleBuffer ? 1 : 0);
            if (load_tile < GEMM::Tiles) {
                if constexpr (!A_t::load_entire_block_into_rf) {
                    A.copy_SM2RF(load_tile);
                }
                if constexpr (!B_t::load_entire_block_into_rf) {
                    B.copy_SM2RF(load_tile);
                }
            }
        }

        warp_fragment_mma_f32_accum<value_t>(A, B, C, tile);
    }
}

}
