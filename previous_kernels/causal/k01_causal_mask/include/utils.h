#pragma once

#include <torch/extension.h>
#include <torch/types.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <type_traits>

namespace flash_attn_lab::causal::k01 {

#define DEVICE_INLINE __forceinline__ __device__

#define WARP_SIZE 32
#define SHFL_ENTIRE_WARP_MASK 0xffffffff

#define B16_BYTES 2
#define BYTES_PER_VEC4_ACCESS 16
#define ELEMS_PER_VEC4_ACCESS (BYTES_PER_VEC4_ACCESS / B16_BYTES)

#define MMA_A_REGS_PER_ROW 2
#define MMA_A_REGS_PER_COL 2
#define MMA_B_REGS_PER_ROW 2
#define MMA_B_REGS_PER_COL 1
#define MMA_C_REGS_PER_ROW 1
#define MMA_C_REGS_PER_COL 2

#define THR_COLS_PER_ACCUM_FRAGMENT 2

#define LDMATRIX_MAT_SIZE 8
#define ROWS_PER_FRAGMENT LDMATRIX_MAT_SIZE
#define COLS_PER_FRAGMENT LDMATRIX_MAT_SIZE

#define N_BUFFER_STAGES 2

#define GSMEM_THR_PER_ROW 8

struct alignas(16) uint128_t {
    uint64_t low;
    uint64_t high;
};

template <typename value_t>
constexpr bool is_supported_mma_input_type() {
    return std::is_same_v<value_t, half> ||
           std::is_same_v<value_t, nv_bfloat16>;
}

template <typename value_t>
constexpr bool is_supported_mma_output_type() {
    return std::is_same_v<value_t, float>;
}

template <typename value_t>
constexpr auto value_storage_type() {
    if constexpr (is_supported_mma_input_type<value_t>()) {
        return uint32_t{};
    } else if constexpr (is_supported_mma_output_type<value_t>()) {
        return float{};
    }
}

template <typename value_t>
constexpr auto value2_storage_type() {
    if constexpr (std::is_same_v<value_t, half>) {
        return half2{};
    } else if constexpr (std::is_same_v<value_t, nv_bfloat16>) {
        return nv_bfloat162{};
    } else if constexpr (std::is_same_v<value_t, float>) {
        return float2{};
    }
}

#define CHECK_CUDA(x)                                                          \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

#ifndef CUDA_CHECK_AND_EXIT
#define CUDA_CHECK_AND_EXIT(error)                                             \
    {                                                                          \
        auto status = static_cast<cudaError_t>(error);                         \
        if (status != cudaSuccess) {                                           \
            std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":"  \
                      << __LINE__ << std::endl;                                \
            std::exit(status);                                                 \
        }                                                                      \
    }
#endif

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

DEVICE_INLINE bool is_cta_leader() { return threadIdx.x == 0; }

inline int cuda_device_num_sms(int device) {
    int sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    return sms;
}

inline int cuda_device_max_smem_bytes(int device) {
    int max_smem;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           device);
    return max_smem;
}

inline int cuda_device_compute_capability(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.major * 10 + prop.minor;
}

constexpr int constexpr_min(int a, int b) { return (a < b) ? a : b; }

constexpr int constexpr_max(int a, int b) { return (a > b) ? a : b; }

constexpr int constexpr_log2_floor(int n) { return std::__bit_width(n) - 1; }

constexpr int binary_to_pm1(int b) { return 2 * b - 1; }

}
