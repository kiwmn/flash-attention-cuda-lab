#pragma once

#include <torch/extension.h>
#include <torch/types.h>
#include <cuda_runtime.h>

namespace flash_attn_lab::standard::k03 {

#define ROWS_PER_FRAGMENT 8
#define COLS_PER_FRAGMENT 8
#define GSM_LDST_ROWS_PER_ITER 4
#define BYTES_PER_VEC4_ACCESS 16
#define ELEMS_PER_VEC4_ACCESS 8

#define BANKS_PER_VEC4_ACCESS 8
#define ELEMS_PER_BANK 8

#define WARP_SIZE 32
#define SHFL_ENTIRE_WARP_MASK 0xffffffff

#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline

#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT2(value) (reinterpret_cast<float2 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

#define N_REGS_PER_F32_ACCUM_FRAGMENT 2
#define MMA_K_FRAGMENTS_PER_ITER 2
#define MMA_M_FRAGMENTS_PER_ITER 2
#define MMA_N_FRAGMENTS_PER_ITER 1

#define CHECK_CUDA(x)                                                          \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

int CEIL_DIV(int a, int b) { return (a - 1) / b + 1; }

inline int cuda_device_compute_capability(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.major * 10 + prop.minor;
}
}
