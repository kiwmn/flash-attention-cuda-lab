#pragma once
#include <map>

#include "kernel_traits.cuh"

namespace flash_attn_lab::paged::k07 {
struct ForwardKernelArgs;
using forward_kernel_fn = void (*)(const ForwardKernelArgs);
using KernelMap = std::map<ForwardKernelConfig, forward_kernel_fn>;
void register_p16_fp16_br64_bc64_pre0(KernelMap &, KernelMap &);
void register_p16_fp16_br64_bc32_pre1(KernelMap &, KernelMap &);
void register_p16_fp16_br64_bc64_pre1(KernelMap &, KernelMap &);
void register_p16_fp16_br128_bc32_pre1(KernelMap &, KernelMap &);
void register_p16_fp16_br64_bc32_pre0(KernelMap &, KernelMap &);
void register_p16_fp16_br128_bc32_pre0(KernelMap &, KernelMap &);
void register_p16_fp16_br128_bc64_pre0(KernelMap &, KernelMap &);
void register_p16_fp16_br128_bc64_pre1(KernelMap &, KernelMap &);
void register_p16_fp16_br64_bc64_pre0_d64(KernelMap &, KernelMap &);
void register_p16_fp16_br64_bc32_pre1_d64(KernelMap &, KernelMap &);
void register_p16_fp16_br64_bc64_pre1_d64(KernelMap &, KernelMap &);
void register_p16_fp16_br128_bc32_pre1_d64(KernelMap &, KernelMap &);
void register_p16_fp16_br64_bc32_pre0_d64(KernelMap &, KernelMap &);
void register_p16_fp16_br128_bc32_pre0_d64(KernelMap &, KernelMap &);
void register_p16_fp16_br128_bc64_pre0_d64(KernelMap &, KernelMap &);
void register_p16_fp16_br128_bc64_pre1_d64(KernelMap &, KernelMap &);
void register_p16_bf16_br64_bc64_pre0(KernelMap &, KernelMap &);
void register_p16_bf16_br64_bc32_pre1(KernelMap &, KernelMap &);
void register_p16_bf16_br64_bc64_pre1(KernelMap &, KernelMap &);
void register_p16_bf16_br128_bc32_pre1(KernelMap &, KernelMap &);
void register_p16_bf16_br64_bc32_pre0(KernelMap &, KernelMap &);
void register_p16_bf16_br128_bc32_pre0(KernelMap &, KernelMap &);
void register_p16_bf16_br128_bc64_pre0(KernelMap &, KernelMap &);
void register_p16_bf16_br128_bc64_pre1(KernelMap &, KernelMap &);
void register_p16_bf16_br64_bc64_pre0_d64(KernelMap &, KernelMap &);
void register_p16_bf16_br64_bc32_pre1_d64(KernelMap &, KernelMap &);
void register_p16_bf16_br64_bc64_pre1_d64(KernelMap &, KernelMap &);
void register_p16_bf16_br128_bc32_pre1_d64(KernelMap &, KernelMap &);
void register_p16_bf16_br64_bc32_pre0_d64(KernelMap &, KernelMap &);
void register_p16_bf16_br128_bc32_pre0_d64(KernelMap &, KernelMap &);
void register_p16_bf16_br128_bc64_pre0_d64(KernelMap &, KernelMap &);
void register_p16_bf16_br128_bc64_pre1_d64(KernelMap &, KernelMap &);
void register_p256_fp16_br64_bc64_pre0(KernelMap &, KernelMap &);
void register_p256_fp16_br64_bc32_pre1(KernelMap &, KernelMap &);
void register_p256_fp16_br64_bc64_pre1(KernelMap &, KernelMap &);
void register_p256_fp16_br128_bc32_pre1(KernelMap &, KernelMap &);
void register_p256_fp16_br64_bc32_pre0(KernelMap &, KernelMap &);
void register_p256_fp16_br128_bc32_pre0(KernelMap &, KernelMap &);
void register_p256_fp16_br128_bc64_pre0(KernelMap &, KernelMap &);
void register_p256_fp16_br128_bc64_pre1(KernelMap &, KernelMap &);
void register_p256_fp16_br64_bc64_pre0_d64(KernelMap &, KernelMap &);
void register_p256_fp16_br64_bc32_pre1_d64(KernelMap &, KernelMap &);
void register_p256_fp16_br64_bc64_pre1_d64(KernelMap &, KernelMap &);
void register_p256_fp16_br128_bc32_pre1_d64(KernelMap &, KernelMap &);
void register_p256_fp16_br64_bc32_pre0_d64(KernelMap &, KernelMap &);
void register_p256_fp16_br128_bc32_pre0_d64(KernelMap &, KernelMap &);
void register_p256_fp16_br128_bc64_pre0_d64(KernelMap &, KernelMap &);
void register_p256_fp16_br128_bc64_pre1_d64(KernelMap &, KernelMap &);
void register_p256_bf16_br64_bc64_pre0(KernelMap &, KernelMap &);
void register_p256_bf16_br64_bc32_pre1(KernelMap &, KernelMap &);
void register_p256_bf16_br64_bc64_pre1(KernelMap &, KernelMap &);
void register_p256_bf16_br128_bc32_pre1(KernelMap &, KernelMap &);
void register_p256_bf16_br64_bc32_pre0(KernelMap &, KernelMap &);
void register_p256_bf16_br128_bc32_pre0(KernelMap &, KernelMap &);
void register_p256_bf16_br128_bc64_pre0(KernelMap &, KernelMap &);
void register_p256_bf16_br128_bc64_pre1(KernelMap &, KernelMap &);
void register_p256_bf16_br64_bc64_pre0_d64(KernelMap &, KernelMap &);
void register_p256_bf16_br64_bc32_pre1_d64(KernelMap &, KernelMap &);
void register_p256_bf16_br64_bc64_pre1_d64(KernelMap &, KernelMap &);
void register_p256_bf16_br128_bc32_pre1_d64(KernelMap &, KernelMap &);
void register_p256_bf16_br64_bc32_pre0_d64(KernelMap &, KernelMap &);
void register_p256_bf16_br128_bc32_pre0_d64(KernelMap &, KernelMap &);
void register_p256_bf16_br128_bc64_pre0_d64(KernelMap &, KernelMap &);
void register_p256_bf16_br128_bc64_pre1_d64(KernelMap &, KernelMap &);
KernelMap forward_kernels;
KernelMap single_page_kernels;
inline void register_forward_kernels() {
    register_p16_fp16_br64_bc64_pre0(forward_kernels, single_page_kernels);
    register_p16_fp16_br64_bc32_pre1(forward_kernels, single_page_kernels);
    register_p16_fp16_br64_bc64_pre1(forward_kernels, single_page_kernels);
    register_p16_fp16_br128_bc32_pre1(forward_kernels, single_page_kernels);
    register_p16_fp16_br64_bc32_pre0(forward_kernels, single_page_kernels);
    register_p16_fp16_br128_bc32_pre0(forward_kernels, single_page_kernels);
    register_p16_fp16_br128_bc64_pre0(forward_kernels, single_page_kernels);
    register_p16_fp16_br128_bc64_pre1(forward_kernels, single_page_kernels);
    register_p16_fp16_br64_bc64_pre0_d64(forward_kernels, single_page_kernels);
    register_p16_fp16_br64_bc32_pre1_d64(forward_kernels, single_page_kernels);
    register_p16_fp16_br64_bc64_pre1_d64(forward_kernels, single_page_kernels);
    register_p16_fp16_br128_bc32_pre1_d64(forward_kernels, single_page_kernels);
    register_p16_fp16_br64_bc32_pre0_d64(forward_kernels, single_page_kernels);
    register_p16_fp16_br128_bc32_pre0_d64(forward_kernels, single_page_kernels);
    register_p16_fp16_br128_bc64_pre0_d64(forward_kernels, single_page_kernels);
    register_p16_fp16_br128_bc64_pre1_d64(forward_kernels, single_page_kernels);
    register_p16_bf16_br64_bc64_pre0(forward_kernels, single_page_kernels);
    register_p16_bf16_br64_bc32_pre1(forward_kernels, single_page_kernels);
    register_p16_bf16_br64_bc64_pre1(forward_kernels, single_page_kernels);
    register_p16_bf16_br128_bc32_pre1(forward_kernels, single_page_kernels);
    register_p16_bf16_br64_bc32_pre0(forward_kernels, single_page_kernels);
    register_p16_bf16_br128_bc32_pre0(forward_kernels, single_page_kernels);
    register_p16_bf16_br128_bc64_pre0(forward_kernels, single_page_kernels);
    register_p16_bf16_br128_bc64_pre1(forward_kernels, single_page_kernels);
    register_p16_bf16_br64_bc64_pre0_d64(forward_kernels, single_page_kernels);
    register_p16_bf16_br64_bc32_pre1_d64(forward_kernels, single_page_kernels);
    register_p16_bf16_br64_bc64_pre1_d64(forward_kernels, single_page_kernels);
    register_p16_bf16_br128_bc32_pre1_d64(forward_kernels, single_page_kernels);
    register_p16_bf16_br64_bc32_pre0_d64(forward_kernels, single_page_kernels);
    register_p16_bf16_br128_bc32_pre0_d64(forward_kernels, single_page_kernels);
    register_p16_bf16_br128_bc64_pre0_d64(forward_kernels, single_page_kernels);
    register_p16_bf16_br128_bc64_pre1_d64(forward_kernels, single_page_kernels);
    register_p256_fp16_br64_bc64_pre0(forward_kernels, single_page_kernels);
    register_p256_fp16_br64_bc32_pre1(forward_kernels, single_page_kernels);
    register_p256_fp16_br64_bc64_pre1(forward_kernels, single_page_kernels);
    register_p256_fp16_br128_bc32_pre1(forward_kernels, single_page_kernels);
    register_p256_fp16_br64_bc32_pre0(forward_kernels, single_page_kernels);
    register_p256_fp16_br128_bc32_pre0(forward_kernels, single_page_kernels);
    register_p256_fp16_br128_bc64_pre0(forward_kernels, single_page_kernels);
    register_p256_fp16_br128_bc64_pre1(forward_kernels, single_page_kernels);
    register_p256_fp16_br64_bc64_pre0_d64(forward_kernels, single_page_kernels);
    register_p256_fp16_br64_bc32_pre1_d64(forward_kernels, single_page_kernels);
    register_p256_fp16_br64_bc64_pre1_d64(forward_kernels, single_page_kernels);
    register_p256_fp16_br128_bc32_pre1_d64(forward_kernels,
                                           single_page_kernels);
    register_p256_fp16_br64_bc32_pre0_d64(forward_kernels, single_page_kernels);
    register_p256_fp16_br128_bc32_pre0_d64(forward_kernels,
                                           single_page_kernels);
    register_p256_fp16_br128_bc64_pre0_d64(forward_kernels,
                                           single_page_kernels);
    register_p256_fp16_br128_bc64_pre1_d64(forward_kernels,
                                           single_page_kernels);
    register_p256_bf16_br64_bc64_pre0(forward_kernels, single_page_kernels);
    register_p256_bf16_br64_bc32_pre1(forward_kernels, single_page_kernels);
    register_p256_bf16_br64_bc64_pre1(forward_kernels, single_page_kernels);
    register_p256_bf16_br128_bc32_pre1(forward_kernels, single_page_kernels);
    register_p256_bf16_br64_bc32_pre0(forward_kernels, single_page_kernels);
    register_p256_bf16_br128_bc32_pre0(forward_kernels, single_page_kernels);
    register_p256_bf16_br128_bc64_pre0(forward_kernels, single_page_kernels);
    register_p256_bf16_br128_bc64_pre1(forward_kernels, single_page_kernels);
    register_p256_bf16_br64_bc64_pre0_d64(forward_kernels, single_page_kernels);
    register_p256_bf16_br64_bc32_pre1_d64(forward_kernels, single_page_kernels);
    register_p256_bf16_br64_bc64_pre1_d64(forward_kernels, single_page_kernels);
    register_p256_bf16_br128_bc32_pre1_d64(forward_kernels,
                                           single_page_kernels);
    register_p256_bf16_br64_bc32_pre0_d64(forward_kernels, single_page_kernels);
    register_p256_bf16_br128_bc32_pre0_d64(forward_kernels,
                                           single_page_kernels);
    register_p256_bf16_br128_bc64_pre0_d64(forward_kernels,
                                           single_page_kernels);
    register_p256_bf16_br128_bc64_pre1_d64(forward_kernels,
                                           single_page_kernels);
}
}
