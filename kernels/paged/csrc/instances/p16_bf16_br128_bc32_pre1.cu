#include "kernel_traits.cuh"
#include "forward_kernel.cuh"
#include "static_kernel_configuration.cuh"
#include <map>

namespace flash_attn_lab::paged::k07 {
using forward_kernel_fn = void (*)(const ForwardKernelArgs);
void register_p16_bf16_br128_bc32_pre1(
    std::map<ForwardKernelConfig, forward_kernel_fn> &normal,
    std::map<ForwardKernelConfig, forward_kernel_fn> &single) {
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            2, 2, 0, false, true, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 2, 2, 0, false, true, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            0, 0, 0, false, false, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 0, 0, 0, false, false, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            0, 0, 0, false, true, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 0, 0, 0, false, true, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            0, 0, 2, false, false, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 0, 0, 2, false, false, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            0, 0, 2, false, true, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 0, 0, 2, false, true, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            0, 0, 2, true, false, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 0, 0, 2, true, false, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            0, 0, 2, true, true, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 0, 0, 2, true, true, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            0, 2, 0, false, false, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 0, 2, 0, false, false, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            0, 2, 0, false, true, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 0, 2, 0, false, true, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            0, 2, 0, true, false, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 0, 2, 0, true, false, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            0, 2, 0, true, true, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 0, 2, 0, true, true, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            0, 2, 2, false, false, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 0, 2, 2, false, false, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            0, 2, 2, false, true, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 0, 2, 2, false, true, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            0, 2, 2, true, false, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 0, 2, 2, true, false, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            0, 2, 2, true, true, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 0, 2, 2, true, true, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            2, 2, 0, false, false, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 2, 2, 0, false, false, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            2, 2, 0, true, false, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 2, 2, 0, true, false, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            2, 2, 0, true, true, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 2, 2, 0, true, true, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            2, 2, 2, false, false, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 2, 2, 2, false, false, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            2, 2, 2, false, true, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 2, 2, 2, false, true, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            2, 2, 2, true, false, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 2, 2, 2, true, false, true, 16}>>);
    normal.emplace(
        ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true,
                            2, 2, 2, true, true, true, 16},
        &flash_attention_forward_kernel<StaticForwardKernelConfig<
            ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                                true, 2, 2, 2, true, true, true, 16}>>);
}
}
