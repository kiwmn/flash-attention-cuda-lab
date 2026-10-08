#pragma once

#include <map>

#include "forward_kernel.cuh"
#include "kernel_traits.cuh"
#include "static_kernel_configuration.cuh"

namespace flash_attn_lab::standard::k03 {

using forward_kernel_fn = void (*)(const ForwardKernelArgs);

inline constexpr ForwardKernelConfig kConfig{torch::kFloat16, 128, 64, 64, 4};
inline constexpr ForwardKernelConfig kConfigD64{torch::kFloat16, 64, 64, 64, 4};

std::map<ForwardKernelConfig, forward_kernel_fn> forward_kernels = {
    {kConfig,
     &flash_attention_forward_kernel<StaticForwardKernelConfig<kConfig>>},
    {kConfigD64,
     &flash_attention_forward_kernel<StaticForwardKernelConfig<kConfigD64>>},
};

}
