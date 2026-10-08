#pragma once
#include <map>

#include "kernel_traits.cuh"
#include "forward_kernel.cuh"
#include "static_kernel_configuration.cuh"

namespace flash_attn_lab::paged::k04 {
using forward_kernel_fn = void (*)(const ForwardKernelArgs);
std::map<ForwardKernelConfig, forward_kernel_fn> forward_kernels = {
    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, false, 0,
                         0, 0, false, false, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false,
                             false, 0, 0, 0, false, false, true, 256}>>},
    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, true, 0,
                         0, 0, false, false, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, true,
                             0, 0, 0, false, false, true, 256}>>},
    {ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 0, false, true, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 0, false, true, true, 256}>>},
    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2,
                         2, 2, true, true, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             2, 2, 2, true, true, true, 256}>>},
    {ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 0, false, true, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true,
                             2, 2, 0, false, true, true, 256}>>},
    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, false, false,
                         0, 0, 0, false, false, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, false,
                             false, 0, 0, 0, false, false, true, 256}>>},
    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, false, true, 0,
                         0, 0, false, false, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, false,
                             true, 0, 0, 0, false, false, true, 256}>>},
    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 0, false, true, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 0, false, true, true, 256}>>},
    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 2,
                         2, 2, true, true, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             2, 2, 2, true, true, true, 256}>>},
    {ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 0, false, true, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                             true, 2, 2, 0, false, true, true, 256}>>},
    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, false, 0,
                         0, 0, false, false, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false,
                             false, 0, 0, 0, false, false, true, 16}>>},
    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, true, 0,
                         0, 0, false, false, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, true,
                             0, 0, 0, false, false, true, 16}>>},
    {ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 0, false, true, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 0, false, true, true, 16}>>},
    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2,
                         2, 2, true, true, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             2, 2, 2, true, true, true, 16}>>},
    {ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 0, false, true, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true,
                             2, 2, 0, false, true, true, 16}>>},
    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, false, false,
                         0, 0, 0, false, false, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, false,
                             false, 0, 0, 0, false, false, true, 16}>>},
    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, false, true, 0,
                         0, 0, false, false, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, false,
                             true, 0, 0, 0, false, false, true, 16}>>},
    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 0, false, true, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 0, false, true, true, 16}>>},
    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 2,
                         2, 2, true, true, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             2, 2, 2, true, true, true, 16}>>},
    {ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 0, false, true, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                             true, 2, 2, 0, false, true, true, 16}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, false, false, 0,
                         0, 0, false, false, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, false, false,
                             0, 0, 0, false, false, true, 256}>>},
    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, false, true, 0,
                         0, 0, false, false, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, false, true,
                             0, 0, 0, false, false, true, 256}>>},
    {ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true, 2, 2,
                         0, false, true, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 0, false, true, true, 256}>>},
    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 2, 2,
                         2, true, true, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             2, 2, 2, true, true, true, 256}>>},
    {ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 0, false, true, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 0, false, true, true, 256}>>},
    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, false, false, 0,
                         0, 0, false, false, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, false,
                             false, 0, 0, 0, false, false, true, 256}>>},
    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, false, true, 0,
                         0, 0, false, false, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, false, true,
                             0, 0, 0, false, false, true, 256}>>},
    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true, 2,
                         2, 0, false, true, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 0, false, true, true, 256}>>},
    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 2,
                         2, 2, true, true, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             2, 2, 2, true, true, true, 256}>>},
    {ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 0, false, true, true, 256},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 0, false, true, true, 256}>>},
    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, false, false, 0,
                         0, 0, false, false, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, false, false,
                             0, 0, 0, false, false, true, 16}>>},
    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, false, true, 0,
                         0, 0, false, false, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, false, true,
                             0, 0, 0, false, false, true, 16}>>},
    {ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true, 2, 2,
                         0, false, true, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 0, false, true, true, 16}>>},
    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 2, 2,
                         2, true, true, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             2, 2, 2, true, true, true, 16}>>},
    {ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 0, false, true, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 0, false, true, true, 16}>>},
    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, false, false, 0,
                         0, 0, false, false, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, false,
                             false, 0, 0, 0, false, false, true, 16}>>},
    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, false, true, 0,
                         0, 0, false, false, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, false, true,
                             0, 0, 0, false, false, true, 16}>>},
    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true, 2,
                         2, 0, false, true, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 0, false, true, true, 16}>>},
    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 2,
                         2, 2, true, true, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             2, 2, 2, true, true, true, 16}>>},
    {ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 0, false, true, true, 16},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 0, false, true, true, 16}>>},
};
}
