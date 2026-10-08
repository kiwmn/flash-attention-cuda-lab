#pragma once

#include <map>

#include "kernel_traits.cuh"
#include "forward_kernel.cuh"
#include "static_kernel_configuration.cuh"

namespace flash_attn_lab::a100::k03 {

using forward_kernel_fn = void (*)(const ForwardKernelArgs);

std::map<ForwardKernelConfig, forward_kernel_fn> forward_kernels = {

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 0, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 0, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 0, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 0, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 2, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, false, 0,
                         0, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false,
                             false, 0, 0, 0, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, true, 0,
                         0, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, true,
                             0, 0, 0, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, false, 0,
                         2, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, false,
                             0, 2, 0, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0,
                         0, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             0, 0, 0, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0,
                         0, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             0, 0, 0, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0,
                         0, 0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             0, 0, 0, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0,
                         0, 0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             0, 0, 0, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0,
                         0, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             0, 0, 2, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0,
                         0, 2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             0, 0, 2, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0,
                         0, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             0, 0, 2, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0,
                         0, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             0, 0, 2, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0,
                         2, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             0, 2, 0, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0,
                         2, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             0, 2, 0, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0,
                         2, 0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             0, 2, 0, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0,
                         2, 0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             0, 2, 0, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0,
                         2, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             0, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0,
                         2, 2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             0, 2, 2, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0,
                         2, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             0, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0,
                         2, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             0, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2,
                         2, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             2, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2,
                         2, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             2, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2,
                         2, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true,
                             2, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true,
                             2, 2, 0, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true,
                             2, 2, 0, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true,
                             2, 2, 0, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true,
                             2, 2, 0, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true,
                             2, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true,
                             2, 2, 2, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true,
                             2, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true,
                             2, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2,
                         2, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true,
                             2, 2, 0, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2,
                         2, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true,
                             2, 2, 0, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2,
                         2, 0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true,
                             2, 2, 0, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2,
                         2, 0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true,
                             2, 2, 0, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2,
                         2, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true,
                             2, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2,
                         2, 2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true,
                             2, 2, 2, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2,
                         2, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true,
                             2, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2,
                         2, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true,
                             2, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 0, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 0, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 0, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 0, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 2, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2,
                         2, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true,
                             2, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, false, false,
                         0, 0, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, false,
                             false, 0, 0, 0, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, false, true, 0,
                         0, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, false,
                             true, 0, 0, 0, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0,
                         0, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             0, 0, 0, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0,
                         0, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             0, 0, 0, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0,
                         0, 0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             0, 0, 0, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0,
                         0, 0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             0, 0, 0, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0,
                         0, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             0, 0, 2, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0,
                         0, 2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             0, 0, 2, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0,
                         0, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             0, 0, 2, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0,
                         0, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             0, 0, 2, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0,
                         2, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             0, 2, 0, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0,
                         2, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             0, 2, 0, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0,
                         2, 0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             0, 2, 0, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0,
                         2, 0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             0, 2, 0, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0,
                         2, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             0, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0,
                         2, 2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             0, 2, 2, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0,
                         2, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             0, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0,
                         2, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             0, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 2,
                         2, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             2, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 2,
                         2, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             2, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 2,
                         2, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true,
                             2, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                             true, 2, 2, 0, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                             true, 2, 2, 0, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                             true, 2, 2, 0, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                             true, 2, 2, 0, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                             true, 2, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                             true, 2, 2, 2, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                             true, 2, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2,
                         2, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true,
                             true, 2, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2,
                         2, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true,
                             true, 2, 2, 0, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2,
                         2, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true,
                             true, 2, 2, 0, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2,
                         2, 0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true,
                             true, 2, 2, 0, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2,
                         2, 0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true,
                             true, 2, 2, 0, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2,
                         2, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true,
                             true, 2, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2,
                         2, 2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true,
                             true, 2, 2, 2, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2,
                         2, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true,
                             true, 2, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2,
                         2, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true,
                             true, 2, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true, 2, 2,
                         0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 0, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true, 2, 2,
                         0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 0, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true, 2, 2,
                         0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 0, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true, 2, 2,
                         0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 0, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true, 2, 2,
                         2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true, 2, 2,
                         2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 2, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true, 2, 2,
                         2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true, 2, 2,
                         2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, false, false, 0,
                         0, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, false, false,
                             0, 0, 0, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, false, true, 0,
                         0, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, false, true,
                             0, 0, 0, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, false, 0,
                         2, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, false,
                             0, 2, 0, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 0, 0,
                         0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             0, 0, 0, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 0, 0,
                         0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             0, 0, 0, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 0, 0,
                         0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             0, 0, 0, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 0, 0,
                         0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             0, 0, 0, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 0, 0,
                         2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             0, 0, 2, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 0, 0,
                         2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             0, 0, 2, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 0, 0,
                         2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             0, 0, 2, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 0, 0,
                         2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             0, 0, 2, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 0, 2,
                         0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             0, 2, 0, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 0, 2,
                         0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             0, 2, 0, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 0, 2,
                         0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             0, 2, 0, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 0, 2,
                         0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             0, 2, 0, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 0, 2,
                         2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             0, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 0, 2,
                         2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             0, 2, 2, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 0, 2,
                         2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             0, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 0, 2,
                         2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             0, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 2, 2,
                         2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             2, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 2, 2,
                         2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             2, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true, 2, 2,
                         2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 64, 64, 4, true, true, true,
                             2, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 0, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 0, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 0, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 0, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 2, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 128, 64, 4, true, true, true, 2,
                         2, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 128, 64, 4, true, true, true,
                             2, 2, 0, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 128, 64, 4, true, true, true, 2,
                         2, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 128, 64, 4, true, true, true,
                             2, 2, 0, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 128, 64, 4, true, true, true, 2,
                         2, 0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 128, 64, 4, true, true, true,
                             2, 2, 0, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 128, 64, 4, true, true, true, 2,
                         2, 0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 128, 64, 4, true, true, true,
                             2, 2, 0, true, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 128, 64, 4, true, true, true, 2,
                         2, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 128, 64, 4, true, true, true,
                             2, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 128, 64, 4, true, true, true, 2,
                         2, 2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 128, 64, 4, true, true, true,
                             2, 2, 2, false, true}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 128, 64, 4, true, true, true, 2,
                         2, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 128, 64, 4, true, true, true,
                             2, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kFloat16, 64, 128, 64, 4, true, true, true, 2,
                         2, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kFloat16, 64, 128, 64, 4, true, true, true,
                             2, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true, 2,
                         2, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 0, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true, 2,
                         2, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 0, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true, 2,
                         2, 0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 0, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true, 2,
                         2, 0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 0, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true, 2,
                         2, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true, 2,
                         2, 2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 2, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true, 2,
                         2, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true, 2,
                         2, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 32, 4, true, true, true,
                             2, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, false, false, 0,
                         0, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, false,
                             false, 0, 0, 0, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, false, true, 0,
                         0, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, false, true,
                             0, 0, 0, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 0,
                         0, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             0, 0, 0, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 0,
                         0, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             0, 0, 0, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 0,
                         0, 0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             0, 0, 0, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 0,
                         0, 0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             0, 0, 0, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 0,
                         0, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             0, 0, 2, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 0,
                         0, 2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             0, 0, 2, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 0,
                         0, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             0, 0, 2, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 0,
                         0, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             0, 0, 2, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 0,
                         2, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             0, 2, 0, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 0,
                         2, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             0, 2, 0, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 0,
                         2, 0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             0, 2, 0, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 0,
                         2, 0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             0, 2, 0, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 0,
                         2, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             0, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 0,
                         2, 2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             0, 2, 2, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 0,
                         2, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             0, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 0,
                         2, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             0, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 2,
                         2, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             2, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 2,
                         2, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             2, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true, 2,
                         2, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 64, 64, 4, true, true, true,
                             2, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 0, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 0, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 0, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 0, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 2, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true, 2,
                         2, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 128, 32, 4, true, true, true,
                             2, 2, 2, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 128, 64, 4, true, true, true, 2,
                         2, 0, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 128, 64, 4, true, true, true,
                             2, 2, 0, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 128, 64, 4, true, true, true, 2,
                         2, 0, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 128, 64, 4, true, true, true,
                             2, 2, 0, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 128, 64, 4, true, true, true, 2,
                         2, 0, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 128, 64, 4, true, true, true,
                             2, 2, 0, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 128, 64, 4, true, true, true, 2,
                         2, 0, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 128, 64, 4, true, true, true,
                             2, 2, 0, true, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 128, 64, 4, true, true, true, 2,
                         2, 2, false, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 128, 64, 4, true, true, true,
                             2, 2, 2, false, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 128, 64, 4, true, true, true, 2,
                         2, 2, false, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 128, 64, 4, true, true, true,
                             2, 2, 2, false, true}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 128, 64, 4, true, true, true, 2,
                         2, 2, true, false},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 128, 64, 4, true, true, true,
                             2, 2, 2, true, false}>>},

    {ForwardKernelConfig{torch::kBFloat16, 64, 128, 64, 4, true, true, true, 2,
                         2, 2, true, true},
     &flash_attention_forward_kernel<StaticForwardKernelConfig<
         ForwardKernelConfig{torch::kBFloat16, 64, 128, 64, 4, true, true, true,
                             2, 2, 2, true, true}>>},
};

}
