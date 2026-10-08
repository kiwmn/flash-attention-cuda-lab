#pragma once

#include <torch/torch.h>

namespace flash_attn_lab::standard::k02 {

struct ForwardKernelConfig {
    const torch::ScalarType dtype;
    const int head_dim;
    const int Br;
    const int Bc;
    const int warp_num;

    size_t smem_bytes() const {
        return (Br + 2 * Bc) * head_dim * torch::elementSize(dtype);
    }

    bool operator<(const ForwardKernelConfig &other) const {
        if (dtype != other.dtype)
            return dtype < other.dtype;
        if (head_dim != other.head_dim)
            return head_dim < other.head_dim;
        if (Br != other.Br)
            return Br < other.Br;
        if (Bc != other.Bc)
            return Bc < other.Bc;
        return warp_num < other.warp_num;
    }
};

}
