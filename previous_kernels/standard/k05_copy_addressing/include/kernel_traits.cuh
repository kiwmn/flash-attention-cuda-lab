#pragma once
#include <torch/torch.h>
#include <algorithm>

namespace flash_attn_lab::standard::k05 {

struct ForwardKernelConfig {
    const torch::ScalarType dtype;

    const int head_dim;
    const int Br;
    const int Bc;
    const int warp_num;

    const bool async_copy;

    const bool prefetch_kv_tiles;
    const bool swizzled;

    const int Q_mma_load_K_fragments;
    const int K_mma_load_K_fragments;
    const int V_mma_load_K_fragments;

    const bool mma_double_buffer_loads;
    const bool optimized_softmax;

    size_t smem_bytes() const {
        size_t bytes = 0;

        bytes += Br * head_dim * torch::elementSize(dtype);

        bytes += Bc * head_dim * torch::elementSize(dtype);
        bytes += Bc * head_dim * torch::elementSize(dtype);

        return bytes;
    }
    bool operator<(const ForwardKernelConfig &other) const {
        if (dtype != other.dtype) {
            return dtype < other.dtype;
        }
        if (head_dim != other.head_dim) {
            return head_dim < other.head_dim;
        }
        if (Br != other.Br) {
            return Br < other.Br;
        }
        if (Bc != other.Bc) {
            return Bc < other.Bc;
        }
        if (warp_num != other.warp_num) {
            return warp_num < other.warp_num;
        }
        if (async_copy != other.async_copy) {
            return async_copy < other.async_copy;
        }
        if (prefetch_kv_tiles != other.prefetch_kv_tiles) {
            return prefetch_kv_tiles < other.prefetch_kv_tiles;
        }
        if (swizzled != other.swizzled) {
            return swizzled < other.swizzled;
        }
        if (Q_mma_load_K_fragments != other.Q_mma_load_K_fragments) {
            return Q_mma_load_K_fragments < other.Q_mma_load_K_fragments;
        }
        if (K_mma_load_K_fragments != other.K_mma_load_K_fragments) {
            return K_mma_load_K_fragments < other.K_mma_load_K_fragments;
        }
        if (V_mma_load_K_fragments != other.V_mma_load_K_fragments) {
            return V_mma_load_K_fragments < other.V_mma_load_K_fragments;
        }
        if (mma_double_buffer_loads != other.mma_double_buffer_loads) {
            return mma_double_buffer_loads < other.mma_double_buffer_loads;
        }
        if (optimized_softmax != other.optimized_softmax) {
            return optimized_softmax < other.optimized_softmax;
        }
        return false;
    }
};

}
