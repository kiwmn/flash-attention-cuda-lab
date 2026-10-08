#pragma once

#include <torch/torch.h>
#include <algorithm>

namespace flash_attn_lab::paged::k07 {

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
    const bool causal;
    const int page_size;

    int smem_bytes(int elem_size = 2) const {
        const int metadata_bytes =
            page_size <= Bc ? 2 * 2 * (Bc / page_size) * int(sizeof(uintptr_t))
                            : 0;
        return (Br + Bc * 2) * head_dim * elem_size + metadata_bytes;
    }

    int num_ctas_per_sm(int max_smem_bytes) const {
        if ((warp_num == 8) || (max_smem_bytes < smem_bytes() * 2)) {
            return 1;
        }

        return 2;
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
        if (causal != other.causal) {
            return causal < other.causal;
        }
        if (page_size != other.page_size) {
            return page_size < other.page_size;
        }
        return false;
    }
};

}
