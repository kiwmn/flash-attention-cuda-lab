#pragma once
#include "load_store.cuh"
#include "utils.h"
#include "kernel_traits.cuh"
#include "gemm.cuh"
#include "tensor.cuh"

namespace flash_attn_lab::standard::k04 {

template <int n, int K, bool double_buffer>
constexpr void static_assert_valid_load_k_fragments() {
    static_assert(((n & (n - 1)) == 0) && n != 1,
                  "load k is power of 2 and DNE 1");

    constexpr int max_frags = (double_buffer ? K / 2 : K) / 8;
    static_assert(n <= max_frags, "load k is <= max fragments");
}

template <ForwardKernelConfig cfg>
constexpr bool valid_config() {
    static_assert_valid_load_k_fragments<cfg.Q_mma_load_K_fragments,
                                         cfg.head_dim,
                                         cfg.mma_double_buffer_loads>();
    static_assert_valid_load_k_fragments<cfg.K_mma_load_K_fragments,
                                         cfg.head_dim,
                                         cfg.mma_double_buffer_loads>();
    static_assert_valid_load_k_fragments<cfg.V_mma_load_K_fragments, cfg.Bc,
                                         cfg.mma_double_buffer_loads>();

    static_assert((cfg.Q_mma_load_K_fragments == cfg.K_mma_load_K_fragments) ||
                  cfg.Q_mma_load_K_fragments == 0);

    return true;
}

template <ForwardKernelConfig CFG>
struct ForwardKernelTileShapes {
    static_assert(valid_config<CFG>());

    static constexpr int d_head_fragments = CFG.head_dim / COLS_PER_FRAGMENT;
    static constexpr int d_head_accum_regs =
        d_head_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT;

    static constexpr int QO_rows_per_warp = CFG.Br / CFG.warp_num;
    static constexpr int QO_fragments_per_warp =
        QO_rows_per_warp / ROWS_PER_FRAGMENT;

    static constexpr int KV_calc_fragments = CFG.Bc / ROWS_PER_FRAGMENT;
    static constexpr int KV_calc_accum_regs =
        KV_calc_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT;

    static constexpr int KV_ldst_fragments_per_warp =
        KV_calc_fragments / CFG.warp_num;
    static constexpr int KV_ldst_rows_per_warp =
        KV_ldst_fragments_per_warp * ROWS_PER_FRAGMENT;

    static constexpr int Q_mma_load_K_fragments =
        CFG.Q_mma_load_K_fragments == 0 ? d_head_fragments
                                        : CFG.Q_mma_load_K_fragments;
    static constexpr int Q_mma_load_stages =
        (CFG.Q_mma_load_K_fragments > 0 && CFG.mma_double_buffer_loads) ? 2 : 1;

    static constexpr int K_mma_load_K_fragments =
        CFG.K_mma_load_K_fragments == 0 ? d_head_fragments
                                        : CFG.K_mma_load_K_fragments;
    static constexpr int K_mma_load_stages =
        (CFG.K_mma_load_K_fragments > 0 && CFG.mma_double_buffer_loads) ? 2 : 1;

    static constexpr int V_mma_load_K_fragments =
        CFG.V_mma_load_K_fragments == 0 ? KV_calc_fragments
                                        : CFG.V_mma_load_K_fragments;
    static constexpr int V_mma_load_stages =
        (CFG.V_mma_load_K_fragments > 0 && CFG.mma_double_buffer_loads) ? 2 : 1;
};

template <ForwardKernelConfig CFG>
struct StaticForwardKernelConfig {
    using accum_t = float;
    using value_t = typename std::conditional_t<CFG.dtype == torch::kBFloat16,
                                                nv_bfloat16, half>;
    using N = ForwardKernelTileShapes<CFG>;

    static constexpr bool async_copy = CFG.async_copy;
    static constexpr int Br = CFG.Br;
    static constexpr int Bc = CFG.Bc;
    static constexpr int head_dim = CFG.head_dim;
    static constexpr bool prefetch_kv_tiles = CFG.prefetch_kv_tiles;
    static constexpr bool optimized_softmax = CFG.optimized_softmax;

    static constexpr LDSTCommon Common{CFG.swizzled, CFG.async_copy};

    static constexpr TensorLDSTConfig
    make_ldst_config(TileLayout gmem_smem_layout, TileLayout register_layout,
                     bool transposed, int block_size, int warp_ldst_rows,
                     bool compute_over_entire_block,
                     bool load_entire_block_into_rf = true,
                     int mma_load_stages = 1) {
        return TensorLDSTConfig{gmem_smem_layout,
                                register_layout,
                                Common,
                                transposed,
                                block_size,
                                CFG.head_dim,
                                warp_ldst_rows,
                                compute_over_entire_block,
                                load_entire_block_into_rf,
                                mma_load_stages};
    }

    static constexpr TensorLDSTConfig Q_LDST =
        make_ldst_config({N::QO_fragments_per_warp, N::d_head_fragments},
                         {N::QO_fragments_per_warp, N::Q_mma_load_K_fragments},
                         false, CFG.Br, N::QO_rows_per_warp, false,
                         CFG.Q_mma_load_K_fragments == 0, N::Q_mma_load_stages);
    using Q_t = MatrixLDST<Q_LDST, value_t>;

    static constexpr TensorLDSTConfig K_LDST =
        make_ldst_config({N::KV_ldst_fragments_per_warp, N::d_head_fragments},
                         {N::KV_calc_fragments, N::K_mma_load_K_fragments},
                         false, CFG.Bc, N::KV_ldst_rows_per_warp, true,
                         CFG.K_mma_load_K_fragments == 0, N::K_mma_load_stages);
    using K_t = MatrixLDST<K_LDST, value_t>;

    static constexpr TensorLDSTConfig V_LDST =
        make_ldst_config({N::KV_ldst_fragments_per_warp, N::d_head_fragments},
                         {N::d_head_fragments, N::V_mma_load_K_fragments}, true,
                         CFG.Bc, N::KV_ldst_rows_per_warp, true,
                         CFG.V_mma_load_K_fragments == 0, N::V_mma_load_stages);
    using V_t = MatrixLDST<V_LDST, value_t>;

    static constexpr TensorLDSTConfig O_LDST =
        make_ldst_config({N::QO_fragments_per_warp, N::d_head_fragments},
                         {N::QO_fragments_per_warp, N::d_head_fragments}, false,
                         CFG.Br, N::QO_rows_per_warp, false, true);
    using O_accum_t = MatrixLDST<O_LDST, accum_t>;
    using O_value_t = MatrixLDST<O_LDST, value_t>;

    static constexpr TensorLDSTConfig S_LDST =
        make_ldst_config({N::QO_fragments_per_warp, N::KV_calc_fragments},
                         {N::QO_fragments_per_warp, N::KV_calc_fragments},
                         false, CFG.Br, 0, false);
    using S_accum_t = MatrixLDST<S_LDST, accum_t>;
    using P_value_t = MatrixLDST<S_LDST, value_t>;

    using S_QK_GEMM = GEMM<Q_t, K_t, S_accum_t, N::d_head_fragments,
                           constexpr_min(N::Q_mma_load_K_fragments,
                                         N::K_mma_load_K_fragments),
                           value_t>;
    using O_PV_GEMM = GEMM<P_value_t, V_t, O_accum_t, N::KV_calc_fragments,
                           N::V_mma_load_K_fragments, value_t>;

    using row_statistics_t = RFVector<accum_t, N::QO_fragments_per_warp>;
};

}
