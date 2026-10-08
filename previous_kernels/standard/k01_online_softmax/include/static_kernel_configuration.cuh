#pragma once

#include "gemm.cuh"
#include "kernel_traits.cuh"
#include "load_store.cuh"
#include "tensor.cuh"
#include "utils.h"

namespace flash_attn_lab::standard::k01 {

template <ForwardKernelConfig CFG>
struct ForwardKernelTileShapes {
    static constexpr int d_head_fragments = CFG.head_dim / COLS_PER_FRAGMENT;
    static constexpr int d_head_accum_regs =
        d_head_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT;
    static constexpr int QO_rows_per_warp = CFG.Br / CFG.warp_num;
    static constexpr int QO_fragments_per_warp =
        QO_rows_per_warp / ROWS_PER_FRAGMENT;
    static constexpr int KV_calc_fragments = CFG.Bc / ROWS_PER_FRAGMENT;
    static constexpr int KV_ldst_fragments_per_warp =
        KV_calc_fragments / CFG.warp_num;
    static constexpr int KV_ldst_rows_per_warp =
        KV_ldst_fragments_per_warp * ROWS_PER_FRAGMENT;
};

template <ForwardKernelConfig CFG>
struct StaticForwardKernelConfig {
    using accum_t = float;
    using value_t = typename std::conditional_t<CFG.dtype == torch::kBFloat16,
                                                nv_bfloat16, half>;
    using N = ForwardKernelTileShapes<CFG>;

    static constexpr int Br = CFG.Br;
    static constexpr int Bc = CFG.Bc;
    static constexpr int head_dim = CFG.head_dim;

    static constexpr TensorLDSTConfig
    make_ldst_config(TileLayout gmem_smem_layout, TileLayout register_layout,
                     bool transposed, int block_size, int warp_ldst_rows,
                     bool compute_over_entire_block) {
        return TensorLDSTConfig{gmem_smem_layout,
                                register_layout,
                                transposed,
                                block_size,
                                CFG.head_dim,
                                warp_ldst_rows,
                                compute_over_entire_block};
    }

    static constexpr TensorLDSTConfig Q_LDST =
        make_ldst_config({N::QO_fragments_per_warp, N::d_head_fragments},
                         {N::QO_fragments_per_warp, N::d_head_fragments}, false,
                         CFG.Br, N::QO_rows_per_warp, false);
    using Q_t = MatrixLDST<Q_LDST, value_t>;

    static constexpr TensorLDSTConfig K_LDST =
        make_ldst_config({N::KV_ldst_fragments_per_warp, N::d_head_fragments},
                         {N::KV_calc_fragments, N::d_head_fragments}, false,
                         CFG.Bc, N::KV_ldst_rows_per_warp, true);
    using K_t = MatrixLDST<K_LDST, value_t>;

    static constexpr TensorLDSTConfig V_LDST =
        make_ldst_config({N::KV_ldst_fragments_per_warp, N::d_head_fragments},
                         {N::d_head_fragments, N::KV_calc_fragments}, true,
                         CFG.Bc, N::KV_ldst_rows_per_warp, true);
    using V_t = MatrixLDST<V_LDST, value_t>;

    static constexpr TensorLDSTConfig O_LDST =
        make_ldst_config({N::QO_fragments_per_warp, N::d_head_fragments},
                         {N::QO_fragments_per_warp, N::d_head_fragments}, false,
                         CFG.Br, N::QO_rows_per_warp, false);
    using O_accum_t = MatrixLDST<O_LDST, accum_t>;
    using O_value_t = MatrixLDST<O_LDST, value_t>;

    static constexpr TensorLDSTConfig S_LDST =
        make_ldst_config({N::QO_fragments_per_warp, N::KV_calc_fragments},
                         {N::QO_fragments_per_warp, N::KV_calc_fragments},
                         false, CFG.Br, 0, false);
    using S_accum_t = MatrixLDST<S_LDST, accum_t>;
    using P_value_t = MatrixLDST<S_LDST, value_t>;

    using S_QK_GEMM = GEMM<Q_t, K_t, S_accum_t, value_t>;
    using O_PV_GEMM = GEMM<P_value_t, V_t, O_accum_t, value_t>;
};

}
