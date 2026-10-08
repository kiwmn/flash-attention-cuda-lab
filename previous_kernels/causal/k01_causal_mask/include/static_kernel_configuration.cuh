#pragma once

#include "utils.h"
#include "kernel_traits.cuh"
#include "gemm.cuh"
#include "layout.cuh"
#include "load_store.cuh"
#include "tensor.cuh"

namespace flash_attn_lab::causal::k01 {

template <int n, int K, bool double_buffer>
constexpr void static_assert_valid_load_k_fragments() {
    static_assert(((n & (n - 1)) == 0) && n != 1,
                  "load k is power of 2 and DNE 1");

    constexpr int max_frags = (double_buffer ? K / 2 : K) / 8;
    static_assert(n <= max_frags, "load k is <= max fragments");
}

template <const ForwardKernelConfig &cfg>
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

    static constexpr int n_threads = CFG.warp_num * WARP_SIZE;

    static constexpr int d_head_fragments = CFG.head_dim / COLS_PER_FRAGMENT;

    static constexpr int QO_rows_per_warp = CFG.Br / CFG.warp_num;
    static constexpr int QO_fragments_per_warp =
        QO_rows_per_warp / ROWS_PER_FRAGMENT;

    static constexpr int KV_calc_fragments = CFG.Bc / ROWS_PER_FRAGMENT;

    static constexpr int KV_ldst_fragments_per_warp =
        KV_calc_fragments / CFG.warp_num;
    static constexpr int KV_ldst_rows_per_warp =
        KV_ldst_fragments_per_warp * ROWS_PER_FRAGMENT;

    static constexpr int get_tile_fragments(int val, int default_val) {
        return val == 0 ? default_val : val;
    }

    static constexpr int QK_rmem_tile_fragments = get_tile_fragments(
        constexpr_max(CFG.Q_mma_load_K_fragments, CFG.K_mma_load_K_fragments),
        d_head_fragments);
    static constexpr int QK_rmem_tile_size =
        QK_rmem_tile_fragments * COLS_PER_FRAGMENT;
    static constexpr int QK_rmem_tiles =
        d_head_fragments / QK_rmem_tile_fragments;
    static constexpr int PV_rmem_tile_fragments =
        get_tile_fragments(CFG.V_mma_load_K_fragments, KV_calc_fragments);
    static constexpr int PV_rmem_tile_size =
        PV_rmem_tile_fragments * COLS_PER_FRAGMENT;
    static constexpr int PV_rmem_tiles =
        KV_calc_fragments / PV_rmem_tile_fragments;

    static constexpr int get_rmem_tile_buffer_size(int load_K_fragments,
                                                   int tiles) {
        if (load_K_fragments == 0) {
            return tiles;
        }
        return CFG.mma_double_buffer_loads ? 2 : 1;
    }

    static constexpr int Q_rmem_tile_buffer_size =
        get_rmem_tile_buffer_size(CFG.Q_mma_load_K_fragments, QK_rmem_tiles);

    static constexpr int K_rmem_tile_buffer_size =
        get_rmem_tile_buffer_size(CFG.K_mma_load_K_fragments, QK_rmem_tiles);

    static constexpr int V_rmem_tile_buffer_size =
        get_rmem_tile_buffer_size(CFG.V_mma_load_K_fragments, PV_rmem_tiles);
};

template <ForwardKernelConfig _CFG>
struct StaticForwardKernelConfig {
    static constexpr ForwardKernelConfig CFG = _CFG;

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
    static constexpr bool causal = CFG.causal;

    static constexpr LDSTCommon common{CFG.swizzled, CFG.async_copy};

    using OpGSMemStride = TStride<N::n_threads / GSMEM_THR_PER_ROW, 64>;

    using OpS2RSmemStride = TStride<16, 16>;
    using OpS2RRmemStride = TStride<1, 1>;

    using OpR2SSMemStride = TStride<8, 8>;
    using OpR2SRmemStride = TStride<1, 1>;

    static_assert(CFG.head_dim % 64 == 0, "head_dim must be a multiple of 64");
    using SmemSwizzle_ =
        CuteSwizzle<3, 3,
                    constexpr_log2_floor(CFG.head_dim) -
                        constexpr_log2_floor(ELEMS_PER_VEC4_ACCESS)>;
    using SmemSwizzle =
        std::conditional_t<CFG.swizzled, SmemSwizzle_, NoSwizzle>;

    using GSMemShapeQO = TShape<CFG.Br, CFG.head_dim, 1>;
    using GSMemShapeKV = TShape<CFG.Bc, CFG.head_dim, 1>;
    using GSMemQKVOStride = TStride<CFG.head_dim, 1, 0>;

    using GSMemLdstConfigQO = GSMemLdstConfig<SmemSwizzle, OpGSMemStride,
                                              GSMemShapeQO, GSMemQKVOStride>;
    using GSMemLdstConfigKV = GSMemLdstConfig<SmemSwizzle, OpGSMemStride,
                                              GSMemShapeKV, GSMemQKVOStride>;

    static constexpr int SRMemTileSize = 16;
    static constexpr int SRMemTileFragments = SRMemTileSize / COLS_PER_FRAGMENT;
    static constexpr int SRMemTilesDHead = CFG.head_dim / SRMemTileSize;
    static constexpr int SRMemFragmentsDHead =
        SRMemTilesDHead * SRMemTileFragments;

    static constexpr int SRMemTilesB_c = CFG.Bc / SRMemTileSize;
    static constexpr int SRMemFragmentsB_c = SRMemTilesB_c * SRMemTileFragments;

    using S2RSmemShapeQ =
        TShape<N::QO_rows_per_warp, SRMemTileSize, SRMemTilesDHead>;
    using S2RSmemStrideQ = TStride<CFG.head_dim, 1, SRMemTileSize>;
    using RmemShapeQ = TShape<N::QO_fragments_per_warp / 2,
                              SRMemTileFragments / 2, SRMemTilesDHead, 2, 2>;

    using S2RMemLdstConfigQ =
        SRMemLdstConfig<SmemSwizzle, OpS2RSmemStride, OpS2RRmemStride,
                        S2RSmemStrideQ, S2RSmemShapeQ>;

    using S2RSmemShapeK = TShape<CFG.Bc, SRMemTileSize, SRMemTilesDHead>;
    using S2RSmemStrideK = TStride<CFG.head_dim, 1, SRMemTileSize>;
    using RmemShapeK = TShape<SRMemFragmentsB_c / 1, SRMemTileFragments / 2,
                              SRMemTilesDHead, 1, 2>;
    using S2RMemLdstConfigK =
        SRMemLdstConfig<SmemSwizzle, OpS2RSmemStride, OpS2RRmemStride,
                        S2RSmemStrideK, S2RSmemShapeK, true>;

    using S2RSmemShapeV = TShape<SRMemTileSize, CFG.head_dim, SRMemTilesB_c>;
    using S2RSmemStrideV =
        TStride<CFG.head_dim, 1, SRMemTileSize * CFG.head_dim>;
    using RmemShapeV = TShape<SRMemFragmentsDHead / 1, SRMemTileFragments / 2,
                              SRMemTilesB_c, 1, 2>;

    using S2RMemLdstConfigV =
        SRMemLdstConfig<SmemSwizzle, OpS2RSmemStride, OpS2RRmemStride,
                        S2RSmemStrideV, S2RSmemShapeV>;

    using R2SSmemShapeO = TShape<N::QO_rows_per_warp, CFG.head_dim, 1>;
    using R2SSmemStrideO = TStride<CFG.head_dim, 1, 0>;
    using RmemShapeOAccum =
        TShape<N::QO_fragments_per_warp / 2,
               N::d_head_fragments * THR_COLS_PER_ACCUM_FRAGMENT / 2, 1, 2, 2>;
    using RmemShapeO =
        TShape<N::QO_fragments_per_warp, N::d_head_fragments, 1, 1, 1>;
    using R2SMemLdstConfigO =
        SRMemLdstConfig<SmemSwizzle, OpR2SSMemStride, OpR2SRmemStride,
                        R2SSmemStrideO, R2SSmemShapeO>;

    using RmemShapeSAccum =
        TShape<N::QO_fragments_per_warp / 2,
               N::KV_calc_fragments * THR_COLS_PER_ACCUM_FRAGMENT / 2, 1, 2, 2>;
    using RmemShapeP = TShape<N::QO_fragments_per_warp / 2,
                              SRMemTileFragments / 2, SRMemTilesB_c, 2, 2>;

    using RmemConfigQ = RmemLdstConfig<RmemShapeQ, N::Q_rmem_tile_buffer_size,
                                       CFG.Q_mma_load_K_fragments == 0, false>;

    using GSRConfigQ =
        GSRMemLdstConfig<RmemConfigQ, CFG.swizzled, CFG.async_copy, false,
                         CFG.Br, CFG.head_dim, N::QO_rows_per_warp, false>;

    using Q_t = GSRBlockTensor<GSRConfigQ, value_t, GSMemLdstConfigQO,
                               S2RMemLdstConfigQ>;

    using RmemConfigK = RmemLdstConfig<RmemShapeK, N::K_rmem_tile_buffer_size,
                                       CFG.K_mma_load_K_fragments == 0, true>;

    using GSRConfigK =
        GSRMemLdstConfig<RmemConfigK, CFG.swizzled, CFG.async_copy, false,
                         CFG.Bc, CFG.head_dim, N::KV_ldst_rows_per_warp, true>;

    using K_t = GSRBlockTensor<GSRConfigK, value_t, GSMemLdstConfigKV,
                               S2RMemLdstConfigK>;

    using RmemConfigV = RmemLdstConfig<RmemShapeV, N::V_rmem_tile_buffer_size,
                                       CFG.V_mma_load_K_fragments == 0, true>;

    using GSRConfigV =
        GSRMemLdstConfig<RmemConfigV, CFG.swizzled, CFG.async_copy, true,
                         CFG.Bc, CFG.head_dim, N::KV_ldst_rows_per_warp, true>;

    using V_t = GSRBlockTensor<GSRConfigV, value_t, GSMemLdstConfigKV,
                               S2RMemLdstConfigV>;

    using RmemConfigSAccum = RmemLdstConfig<RmemShapeSAccum, 1, true, true>;

    using S_accum_t = RmemBlockTensor<RmemConfigSAccum, accum_t>;

    using RmemConfigP =
        RmemLdstConfig<RmemShapeP, RmemShapeP::tiles(), true, false>;

    using P_value_t = RmemBlockTensor<RmemConfigP, value_t>;

    using RmemConfigOAccum = RmemLdstConfig<RmemShapeOAccum, 1, true, true>;

    using O_accum_t = RmemBlockTensor<RmemConfigOAccum, accum_t>;

    using RmemConfigO = RmemLdstConfig<RmemShapeO, 1, true, true>;

    using GSRConfigO =
        GSRMemLdstConfig<RmemConfigO, CFG.swizzled, CFG.async_copy, false,
                         CFG.Br, CFG.head_dim, N::QO_rows_per_warp, false>;

    using O_value_t = GSRBlockTensor<GSRConfigO, value_t, GSMemLdstConfigQO,
                                     R2SMemLdstConfigO>;

    using S_QK_GEMM = GEMM<Q_t, K_t, S_accum_t, SRMemTilesDHead, value_t>;
    using O_PV_GEMM = GEMM<P_value_t, V_t, O_accum_t, SRMemTilesB_c, value_t>;

    using row_statistics_t = ArrayAligned<N::QO_fragments_per_warp, accum_t>;
};

}
