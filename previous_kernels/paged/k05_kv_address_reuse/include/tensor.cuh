#pragma once

// Tensor wrappers bind global/shared-memory tiles to per-thread register
// fragments.

#include "array.cuh"
#include "utils.h"
#include "load_store.cuh"
#include "tensor_view.cuh"

namespace flash_attn_lab::paged::k05 {

template <typename RmemConfig, typename value_t_, typename index_t = int64_t>
struct RmemBlockTensor {
    using Shape = typename RmemConfig::rmem_shape;
    using Stride =
        decltype(stride_for_shape<Shape, RmemConfig::row_major_op_tile>());
    using Layout = Layout<Stride, Shape>;
    using Layout2x2 = decltype(Layout::layout_as_2x2_op_tiled());

    using value_t = value_t_;
    using storage_t = decltype(value_storage_type<value_t>());

    using ViewType = TensorView<value_t, Layout>;
    using ViewType2x2 = TensorView<value_t, Layout2x2>;

    static constexpr int StorageSize = Shape::size();

    static constexpr int rmem_tile_buffer_size =
        RmemConfig::rmem_tile_buffer_size;
    static constexpr bool load_entire_block_into_rf =
        RmemConfig::load_entire_block_into_rf;

    ArrayAligned<StorageSize, storage_t> storage;

    DEVICE_INLINE constexpr void zero() { storage.zero(); }

    DEVICE_INLINE constexpr ViewType view() { return ViewType(storage.data()); }
    DEVICE_INLINE constexpr ViewType2x2 view2x2() {
        return ViewType2x2(storage.data());
    }
    DEVICE_INLINE constexpr auto view_as_type2() { return view().as_type2(); }

    DEVICE_INLINE constexpr auto view_with_op_tiling_removed() {
        return view().with_op_tiling_removed();
    }
};

template <typename GSRConfig, typename value_t, typename gsmem, typename srmem,
          typename index_t = int64_t>
struct GSRBlockTensor
    : public RmemBlockTensor<typename GSRConfig::rmem, value_t, index_t> {
    using Base = RmemBlockTensor<typename GSRConfig::rmem, value_t, index_t>;
    using GM2SM_op = std::conditional_t<GSRConfig::common.async_copy,
                                        GM2SMAsync<value_t>, GM2SM<value_t>>;

    using SM2GM_op = SM2GM<value_t>;

    value_t *gmem_ptr;
    RuntimeStride gmem_stride;
    int row_begin;
    int valid_rows;

    value_t *smem_gsm_ptr;

    value_t *smem_s2r_ptr;
    value_t *smem_r2s_ptr;

    SwizzleStride s2rmem_swizzle_stride;
    SwizzleStride r2smem_swizzle_stride;

    DEVICE_INLINE GSRBlockTensor(value_t *gmem_block_ptr,
                                 index_t _gmem_seq_stride, value_t *_smem_ptr,
                                 int _valid_rows, int _row_begin = 0)
        : Base(), gmem_stride(RuntimeStride{_gmem_seq_stride, 1, 0}),
          row_begin(_row_begin), valid_rows(_valid_rows) {
        const int tid = threadIdx.x;

        gmem_ptr = gmem_block_ptr;
        smem_gsm_ptr = _smem_ptr + gsmem::smem_thr_offset(tid);

        const int lane_id = tid % WARP_SIZE;
        const int warp_idx = tid / WARP_SIZE;
        s2rmem_swizzle_stride =
            srmem::lane_to_thr_swizzle_stride_s2rmem(lane_id);
        r2smem_swizzle_stride =
            srmem::lane_to_thr_swizzle_stride_r2smem(lane_id);

        auto smem_srmem_ptr =
            _smem_ptr + (GSRConfig::compute_over_entire_block
                             ? 0
                             : GSRConfig().warp_ldst_rows * warp_idx *
                                   GSRConfig().smem_cols);

        smem_s2r_ptr =
            smem_srmem_ptr + srmem::lane_to_thr_offset_s2rmem(lane_id);
        smem_r2s_ptr =
            smem_srmem_ptr + srmem::lane_to_thr_offset_r2smem(lane_id);
    }

    DEVICE_INLINE constexpr void advance_gmem_block() {
        row_begin += GSRConfig().block_size;
    }

    DEVICE_INLINE constexpr void copy_GM2SM() {
        static_assert(!GSRConfig::compute_over_entire_block,
                      "K/V must load through the shared page cache");
        const ContiguousRowAccessor<value_t> accessor{gmem_ptr,
                                                      gmem_stride.row};
        copy_block_GSM<GM2SM_op, gsmem>(accessor, smem_gsm_ptr, row_begin,
                                        valid_rows);
    }

    template <typename Pages>
    DEVICE_INLINE void copy_GM2SM(Pages &pages) {
        if (row_begin + GSRConfig::block_size <= valid_rows) {
            copy_paged_GM2SM<true>(pages);
        } else {
            copy_paged_GM2SM<false>(pages);
        }
    }

    template <bool FullTile, typename Pages>
    DEVICE_INLINE void copy_paged_GM2SM(Pages &pages) {
        static_assert(GSRConfig::compute_over_entire_block);
        static_assert(Pages::rows_per_page_chunk % gsmem::OpStride::row() == 0);

        if constexpr (!GSRConfig::transposed) {
            pages.prepare(row_begin, valid_rows);
        }
        struct PageChunkConfig : gsmem {
            using OpIters =
                TShape<Pages::rows_per_page_chunk / gsmem::OpStride::row(),
                       gsmem::OpIters::cols()>;
        };
        const int first_page_begin =
            (row_begin / Pages::page_size) * Pages::page_size;
        const int offset_in_page = row_begin % Pages::page_size;
#pragma unroll
        for (int p = 0; p < Pages::pages_per_tile; ++p) {
            const auto accessor =
                pages.template accessor<GSRConfig::transposed>(p);
            copy_block_GSM<GM2SM_op, PageChunkConfig, FullTile>(
                accessor,
                smem_gsm_ptr + p * Pages::page_size * gsmem::SmemStride::row(),
                offset_in_page,
                valid_rows - first_page_begin - p * Pages::page_size);
        }
    }

    DEVICE_INLINE constexpr void copy_SM2GM() {
        const ContiguousRowAccessor<value_t> accessor{gmem_ptr,
                                                      gmem_stride.row};
        copy_block_GSM<SM2GM_op, gsmem>(accessor, smem_gsm_ptr, row_begin,
                                        valid_rows);
    }

    DEVICE_INLINE constexpr void copy_SM2RF(int tile = 0) {
        if constexpr (!GSRConfig::transposed) {
            copy_warp_fragment_SM2RF<srmem>(*this, smem_s2r_ptr,
                                            s2rmem_swizzle_stride, tile);
        } else {
            copy_warp_fragment_transposed_SM2RF<srmem>(
                *this, smem_s2r_ptr, s2rmem_swizzle_stride, tile);
        }
    }

    DEVICE_INLINE constexpr void copy_SM2RF_all_tiles() {
        for (int tile = 0; tile < Base::Shape::tiles(); ++tile) {
            copy_SM2RF(tile);
        }
    }

    DEVICE_INLINE constexpr void copy_RF2SM() {
        copy_warp_fragment_RF2SM<srmem>(*this, smem_r2s_ptr,
                                        r2smem_swizzle_stride);
    }
};

}
