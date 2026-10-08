#pragma once

// Tensor wrappers bind global/shared-memory tiles to per-thread register
// fragments.

#include "array.cuh"
#include "utils.h"
#include "load_store.cuh"
#include "tensor_view.cuh"

namespace flash_attn_lab::paged::k07 {

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

    template <int TileInPage, typename Pages>
    DEVICE_INLINE void copy_GM2SM(Pages &pages) {
        if (row_begin + GSRConfig::block_size <= valid_rows) {
            copy_paged_GM2SM<true, TileInPage>(pages);
        } else {
            copy_paged_GM2SM<false, TileInPage>(pages);
        }
    }

    template <int PageIndex, bool FullTile, typename Pages>
    DEVICE_INLINE void copy_one_page(Pages &pages, int first_page_begin,
                                     int offset_in_page, int valid_rows) {
        struct PageChunkConfig : gsmem {
            using OpIters =
                TShape<Pages::rows_per_page_chunk / gsmem::OpStride::row(),
                       gsmem::OpIters::cols()>;
        };
        const auto accessor =
            pages.template accessor<GSRConfig::transposed>(PageIndex);
        copy_block_GSM<GM2SM_op, PageChunkConfig, FullTile>(
            accessor,
            smem_gsm_ptr +
                PageIndex * Pages::page_size * gsmem::SmemStride::row(),
            offset_in_page,
            valid_rows - first_page_begin - PageIndex * Pages::page_size);
    }

    template <bool FullTile, typename Pages>
    DEVICE_INLINE void
    copy_block_small_pages(Pages &pages, int first_page_begin, int valid_rows) {
        static_assert(Pages::page_size <= GSRConfig::block_size);
        static_assert(Pages::pages_per_tile == 1 ||
                      Pages::pages_per_tile == 2 ||
                      Pages::pages_per_tile == 4 || Pages::pages_per_tile == 8);
        if constexpr (Pages::pages_per_tile == 1) {
            copy_one_page<0, FullTile>(pages, first_page_begin, 0, valid_rows);
        } else if constexpr (Pages::pages_per_tile == 2) {
            copy_one_page<0, FullTile>(pages, first_page_begin, 0, valid_rows);
            copy_one_page<1, FullTile>(pages, first_page_begin, 0, valid_rows);
        } else if constexpr (Pages::pages_per_tile == 4) {
            copy_one_page<0, FullTile>(pages, first_page_begin, 0, valid_rows);
            copy_one_page<1, FullTile>(pages, first_page_begin, 0, valid_rows);
            copy_one_page<2, FullTile>(pages, first_page_begin, 0, valid_rows);
            copy_one_page<3, FullTile>(pages, first_page_begin, 0, valid_rows);
        } else {
            copy_one_page<0, FullTile>(pages, first_page_begin, 0, valid_rows);
            copy_one_page<1, FullTile>(pages, first_page_begin, 0, valid_rows);
            copy_one_page<2, FullTile>(pages, first_page_begin, 0, valid_rows);
            copy_one_page<3, FullTile>(pages, first_page_begin, 0, valid_rows);
            copy_one_page<4, FullTile>(pages, first_page_begin, 0, valid_rows);
            copy_one_page<5, FullTile>(pages, first_page_begin, 0, valid_rows);
            copy_one_page<6, FullTile>(pages, first_page_begin, 0, valid_rows);
            copy_one_page<7, FullTile>(pages, first_page_begin, 0, valid_rows);
        }
    }

    template <bool FullTile, int TileInPage, typename Pages>
    DEVICE_INLINE void copy_paged_GM2SM(Pages &pages) {
        static_assert(GSRConfig::compute_over_entire_block);
        static_assert(Pages::rows_per_page_chunk % gsmem::OpStride::row() == 0);
        constexpr int warp_num =
            gsmem::OpStride::row() * gsmem::threads_per_row / WARP_SIZE;
        if constexpr (Pages::page_size <= GSRConfig::block_size) {
            static_assert(TileInPage == 0);
            if constexpr (!GSRConfig::transposed) {
                pages.prepare_block_pages(row_begin, valid_rows,
                                          gsmem::tid_to_thr_row(threadIdx.x));
            }
            copy_block_small_pages<FullTile>(pages, row_begin, valid_rows);
        } else {
            static_assert(Pages::page_size % GSRConfig::block_size == 0);
            if constexpr (!GSRConfig::transposed) {
                pages.template prepare<TileInPage>(
                    row_begin, valid_rows, gsmem::tid_to_thr_row(threadIdx.x));
            }
            const int offset_in_page =
                TileInPage < 0 ? row_begin : TileInPage * GSRConfig::block_size;
            const int first_page_begin = row_begin - offset_in_page;
            copy_one_page<0, FullTile>(pages, first_page_begin, offset_in_page,
                                       valid_rows);
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
