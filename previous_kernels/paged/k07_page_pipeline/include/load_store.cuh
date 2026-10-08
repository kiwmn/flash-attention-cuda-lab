#pragma once

// Copies preserve the shared-memory layout used by ldmatrix and MMA operand
// fragments.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>

#include "utils.h"
#include "layout.cuh"
#include "kv_accessor.cuh"
#include "ptx_functions.cuh"
#include "swizzling.cuh"

namespace flash_attn_lab::paged::k07 {

struct LDSTCommon {
    const bool swizzled;
    const bool async_copy;
};

template <typename ShapeT, int TileBufferSize, bool LoadEntireBlockIntoRF,
          bool RowMajorOpTile = true>
struct RmemLdstConfig {
    using rmem_shape = ShapeT;
    static constexpr int rmem_tile_buffer_size = TileBufferSize;
    static constexpr bool load_entire_block_into_rf = LoadEntireBlockIntoRF;
    static constexpr bool row_major_op_tile = RowMajorOpTile;
};

template <typename RmemConfigT, bool Swizzled, bool AsyncCopy, bool Transposed,
          int BlockSize, int SmemCols, int WarpLdstRows,
          bool ComputeOverEntireBlock>
struct GSRMemLdstConfig {
    using rmem = RmemConfigT;
    static constexpr LDSTCommon common{Swizzled, AsyncCopy};
    static constexpr bool transposed = Transposed;
    static constexpr int block_size = BlockSize;
    static constexpr int smem_cols = SmemCols;

    static constexpr int warp_ldst_rows = WarpLdstRows;

    static constexpr bool compute_over_entire_block = ComputeOverEntireBlock;
};

template <typename T>
struct GM2SMAsync {
    DEVICE_INLINE constexpr void operator()(T *gmem, T *smem, bool valid) {
        cp_async<BYTES_PER_VEC4_ACCESS>(smem, gmem, valid);
    }
};

template <typename T>
struct GM2SM {
    DEVICE_INLINE constexpr void operator()(T *gmem, T *smem, bool valid) {
        reinterpret_cast<uint4 *>(smem)[0] =
            valid ? reinterpret_cast<uint4 *>(gmem)[0] : make_uint4(0, 0, 0, 0);
    }
};

template <typename T>
struct SM2GM {
    DEVICE_INLINE constexpr void operator()(T *gmem, T *smem, bool valid) {
        if (valid) {
            reinterpret_cast<uint4 *>(gmem)[0] =
                reinterpret_cast<uint4 *>(smem)[0];
        }
    }
};

template <typename Swizzle_, typename OpStride_, typename TensorShape_,
          typename SmemStride_>
struct GSMemLdstConfig {
    using Swizzle = Swizzle_;
    using OpStride = OpStride_;
    using TensorShape = TensorShape_;
    using SmemStride = SmemStride_;

    using OpIters = TShape<TensorShape::rows() / OpStride::row(),
                           TensorShape::cols() / OpStride::col()>;

    static constexpr int threads_per_row = 8;

    static constexpr int tid_to_thr_row(int tid) {
        return tid / threads_per_row;
    }

    static constexpr int tid_to_thr_col(int tid) {
        return (tid % threads_per_row) * COLS_PER_FRAGMENT;
    }

    static constexpr int64_t gmem_thr_offset(int tid, RuntimeStride stride) {
        return tid_to_thr_row(tid) * stride.row +
               tid_to_thr_col(tid) * stride.col;
    }

    static constexpr int smem_thr_offset(int tid) {
        return Swizzle::apply(tid_to_thr_row(tid) * SmemStride::row() +
                              tid_to_thr_col(tid) * SmemStride::col());
    }
};

template <typename op, typename Cfg, bool FullTile = false, typename Accessor,
          typename value_t>
DEVICE_INLINE constexpr void copy_block_GSM(const Accessor &accessor,
                                            value_t *smem, int row_begin,
                                            int valid_rows) {
    const int thread_row = Cfg::tid_to_thr_row(threadIdx.x);
    const int thread_col = Cfg::tid_to_thr_col(threadIdx.x);
    const int first_row = row_begin + thread_row;
    auto row_cursor = accessor.template make_row_cursor<Cfg::threads_per_row>(
        first_row, FullTile || first_row < valid_rows);
#pragma unroll
    for (int ir = 0; ir < Cfg::OpIters::rows(); ++ir) {
        const int r = ir * Cfg::OpStride::row();
        const int row = row_begin + thread_row + r;
        const bool valid = FullTile || row < valid_rows;
        value_t *row_ptr = row_cursor.row_ptr(valid);
#pragma unroll
        for (int ic = 0; ic < Cfg::OpIters::cols(); ++ic) {
            const int c = ic * Cfg::OpStride::col();
            const int smem_idx =
                r * Cfg::SmemStride::row() + c * Cfg::SmemStride::col();

            value_t *src = valid ? row_ptr + thread_col + c : row_ptr;
            op()(src, smem + smem_idx, valid);
        }
        if (ir + 1 < Cfg::OpIters::rows()) {
            const int next_row = row + Cfg::OpStride::row();
            row_cursor.advance(Cfg::OpStride::row(),
                               FullTile || next_row < valid_rows);
        }
    }
}

template <typename Swizzle_, typename OpSmemStride_, typename OpRmemStride_,
          typename SmemStride_, typename SmemShape_,
          bool SmemRowMajorLdmatrix_ = false>
struct SRMemLdstConfig {
    using Swizzle = Swizzle_;
    using OpSmemStride = OpSmemStride_;
    using OpRmemStride = OpRmemStride_;
    using SmemStride = SmemStride_;
    using SmemShape = SmemShape_;
    static constexpr bool smem_row_major_ldmatrix = SmemRowMajorLdmatrix_;
    using OpIters = TShape<SmemShape::rows() / OpSmemStride::row(),
                           SmemShape::cols() / OpSmemStride::col()>;

    static constexpr int smem_col_fragments_per_tile = SmemShape::cols() / 8;

    static constexpr int lane_to_thr_offset_s2rmem(int lane_id) {
        int thread_row, thread_col;
        if constexpr (!smem_row_major_ldmatrix) {
            thread_row = lane_id % 16;
            thread_col = (lane_id / 16) * COLS_PER_FRAGMENT;
        } else {
            thread_row = (lane_id % 8) + 8 * (lane_id / 16);
            thread_col = lane_id & 8;
        }
        return Swizzle::apply(thread_row * SmemStride::row() +
                              thread_col * SmemStride::col());
    }

    static constexpr int lane_to_thr_offset_r2smem(int lane_id) {
        constexpr int threads_per_row = 4;
        constexpr int elems_per_thread = 2;
        int thread_row = lane_id / threads_per_row;
        int thread_col = (lane_id % threads_per_row) * elems_per_thread;
        return Swizzle::apply(thread_row * SmemStride::row() +
                              thread_col * SmemStride::col());
    }

    static constexpr SwizzleStride
    lane_to_thr_swizzle_stride_s2rmem(int lane_id) {
        if constexpr (std::is_same_v<Swizzle, NoSwizzle>) {
            return SwizzleStride{64, 32, 16};
        } else {
            int base_swizzle_offset = lane_to_thr_offset_s2rmem(lane_id);

            int base_offset_cmp = Swizzle::yy_mask_lowest_bit << 1;
            int s1 = 32 * binary_to_pm1((base_swizzle_offset &
                                         (base_offset_cmp << 1)) == 0);
            int s2 = 16 * binary_to_pm1(
                              (base_swizzle_offset & base_offset_cmp) == 0);

            return SwizzleStride{64, s1, s2};
        }
    }

    static constexpr SwizzleStride
    lane_to_thr_swizzle_stride_r2smem(int lane_id) {
        if constexpr (std::is_same_v<Swizzle, NoSwizzle>) {
            return SwizzleStride{64, 32, 16, 8};
        } else {
            int base_swizzle_offset = lane_to_thr_offset_r2smem(lane_id);

            int base_offset_cmp = Swizzle::yy_mask_lowest_bit;
            int s1 = 32 * binary_to_pm1((base_swizzle_offset &
                                         (base_offset_cmp << 2)) == 0);
            int s2 = 16 * binary_to_pm1((base_swizzle_offset &
                                         (base_offset_cmp << 1)) == 0);
            int s3 =
                8 * binary_to_pm1((base_swizzle_offset & base_offset_cmp) == 0);

            return SwizzleStride{64, s1, s2, s3};
        }
    }
};

template <typename Cfg, typename RmemType, typename value_t>
DEVICE_INLINE constexpr void
copy_warp_fragment_SM2RF(RmemType &rmem, value_t *smem,
                         const SwizzleStride &swizzle_stride, const int &tile) {
    static_assert(is_supported_mma_input_type<value_t>(),
                  "value_t must be half or bfloat16");
    static_assert(Cfg::OpIters::rows() == RmemType::ViewType2x2::Shape::rows() /
                                              Cfg::OpRmemStride::row(),
                  "OpIters.rows must be equal to RmemType::Shape.rows / "
                  "Cfg::OpRmemStride.row");
    static_assert(RmemType::Shape::cols() == 1,
                  "RmemType::Shape.cols must be 2");
    auto rmem_uint = rmem.view2x2();
    int swizzle_offset = swizzle_stride.offset_s2rmem(tile);

#pragma unroll
    for (int ir = 0; ir < Cfg::OpIters::rows(); ++ir) {
        int smem_offset =
            ir * Cfg::OpSmemStride::row() * Cfg::SmemStride::row() +
            swizzle_offset;
        int rmem_row = ir * Cfg::OpRmemStride::row();
        if constexpr (!Cfg::smem_row_major_ldmatrix) {
            ldmatrix_x4(rmem_uint(rmem_row, 0, tile, 0, 0),
                        rmem_uint(rmem_row, 0, tile, 1, 0),
                        rmem_uint(rmem_row, 0, tile, 0, 1),
                        rmem_uint(rmem_row, 0, tile, 1, 1), &smem[smem_offset]);
        } else {
            ldmatrix_x4(rmem_uint(rmem_row, 0, tile, 0, 0),
                        rmem_uint(rmem_row, 0, tile, 0, 1),
                        rmem_uint(rmem_row, 0, tile, 1, 0),
                        rmem_uint(rmem_row, 0, tile, 1, 1), &smem[smem_offset]);
        }
    }
}

template <typename Cfg, typename RmemType, typename value_t>
DEVICE_INLINE constexpr void
copy_warp_fragment_transposed_SM2RF(RmemType &rmem, value_t *smem,
                                    const SwizzleStride &swizzle_stride,
                                    const int &tile) {
    static_assert(is_supported_mma_input_type<value_t>(),
                  "value_t must be half or bfloat16");
    using Swizzle = typename Cfg::Swizzle;
    static_assert(RmemType::Shape::cols() == 1,
                  "RmemType::Shape.cols must be 2");
    static_assert(Cfg::OpIters::cols() == RmemType::ViewType2x2::Shape::rows() /
                                              Cfg::OpRmemStride::col(),
                  "OpIters.cols must be equal to RmemType::Shape.rows / "
                  "Cfg::OpRmemStride.col");
    auto rmem_uint = rmem.view2x2();

    int base_offset = tile * Cfg::SmemStride::tile();

#pragma unroll
    for (int ic = 0; ic < Cfg::OpIters::cols(); ++ic) {
        int swizzle_offset = swizzle_stride.offset_s2rmem(ic);
        int smem_offset = base_offset + swizzle_offset;

        int rmem_row = ic * Cfg::OpRmemStride::row();
        ldmatrix_x4_transpose(rmem_uint(rmem_row, 0, tile, 0, 0),
                              rmem_uint(rmem_row, 0, tile, 0, 1),
                              rmem_uint(rmem_row, 0, tile, 1, 0),
                              rmem_uint(rmem_row, 0, tile, 1, 1),
                              &smem[smem_offset]);
    }
}

template <typename Cfg, typename RmemType, typename value_t>
DEVICE_INLINE constexpr void
copy_warp_fragment_RF2SM(RmemType &rmem, value_t *smem,
                         const SwizzleStride &swizzle_stride) {
    static_assert(is_supported_mma_input_type<value_t>(),
                  "value_t must be half or bfloat16");
    using Swizzle = typename Cfg::Swizzle;
    auto rmem_uint = rmem.view();

#pragma unroll
    for (int ir = 0; ir < Cfg::OpIters::rows(); ++ir) {
#pragma unroll
        for (int ic = 0; ic < Cfg::OpIters::cols(); ++ic) {
            int smem_offset =
                ir * Cfg::OpSmemStride::row() * Cfg::SmemStride::row() +
                swizzle_stride.offset_r2smem(ic);

            reinterpret_cast<uint32_t *>(&smem[smem_offset])[0] = rmem_uint(
                ir * Cfg::OpRmemStride::row(), ic * Cfg::OpRmemStride::col());
        }
    }
}

template <typename SrcType, typename DstType>
DEVICE_INLINE constexpr void convert_to_16_bit_dtype(SrcType &src_view,
                                                     DstType &dst_view) {
    static_assert(std::is_same_v<typename SrcType::value_t, float>,
                  "Input tensor must be float type");
    static_assert(std::is_same_v<typename DstType::value_t, half> ||
                      std::is_same_v<typename DstType::value_t, nv_bfloat16>,
                  "Output tensor must be half or bfloat16 type");
    using value_t = typename DstType::value_t;

    auto src = src_view.with_op_tiling_removed();
    auto dst2 = dst_view.with_op_tiling_removed().as_type2();
    using SrcShape = decltype(src)::Layout::Shape;
    using DstShape = decltype(dst2)::Layout::Shape;

    static_assert(SrcShape::tiles() == 1, "Src must have 1 tile");
    static_assert(SrcShape::cols() * SrcShape::tiles() ==
                      DstShape::cols() * DstShape::tiles() * 2,
                  "A and B must have the same shape");
    static_assert(SrcShape::rows() == DstShape::rows(),
                  "A and B must have the same shape");

#pragma unroll
    for (int tile = 0; tile < DstShape::tiles(); ++tile) {
        int tile_offset = 2 * tile * DstShape::cols();
#pragma unroll
        for (int m = 0; m < DstShape::rows(); ++m) {
#pragma unroll
            for (int k = 0; k < DstShape::cols(); ++k) {
                int src_k = tile_offset + 2 * k;
                float2 src_val{src(m, src_k, 0), src(m, src_k + 1, 0)};
                if constexpr (std::is_same_v<value_t, half>) {
                    dst2(m, k, tile) = __float22half2_rn(src_val);
                } else {
                    dst2(m, k, tile) = __float22bfloat162_rn(src_val);
                }
            }
        }
    }
}

}
