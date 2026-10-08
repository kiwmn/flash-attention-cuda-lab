#pragma once

// Compile-time shapes and strides map logical fragments to registers and shared
// memory.

#include <stdio.h>

#include "utils.h"

namespace flash_attn_lab::a100::k04 {

template <int Row, int Col, int Tile = 0, int OpRow = 0, int OpCol = 0>
struct TStride {
    DEVICE_INLINE constexpr static int row() { return Row; }
    DEVICE_INLINE constexpr static int col() { return Col; }
    DEVICE_INLINE constexpr static int tile() { return Tile; }
    DEVICE_INLINE constexpr static int op_row() { return OpRow; }
    DEVICE_INLINE constexpr static int op_col() { return OpCol; }
};

struct SwizzleStride {
    int s0;
    int s1;
    int s2;
    int s3;

    constexpr int offset_s2rmem(int iter) const {
        int i0 = (iter >> 2) & 1;
        int i1 = (iter >> 1) & 1;
        int i2 = iter & 1;
        return i0 * s0 + i1 * s1 + i2 * s2;
    }

    constexpr int offset_r2smem(int iter) const {
        int i0 = (iter >> 3) & 1;
        int i1 = (iter >> 2) & 1;
        int i2 = (iter >> 1) & 1;
        int i3 = iter & 1;
        return i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3;
    }
};

template <typename index_t = int64_t>
struct RuntimeStride {
    index_t row;
    index_t col;
    index_t tile = 0;
};

template <int Rows, int Cols, int Tiles = 1, int OpRows = 1, int OpCols = 1,
          bool op_tiling_removed = false>
struct TShape {
    DEVICE_INLINE constexpr static int rows() {
        return op_tiling_removed ? Rows * OpRows : Rows;
    }

    DEVICE_INLINE constexpr static int cols() {
        return op_tiling_removed ? Cols * OpCols : Cols;
    }

    DEVICE_INLINE constexpr static int tiles() { return Tiles; }

    DEVICE_INLINE constexpr static int op_rows() {
        return op_tiling_removed ? 1 : OpRows;
    }

    DEVICE_INLINE constexpr static int op_cols() {
        return op_tiling_removed ? 1 : OpCols;
    }

    DEVICE_INLINE constexpr static int op_size() {
        return op_tiling_removed ? 1 : OpRows * OpCols;
    }

    DEVICE_INLINE constexpr static int tile_size() {
        return rows() * cols() * op_size();
    }

    DEVICE_INLINE constexpr static int size() { return Tiles * tile_size(); }

private:
    template <typename, typename, bool>
    friend struct Layout;

    static constexpr int _op_rows = OpRows;
    static constexpr int _op_cols = OpCols;
};

template <typename Stride_, typename Shape_, bool OpTilingRemoved = false>
struct Layout {
    using Stride = Stride_;
    using Shape = Shape_;
    static constexpr bool op_tiling_removed = OpTilingRemoved;

    DEVICE_INLINE constexpr static auto layout_as_2x2_op_tiled() {
        if constexpr (Shape::op_rows() == 2 && Shape::op_cols() == 2) {
            return Layout<Stride, Shape>{};
        } else {
            using NewShape =
                TShape<Shape::rows() / 2, Shape::cols(), Shape::tiles(),
                       Shape::op_rows() * 2, Shape::op_cols()>;
            using NewStride =
                TStride<Stride::row() * 2, Stride::col(), Stride::tile(),
                        Stride::op_row(), Stride::op_col()>;
            return flash_attn_lab::a100::k04::Layout<NewStride, NewShape>{};
        }
    }

    DEVICE_INLINE constexpr static auto layout_as_type2() {
        using NewStride =
            TStride<Stride::row() / 2, Stride::col(), Stride::tile() / 2,
                    Stride::op_row(), Stride::op_col()>;
        using NewShape =
            TShape<Shape::rows(), Shape::cols() / 2, Shape::tiles(),
                   Shape::op_rows(), Shape::op_cols()>;
        return flash_attn_lab::a100::k04::Layout<NewStride, NewShape>{};
    }

    DEVICE_INLINE constexpr static auto layout_with_op_tiling_removed() {
        using NewShape = TShape<Shape::rows(), Shape::cols(), Shape::tiles(),
                                Shape::op_rows(), Shape::op_cols(), true>;
        return flash_attn_lab::a100::k04::Layout<Stride, NewShape, true>{};
    }

    DEVICE_INLINE constexpr static auto tiled_layout_with_2_cols_per_tile() {
        using NewShape = TShape<Shape::rows(), 2, Shape::cols() / 2,
                                Shape::op_rows(), Shape::op_cols()>;
        using NewStride =
            TStride<Stride::row(), Stride::col() * 2, Stride::tile(),
                    Stride::op_row(), Stride::op_col()>;
        return flash_attn_lab::a100::k04::Layout<NewStride, NewShape>{};
    }

    DEVICE_INLINE constexpr static int crd2idx(int row, int col, int tile,
                                               int op_row = 0, int op_col = 0) {
        int _row = row;
        int _col = col;
        int _op_row = op_row;
        int _op_col = op_col;

        if constexpr (op_tiling_removed) {
            op_row = row % Shape::_op_rows;
            row = row / Shape::_op_rows;
            op_col = col % Shape::_op_cols;
            col = col / Shape::_op_cols;
        }

        auto offset = tile * Stride::tile() + row * Stride::row() +
                      col * Stride::col() + op_row * Stride::op_row() +
                      op_col * Stride::op_col();
        return offset;
    }
};

template <typename Shape>
constexpr auto row_major_stride() {
    static_assert(1 <= Shape::op_rows() && Shape::op_rows() <= 2);
    static_assert(1 <= Shape::op_cols() && Shape::op_cols() <= 2);

    constexpr int op_row_stride = 1;
    constexpr int op_col_stride = Shape::op_rows();

    return TStride<Shape::cols() * Shape::op_size(), 1, Shape::tile_size(),
                   op_row_stride, op_col_stride>{};
}

template <typename Shape, bool op_row_major>
constexpr auto stride_for_shape() {
    constexpr int tile_stride = Shape::tile_size();
    constexpr int row_stride = Shape::op_size() * Shape::cols();
    constexpr int col_stride = Shape::op_size();
    constexpr int op_row_stride =
        op_row_major ? (Shape::op_rows() == 1 ? row_stride : Shape::op_cols())
                     : 1;
    constexpr int op_col_stride = op_row_major ? 1 : Shape::op_rows();
    return TStride<row_stride, col_stride, tile_stride, op_row_stride,
                   op_col_stride>{};
}

}
