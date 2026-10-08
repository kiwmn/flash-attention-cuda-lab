#pragma once

#include <type_traits>

#include "utils.h"
#include "layout.cuh"

namespace flash_attn_lab::a100::k07 {

template <typename value_t_, typename Layout_,
          typename storage_t_ = decltype(value_storage_type<value_t_>())>
struct TensorView {
    static_assert(is_supported_mma_input_type<value_t_>() ||
                      is_supported_mma_output_type<value_t_>(),
                  "value_t must be half, nv_bfloat16, or float");
    using value_t = value_t_;
    using storage_t = storage_t_;
    using Layout = Layout_;
    using Shape = typename Layout::Shape;
    using Stride = typename Layout::Stride;

    storage_t *data;

    DEVICE_INLINE constexpr TensorView(storage_t *data) : data(data) {}

    template <typename NewLayout>
    DEVICE_INLINE constexpr TensorView<value_t, NewLayout> with_layout() {
        static_assert(NewLayout::Shape::size() == Shape::size(),
                      "Shape size mismatch");
        return TensorView<value_t, NewLayout>(data);
    }

    DEVICE_INLINE constexpr auto as_type2() {
        if constexpr (is_supported_mma_input_type<value_t>()) {
            using storage_t2 = std::conditional_t<std::is_same_v<value_t, half>,
                                                  half2, nv_bfloat162>;
            return as_storage_type<storage_t2, Layout>();
        } else if constexpr (std::is_same_v<value_t, float>) {
            using NewLayout = decltype(Layout::layout_as_type2());

            return as_storage_type<float2, NewLayout>();
        }
    }

    DEVICE_INLINE constexpr auto with_op_tiling_removed() {
        if constexpr (Layout::op_tiling_removed) {
            return *this;
        } else {
            using NewLayout = decltype(Layout::layout_with_op_tiling_removed());
            return with_layout<NewLayout>();
        }
    }

    DEVICE_INLINE constexpr storage_t &
    operator()(int row, int col, int tile = 0, int op_row = 0, int op_col = 0) {
        return data[Layout::crd2idx(row, col, tile, op_row, op_col)];
    }
    DEVICE_INLINE constexpr storage_t operator()(int row, int col, int tile = 0,
                                                 int op_row = 0,
                                                 int op_col = 0) const {
        return data[Layout::crd2idx(row, col, tile, op_row, op_col)];
    }

private:
    template <typename T, typename NewLayout>
    DEVICE_INLINE constexpr auto as_storage_type() {
        return TensorView<value_t, NewLayout, T>(reinterpret_cast<T *>(data));
    }
};

}
