#pragma once

#include "utils.h"

namespace flash_attn_lab::paged::k04 {

template <int N, typename value_t>
struct Array {
    value_t regs[N];

    DEVICE_INLINE constexpr value_t *data() { return regs; }
    DEVICE_INLINE constexpr const value_t *data() const { return regs; }

    DEVICE_INLINE constexpr void fill(value_t val) {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            regs[i] = value_t(val);
        }
    }

    DEVICE_INLINE constexpr void zero() { fill(0); }

    DEVICE_INLINE constexpr value_t &operator[](int idx) { return regs[idx]; }
    DEVICE_INLINE constexpr value_t operator[](int idx) const {
        return regs[idx];
    }

    DEVICE_INLINE constexpr static int size() { return N; }
};

template <int N, typename value_t, int Alignment = 16>
struct __align__(Alignment) ArrayAligned : public Array<N, value_t> {};

}
