#pragma once
#include "utils.h"
template <typename value_t>
__device__ void
mma_m16n8k16_f32_accum(float &d1, float &d2, float &d3, float &d4,
                       uint32_t const &a1, uint32_t const &a2,
                       uint32_t const &a3, uint32_t const &a4,
                       uint32_t const &b1, uint32_t const &b2, float const &c1,
                       float const &c2, float const &c3, float const &c4) {
    static_assert(
        std::is_same_v<value_t, half> || std::is_same_v<value_t, nv_bfloat16>,
        "mma_m16n8k16_f32_accum only supports half and bfloat16 types");
    if constexpr ((std::is_same_v<value_t, nv_bfloat16>)) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     " {%0, %1, %2, %3}, "
                     " {%4, %5, %6, %7}, "
                     " {%8, %9}, "
                     " {%10, %11, %12, %13};"
                     : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                     : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(b1), "r"(b2),
                       "f"(c1), "f"(c2), "f"(c3), "f"(c4));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     " {%0, %1, %2, %3}, "
                     " {%4, %5, %6, %7}, "
                     " {%8, %9}, "
                     " {%10, %11, %12, %13};"
                     : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                     : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(b1), "r"(b2),
                       "f"(c1), "f"(c2), "f"(c3), "f"(c4));
    }
}

DEVICE_INLINE void cp_async_commit() { asm volatile("cp.async.commit_group;"); }

template <int ngroups>
DEVICE_INLINE void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(ngroups));
}

template <int size, typename T>
__device__ void cp_async(T *smem_dst, const T *gmem_src) {
    static_assert(size == 16);
    uint32_t smem_ptr = __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                 :
                 : "r"(smem_ptr), "l"(gmem_src), "n"(size));
}

template <typename T>
__device__ void ldmatrix_x4(uint32_t &d1, uint32_t &d2, uint32_t &d3,
                            uint32_t &d4, const T *smem_src) {
    uint32_t smem_ptr = __cvta_generic_to_shared(smem_src);

    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                 " {%0, %1, %2, %3}, "
                 " [%4];"
                 : "=r"(d1), "=r"(d2), "=r"(d3), "=r"(d4)
                 : "r"(smem_ptr));
}

template <typename T>
__device__ void ldmatrix_x4_transpose(uint32_t &d1, uint32_t &d2, uint32_t &d3,
                                      uint32_t &d4, const T *smem_src) {
    uint32_t smem_ptr = __cvta_generic_to_shared(smem_src);

    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 "
                 " {%0, %1, %2, %3}, "
                 " [%4];"
                 : "=r"(d1), "=r"(d2), "=r"(d3), "=r"(d4)
                 : "r"(smem_ptr));
}
