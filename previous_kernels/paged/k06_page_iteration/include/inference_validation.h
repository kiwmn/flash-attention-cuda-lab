#pragma once

// Validate tensor layout on the host; optional metadata checks synchronize
// device values.

#include <algorithm>
#include <limits>
#include <vector>
#include <torch/torch.h>

namespace flash_attn_lab::paged::k06 {

inline void check_inference_tensor(const at::Tensor &tensor,
                                   const at::Tensor &q, int ndim) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == q.device(),
                "All tensors must be on the same CUDA device");
    TORCH_CHECK(tensor.scalar_type() == q.scalar_type(), "Dtype mismatch");
    TORCH_CHECK(tensor.dim() == ndim &&
                    (tensor.size(-1) == 64 || tensor.size(-1) == 128),
                "Expected D=64 or D=128 and rank ", ndim);
    TORCH_CHECK(tensor.size(-1) == q.size(-1),
                "Q/K/V/output head dimensions must match");
    TORCH_CHECK(tensor.stride(-1) == 1, "Last dimension must be contiguous");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
                "Tensor data pointer must be 16-byte aligned");
    std::vector<std::pair<int64_t, int64_t>> dims;
    for (int axis = 0; axis < ndim - 1; ++axis) {
        TORCH_CHECK(tensor.stride(axis) > 0 && tensor.stride(axis) % 8 == 0,
                    "Outer strides must be positive multiples of 8 elements");
        if (tensor.size(axis) > 1) {
            dims.emplace_back(tensor.stride(axis), tensor.size(axis));
        }
    }
    std::sort(dims.begin(), dims.end());
    int64_t span = tensor.size(-1);
    for (auto [stride, size] : dims) {
        TORCH_CHECK(stride >= span, "Unsupported overlapping tensor view");
        span += (size - 1) * stride;
    }
}

inline void check_output_disjoint(const at::Tensor &out,
                                  const at::Tensor &input) {
    if (out.numel() == 0 || input.numel() == 0) {
        return;
    }
    auto interval_end = [](const at::Tensor &t) {
        int64_t last = 0;
        for (int axis = 0; axis < t.dim(); ++axis) {
            last += (t.size(axis) - 1) * t.stride(axis);
        }
        return reinterpret_cast<uintptr_t>(t.data_ptr()) +
               (last + 1) * t.element_size();
    };
    TORCH_CHECK(
        interval_end(out) <= reinterpret_cast<uintptr_t>(input.data_ptr()) ||
            interval_end(input) <= reinterpret_cast<uintptr_t>(out.data_ptr()),
        "Output must not overlap input memory location");
}

inline void check_metadata(const at::Tensor &tensor, const at::Tensor &q,
                           int64_t length) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == q.device() &&
                    tensor.scalar_type() == at::kInt && tensor.dim() == 1 &&
                    tensor.is_contiguous() && tensor.numel() == length,
                "Metadata must be contiguous CUDA int32 with expected length");
}

inline void validate_paged_values(const at::Tensor &q_starts,
                                  const at::Tensor &lengths,
                                  const at::Tensor &table, int64_t total_q,
                                  int64_t physical_pages, int page_size,
                                  int max_q, int max_kv) {
    auto qc = q_starts.cpu(), lc = lengths.cpu();
    auto tc = table.cpu().contiguous();
    const int *qs = qc.data_ptr<int>(), *ls = lc.data_ptr<int>();
    const int *pages = tc.data_ptr<int>();
    const int64_t batch = lengths.numel();
    TORCH_CHECK(qs[0] == 0 && qs[batch] == total_q,
                "Prefix sums must begin at 0 and end at tensor token count");
    for (int64_t i = 0; i < batch; ++i) {
        const int64_t nq = int64_t(qs[i + 1]) - qs[i];
        const int64_t nk = ls[i];
        TORCH_CHECK(nq > 0 && nk >= nq && nq <= max_q && nk <= max_kv,
                    "Invalid request lengths or underestimated max lengths");
        const int64_t needed = (nk + page_size - 1) / page_size;
        TORCH_CHECK(needed <= table.size(1),
                    "Insufficient block_table capacity");
        for (int64_t page = 0; page < needed; ++page) {
            const int physical = pages[i * table.size(1) + page];
            TORCH_CHECK(physical >= 0 && physical < physical_pages,
                        "Invalid physical page in block_table");
        }
    }
}

}
