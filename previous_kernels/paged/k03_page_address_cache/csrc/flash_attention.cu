#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include <limits>
#include <utility>
#include <vector>

#include "utils.h"
#include "kernel_traits.cuh"
#include "flash_kernels.cuh"
#include "inference_validation.h"

using namespace flash_attn_lab::paged::k03;

ForwardKernelConfig py_to_cpp_kernel_config(const py::object &py_cfg) {
    return ForwardKernelConfig{
        py::cast<torch::ScalarType>(
            py_cfg.attr("dtype").attr("to_torch_dtype")()),
        py::cast<int>(py_cfg.attr("head_dim")),
        py::cast<int>(py_cfg.attr("Br")),
        py::cast<int>(py_cfg.attr("Bc")),
        py::cast<int>(py_cfg.attr("warp_num")),
        py::cast<bool>(py_cfg.attr("async_copy")),
        py::cast<bool>(py_cfg.attr("prefetch_kv_tiles")),
        py::cast<bool>(py_cfg.attr("swizzled")),
        py::cast<int>(py_cfg.attr("Q_mma_load_K_tiles")),
        py::cast<int>(py_cfg.attr("K_mma_load_K_tiles")),
        py::cast<int>(py_cfg.attr("V_mma_load_K_tiles")),
        py::cast<bool>(py_cfg.attr("mma_double_buffer_loads")),
        py::cast<bool>(py_cfg.attr("optimized_softmax")),
        true,
        py::hasattr(py_cfg, "page_size")
            ? py::cast<int>(py_cfg.attr("page_size"))
            : 256};
}

std::tuple<at::Tensor, float> launch_forward(const ForwardKernelConfig &cfg,
                                             const ForwardKernelArgs &args,
                                             const at::Tensor &out,
                                             dim3 gridDim, bool benchmark) {
    dim3 blockDim(cfg.warp_num * WARP_SIZE);
    float runtime = 0.0f;
    cudaEvent_t start, stop;

    const int smem_bytes = cfg.smem_bytes();
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (benchmark) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, stream);
    }

    forward_kernels.at(cfg)<<<gridDim, blockDim, smem_bytes, stream>>>(args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (benchmark) {
        cudaEventRecord(stop, stream);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runtime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return std::make_tuple(out, runtime);
}

std::tuple<at::Tensor, float> flash_attention_forward(
    const py::object &py_cfg, const at::Tensor &q, const at::Tensor &k,
    const at::Tensor &v, const at::Tensor &q_starts, const at::Tensor &seq_lens,
    const at::Tensor &block_table, int max_q, int max_kv,
    std::optional<at::Tensor> out, bool benchmark, bool validate_metadata) {
    TORCH_CHECK(q.is_cuda(), "Q must be CUDA");
    at::cuda::CUDAGuard guard(q.device());
    TORCH_CHECK(cuda_device_compute_capability(q.device().index()) >= 80,
                "Flash Attention requires SM_80 or higher");
    TORCH_CHECK(q.scalar_type() == at::kHalf ||
                    q.scalar_type() == at::kBFloat16,
                "Only fp16 and bf16 are supported");
    check_inference_tensor(q, q, 3);
    check_inference_tensor(k, q, 4);
    check_inference_tensor(v, q, 4);
    TORCH_CHECK(k.sizes() == v.sizes(), "K/V shapes must match");
    const int64_t hq = q.size(1), hkv = k.size(2);
    TORCH_CHECK(hq > 0 && hkv > 0 && hq % hkv == 0,
                "Require Hq % Hkv == 0 and positive head counts");
    TORCH_CHECK(q_starts.dim() == 1 && q_starts.numel() >= 1,
                "query_start_loc must have B+1 entries");
    const int64_t batch = q_starts.numel() - 1;
    check_metadata(q_starts, q, batch + 1);
    check_metadata(seq_lens, q, batch);
    constexpr int length_limit = std::numeric_limits<int>::max() - 128;
    TORCH_CHECK(batch <= 65535 && hq <= 65535 && max_q >= 0 &&
                    max_kv >= max_q && max_kv <= length_limit &&
                    q.size(0) <= length_limit && k.size(0) <= length_limit &&
                    (batch == 0
                         ? q.size(0) == 0
                         : max_q > 0 && q.size(0) >= batch && k.size(0) > 0),
                "Invalid capacity, max lengths or CUDA grid limits");
    const int64_t page_size = k.size(1);
    TORCH_CHECK(page_size == 16 || page_size == 32 || page_size == 64 ||
                    page_size == 256,
                "Supported page sizes are 16, 32, 64, 256");
    TORCH_CHECK(block_table.is_cuda() && block_table.device() == q.device() &&
                    block_table.scalar_type() == at::kInt &&
                    block_table.dim() == 2 && block_table.size(0) == batch &&
                    block_table.stride(1) == 1 &&
                    block_table.stride(0) >= block_table.size(1),
                "block_table must be CUDA int32 [B,max_blocks] with contiguous "
                "columns");
    TORCH_CHECK(block_table.size(1) >=
                    (int64_t(max_kv) + page_size - 1) / page_size,
                "Insufficient block_table capacity for max_kv_len");
    const auto cfg = py_to_cpp_kernel_config(py_cfg);
    TORCH_CHECK(page_size == cfg.page_size,
                "Cache page size must match compile-time kernel page_size");
    TORCH_CHECK(forward_kernels.contains(cfg) && cfg.dtype == q.scalar_type() &&
                    cfg.head_dim == q.size(-1),
                "Unsupported kernel configuration or dtype");
    at::Tensor result =
        out.has_value() ? *out : torch::empty(q.sizes(), q.options());
    check_inference_tensor(result, q, 3);
    TORCH_CHECK(result.sizes() == q.sizes(), "Output shape must match Q");
    for (const auto &input : {q, k, v, q_starts, seq_lens, block_table}) {
        check_output_disjoint(result, input);
    }
    if (validate_metadata) {
        validate_paged_values(q_starts, seq_lens, block_table, q.size(0),
                              k.size(0), page_size, max_q, max_kv);
    }
    if (batch == 0) {
        return {result, 0.0f};
    }
    auto strides = [](const at::Tensor &t) {
        return ForwardKernelArgs::Strides{t.stride(1), t.stride(0)};
    };
    ForwardKernelArgs args{};
    args.q = q.data_ptr();
    args.k = k.data_ptr();
    args.v = v.data_ptr();
    args.o = result.data_ptr();
    args.q_stride = strides(q);
    args.o_stride = strides(result);
    args.query_start_loc = q_starts.data_ptr<int>();
    args.seq_lens = seq_lens.data_ptr<int>();
    args.heads_per_kv = hq / hkv;

    args.k_stride = {k.stride(0), k.stride(2), k.stride(1)};
    args.v_stride = {v.stride(0), v.stride(2), v.stride(1)};
    args.block_table = block_table.data_ptr<int>();
    args.table_stride = block_table.stride(0);
    dim3 grid((max_q + cfg.Br - 1) / cfg.Br, hq, batch);
    return launch_forward(cfg, args, result, grid, benchmark);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward, py::arg("kernel_cfg"),
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("query_start_loc"),
          py::arg("seq_lens"), py::arg("block_table"), py::arg("max_query_len"),
          py::arg("max_kv_len"), py::arg("out") = py::none(),
          py::arg("benchmark") = false, py::arg("validate_metadata") = false);

    for (const auto &[cfg, kernel] : forward_kernels) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            cfg.smem_bytes()));
    }
}
