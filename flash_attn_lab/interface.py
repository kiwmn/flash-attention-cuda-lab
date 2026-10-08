"""Python interfaces and lazy compilation for the four attention scenarios."""

from __future__ import annotations

from functools import lru_cache
import os
from threading import Lock

from helper.kernel_configs import ROOT, kernel_directory


_BUILD_LOCK = Lock()


def get_build_cflags() -> list[str]:
    return ["-O3", "-std=c++20"]


def get_build_cuda_cflags() -> list[str]:
    # Ampere SM80 code also runs on SM86; keep one code-generation target.
    flags = [
        "-lineinfo", "-O3", "-std=c++20",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
        "-gencode=arch=compute_80,code=sm_80", "-diag-suppress=177",
    ]
    if os.environ.get("FLASH_ATTN_LAB_PTXAS_VERBOSE") == "1":
        flags.append("-Xptxas=-v")
    return flags


def get_build_sources(scenario: str, version: int | None = None) -> list[str]:
    directory = kernel_directory(scenario, version)
    return [str(directory / "csrc" / "flash_attention.cu")] + [
        str(path) for path in sorted((directory / "csrc" / "instances").glob("*.cu"))
    ]


@lru_cache(maxsize=None)
def load_extension(scenario: str, version: int | None = None):
    """Compile a kernel on first use; PyTorch reuses its cached build thereafter."""
    from torch.utils.cpp_extension import load

    directory = kernel_directory(scenario, version)
    suffix = "final" if version is None else f"k{version:02d}"
    name = f"flash_attention_cuda_lab_{scenario}_{suffix}"
    build_directory = ROOT / "build" / name
    build_directory.mkdir(parents=True, exist_ok=True)
    # PyTorch can add architecture flags independently of extra_cuda_cflags.
    with _BUILD_LOCK:
        previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
        try:
            return load(
                name=name,
                build_directory=str(build_directory),
                sources=get_build_sources(scenario, version),
                extra_include_paths=[str(directory / "include")],
                extra_cflags=get_build_cflags(),
                extra_cuda_cflags=get_build_cuda_cflags(),
                extra_ldflags=["-Wl,--no-as-needed", "-lcuda"],
                with_cuda=True,
                verbose=os.environ.get("FLASH_ATTN_LAB_VERBOSE_BUILD") == "1",
            )
        finally:
            if previous_arch is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch


def forward(
    cfg, q, k, v, *, scenario: str = "standard", version: int | None = None,
    o=None, benchmark: bool = False,
):
    """Compute self-attention for contiguous FP16/BF16 ``[B, H, N, D]`` tensors.

    ``scenario`` is ``standard``, ``a100``, or ``causal``. ``version=None`` uses
    the final kernel; integers select the numbered steps in that scenario.
    ``D`` is 64 or 128, and ``N`` must be divisible by both ``cfg.Br`` and
    ``cfg.Bc``. The result has the same shape and dtype as ``q``. Pass ``o`` to
    reuse an output buffer. ``benchmark=True`` returns ``(output, milliseconds)``.
    Only the forward pass is implemented.
    """
    import torch

    if scenario not in ("standard", "a100", "causal"):
        raise ValueError("forward scenario must be standard, a100, or causal")
    tensors = (q, k, v) if o is None else (q, k, v, o)
    for tensor in tensors:
        if tensor.ndim != 4 or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("dense attention requires contiguous CUDA [B, H, N, D] tensors")
        if tensor.device != q.device or tensor.dtype != q.dtype or tensor.shape != q.shape:
            raise ValueError("Q, K, V and output must have the same shape, dtype, and device")
        if tensor.requires_grad:
            raise ValueError("only the forward pass is implemented; inputs must not require gradients")
    if q.dtype not in (torch.float16, torch.bfloat16) or min(q.shape) <= 0:
        raise ValueError("inputs must be nonempty FP16 or BF16 tensors")
    if o is not None:
        output_start = o.data_ptr()
        output_end = output_start + o.numel() * o.element_size()
        if any(
            output_start < tensor.data_ptr() + tensor.numel() * tensor.element_size()
            and tensor.data_ptr() < output_end
            for tensor in (q, k, v)
        ):
            raise ValueError("output must not overlap Q, K, or V")
    with torch.cuda.device(q.device):
        out, milliseconds = load_extension(scenario, version).forward(cfg, q, k, v, o, benchmark)
    return (out, milliseconds) if benchmark else out


def paged_attention(
    cfg, q, k_cache, v_cache, query_start_loc, seq_lens, block_table, *,
    max_query_len: int, max_kv_len: int, version: int | None = None, o=None,
    benchmark: bool = False, validate_metadata: bool = False,
):
    """Compute causal attention over a read-only paged KV cache.

    Q/output use ``[total_queries, query_heads, D]``; K/V use
    ``[physical_pages, page_size, kv_heads, D]``. Metadata are CUDA int32 tensors:
    ``query_start_loc`` is the query prefix sum, ``seq_lens`` stores each
    request's KV length, and ``block_table`` maps logical to physical pages.
    Query positions align to the end of the corresponding KV sequence.
    ``max_query_len`` and ``max_kv_len`` are host upper bounds on request lengths.
    Set ``validate_metadata=True`` to synchronize and check metadata values.
    See ``docs/paged_performance.md`` for layout and page-size constraints.
    ``benchmark=True`` returns ``(output, milliseconds)``.
    """
    import torch

    if not q.is_cuda:
        raise ValueError("paged attention requires CUDA tensors")
    if any(tensor.requires_grad for tensor in (q, k_cache, v_cache)):
        raise ValueError("only the forward pass is implemented; inputs must not require gradients")
    with torch.cuda.device(q.device):
        out, milliseconds = load_extension("paged", version).forward(
            cfg, q, k_cache, v_cache, query_start_loc, seq_lens, block_table,
            max_query_len, max_kv_len, o, benchmark, validate_metadata,
        )
    return (out, milliseconds) if benchmark else out
