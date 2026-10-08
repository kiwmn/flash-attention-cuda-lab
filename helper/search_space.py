"""Configuration combinations used to generate the tunable CUDA registries."""

from dataclasses import asdict, replace
from itertools import product

from .kernel_configs import DType, ForwardKernelConfig, PagedForwardKernelConfig


def dense_search_space(head_dim: int = 128) -> list[ForwardKernelConfig]:
    """The 91 FP16/BF16 configurations compiled by dense steps 4 onward."""
    if head_dim not in (64, 128):
        raise ValueError("head_dim must be 64 or 128")
    base = ForwardKernelConfig(DType.FP16, head_dim, 64, 64, 4, True,
                               False, False, 0, 0, 0, False, False)
    prefetched = replace(base, swizzled=True, prefetch_kv_tiles=True)
    partial = replace(prefetched, Q_mma_load_K_tiles=2,
                      K_mma_load_K_tiles=2, V_mma_load_K_tiles=2)
    buffered = replace(partial, mma_double_buffer_loads=True)
    fixed = [base, replace(base, swizzled=True), prefetched, partial,
             buffered, replace(buffered, optimized_softmax=True)]
    configs = {replace(cfg, dtype=dtype) for cfg in fixed for dtype in DType}
    configs.add(replace(partial, Q_mma_load_K_tiles=0, optimized_softmax=True))
    configs.add(replace(prefetched, swizzled=False, K_mma_load_K_tiles=2,
                        optimized_softmax=True))
    for dtype, br, bc, q, k, v, buffer, softmax in product(
        DType, (64, 128), (32, 64), (0, 2), (0, 2), (0, 2),
        (False, True), (False, True),
    ):
        if q != k and q != 0:
            continue
        if br == 64 and ((bc == 32 and q == 0) or (bc == 64 and q != 0)):
            continue
        if br == 128 and q == 0:
            continue
        configs.add(ForwardKernelConfig(dtype, head_dim, br, bc, 4, True,
                                       True, True, q, k, v, buffer, softmax))
    return sorted(configs)


def paged_search_space(head_dim: int = 128) -> list[PagedForwardKernelConfig]:
    """Tile, load, buffering and softmax combinations for the final paged kernel."""
    if head_dim not in (64, 128):
        raise ValueError("head_dim must be 64 or 128")
    configs = []
    for page_size, dtype in product((16, 256), DType):
        entries = [
            ForwardKernelConfig(dtype, head_dim, 64, 64, 4, True, False, False,
                                0, 0, 0, False, False),
            ForwardKernelConfig(dtype, head_dim, 64, 64, 4, True, False, True,
                                0, 0, 0, False, False),
            ForwardKernelConfig(dtype, head_dim, 64, 32, 4, True, True, True,
                                2, 2, 0, False, True),
            ForwardKernelConfig(dtype, head_dim, 64, 64, 4, True, True, True,
                                2, 2, 2, True, True),
            ForwardKernelConfig(dtype, head_dim, 128, 32, 4, True, True, True,
                                2, 2, 0, False, True),
        ]
        for (br, bc), prefetch, (q, k), v, buffer, softmax in product(
            ((64, 32), (64, 64), (128, 32), (128, 64)),
            (False, True), ((0, 0), (0, 2), (2, 2)), (0, 2),
            (False, True), (False, True),
        ):
            if q == k == v == 0 and buffer:
                continue
            entries.append(ForwardKernelConfig(dtype, head_dim, br, bc, 4, True,
                                               prefetch, True, q, k, v, buffer, softmax))
        configs.extend(PagedForwardKernelConfig(**asdict(cfg), page_size=page_size)
                       for cfg in dict.fromkeys(entries))
    return configs
