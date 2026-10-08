"""Configuration objects and the exact configurations compiled by each kernel."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("standard", "a100", "causal", "paged")


class DType(IntEnum):
    FP16 = 5
    BF16 = 15

    def to_cpp_str(self) -> str:
        return "torch::kFloat16" if self == DType.FP16 else "torch::kBFloat16"

    def to_torch_dtype(self):
        import torch

        return torch.float16 if self == DType.FP16 else torch.bfloat16


@dataclass(frozen=True, order=True)
class ForwardKernelConfig:
    dtype: DType
    head_dim: int
    Br: int
    Bc: int
    warp_num: int
    async_copy: bool
    prefetch_kv_tiles: bool
    swizzled: bool
    Q_mma_load_K_tiles: int
    K_mma_load_K_tiles: int
    V_mma_load_K_tiles: int
    mma_double_buffer_loads: bool
    optimized_softmax: bool

    def short_form(self, include_d_head=True, include_tup=True) -> str:
        dimension = f"{self.head_dim}, " if include_d_head else ""
        prefix = (
            f"({self.dtype.name}, {dimension}{self.Br}, {self.Bc}, {self.warp_num}): "
            if include_tup else ""
        )
        features = [
            name for enabled, name in (
                (self.async_copy, "async"),
                (self.prefetch_kv_tiles, "prefetch_kv"),
                (self.swizzled, "swizzled"),
            ) if enabled
        ]
        features.append(
            f"load_{self.Q_mma_load_K_tiles}_{self.K_mma_load_K_tiles}_"
            f"{self.V_mma_load_K_tiles}_tiles"
        )
        if self.mma_double_buffer_loads:
            features.append("buffer")
        if self.optimized_softmax:
            features.append("opt_softmax")
        return prefix + "+".join(features)

    def __str__(self) -> str:
        return self.short_form()


@dataclass(frozen=True, order=True)
class PagedForwardKernelConfig(ForwardKernelConfig):
    page_size: int = 256

    def short_form(self, include_d_head=True, include_tup=True) -> str:
        return super().short_form(include_d_head, include_tup) + f"+page_{self.page_size}"


def _check_scenario(scenario: str) -> None:
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")


def list_versions(scenario: str) -> dict[int, str]:
    """Map each local step number to its directory name, in reading order."""
    _check_scenario(scenario)
    return {
        int(path.name[1:3]): path.name
        for path in sorted((ROOT / "previous_kernels" / scenario).glob("k[0-9][0-9]_*"))
        if path.is_dir()
    }


def kernel_directory(scenario: str, version: int | None = None) -> Path:
    """Resolve the final kernel, or a numbered step in the selected scenario."""
    _check_scenario(scenario)
    if version is None:
        path = ROOT / "kernels" / scenario
    else:
        if isinstance(version, bool) or not isinstance(version, int):
            raise TypeError("version must be a local integer step number or None")
        versions = list_versions(scenario)
        if version not in versions:
            raise ValueError(f"unknown {scenario} version {version}; available: {list(versions)}")
        path = ROOT / "previous_kernels" / scenario / versions[version]
    if not (path / "csrc" / "flash_attention.cu").is_file():
        raise FileNotFoundError(
            f"kernel sources not found at {path}; install this source checkout with pip install -e ."
        )
    return path


def _dtype(value) -> DType:
    if isinstance(value, DType):
        return value
    names = {
        "fp16": DType.FP16, "float16": DType.FP16, "torch.float16": DType.FP16,
        "bf16": DType.BF16, "bfloat16": DType.BF16, "torch.bfloat16": DType.BF16,
    }
    try:
        return names[str(value).lower()]
    except KeyError:
        raise ValueError(f"unsupported dtype: {value!r}") from None


@lru_cache(maxsize=None)
def _registered_configs(scenario: str, version: int | None) -> tuple[ForwardKernelConfig, ...]:
    directory = kernel_directory(scenario, version)
    sources = [directory / "include" / "flash_kernels.cuh"]
    sources.extend(sorted((directory / "csrc" / "instances").glob("*.cu")))
    configs = {}
    # Read the actual C++ registry so Python cannot select an uncompiled combination.
    pattern = re.compile(r"\bForwardKernelConfig\s*(?:\w+\s*)?\{([^{}]+)\}")
    for source in sources:
        code = re.sub(r"//[^\n]*|/\*.*?\*/", "", source.read_text(), flags=re.S)
        for entry in pattern.findall(code):
            fields = [field.strip() for field in entry.split(",")]
            if not fields[0].startswith("torch::k"):
                continue
            dtype = {"torch::kFloat16": DType.FP16, "torch::kBFloat16": DType.BF16}[fields[0]]
            values = [dtype] + [
                {"true": True, "false": False}[field] if field in ("true", "false") else int(field)
                for field in fields[1:]
            ]
            if len(values) == 5:
                # The first three steps have a fixed feature set and a five-field registry.
                values += [True, version == 3, version in (2, 3), 0, 0, 0, False, False]
            if len(values) not in (13, 14, 15):
                raise ValueError(f"unsupported registry entry in {source}: {entry}")
            if len(values) == 15:
                cfg = PagedForwardKernelConfig(*values[:13], page_size=values[14])
            else:
                cfg = ForwardKernelConfig(*values[:13])
            configs[cfg] = None
    if not configs:
        raise RuntimeError(f"no configurations found in {directory}")
    return tuple(configs)


def get_configs(
    scenario: str,
    version: int | None = None,
    dtype=None,
    head_dim: int = 128,
    page_size: int | None = None,
) -> list[ForwardKernelConfig]:
    """Return compiled configurations, optionally filtered by precision and page size.

    ``version=None`` selects the final implementation. Standard steps 1–3 have
    fixed FP16 configurations. Starting at step 4, benchmark all returned
    configurations to select the fastest for the input being measured.
    An empty list means that precision, dimension, or page size is unavailable.
    """
    if head_dim not in (64, 128):
        raise ValueError("head_dim must be 64 or 128")
    if page_size is not None and scenario != "paged":
        raise ValueError("page_size only applies to the paged scenario")
    if page_size is not None and page_size not in (16, 32, 64, 256):
        raise ValueError("page_size must be 16, 32, 64, or 256")
    precision = _dtype(dtype) if dtype is not None else None
    return [
        cfg for cfg in _registered_configs(scenario, version)
        if cfg.head_dim == head_dim
        and (precision is None or cfg.dtype == precision)
        and (page_size is None or getattr(cfg, "page_size", page_size) == page_size)
    ]
