# FlashAttention CUDA Lab

CUDA attention kernels for NVIDIA Ampere, organized as a sequence of readable
implementations with configuration search and performance comparisons.
The four scenarios cover non-causal self-attention on RTX 3090 and A100,
causal self-attention, and attention over a paged KV cache.

Read the complete optimization sequences in [previous_kernels/](previous_kernels/README.md),
or use the final implementations in [kernels/](kernels/README.md).

## Supported features

| Scenario | Optimization target | Input layout | Features |
|---|---|---|---|
| [Standard](kernels/standard/README.md) | RTX 3090 | Q/K/V: `[B,H,N,D]` | Non-causal self-attention |
| [A100](kernels/a100/README.md) | A100 | Q/K/V: `[B,H,N,D]` | Non-causal self-attention |
| [Causal](kernels/causal/README.md) | RTX 3090 | Q/K/V: `[B,H,N,D]` | Causal self-attention |
| [Paged](kernels/paged/README.md) | RTX 3090 | Q: `[T,Hq,D]`; KV: `[P,S,Hkv,D]` | Variable lengths, causal prefill/decode, GQA/MQA, page sizes 16 and 256 |

Final kernels support FP16/BF16 input and output, FP32 accumulation, and
head dimensions 64 and 128. Only the forward pass is implemented; scale is
`1/sqrt(D)` and dropout is zero. Dense inputs must be contiguous, have identical
shapes, and have a positive sequence length divisible by the selected `Br` and `Bc`.
Paged attention reads an existing KV cache and supports partial tiles.

## Installation

Use Python 3.10+, CUDA-enabled PyTorch, the CUDA toolkit, Ninja, and a C++20
compiler. The kernels target Ampere (SM80/SM86). Install from a source checkout:

```bash
cd flash-attention-cuda-lab
python -m pip install -e . --no-build-isolation
python -m pip install -r requirements-dev.txt
```

Install a PyTorch build compatible with your CUDA toolkit first. Set `CUDA_HOME`
if the toolkit is outside its default location. CUDA extensions compile on first
use and are cached under `build/`. `MAX_JOBS=2` can limit build memory usage.

## Quick start

```python
import torch
from flash_attn_lab import forward
from helper import get_configs

cfg = get_configs("standard", dtype="fp16", head_dim=128)[0]
q, k, v = [torch.randn(1, 2, 512, 128, device="cuda", dtype=torch.float16)
           for _ in range(3)]
out = forward(cfg, q, k, v)
```

The first configuration is a runnable starting point. Search the available
configurations with the benchmark commands below to find a faster one for your
input. Use `dtype="bf16"` with `torch.bfloat16` tensors for BF16.

Select another final kernel by passing the same scenario to both calls:

```python
cfg = get_configs("a100", dtype="fp16", head_dim=128)[0]
out = forward(cfg, q, k, v, scenario="a100")

cfg = get_configs("causal", dtype="fp16", head_dim=128)[0]
out = forward(cfg, q, k, v, scenario="causal")
```

To run a numbered step, pass its local number to both configuration lookup and
execution, for example `get_configs("standard", version=4, dtype="fp16")` and
`forward(cfg, q, k, v, version=4)`. Omitting `version` selects the final kernel.
Optional `o=` reuses an output buffer, and `benchmark=True` returns
`(output, milliseconds)`.

See the [paged example and input contract](docs/paged_performance.md#usage)
for variable-length requests and a paged KV cache.

## Performance

The tables show performance relative to FlashAttention-2: higher is better.
All dense tables use `H=16`, `D=128`, and `B × seq_len = 16384`.
Each row selects the fastest configuration for each input shape and precision,
except standard steps 01–03, which use a fixed `Br=64`, `Bc=64`, four-warp configuration.

### Standard attention · RTX 3090

| Kernel step | FP16 · seq_len = 4096 | FP16 · harm. mean* | BF16 · seq_len = 4096 | BF16 · harm. mean* |
|---|---:|---:|---:|---:|
| [01 · Online softmax baseline](previous_kernels/standard/k01_online_softmax/README.md) | 49.67% | 49.42% | N/A | N/A |
| [02 · Shared-memory swizzling](previous_kernels/standard/k02_smem_swizzle/README.md) | 98.26% | 98.37% | N/A | N/A |
| [03 · K/V prefetching](previous_kernels/standard/k03_kv_prefetch/README.md) | 98.65% | 98.08% | N/A | N/A |
| [04 · Register pipeline, softmax optimization and auto-tuning](previous_kernels/standard/k04_register_pipeline/README.md) | 101.48% | 101.31% | 101.61% | 101.40% |
| [05 · Copy address calculation](previous_kernels/standard/k05_copy_addressing/README.md) | 101.49% | 101.36% | 101.51% | 101.52% |
| [06 · Register layout and MMA traversal](previous_kernels/standard/k06_mma_layout/README.md) | 100.79% | 100.96% | 101.22% | 100.93% |
| [07 · First KV tile specialization (final)](previous_kernels/standard/k07_first_tile_specialization/README.md) | 102.78% | 102.42% | 102.80% | 102.39% |

Step 04 combines the register pipeline, softmax optimization, and configuration
auto-tuning in one step. Subsequent steps also use their best configurations.
`N/A` means the precision is not registered for that step.

[Final kernel performance by sequence length](docs/standard_performance.md).

### Standard attention · A100

| Kernel step | FP16 · seq_len = 4096 | FP16 · harm. mean* | BF16 · seq_len = 4096 | BF16 · harm. mean* |
|---|---:|---:|---:|---:|
| [01 · Standard attention baseline](previous_kernels/a100/k01_baseline/README.md) | 91.12% | 92.23% | 91.11% | 92.27% |
| [02 · Global-memory addressing and normalization](previous_kernels/a100/k02_gmem_addressing/README.md) | 94.82% | 95.98% | 94.73% | 95.68% |
| [03 · Reverse KV traversal](previous_kernels/a100/k03_reverse_kv/README.md) | 97.96% | 99.35% | 99.24% | 99.37% |
| [04 · Output-store cache policy](previous_kernels/a100/k04_output_cache_policy/README.md) | 98.54% | 99.19% | 99.32% | 99.35% |
| [05 · Head-dimension tiling](previous_kernels/a100/k05_head_dim_tiling/README.md) | 97.73% | 99.66% | 98.73% | 99.77% |
| [06 · QK load scheduling](previous_kernels/a100/k06_qk_load_schedule/README.md) | 99.26% | 100.10% | 99.25% | 100.10% |
| [07 · QK row-first traversal (final)](previous_kernels/a100/k07_qk_row_traversal/README.md) | 100.34% | 101.04% | 100.30% | 101.04% |
| [08 · Bounded delayed softmax rescaling (additional)](previous_kernels/a100/k08_delayed_softmax/README.md) | 102.25% | 102.89% | 102.16% | 102.89% |

Step 01 is the final RTX 3090 standard implementation, evaluated on A100.
All A100 steps use their best configurations. Step 08 is an additional branch
from step 06; step 07 remains the default final kernel.

[Final kernel performance by sequence length](docs/a100_performance.md).

### Causal attention · RTX 3090

| Kernel step | FP16 · seq_len = 4096 | FP16 · harm. mean* | BF16 · seq_len = 4096 | BF16 · harm. mean* |
|---|---:|---:|---:|---:|
| [01 · Causal mask](previous_kernels/causal/k01_causal_mask/README.md) | 99.95% | 100.35% | 100.20% | 100.53% |
| [02 · Separate full and masked tiles (final)](previous_kernels/causal/k02_mask_boundary_split/README.md) | 102.26% | 102.07% | 101.49% | 101.93% |

[Final kernel performance by sequence length](docs/causal_performance.md).

### Paged attention · RTX 3090

#### Page size 16

| Kernel step | FP16 · kv_len = 4096 | FP16 · harm. mean* | BF16 · kv_len = 4096 | BF16 · harm. mean* |
|---|---:|---:|---:|---:|
| [01 · Paged KV baseline](previous_kernels/paged/k01_paged_kv/README.md) | 89.83% | 92.51% | 91.87% | 93.91% |
| [02 · Shared row addresses](previous_kernels/paged/k02_shared_row_address/README.md) | 92.35% | 92.66% | 91.65% | 91.75% |
| [03 · Cached page addresses](previous_kernels/paged/k03_page_address_cache/README.md) | N/A | N/A | N/A | N/A |
| [04 · Full-tile loading](previous_kernels/paged/k04_full_tile_loads/README.md) | 96.03% | 96.70% | 96.00% | 97.03% |
| [05 · K/V address reuse](previous_kernels/paged/k05_kv_address_reuse/README.md) | 95.86% | 96.66% | 95.57% | 96.61% |
| [06 · Page traversal and single-page specialization](previous_kernels/paged/k06_page_iteration/README.md) | 98.11% | 99.28% | 97.13% | 99.60% |
| [07 · CTA page loading and configuration search (final)](previous_kernels/paged/k07_page_pipeline/README.md) | 102.94% | 103.46% | 102.60% | 103.48% |

Step 03 registers page size 256 only (`N/A`).

#### Page size 256

| Kernel step | FP16 · kv_len = 4096 | FP16 · harm. mean* | BF16 · kv_len = 4096 | BF16 · harm. mean* |
|---|---:|---:|---:|---:|
| [01 · Paged KV baseline](previous_kernels/paged/k01_paged_kv/README.md) | 90.25% | 92.67% | 89.93% | 92.62% |
| [02 · Shared row addresses](previous_kernels/paged/k02_shared_row_address/README.md) | 96.02% | 96.18% | 94.72% | 95.27% |
| [03 · Cached page addresses](previous_kernels/paged/k03_page_address_cache/README.md) | 98.23% | 100.43% | 101.12% | 101.14% |
| [04 · Full-tile loading](previous_kernels/paged/k04_full_tile_loads/README.md) | 100.17% | 101.80% | 101.78% | 102.05% |
| [05 · K/V address reuse](previous_kernels/paged/k05_kv_address_reuse/README.md) | 103.24% | 102.15% | 100.11% | 102.33% |
| [06 · Page traversal and single-page specialization](previous_kernels/paged/k06_page_iteration/README.md) | 102.60% | 103.26% | 103.75% | 103.25% |
| [07 · CTA page loading and configuration search (final)](previous_kernels/paged/k07_page_pipeline/README.md) | 104.39% | 105.17% | 105.17% | 104.88% |

Both page sizes use `Hq=Hkv=8`, `D=128`. At `kv_len=4096`, four requests use
`q_len=[1024,2048,3072,4096]`. The total KV budget is 16384 tokens.
[Complete workload and final results for page sizes 16/256](docs/paged_performance.md).

*`harm. mean` is the harmonic mean of the per-length percentages, selecting each
length's best configuration independently. Dense lengths are
512, 1024, 2048, 4096, 8192, and 16384; paged lengths also include 128.

## Testing and configuration search

```bash
python -m pytest tests/ -q

# Standard step 04: enumerate the compiled configurations and rank their timings.
python benchmarks/attention.py --scenario standard --version k04 --configs all

# Final implementations, both precisions, selected lengths.
python benchmarks/attention.py --scenario standard --dtype both --seq-lens 512 4096
python benchmarks/attention.py --scenario a100 --dtype both --seq-lens 512 4096
python benchmarks/attention.py --scenario causal --dtype both --seq-lens 512 4096
python benchmarks/attention.py --scenario paged --dtype both --seq-lens 128 4096
```

Auto-tuning is part of the benchmark workflow: enumerate the legal compiled
configurations, time each candidate, and select the fastest for the current
shape and precision. The Python call uses the configuration you pass explicitly.
See [benchmarks/README.md](benchmarks/README.md) for the FlashAttention-2 baseline,
Nsight Compute profiling, and measurement options. Benchmark outputs are local
artifacts; this repository keeps the scripts and the performance tables.

## Project structure

```text
kernels/           Final kernels for the four scenarios
previous_kernels/  Complete retained optimization sequences, including final steps
flash_attn_lab/    Python interface and extension loading
helper/            Configuration lookup and input construction
benchmarks/        Performance measurement and configuration search scripts
tests/             Correctness tests and reference calculations
tools/             Source-generation and build utilities
docs/              Final performance tables and input details
```

## References

- [sonnyli/flash_attention_from_scratch](https://github.com/sonnyli/flash_attention_from_scratch) — CUDA optimization roadmap and README structure.
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — reference implementation and performance baseline.

## License

[MIT](LICENSE).
