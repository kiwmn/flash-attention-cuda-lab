# Paged attention · RTX 3090

[Project overview](../README.md) · [Final implementation](../kernels/paged/README.md)

## Workload

These tables use the final kernel with `Hq=Hkv=8`, `D=128`, FP16/BF16, and
right-aligned causal attention. Each request has the same `kv_len=N` within a
measurement, and the number of requests is `B=16384/N`. For request index
`i=1..B`, `q_len=ceil(N*i/B)` and the existing history length is `N-q_len`.
This produces the following query lengths:

| KV length | Requests | Query lengths | Total query tokens |
|---:|---:|---|---:|
| 128 | 128 | 1, 2, …, 128 | 8256 |
| 512 | 32 | 16, 32, …, 512 | 8448 |
| 1024 | 16 | 64, 128, …, 1024 | 8704 |
| 2048 | 8 | 256, 512, …, 2048 | 9216 |
| 4096 | 4 | 1024, 2048, 3072, 4096 | 10240 |
| 8192 | 2 | 4096, 8192 | 12288 |
| 16384 | 1 | 16384 | 16384 |

The workload mixes queries with cached history and full prefill; the shortest
case also includes a one-token decode request. It is not a decode-only benchmark.
K/V are already present in the cache. Cache writes and request scheduling are
outside the measurement.

The reference is FlashAttention-2 paged attention with page size 256 and
`num_splits=1`. The custom kernel is measured separately at page sizes 256 and
16, with identical logical inputs. The page-size-16 table also uses the
page-size-256 FA2 baseline. Percentages are `100 × FA2 time / kernel time`.
Each length and precision uses its fastest measured configuration from 176
candidates. The harmonic mean uses these per-length percentages. Nsight Compute
uses base clocks and cache flushing; each duration averages three captures.

## FP16

### Page size 16

| KV length | Requests | Relative to FA2 | Br × Bc | Q/K/V load | Prefetch | RF buffer | Optimized softmax |
|---:|---:|---:|---|---|---|---|---|
| 128 | 128 | 107.53% | 64 × 32 | 2/2/2 | Yes | No | Yes |
| 512 | 32 | 104.04% | 64 × 32 | 2/2/2 | Yes | Yes | No |
| 1024 | 16 | 101.61% | 64 × 64 | 2/2/2 | Yes | Yes | Yes |
| 2048 | 8 | 103.40% | 64 × 64 | 0/2/2 | Yes | No | No |
| 4096 | 4 | 102.94% | 64 × 64 | 0/2/0 | Yes | Yes | No |
| 8192 | 2 | 103.31% | 64 × 64 | 0/2/2 | Yes | No | No |
| 16384 | 1 | 101.60% | 64 × 64 | 0/2/2 | Yes | Yes | No |

Harmonic mean: **103.46%**.

### Page size 256

| KV length | Requests | Relative to FA2 | Br × Bc | Q/K/V load | Prefetch | RF buffer | Optimized softmax |
|---:|---:|---:|---|---|---|---|---|
| 128 | 128 | 110.33% | 64 × 32 | 2/2/2 | Yes | Yes | Yes |
| 512 | 32 | 102.40% | 64 × 64 | 0/2/2 | Yes | No | No |
| 1024 | 16 | 105.40% | 64 × 64 | 0/2/2 | Yes | No | Yes |
| 2048 | 8 | 104.45% | 64 × 64 | 0/2/2 | Yes | No | No |
| 4096 | 4 | 104.39% | 64 × 64 | 0/2/2 | Yes | Yes | Yes |
| 8192 | 2 | 104.88% | 64 × 64 | 0/2/2 | Yes | No | No |
| 16384 | 1 | 104.69% | 64 × 64 | 0/2/2 | Yes | Yes | Yes |

Harmonic mean: **105.17%**.

## BF16

### Page size 16

| KV length | Requests | Relative to FA2 | Br × Bc | Q/K/V load | Prefetch | RF buffer | Optimized softmax |
|---:|---:|---:|---|---|---|---|---|
| 128 | 128 | 107.38% | 64 × 32 | 2/2/0 | Yes | Yes | Yes |
| 512 | 32 | 104.22% | 64 × 32 | 2/2/2 | Yes | Yes | Yes |
| 1024 | 16 | 102.56% | 64 × 64 | 0/2/2 | Yes | Yes | No |
| 2048 | 8 | 103.26% | 64 × 64 | 0/2/2 | Yes | No | Yes |
| 4096 | 4 | 102.60% | 64 × 64 | 0/2/2 | Yes | No | No |
| 8192 | 2 | 103.02% | 64 × 64 | 0/2/2 | Yes | No | No |
| 16384 | 1 | 101.49% | 64 × 64 | 0/2/0 | Yes | No | Yes |

Harmonic mean: **103.48%**.

### Page size 256

| KV length | Requests | Relative to FA2 | Br × Bc | Q/K/V load | Prefetch | RF buffer | Optimized softmax |
|---:|---:|---:|---|---|---|---|---|
| 128 | 128 | 109.21% | 64 × 32 | 2/2/0 | Yes | Yes | Yes |
| 512 | 32 | 102.79% | 64 × 64 | 0/2/2 | Yes | No | Yes |
| 1024 | 16 | 103.80% | 64 × 64 | 0/2/2 | Yes | No | Yes |
| 2048 | 8 | 104.48% | 64 × 64 | 2/2/2 | Yes | Yes | No |
| 4096 | 4 | 105.17% | 64 × 64 | 0/2/2 | Yes | Yes | No |
| 8192 | 2 | 104.75% | 64 × 64 | 0/2/2 | Yes | Yes | Yes |
| 16384 | 1 | 104.23% | 64 × 64 | 0/2/2 | Yes | No | Yes |

Harmonic mean: **104.88%**.

## Configuration fields

All rows use four warps, asynchronous copies, and shared-memory swizzling.
`Br` and `Bc` are query and KV tile sizes. A Q/K/V load value of `0` retains
the full operand in registers; `2` loads two MMA K-tiles at a time.
The remaining columns control KV prefetching, register double buffering, and
softmax optimization. Configuration selection is explicit in the Python API.

## Usage

```python
import torch
from flash_attn_lab import paged_attention
from helper import get_configs

cfg = get_configs("paged", dtype="fp16", head_dim=128, page_size=256)[0]
query_start_loc = torch.tensor([0, 1, 5, 8, 10], device="cuda", dtype=torch.int32)
seq_lens = torch.tensor([101, 4, 9, 10], device="cuda", dtype=torch.int32)
block_table = torch.arange(4, device="cuda", dtype=torch.int32).reshape(4, 1)
q = torch.randn(10, 8, 128, device="cuda", dtype=torch.float16)
k_cache = torch.randn(4, 256, 4, 128, device="cuda", dtype=q.dtype)
v_cache = torch.randn_like(k_cache)

out = paged_attention(
    cfg, q, k_cache, v_cache, query_start_loc, seq_lens, block_table,
    max_query_len=4, max_kv_len=101, validate_metadata=True,
)
```

Q and output use packed `[total_queries, query_heads, D]` layout. K/V caches
use `[physical_pages, page_size, kv_heads, D]`. The final kernel supports
`D=64/128`, FP16/BF16, registered page sizes 16/256, and GQA/MQA when
`query_heads % kv_heads == 0`.

`query_start_loc` is the prefix sum of query lengths. `seq_lens` holds each
request's KV length, and `block_table` maps logical pages to physical cache
pages. These metadata tensors are CUDA int32 on the same device as Q/K/V;
the prefix sum and lengths are contiguous, and block-table columns are
contiguous. Unused table entries are not read. A request must satisfy
`kv_len >= q_len > 0`; empty batches return an empty output.

For each query position `q_position` within a request, visible keys satisfy
`kv_position <= kv_len - q_len + q_position`. Physical pages can appear in any
order, and requests can share read-only prefix pages. The cache must contain
the valid keys and values before calling the kernel.

The last tensor dimension must be contiguous. Outer strides must be positive,
nonoverlapping, and multiples of eight elements; data pointers must be
16-byte aligned. K/V may use different strides. Output storage must not
overlap inputs or metadata. `max_query_len` and `max_kv_len` are host upper
bounds and must cover the true request lengths. `validate_metadata=True`
synchronizes to validate values; leave it disabled only when the caller
already guarantees valid metadata.

## Run the benchmark

```bash
python benchmarks/ncu.py --runs 3 --scenario paged --page-size 256 --dtype both --configs all \
  --seq-lens 128 512 1024 2048 4096 8192 16384
python benchmarks/ncu.py --runs 3 --scenario paged --page-size 16 --dtype both --configs all \
  --seq-lens 128 512 1024 2048 4096 8192 16384
```

See [benchmarks/README.md](../benchmarks/README.md) for the paged
FlashAttention-2 baseline and profiling setup.
