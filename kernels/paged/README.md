# Paged causal attention

Final forward implementation for variable-length mixed prefill/decode inputs on
RTX 3090. The kernel reads an already populated KV cache; the caller writes the
cache and supplies the request metadata.

## Inputs

| Argument | Shape and meaning |
|---|---|
| `q` / `o` | `[T, Hq, D]`, query tokens packed across requests |
| `k_cache` / `v_cache` | `[P, S, Hkv, D]`, physical KV pages |
| `query_start_loc` | CUDA int32 `[B+1]`, query prefix sums from 0 to T |
| `seq_lens` | CUDA int32 `[B]`, valid KV length per request |
| `block_table` | CUDA int32 `[B, max_pages]`, logical-to-physical page indices |

Supports FP16/BF16, D64/D128, page sizes 16/256 and `Hq % Hkv == 0`
(MHA/GQA/MQA). Accumulation is FP32; scale is `1/sqrt(D)`. A request must have
`kv_len >= q_len > 0`. Query position `i` reads key positions
`j <= kv_len - q_len + i` (right-aligned causal masking).

`max_query_len` and `max_kv_len` are host-side upper bounds. They must cover
all requests, and the block table must have at least `ceil(max_kv_len / S)` columns.
Unused page-table entries are ignored; used entries must index valid physical pages.
An empty batch returns an empty output.

The last tensor dimension must be contiguous. Data pointers are 16-byte aligned;
outer strides are positive multiples of eight elements, with no overlapping elements.
Non-overlapping padded or permuted outer dimensions are supported. Output storage
must not overlap any input. Metadata tensors share the CUDA device; prefix sums
and lengths are contiguous, while the block table may have a padded row stride.

`validate_metadata=True` synchronizes to check device metadata values. Without
this option, the caller must provide valid values. Execution uses the current
CUDA stream.

## Example

```python
import torch
from flash_attn_lab import paged_attention
from helper import DType, get_configs

cfg = get_configs("paged", dtype=DType.FP16, head_dim=128, page_size=16)[0]
q = torch.randn(10, 8, 128, device="cuda", dtype=torch.float16)
k_cache = torch.randn(10, 16, 4, 128, device="cuda", dtype=q.dtype)
v_cache = torch.randn_like(k_cache)
query_start_loc = torch.tensor([0, 1, 5, 8, 10], device="cuda", dtype=torch.int32)
seq_lens = torch.tensor([101, 4, 9, 10], device="cuda", dtype=torch.int32)
block_table = torch.full((4, 7), -1, device="cuda", dtype=torch.int32)
block_table[0] = torch.arange(7, device="cuda", dtype=torch.int32)
block_table[1:, 0] = torch.tensor([7, 8, 9], device="cuda", dtype=torch.int32)
out = paged_attention(
    cfg, q, k_cache, v_cache, query_start_loc, seq_lens, block_table,
    max_query_len=4, max_kv_len=101, validate_metadata=True,
)
```

Use `o=` for a separate output buffer. `benchmark=True` returns `(out, kernel_time_ms)`.
Omitting `version` selects the final implementation. The [baseline](../../previous_kernels/paged/k01_paged_kv/README.md)
also supports runtime page sizes 32/64; later steps specialize registered page sizes.

[Optimization steps](../../previous_kernels/paged/README.md) ·
[Performance and workloads](../../docs/paged_performance.md) ·
[Benchmark commands](../../benchmarks/README.md)
