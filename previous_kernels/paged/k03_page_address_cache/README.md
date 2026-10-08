# Paged · 03 · Cached page addresses

Specialize page size at compile time and reuse physical page addresses.

This step registers page size 256 only. The next step also supports page size 16.

Based on [Shared row addresses](../k02_shared_row_address/README.md).

```python
paged_attention(cfg, q, k_cache, v_cache, query_start_loc, seq_lens, block_table,
                max_query_len=max_query_len, max_kv_len=max_kv_len, version=3)
```

Select a registered configuration with `get_configs("paged", version=3, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/paged/README.md) · [Performance](../../../docs/paged_performance.md)
