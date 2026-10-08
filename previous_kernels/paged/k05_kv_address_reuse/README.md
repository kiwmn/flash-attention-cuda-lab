# Paged · 05 · K/V address reuse

Reuse the page information prepared for K when loading V.

Based on [Full-tile loading](../k04_full_tile_loads/README.md).

```python
paged_attention(cfg, q, k_cache, v_cache, query_start_loc, seq_lens, block_table,
                max_query_len=max_query_len, max_kv_len=max_kv_len, version=5)
```

Select a registered configuration with `get_configs("paged", version=5, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/paged/README.md) · [Performance](../../../docs/paged_performance.md)
