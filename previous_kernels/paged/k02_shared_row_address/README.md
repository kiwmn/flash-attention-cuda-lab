# Paged · 02 · Shared row addresses

Share KV row addresses between threads loading the same row.

Based on [Paged KV baseline](../k01_paged_kv/README.md).

```python
paged_attention(cfg, q, k_cache, v_cache, query_start_loc, seq_lens, block_table,
                max_query_len=max_query_len, max_kv_len=max_kv_len, version=2)
```

Select a registered configuration with `get_configs("paged", version=2, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/paged/README.md) · [Performance](../../../docs/paged_performance.md)
