# Paged · 01 · Paged KV baseline

Read paged KV caches for packed variable-length queries with right-aligned causal masking, GQA/MQA and partial tiles.

```python
paged_attention(cfg, q, k_cache, v_cache, query_start_loc, seq_lens, block_table,
                max_query_len=max_query_len, max_kv_len=max_kv_len, version=1)
```

Select a registered configuration with `get_configs("paged", version=1, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/paged/README.md) · [Performance](../../../docs/paged_performance.md)
