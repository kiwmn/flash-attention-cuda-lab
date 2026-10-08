# Paged · 04 · Full-tile loading

Separate complete KV tiles from partial and masked boundary tiles.

Based on [Cached page addresses](../k03_page_address_cache/README.md).

```python
paged_attention(cfg, q, k_cache, v_cache, query_start_loc, seq_lens, block_table,
                max_query_len=max_query_len, max_kv_len=max_kv_len, version=4)
```

Select a registered configuration with `get_configs("paged", version=4, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/paged/README.md) · [Performance](../../../docs/paged_performance.md)
