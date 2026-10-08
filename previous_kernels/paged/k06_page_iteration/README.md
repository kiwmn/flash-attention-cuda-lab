# Paged · 06 · Page traversal and single-page specialization

Increment addresses within a page, traverse KV tiles page by page and use a specialized loop when all KV tokens fit within one page.

Based on [K/V address reuse](../k05_kv_address_reuse/README.md).

```python
paged_attention(cfg, q, k_cache, v_cache, query_start_loc, seq_lens, block_table,
                max_query_len=max_query_len, max_kv_len=max_kv_len, version=6)
```

Select a registered configuration with `get_configs("paged", version=6, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/paged/README.md) · [Performance](../../../docs/paged_performance.md)
