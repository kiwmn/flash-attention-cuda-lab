# Paged · 07 · CTA page loading and configuration search

Prepare page addresses cooperatively, double-buffer current and next tile metadata, and load each small page with the entire CTA. Expand the registered configurations for per-shape performance search.

Based on [Page traversal and single-page specialization](../k06_page_iteration/README.md).

```python
paged_attention(cfg, q, k_cache, v_cache, query_start_loc, seq_lens, block_table,
                max_query_len=max_query_len, max_kv_len=max_kv_len, version=7)
```

Select a registered configuration with `get_configs("paged", version=7, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/paged/README.md) · [Performance](../../../docs/paged_performance.md)
