# Paged optimization steps

| Step | Implementation | Change |
|---|---|---|
| 01 | [Paged KV baseline](k01_paged_kv/README.md) | Read paged KV caches for packed variable-length queries with right-aligned causal masking, GQA/MQA and partial tiles. |
| 02 | [Shared row addresses](k02_shared_row_address/README.md) | Share KV row addresses between threads loading the same row. |
| 03 | [Cached page addresses](k03_page_address_cache/README.md) | Specialize page size at compile time and reuse physical page addresses. |
| 04 | [Full-tile loading](k04_full_tile_loads/README.md) | Separate complete KV tiles from partial and masked boundary tiles. |
| 05 | [K/V address reuse](k05_kv_address_reuse/README.md) | Reuse the page information prepared for K when loading V. |
| 06 | [Page traversal and single-page specialization](k06_page_iteration/README.md) | Increment addresses within a page, traverse KV tiles page by page and use a specialized loop when all KV tokens fit within one page. |
| 07 | [CTA page loading and configuration search](k07_page_pipeline/README.md) | Prepare page addresses cooperatively, double-buffer current and next tile metadata, and load each small page with the entire CTA. Expand the registered configurations for per-shape performance search. |

[Final implementation](../../kernels/paged/README.md) · [Performance](../../docs/paged_performance.md)
