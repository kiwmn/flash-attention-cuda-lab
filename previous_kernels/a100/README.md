# A100 optimization steps

| Step | Implementation | Change |
|---|---|---|
| 01 | [Standard attention baseline](k01_baseline/README.md) | Use the final standard-attention implementation as the starting point for A100 optimization. |
| 02 | [Global-memory addressing and normalization](k02_gmem_addressing/README.md) | Calculate KV addresses from block indices and row offsets, carry global-memory strides in 64-bit integers, bound next-block prefetching and reuse one reciprocal per output row. |
| 03 | [Reverse KV traversal](k03_reverse_kv/README.md) | Traverse KV tiles from the end of the sequence and prefetch the preceding tile. |
| 04 | [Output-store cache policy](k04_output_cache_policy/README.md) | Use an L1 no-allocation cache hint for output stores. |
| 05 | [Head-dimension tiling](k05_head_dim_tiling/README.md) | Organize shared-memory tiles into 64-column regions along the head dimension; apply the same mapping to copies, operand loads and output stores. |
| 06 | [QK load scheduling](k06_qk_load_schedule/README.md) | Finish the current QK slice before loading the next slice into the register buffers. |
| 07 | [QK row-first traversal (final)](k07_qk_row_traversal/README.md) | Visit Q row fragments first and alternate the K column direction between adjacent rows. |
| 08 | [Bounded delayed softmax rescaling (additional)](k08_delayed_softmax/README.md) | Based on step 06; retain the softmax offset within four log2 units and skip unchanged warp-wide rescaling. |

Step 07 is the selected final kernel. Step 08 is an additional branch from step 06.

[Final implementation](../../kernels/a100/README.md) · [Performance](../../docs/a100_performance.md)
