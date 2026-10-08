# Standard optimization steps

| Step | Implementation | Change |
|---|---|---|
| 01 | [Online softmax baseline](k01_online_softmax/README.md) | Compute tiled attention with Tensor Core MMA and FP32 online softmax. |
| 02 | [Shared-memory swizzling](k02_smem_swizzle/README.md) | Apply an XOR address mapping to shared-memory tiles. |
| 03 | [K/V prefetching](k03_kv_prefetch/README.md) | Overlap global-to-shared K/V copies with attention computation. |
| 04 | [Register pipeline, softmax optimization and auto-tuning](k04_register_pipeline/README.md) | Combine partial register loads, register double buffering and optimized softmax. Enumerate registered configurations with the benchmark runner and select the fastest for each input shape and dtype. |
| 05 | [Copy address calculation](k05_copy_addressing/README.md) | Use CTA-wide copies and precompute per-thread memory offsets. |
| 06 | [Register layout and MMA traversal](k06_mma_layout/README.md) | Store register fragments in tile-major order and traverse MMA fragments in a serpentine order. |
| 07 | [First KV tile specialization](k07_first_tile_specialization/README.md) | Initialize online-softmax state directly for the first KV tile, avoiding rescaling an empty accumulator. |

[Final implementation](../../kernels/standard/README.md) · [Performance](../../docs/standard_performance.md)
