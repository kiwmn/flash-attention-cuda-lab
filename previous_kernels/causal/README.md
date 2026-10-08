# Causal optimization steps

| Step | Implementation | Change |
|---|---|---|
| 01 | [Causal mask](k01_causal_mask/README.md) | Restrict KV traversal to visible keys and mask future positions within the boundary tile. |
| 02 | [Separate full and masked tiles](k02_mask_boundary_split/README.md) | Use separate loops for fully visible KV tiles and tiles requiring a causal mask. |

[Final implementation](../../kernels/causal/README.md) · [Performance](../../docs/causal_performance.md)
