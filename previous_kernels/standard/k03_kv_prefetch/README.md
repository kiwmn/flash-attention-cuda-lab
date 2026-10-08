# Standard · 03 · K/V prefetching

Overlap global-to-shared K/V copies with attention computation.

Based on [Shared-memory swizzling](../k02_smem_swizzle/README.md).

```python
forward(cfg, q, k, v, scenario="standard", version=3)
```

Select a registered configuration with `get_configs("standard", version=3, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/standard/README.md) · [Performance](../../../docs/standard_performance.md)
