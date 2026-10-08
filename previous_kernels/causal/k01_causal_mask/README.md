# Causal · 01 · Causal mask

Restrict KV traversal to visible keys and mask future positions within the boundary tile.

Based on the [standard final implementation](../../../kernels/standard/README.md).

```python
forward(cfg, q, k, v, scenario="causal", version=1)
```

Select a registered configuration with `get_configs("causal", version=1, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/causal/README.md) · [Performance](../../../docs/causal_performance.md)
