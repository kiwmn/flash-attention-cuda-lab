# A100 · 05 · Head-dimension tiling

Organize shared-memory tiles into 64-column regions along the head dimension; apply the same mapping to copies, operand loads and output stores.

Based on [Output-store cache policy](../k04_output_cache_policy/README.md).

```python
forward(cfg, q, k, v, scenario="a100", version=5)
```

Select a registered configuration with `get_configs("a100", version=5, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/a100/README.md) · [Performance](../../../docs/a100_performance.md)
