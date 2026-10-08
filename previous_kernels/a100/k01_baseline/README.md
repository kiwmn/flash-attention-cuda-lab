# A100 · 01 · Standard attention baseline

Use the final standard-attention implementation as the starting point for A100 optimization.

Based on the [standard final implementation](../../../kernels/standard/README.md).

```python
forward(cfg, q, k, v, scenario="a100", version=1)
```

Select a registered configuration with `get_configs("a100", version=1, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/a100/README.md) · [Performance](../../../docs/a100_performance.md)
