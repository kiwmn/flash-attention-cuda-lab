# A100 · 03 · Reverse KV traversal

Traverse KV tiles from the end of the sequence and prefetch the preceding tile.

Based on [Global-memory addressing and normalization](../k02_gmem_addressing/README.md).

```python
forward(cfg, q, k, v, scenario="a100", version=3)
```

Select a registered configuration with `get_configs("a100", version=3, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/a100/README.md) · [Performance](../../../docs/a100_performance.md)
