# Standard · 07 · First KV tile specialization

Initialize online-softmax state directly for the first KV tile, avoiding rescaling an empty accumulator.

Based on [Register layout and MMA traversal](../k06_mma_layout/README.md).

```python
forward(cfg, q, k, v, scenario="standard", version=7)
```

Select a registered configuration with `get_configs("standard", version=7, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/standard/README.md) · [Performance](../../../docs/standard_performance.md)
