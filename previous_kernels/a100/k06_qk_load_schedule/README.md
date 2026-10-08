# A100 · 06 · QK load scheduling

Finish the current QK slice before loading the next slice into the register buffers.

Based on [Head-dimension tiling](../k05_head_dim_tiling/README.md).

```python
forward(cfg, q, k, v, scenario="a100", version=6)
```

Select a registered configuration with `get_configs("a100", version=6, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/a100/README.md) · [Performance](../../../docs/a100_performance.md)
