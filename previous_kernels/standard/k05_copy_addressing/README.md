# Standard · 05 · Copy address calculation

Use CTA-wide copies and precompute per-thread memory offsets.

Based on [Register pipeline, softmax optimization and auto-tuning](../k04_register_pipeline/README.md).

```python
forward(cfg, q, k, v, scenario="standard", version=5)
```

Select a registered configuration with `get_configs("standard", version=5, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/standard/README.md) · [Performance](../../../docs/standard_performance.md)
