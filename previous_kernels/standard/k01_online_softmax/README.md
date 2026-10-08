# Standard · 01 · Online softmax baseline

Compute tiled attention with Tensor Core MMA and FP32 online softmax.

```python
forward(cfg, q, k, v, scenario="standard", version=1)
```

Select a registered configuration with `get_configs("standard", version=1, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/standard/README.md) · [Performance](../../../docs/standard_performance.md)
