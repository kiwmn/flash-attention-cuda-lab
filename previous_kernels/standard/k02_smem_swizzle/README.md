# Standard · 02 · Shared-memory swizzling

Apply an XOR address mapping to shared-memory tiles.

Based on [Online softmax baseline](../k01_online_softmax/README.md).

```python
forward(cfg, q, k, v, scenario="standard", version=2)
```

Select a registered configuration with `get_configs("standard", version=2, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/standard/README.md) · [Performance](../../../docs/standard_performance.md)
