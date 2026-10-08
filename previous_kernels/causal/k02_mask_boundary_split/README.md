# Causal · 02 · Separate full and masked tiles

Use separate loops for fully visible KV tiles and tiles requiring a causal mask.

Based on [Causal mask](../k01_causal_mask/README.md).

```python
forward(cfg, q, k, v, scenario="causal", version=2)
```

Select a registered configuration with `get_configs("causal", version=2, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/causal/README.md) · [Performance](../../../docs/causal_performance.md)
