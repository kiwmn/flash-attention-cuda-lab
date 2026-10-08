# Standard · 06 · Register layout and MMA traversal

Store register fragments in tile-major order and traverse MMA fragments in a serpentine order.

Based on [Copy address calculation](../k05_copy_addressing/README.md).

```python
forward(cfg, q, k, v, scenario="standard", version=6)
```

Select a registered configuration with `get_configs("standard", version=6, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/standard/README.md) · [Performance](../../../docs/standard_performance.md)
