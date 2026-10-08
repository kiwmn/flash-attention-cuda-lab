# A100 · 07 · QK row-first traversal

Visit Q row fragments first and alternate the K column direction between adjacent rows.

Based on [QK load scheduling](../k06_qk_load_schedule/README.md).

```python
forward(cfg, q, k, v, scenario="a100", version=7)
```

Select a registered configuration with `get_configs("a100", version=7, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/a100/README.md) · [Performance](../../../docs/a100_performance.md)
