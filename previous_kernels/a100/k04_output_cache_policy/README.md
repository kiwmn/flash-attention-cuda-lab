# A100 · 04 · Output-store cache policy

Use an L1 no-allocation cache hint for output stores.

Based on [Reverse KV traversal](../k03_reverse_kv/README.md).

```python
forward(cfg, q, k, v, scenario="a100", version=4)
```

Select a registered configuration with `get_configs("a100", version=4, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/a100/README.md) · [Performance](../../../docs/a100_performance.md)
