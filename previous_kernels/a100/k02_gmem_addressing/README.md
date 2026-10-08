# A100 · 02 · Global-memory addressing and normalization

Calculate KV addresses from block indices and row offsets, carry global-memory strides in 64-bit integers, bound next-block prefetching and reuse one reciprocal per output row.

Based on [Standard attention baseline](../k01_baseline/README.md).

```python
forward(cfg, q, k, v, scenario="a100", version=2)
```

Select a registered configuration with `get_configs("a100", version=2, dtype=..., head_dim=...)`.

[Interface and input constraints](../../../kernels/a100/README.md) · [Performance](../../../docs/a100_performance.md)
