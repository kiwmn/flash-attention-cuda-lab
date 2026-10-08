# Standard · 04 · Register pipeline, softmax optimization and auto-tuning

Combine partial register loads, register double buffering and optimized softmax. Enumerate registered configurations with the benchmark runner and select the fastest for each input shape and dtype.

Based on [K/V prefetching](../k03_kv_prefetch/README.md).

```python
forward(cfg, q, k, v, scenario="standard", version=4)
```

Select a registered configuration with `get_configs("standard", version=4, dtype=..., head_dim=...)`.

Configuration search is part of this step. Run `python -m benchmarks.attention --scenario standard --version 4 --configs all` from the project root. The benchmark measures registered configurations and ranks them for each shape and dtype.

[Interface and input constraints](../../../kernels/standard/README.md) · [Performance](../../../docs/standard_performance.md)
