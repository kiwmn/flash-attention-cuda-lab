# A100 attention

Final non-causal forward implementation, optimized for A100.

Inputs are contiguous CUDA tensors `[B, H, N, D]` with matching FP16 or BF16
dtype. Accumulation uses FP32. Supported head dimensions are 64 and 128;
`N` must be divisible by both `Br` and `Bc`. The scale is `1/sqrt(D)`.
This implementation has no backward pass or dropout.

```python
import torch
from flash_attn_lab import forward
from helper import DType, get_configs

cfg = get_configs("a100", dtype=DType.FP16, head_dim=128)[0]
q, k, v = [torch.randn(1, 2, 512, 128, device="cuda", dtype=torch.float16)
           for _ in range(3)]
out = forward(cfg, q, k, v, scenario="a100")
```

The first registered configuration is a runnable example. Search all registered
configurations with the [benchmark scripts](../../benchmarks/README.md) to
select the fastest for a given input.

Use `DType.BF16` with BF16 inputs. `o=` accepts a separate contiguous output
buffer with the same shape, dtype and device. `benchmark=True` returns
`(out, kernel_time_ms)`. Omitting `version` selects this final implementation;
a local step number selects a source tree under `previous_kernels`.

[Optimization steps](../../previous_kernels/a100/README.md) ·
[Performance](../../docs/a100_performance.md)

[Additional delayed-softmax optimization](../../previous_kernels/a100/k08_delayed_softmax/README.md)
is available with `version=8`. The default final implementation remains step 07.
