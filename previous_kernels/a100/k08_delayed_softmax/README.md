# A100 · 08 · Bounded delayed softmax rescaling

Additional optimization based on [step 06](../k06_qk_load_schedule/README.md).
Step 07 remains the [default final kernel](../../../kernels/a100/README.md).
This version uses step 06's QK traversal and changes the online-softmax update.

After the first KV tile, keep the normalization offset while the scaled tile
maximum is at most four log2 units above it. Otherwise update the offset and
rescale the running exponential sum and output accumulator. When every row in
a warp keeps its offset, skip rescaling for the entire warp.

In exact arithmetic, the unnormalized exponential weights are bounded by 16.
FP32 sums use the same offset as the output accumulator. FP16/BF16 conversion
and intermediate ranges differ from the default kernel, so results need not be
bitwise identical; very large BF16 values have different overflow exposure.
The amount of skipped work depends on the input. The delay is always enabled in
this version; `optimized_softmax` still selects its existing initialization path.

Threshold-based rescaling is described in
[FlashAttention-4, Section 3.1.4](https://arxiv.org/html/2603.05451v1#S3.SS1.SSS4).
This implementation uses a four-log2-unit threshold in the Ampere kernel.

```python
from flash_attn_lab import forward
from helper import get_configs

cfg = get_configs("a100", version=8, dtype="fp16", head_dim=128)[0]
out = forward(cfg, q, k, v, scenario="a100", version=8)
```

Q/K/V are contiguous `[B,H,N,D]` CUDA tensors with the same FP16/BF16 dtype.
D is 64 or 128; N must be divisible by both tile sizes. Accumulation is FP32,
scale is `1/sqrt(D)`, and dropout is zero.

[Performance](../../../docs/a100_performance.md#additional-optimization-bounded-delayed-softmax-rescaling) ·
[Configuration search](../../../benchmarks/README.md)
