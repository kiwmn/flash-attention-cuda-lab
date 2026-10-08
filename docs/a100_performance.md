# Standard attention · A100

[Project overview](../README.md) · [Final implementation](../kernels/a100/README.md)

## Workload

The final kernel uses contiguous `[B,H,N,D]` Q/K/V tensors, `H=16`, `D=128`,
and `B × N=16384`. FP16 and BF16 use separate configurations and baselines.
Attention is non-causal, with scale `1/sqrt(D)` and no dropout.

Percentages are `100 × FA2 time / kernel time`. Each length uses the fastest
configuration in the measured search space. The harmonic mean is computed from
these per-length percentages. Timing covers the attention kernel, excluding
compilation, input preparation, and layout conversion. Nsight Compute uses
`clock-control=base` and `cache-control=all`.

## FP16

| Sequence length | B | Relative to FA2 | Br × Bc | Q/K/V load | Prefetch | RF buffer | Optimized softmax |
|---:|---:|---:|---|---|---|---|---|
| 512 | 32 | 103.56% | 128 × 64 | 2/2/2 | Yes | Yes | Yes |
| 1024 | 16 | 101.66% | 128 × 64 | 2/2/2 | Yes | Yes | Yes |
| 2048 | 8 | 100.77% | 128 × 64 | 2/2/2 | Yes | Yes | Yes |
| 4096 | 4 | 100.34% | 128 × 64 | 2/2/2 | Yes | Yes | Yes |
| 8192 | 2 | 100.07% | 128 × 64 | 2/2/2 | Yes | Yes | Yes |
| 16384 | 1 | 99.95% | 128 × 64 | 2/2/2 | Yes | Yes | Yes |

Harmonic mean: **101.04%**.

## BF16

| Sequence length | B | Relative to FA2 | Br × Bc | Q/K/V load | Prefetch | RF buffer | Optimized softmax |
|---:|---:|---:|---|---|---|---|---|
| 512 | 32 | 103.52% | 128 × 64 | 2/2/2 | Yes | Yes | Yes |
| 1024 | 16 | 101.67% | 128 × 64 | 2/2/2 | Yes | Yes | Yes |
| 2048 | 8 | 100.77% | 128 × 64 | 2/2/2 | Yes | Yes | Yes |
| 4096 | 4 | 100.30% | 128 × 64 | 2/2/2 | Yes | Yes | Yes |
| 8192 | 2 | 100.08% | 128 × 64 | 2/2/2 | Yes | Yes | Yes |
| 16384 | 1 | 99.96% | 128 × 64 | 2/2/2 | Yes | Yes | Yes |

Harmonic mean: **101.04%**.

## Configuration fields

All listed configurations use four warps, asynchronous global-to-shared copies,
and shared-memory swizzling. `Br` and `Bc` are the query and KV tile sizes.
A Q/K/V load value of `0` keeps the full operand in registers; `2` loads the
operand in groups of two MMA K-tiles. `RF buffer` enables register double
buffering, and `Optimized softmax` selects the configured softmax optimization.

Use `get_configs("a100", dtype="fp16", head_dim=128)` to obtain the
registered configurations. Select a row's tile sizes, load values, and flags;
then pass that configuration to `forward(..., scenario="a100")`.

## Run the benchmark

```bash
python benchmarks/ncu.py --scenario a100 --dtype both --configs all \
  --seq-lens 512 1024 2048 4096 8192 16384
```

The final kernel is selected by default. `--version kNN` profiles a numbered
step. See the [benchmark guide](../benchmarks/README.md) for installation and
profiling requirements.

## Additional optimization: bounded delayed softmax rescaling

[Step 08](../previous_kernels/a100/k08_delayed_softmax/README.md) builds on step 06
and uses a four-log2-unit threshold for updating the softmax normalization offset.
It skips rescaling when every row in a warp retains its offset. Step 07 above
remains the default final implementation.

The following results use the same six input shapes and per-shape configuration
selection. All selected configurations use `Br=128`, `Bc=64`, four warps,
asynchronous copies, KV prefetching, shared-memory swizzling, `load_2_2_2_tiles`,
register double buffering, and `optimized_softmax=True`.

### FP16

| Sequence length | B | Relative to FA2 |
|---:|---:|---:|
| 512 | 32 | 105.00% |
| 1024 | 16 | 103.46% |
| 2048 | 8 | 102.67% |
| 4096 | 4 | 102.25% |
| 8192 | 2 | 102.07% |
| 16384 | 1 | 101.95% |

Harmonic mean: **102.89%**.

### BF16

| Sequence length | B | Relative to FA2 |
|---:|---:|---:|
| 512 | 32 | 105.10% |
| 1024 | 16 | 103.55% |
| 2048 | 8 | 102.62% |
| 4096 | 4 | 102.16% |
| 8192 | 2 | 102.04% |
| 16384 | 1 | 101.93% |

Harmonic mean: **102.89%**.

The performance benefit depends on how often the offset changes. Delayed updates
change intermediate ranges and FP16/BF16 rounding; unnormalized exponential weights
can reach 16 in exact arithmetic. See the [implementation notes](../previous_kernels/a100/k08_delayed_softmax/README.md)
for the numerical behavior and the FlashAttention-4 reference.

```python
cfg = get_configs("a100", version=8, dtype="fp16", head_dim=128)
cfg = next(c for c in cfg if c.Br == 128 and c.Bc == 64
           and c.Q_mma_load_K_tiles == c.K_mma_load_K_tiles == c.V_mma_load_K_tiles == 2
           and c.prefetch_kv_tiles and c.mma_double_buffer_loads and c.optimized_softmax)
out = forward(cfg, q, k, v, scenario="a100", version=8)
```

```bash
python benchmarks/ncu.py --scenario a100 --version k08 --dtype both --configs all \
  --seq-lens 512 1024 2048 4096 8192 16384
```
