# Standard attention · RTX 3090

[Project overview](../README.md) · [Final implementation](../kernels/standard/README.md)

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
| 512 | 32 | 101.11% | 128 × 32 | 2/2/0 | Yes | Yes | Yes |
| 1024 | 16 | 101.99% | 128 × 32 | 2/2/0 | Yes | No | Yes |
| 2048 | 8 | 102.59% | 128 × 32 | 2/2/2 | Yes | Yes | Yes |
| 4096 | 4 | 102.78% | 128 × 32 | 2/2/0 | Yes | Yes | No |
| 8192 | 2 | 103.01% | 128 × 32 | 2/2/2 | Yes | Yes | No |
| 16384 | 1 | 103.08% | 128 × 32 | 2/2/2 | Yes | Yes | No |

Harmonic mean: **102.42%**.

## BF16

| Sequence length | B | Relative to FA2 | Br × Bc | Q/K/V load | Prefetch | RF buffer | Optimized softmax |
|---:|---:|---:|---|---|---|---|---|
| 512 | 32 | 101.15% | 128 × 32 | 2/2/2 | Yes | Yes | Yes |
| 1024 | 16 | 101.97% | 128 × 32 | 2/2/0 | Yes | No | Yes |
| 2048 | 8 | 102.57% | 128 × 32 | 2/2/0 | Yes | Yes | Yes |
| 4096 | 4 | 102.80% | 128 × 32 | 2/2/2 | Yes | Yes | No |
| 8192 | 2 | 102.94% | 128 × 32 | 2/2/2 | Yes | Yes | No |
| 16384 | 1 | 102.93% | 128 × 32 | 2/2/0 | Yes | Yes | No |

Harmonic mean: **102.39%**.

## Configuration fields

All listed configurations use four warps, asynchronous global-to-shared copies,
and shared-memory swizzling. `Br` and `Bc` are the query and KV tile sizes.
A Q/K/V load value of `0` keeps the full operand in registers; `2` loads the
operand in groups of two MMA K-tiles. `RF buffer` enables register double
buffering, and `Optimized softmax` selects the configured softmax optimization.

Use `get_configs("standard", dtype="fp16", head_dim=128)` to obtain the
registered configurations. Select a row's tile sizes, load values, and flags;
then pass that configuration to `forward(..., scenario="standard")`.

## Run the benchmark

```bash
python benchmarks/ncu.py --runs 3 --scenario standard --dtype both --configs all \
  --seq-lens 512 1024 2048 4096 8192 16384
```

The final kernel is selected by default. `--version kNN` profiles a numbered
step. See the [benchmark guide](../benchmarks/README.md) for installation and
profiling requirements.
