# Causal attention · RTX 3090

[Project overview](../README.md) · [Final implementation](../kernels/causal/README.md)

## Workload

The final kernel uses contiguous `[B,H,N,D]` Q/K/V tensors, `H=16`, `D=128`,
and `B × N=16384`. FP16 and BF16 use separate configurations and baselines.
Attention is causal, with scale `1/sqrt(D)` and no dropout.

Percentages are `100 × FA2 time / kernel time`. Each length uses the fastest
configuration in the measured search space. The harmonic mean is computed from
these per-length percentages. Timing covers the attention kernel, excluding
compilation, input preparation, and layout conversion. Nsight Compute uses
`clock-control=base` and `cache-control=all`.

## FP16

| Sequence length | B | Relative to FA2 | Br × Bc | Q/K/V load | Prefetch | RF buffer | Optimized softmax |
|---:|---:|---:|---|---|---|---|---|
| 512 | 32 | 102.94% | 64 × 32 | 2/2/0 | Yes | Yes | Yes |
| 1024 | 16 | 102.27% | 64 × 32 | 2/2/0 | Yes | No | Yes |
| 2048 | 8 | 101.99% | 64 × 32 | 2/2/0 | Yes | Yes | No |
| 4096 | 4 | 102.26% | 64 × 32 | 2/2/2 | Yes | No | Yes |
| 8192 | 2 | 101.76% | 64 × 32 | 2/2/2 | Yes | Yes | Yes |
| 16384 | 1 | 101.23% | 64 × 32 | 2/2/0 | Yes | No | Yes |

Harmonic mean: **102.07%**.

## BF16

| Sequence length | B | Relative to FA2 | Br × Bc | Q/K/V load | Prefetch | RF buffer | Optimized softmax |
|---:|---:|---:|---|---|---|---|---|
| 512 | 32 | 103.53% | 64 × 32 | 2/2/0 | Yes | Yes | No |
| 1024 | 16 | 102.34% | 64 × 32 | 2/2/2 | Yes | No | Yes |
| 2048 | 8 | 101.86% | 64 × 32 | 2/2/0 | Yes | No | Yes |
| 4096 | 4 | 101.49% | 64 × 32 | 2/2/2 | Yes | No | No |
| 8192 | 2 | 101.21% | 64 × 32 | 2/2/2 | Yes | No | No |
| 16384 | 1 | 101.20% | 64 × 32 | 2/2/2 | Yes | Yes | No |

Harmonic mean: **101.93%**.

## Configuration fields

All listed configurations use four warps, asynchronous global-to-shared copies,
and shared-memory swizzling. `Br` and `Bc` are the query and KV tile sizes.
A Q/K/V load value of `0` keeps the full operand in registers; `2` loads the
operand in groups of two MMA K-tiles. `RF buffer` enables register double
buffering, and `Optimized softmax` selects the configured softmax optimization.

Use `get_configs("causal", dtype="fp16", head_dim=128)` to obtain the
registered configurations. Select a row's tile sizes, load values, and flags;
then pass that configuration to `forward(..., scenario="causal")`.

## Run the benchmark

```bash
python benchmarks/ncu.py --runs 3 --scenario causal --dtype both --configs all \
  --seq-lens 512 1024 2048 4096 8192 16384
```

The final kernel is selected by default. `--version kNN` profiles a numbered
step. See the [benchmark guide](../benchmarks/README.md) for installation and
profiling requirements.
