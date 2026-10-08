# Benchmarks and configuration search

Run commands from the repository root after installing the project. A CUDA GPU is
required for measurements. Install `flash-attn` to include the FlashAttention-2
baseline, or pass `--reference none` to measure only this project's kernels.

## Configuration search

The same benchmark script measures one configuration or searches all registered
configurations. Starting with standard k04, configuration search is part of the
optimization step; it has no separate tuning package.

```bash
python benchmarks/attention.py --scenario standard --version k04 --configs all
python benchmarks/attention.py --scenario a100 --configs all
python benchmarks/attention.py --scenario causal --configs all
python benchmarks/attention.py --scenario paged --configs all
```

Omit `--version` to use `kernels/<scenario>/`; use `--version kNN` to run a retained
step in `previous_kernels/<scenario>/`. Each shape and dtype is ranked separately.
The first row is its fastest configuration. The final summary reports the harmonic
mean of these per-length FA2-relative percentages. Configuration search does not
modify the kernel or automatically install a runtime dispatch policy.

Standard k01–k03 each register one fixed FP16 configuration. Use `--dtype fp16`
for those steps. Later versions support FP16 and BF16.

List configurations without compiling or running CUDA, then measure one by index:

```bash
python benchmarks/attention.py --scenario standard --dtype fp16 --list-configs
python benchmarks/attention.py --scenario standard --dtype fp16 --seq-lens 4096 --configs 0
```

The index belongs to the selected scenario, version, dtype, head dimension and
page size; the script also prints the full configuration. Use the corresponding
entry from `helper.kernel_configs.get_configs()` in Python.

## Workloads and timing

Dense attention defaults to `B * seq_len = 16384`, 16 heads, head dimension 128 and
lengths 512, 1024, 2048, 4096, 8192 and 16384. `causal` uses a causal FA2 baseline.
FA2 receives contiguous `[B,N,H,D]` inputs and a preallocated output buffer;
layout conversion and allocation occur before timing.

Paged attention uses the same length list as `kv_len`, with
`requests = 16384 / kv_len`, 8 query/KV heads and 256-token pages. Request `i`
(numbered from 1) has `q_len = ceil(kv_len * i / requests)`. The cache already
contains its current K/V tokens; causal masks align to the bottom right. Physical
pages are shuffled. `--kv-heads 1` selects MQA and `--kv-heads 4` selects GQA.
`--page-size 16` changes the custom kernel's cache layout; FA2 uses the same logical
inputs repacked into 256-token pages. See [paged performance](../docs/paged_performance.md).

The paged baseline uses one KV split. It calls the installed FA2 CUDA extension
with `num_splits=1` when that argument is exposed. The 21-argument bindings in
FA2 2.7.4 and 2.8.3 also use one split for these prefill workloads and are supported.
Their one-token GQA path is rejected because it can select multiple splits.
No adjacent source checkout is required. One compatible installation is:

```bash
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

Pass `--reference none` when only measuring the custom kernel.

CUDA-event timing uses three warmups and the median of 20 samples. A 128 MiB
buffer is written before each sample to evict previous cache contents. Use
`--warm-cache` to retain cache contents, and `--warmups`/`--repeats` to change the
sample counts. Compilation, input creation and cache packing occur outside timing.

```bash
python benchmarks/attention.py --scenario paged --seq-lens 4096 --dtype bf16 \
    --kv-heads 4 --page-size 16 --configs all --csv build/paged.csv
```

Output files are optional local artifacts. The repository contains no archived
measurement results. Run numerical tests before measuring a changed kernel:

```bash
python -m pytest tests --scenario standard
python -m pytest tests --scenario standard --kernel-version k04 --all-configs
python -m pytest tests --scenario paged --all-configs
```

Default tests cover representative operand-loading and buffering configurations.
`--all-configs` validates the complete selected registry; `--kernel-version all`
includes the retained versions. `--head-dim 64` selects the smaller head dimension
for versions that support it.

## Nsight Compute

The profiling entry point warms each selected configuration before opening its
CUDA profiler capture. Each capture has an NVTX label with dtype, length and
configuration. Nsight Compute uses base clocks and clears caches between replay
passes. The script reads `gpu__time_duration.sum`, converts it to milliseconds,
and ranks configurations separately for each length and dtype. `--runs 3`
averages three captures per configuration. FA2 ratios use the matching precision
and shape; the summary reports the best-per-length harmonic mean.

Event timings and profiler timings are separate measurement methods. The
published performance tables use Nsight Compute; use this entry point to repeat
that method. No CUDA-event measurements run while profiling.

```bash
python benchmarks/ncu.py --output build/ncu/standard -- \
    --scenario standard --dtype fp16 --seq-lens 4096 --configs 0

python benchmarks/ncu.py --set full --runs 3 --output build/ncu/paged -- \
    --scenario paged --dtype bf16 --seq-lens 4096 --configs 0 --reference none
```

By default profiling uses one length (4096), FP16 and the first configuration.
Pass `--configs all` to capture a complete registry. Use `--csv build/summary.csv`
before `--` to save the ranked summary. Open the generated
`.ncu-rep` file in Nsight Compute, or export its raw metrics:

```bash
ncu --import build/ncu/standard.ncu-rep --page raw --csv > build/ncu/standard.csv
```
