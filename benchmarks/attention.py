#!/usr/bin/env python3
"""Measure registered configurations and select the fastest for each workload."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flash_attn_lab import forward, paged_attention
from helper.kernel_configs import get_configs
from helper.paged_inputs import make_inverse_q_history_lengths, make_paged_cache


@dataclass
class Workload:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    output: torch.Tensor
    batch: int
    query_starts: torch.Tensor | None = None
    kv_starts: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    max_query_len: int = 0
    max_kv_len: int = 0


def version_argument(value):
    if value == "final":
        return None
    try:
        number = int(value.lower().removeprefix("k"))
        if number > 0:
            return number
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("use final or a positive version such as k04")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("standard", "a100", "causal", "paged"), default="standard")
    parser.add_argument("--version", type=version_argument, default=None, help="final (default) or kNN")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="both")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384],
                        help="Sequence lengths; KV lengths for paged attention")
    parser.add_argument("--tokens", type=int, default=16384, help="Total Q tokens (dense) or logical KV tokens (paged)")
    parser.add_argument("--batch", type=int, help="Override tokens/sequence_length batch or request count")
    parser.add_argument("--heads", type=int, help="Query heads (dense: 16, paged: 8)")
    parser.add_argument("--kv-heads", type=int, help="KV heads for paged attention (default: same as query heads)")
    parser.add_argument("--head-dim", choices=(64, 128), type=int, default=128)
    parser.add_argument("--page-size", choices=(16, 256), type=int, default=256)
    parser.add_argument("--configs", default="all", help="all, first, or a zero-based configuration index")
    parser.add_argument("--list-configs", action="store_true", help="Print configuration indices without using CUDA")
    parser.add_argument("--reference", choices=("fa2", "none"), default="fa2")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warm-cache", action="store_true", help="Keep cache contents between timing samples")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--csv", type=Path, help="Optional local output CSV")
    parser.add_argument("--profile", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile-runs", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--profile-manifest", type=Path, help=argparse.SUPPRESS)
    return parser


def select_configs(args, dtype):
    configs = get_configs(args.scenario, version=args.version, dtype=dtype,
                          head_dim=args.head_dim,
                          page_size=args.page_size if args.scenario == "paged" else None)
    entries = list(enumerate(configs))
    if args.configs == "all":
        return entries
    if args.configs == "first":
        return entries[:1]
    try:
        index = int(args.configs)
        if index >= 0:
            return [entries[index]]
    except (ValueError, IndexError):
        pass
    raise ValueError(f"configuration {args.configs!r} unavailable for {dtype}; use --list-configs")


def make_workload(args, seq_len, dtype, *, page_size=None):
    device = torch.device("cuda", args.device)
    if args.batch is None and args.tokens % seq_len:
        raise ValueError("each sequence length must divide --tokens, or specify --batch")
    batch = args.batch if args.batch is not None else args.tokens // seq_len
    heads = args.heads if args.heads is not None else (8 if args.scenario == "paged" else 16)
    generator = torch.Generator(device=device).manual_seed(1234)

    def randn(shape):
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)

    if args.scenario != "paged":
        q, k, v = [randn((batch, heads, seq_len, args.head_dim)) for _ in range(3)]
        return Workload(q, k, v, torch.empty_like(q), batch)
    q_lens, _, seq_lens, starts = make_inverse_q_history_lengths(batch * seq_len, batch, device=device)
    kv_starts = torch.arange(batch + 1, device=device, dtype=torch.int32) * seq_len
    kv_heads = args.kv_heads if args.kv_heads is not None else heads
    if heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    q = randn((int(starts[-1]), heads, args.head_dim))
    k = randn((batch * seq_len, kv_heads, args.head_dim))
    v = randn(k.shape)
    k_cache, v_cache, table = make_paged_cache(k, v, kv_starts, page_size or args.page_size)
    return Workload(q, k_cache, v_cache, torch.empty_like(q), batch,
                    starts, kv_starts, seq_lens, table, int(q_lens[-1]), seq_len)


def custom_call(args, data, cfg):
    if args.scenario == "paged":
        return partial(paged_attention, cfg, data.q, data.k, data.v,
                       data.query_starts, data.seq_lens, data.block_table,
                       max_query_len=data.max_query_len, max_kv_len=data.max_kv_len,
                       version=args.version, o=data.output)
    return partial(forward, cfg, data.q, data.k, data.v,
                   scenario=args.scenario, version=args.version, o=data.output)


def reference_call(args, data):
    try:
        from flash_attn import flash_attn_func
        from flash_attn.flash_attn_interface import flash_attn_gpu
    except ImportError as exc:
        raise RuntimeError("Install flash-attn for the FA2 baseline, or use --reference none") from exc
    if args.scenario == "paged":
        signature = flash_attn_gpu.varlen_fwd.__doc__ or ""
        supports_split_count = "arg21:" in signature
        legacy_single_split = "arg20:" in signature and not supports_split_count
        if not supports_split_count and not legacy_single_split:
            raise RuntimeError("Unsupported FA2 varlen_fwd binding; use flash-attn 2.7.4 or 2.8.x")
        if legacy_single_split and data.max_query_len == 1 and data.q.shape[1] > data.k.shape[2]:
            raise RuntimeError("This FA2 build cannot pin num_splits=1 for one-token GQA; use --reference none")

        def paged_fa2():
            # The public varlen wrapper does not expose the split count.
            try:
                arguments = (
                    data.q, data.k, data.v, None, data.query_starts, data.kv_starts,
                    data.seq_lens, None, data.block_table, None,
                    data.max_query_len, data.max_kv_len, 0.0, data.q.shape[-1] ** -0.5,
                    False, True, -1, -1, 0.0, False, None,
                )
                if supports_split_count:
                    arguments += (1,)
                # FA2 2.7/2.8's 21-argument entry uses one split outside
                # the one-token GQA path, which is rejected above.
                return flash_attn_gpu.varlen_fwd(*arguments)[0]
            except TypeError as exc:
                raise RuntimeError(
                    "Incompatible FA2 CUDA varlen_fwd binding. Use flash-attn 2.7.4/2.8.x "
                    "or pass --reference none."
                ) from exc
        return paged_fa2
    # Prepare native FA2 layout and output storage outside the measured region.
    q, k, v = [tensor.transpose(1, 2).contiguous() for tensor in (data.q, data.k, data.v)]
    output = torch.empty_like(q)

    def dense_fa2():
        return flash_attn_gpu.fwd(
            q, k, v, output, None, 0.0, q.shape[-1] ** -0.5,
            args.scenario == "causal", -1, -1, 0.0, False, None,
        )[0]

    return dense_fa2


def measure(call, *, warmups, repeats, flush_buffer):
    # Load/compile even when the user requests zero additional warmups.
    call()
    for _ in range(warmups):
        call()
    torch.cuda.synchronize()
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    samples = []
    for _ in range(repeats):
        if flush_buffer is not None:
            flush_buffer.zero_()
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def profile(call, name, runs):
    call()
    torch.cuda.synchronize()
    print(f"Profiling {name}", flush=True)
    for _ in range(runs):
        torch.cuda.nvtx.range_push(name)
        torch.cuda.profiler.start()
        try:
            call()
            torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()
            torch.cuda.nvtx.range_pop()


def workload_row(args, data, dtype, seq_len, version):
    paged = args.scenario == "paged"
    return dict(scenario=args.scenario, version=version, dtype=dtype, seq_len=seq_len,
                batch=data.batch, heads=data.q.shape[-2] if paged else data.q.shape[1],
                kv_heads=data.k.shape[-2] if paged else data.k.shape[1],
                head_dim=args.head_dim, page_size=args.page_size if paged else "",
                query_tokens=data.q.shape[0] if paged else data.batch * seq_len)


@torch.inference_mode()
def run(args):
    dtypes = ["fp16", "bf16"] if args.dtype == "both" else [args.dtype]
    if args.list_configs:
        for dtype in dtypes:
            for index, cfg in select_configs(args, dtype):
                print(f"{dtype} [{index}] {cfg}")
        return []
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if (min(args.seq_lens) <= 0 or args.tokens <= 0 or args.repeats <= 0
            or args.profile_runs <= 0 or args.warmups < 0):
        raise ValueError("lengths, tokens and repeat counts must be positive; warmups must be nonnegative")
    if any(x is not None and x <= 0 for x in (args.batch, args.heads, args.kv_heads)):
        raise ValueError("batch and head counts must be positive")
    torch.cuda.set_device(args.device)
    torch.manual_seed(1234)
    version = "final" if args.version is None else f"k{args.version:02d}"
    print(f"GPU: {torch.cuda.get_device_name()}; {args.scenario}/{version}")
    flush_buffer = None if args.warm_cache or args.profile else torch.empty(
        128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    results, manifest, best_percent = [], [], {dtype: [] for dtype in dtypes}
    for dtype_name in dtypes:
        configs = select_configs(args, dtype_name)
        if not configs:
            print(f"{dtype_name}: no registered configurations")
            continue
        dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
        for seq_len in args.seq_lens:
            data = make_workload(args, seq_len, dtype)
            metadata = workload_row(args, data, dtype_name, seq_len, version)
            baseline_ms = None
            if args.reference == "fa2":
                reference_data = data
                if args.scenario == "paged" and args.page_size != 256:
                    # FA2 uses 256-token pages containing the same logical Q/K/V.
                    reference_data = make_workload(args, seq_len, dtype, page_size=256)
                baseline = reference_call(args, reference_data)
                if args.profile:
                    profile(baseline, f"{dtype_name}/n{seq_len}/fa2", args.profile_runs)
                    manifest.extend([dict(metadata, backend="fa2", config_index="", config="FA2")]
                                    * args.profile_runs)
                else:
                    baseline_ms = measure(baseline, warmups=args.warmups, repeats=args.repeats,
                                          flush_buffer=flush_buffer)
            rows = []
            for index, cfg in configs:
                if args.scenario != "paged" and (seq_len % cfg.Br or seq_len % cfg.Bc):
                    raise ValueError(f"N={seq_len} must be divisible by Br={cfg.Br} and Bc={cfg.Bc}")
                call = custom_call(args, data, cfg)
                if args.profile:
                    profile(call, f"{dtype_name}/n{seq_len}/config{index}/{cfg}", args.profile_runs)
                    manifest.extend([dict(metadata, backend="custom", config_index=index, config=str(cfg))]
                                    * args.profile_runs)
                    continue
                time_ms = measure(call, warmups=args.warmups, repeats=args.repeats,
                                  flush_buffer=flush_buffer)
                rows.append(dict(
                    **metadata,
                    config_index=index, config=str(cfg), time_ms=time_ms,
                    fa2_ms=baseline_ms if baseline_ms is not None else "",
                    fa2_percent=100 * baseline_ms / time_ms if baseline_ms is not None else "",
                ))
            if rows:
                rows.sort(key=lambda row: row["time_ms"])
                print(f"\n{dtype_name} {'kv_len' if args.scenario == 'paged' else 'seq_len'}={seq_len}; batch={data.batch}")
                for rank, row in enumerate(rows, 1):
                    ratio = f"{row['fa2_percent']:.2f}%" if baseline_ms is not None else "—"
                    print(f"{rank:3d}  {row['time_ms']:.6f} ms  {ratio:>9}  [{row['config_index']}] {row['config']}")
                if baseline_ms is not None:
                    best_percent[dtype_name].append(rows[0]["fa2_percent"])
                results.extend(rows)
            del data
    for dtype_name, values in best_percent.items():
        if values:
            print(f"{dtype_name} best-per-length harmonic mean: {statistics.harmonic_mean(values):.2f}%")
    if args.csv and results:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved {args.csv}")
    if args.profile_manifest:
        args.profile_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return results


def main():
    parser = make_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
