#!/usr/bin/env python3
"""Profile configurations with Nsight Compute and rank measured GPU durations."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def read_durations(text):
    """Read one gpu__time_duration.sum per launch, normalized to milliseconds."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if line.startswith('\"ID\",') or line.startswith("ID,")), None)
    if start is None:
        raise ValueError("Nsight Compute returned no metric table")
    units = {"ns": 1e-6, "nsecond": 1e-6, "us": 1e-3, "usecond": 1e-3,
             "ms": 1.0, "msecond": 1.0, "s": 1e3, "second": 1e3}
    launches = {}
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    metric = "gpu__time_duration.sum"
    wide = metric in reader.fieldnames
    raw_unit = next(reader)[metric] if wide else None
    for row in reader:
        if not wide and row.get("Metric Name") != metric:
            continue
        if not row.get("ID", "").isdigit():
            continue
        unit = raw_unit if wide else row["Metric Unit"]
        if unit not in units:
            raise ValueError(f"unsupported Nsight Compute duration unit: {unit}")
        value = row[metric] if wide else row["Metric Value"]
        duration = float(value.replace(",", "")) * units[unit]
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Nsight Compute returned a nonpositive or nonfinite duration")
        launch_id = int(row["ID"])
        if launch_id in launches:
            raise ValueError(f"duplicate GPU duration for launch {launch_id}")
        launches[launch_id] = (row["Kernel Name"], duration)
    if not launches:
        raise ValueError("Nsight Compute returned no GPU durations")
    return [launches[index] for index in sorted(launches)]


def summarize(launches, manifest):
    """Match ordered captures to their configurations and keep dtype baselines separate."""
    if len(launches) != len(manifest):
        raise ValueError(f"expected {len(manifest)} CUDA launches, captured {len(launches)}")
    groups = {}
    for (kernel_name, time_ms), entry in zip(launches, manifest):
        expected = f"flash_attn_lab::{entry['scenario']}::" if entry["backend"] == "custom" else (
            "flash_fwd_splitkv_kernel" if entry["scenario"] == "paged" else "flash_fwd_kernel")
        if expected not in kernel_name:
            raise ValueError(f"unexpected CUDA kernel for {entry['config']}: {kernel_name}")
        key = (entry["scenario"], entry["version"], entry["dtype"], entry["seq_len"], entry["config"])
        if key not in groups:
            groups[key] = (entry, [])
        groups[key][1].append(time_ms)
    baselines = {}
    rows = []
    for key, (entry, samples) in groups.items():
        duration = statistics.fmean(samples)
        if entry["backend"] == "fa2":
            baselines[key[:-1]] = duration
        else:
            row = dict(entry, time_ms=duration, captures=len(samples))
            row.pop("backend")
            rows.append(row)
    for row in rows:
        key = (row["scenario"], row["version"], row["dtype"], row["seq_len"])
        baseline = baselines.get(key)
        row["fa2_ms"] = baseline if baseline is not None else ""
        row["fa2_percent"] = 100 * baseline / row["time_ms"] if baseline is not None else ""
    return sorted(rows, key=lambda row: (row["dtype"], row["seq_len"], row["time_ms"]))


def print_summary(rows):
    previous = None
    best = {}
    rank = 0
    for row in rows:
        key = (row["dtype"], row["seq_len"])
        if key != previous:
            print(f"\n{row['dtype']} {'kv_len' if row['scenario'] == 'paged' else 'seq_len'}={row['seq_len']}")
            previous, rank = key, 0
            if row["fa2_percent"] != "":
                best.setdefault(row["dtype"], []).append(row["fa2_percent"])
        rank += 1
        relative = f"{row['fa2_percent']:.2f}%" if row["fa2_percent"] != "" else "—"
        print(f"{rank:3d}  {row['time_ms']:.6f} ms  {relative:>9}  [{row['config_index']}] {row['config']}")
    for dtype, percentages in best.items():
        print(f"{dtype} best-per-length harmonic mean: {statistics.harmonic_mean(percentages):.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="Pass benchmark options after --; see benchmarks/attention.py --help.")
    parser.add_argument("--output", type=Path, default=Path("build/ncu/attention"),
                        help="Nsight Compute report basename")
    parser.add_argument("--set", choices=("basic", "full"), default="basic")
    parser.add_argument("--ncu", default="ncu", help="Nsight Compute executable")
    parser.add_argument("--runs", type=int, default=1, help="Average this many captures per configuration")
    parser.add_argument("--csv", type=Path, help="Optional ranked summary CSV")
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if args.runs <= 0:
        parser.error("--runs must be positive")
    executable = shutil.which(args.ncu)
    if executable is None:
        parser.error(f"Nsight Compute executable not found: {args.ncu}")
    help_text = subprocess.run([executable, "--help"], capture_output=True, text=True,
                               check=True).stdout
    rename_flags = ["--rename-kernels", "off"] if "--rename-kernels" in help_text else []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("attention.py")
    with tempfile.TemporaryDirectory(prefix="attention-ncu-") as scratch:
        raw = Path(scratch) / "metrics.csv"
        manifest_path = Path(scratch) / "launches.json"
        command = [executable, "--profile-from-start", "off", "--target-processes", "all",
                   "--clock-control", "base", "--cache-control", "all", "--nvtx",
                   *rename_flags,
                   "--set", args.set, "--metrics", "gpu__time_duration.sum",
                   "--page", "raw",
                   "--print-units", "base", "--print-fp",
                   "--print-kernel-base", "demangled",
                   "--csv", "--log-file", str(raw), "--force-overwrite",
                   "--export", str(args.output),
                   sys.executable, str(runner), "--seq-lens", "4096", "--dtype", "fp16",
                   "--configs", "first", *forwarded, "--profile",
                   "--profile-runs", str(args.runs), "--profile-manifest", str(manifest_path)]
        result = subprocess.run(command, check=False)
        if result.returncode:
            if raw.exists():
                print(raw.read_text(), file=sys.stderr)
            return result.returncode
        try:
            manifest = json.loads(manifest_path.read_text())
            rows = summarize(read_durations(raw.read_text()), manifest)
        except (OSError, ValueError, KeyError) as exc:
            parser.exit(1, f"error: incomplete or incompatible Nsight Compute capture: {exc}\n")
    print_summary(rows)
    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {args.csv}")
    print(f"Nsight Compute report: {args.output}.ncu-rep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
