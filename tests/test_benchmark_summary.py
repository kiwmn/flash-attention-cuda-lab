import csv
import io

import pytest

from benchmarks.ncu import read_durations, summarize


def test_ncu_units_order_and_dtype_baselines():
    stream = io.StringIO()
    writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
    writer.writerow(["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
    names = ["flash::flash_fwd_kernel", "flash_attn_lab::standard::k07::flash_attention_forward_kernel"] * 2
    values = [("nsecond", "2,000,000"), ("usecond", "1,000"), ("msecond", "3"), ("second", "0.001")]
    manifest = []
    for index in reversed(range(4)):
        unit, value = values[index]
        writer.writerow([index, names[index], "gpu__time_duration.sum", unit, value])
    for index in range(4):
        reference = index % 2 == 0
        manifest.append(dict(scenario="standard", version="final", dtype="fp16" if index < 2 else "bf16",
                             seq_len=4096, backend="fa2" if reference else "custom",
                             config="FA2" if reference else "configuration", config_index=0))
    rows = summarize(read_durations(stream.getvalue()), manifest)
    assert {row["dtype"]: row["fa2_percent"] for row in rows} == {"fp16": 200.0, "bf16": 300.0}


def test_ncu_extra_launch_is_rejected():
    with pytest.raises(ValueError, match="expected 0 CUDA launches"):
        summarize([("unrelated_kernel", 1.0)], [])


def test_ncu_raw_columns():
    text = ('"ID","Kernel Name","gpu__time_duration.sum"\n'
            '"","","nsecond"\n'
            '"0","kernel","1,234,000"\n')
    assert read_durations(text) == [("kernel", 1.234)]
