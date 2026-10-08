#!/usr/bin/env python3
"""Compile one scenario's final kernel or one numbered optimization step."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flash_attn_lab.interface import load_extension
from helper.kernel_configs import SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="standard")
    parser.add_argument("--version", type=int, default=None, help="local step number; default: final kernel")
    args = parser.parse_args()
    extension = load_extension(args.scenario, args.version)
    print(extension.__file__)


if __name__ == "__main__":
    main()
