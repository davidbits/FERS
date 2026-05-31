#!/usr/bin/env python3
"""Capture the CW VITA49 example to a local UDP JSONL file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vita49_common import add_capture_args, run_capture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_capture_args(parser, default_port=4992, default_fullscale="1.0")
    args = parser.parse_args()
    scenario_dir = Path(__file__).resolve().parent
    return run_capture(
        scenario_dir=scenario_dir,
        scenario_file="example.fersxml",
        host=args.host,
        port=args.port,
        fullscale=args.fullscale,
        epoch=args.epoch,
        out_dir=args.out_dir,
        capture_file=args.capture_file,
        run_json=args.run_json,
        log_level=args.log_level,
        threads=args.threads,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
