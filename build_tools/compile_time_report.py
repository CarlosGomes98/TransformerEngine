#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Summarize compile times from a Ninja .ninja_log file."""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys


Entry = collections.namedtuple("Entry", ["elapsed_ms", "output"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ninja-log", type=pathlib.Path, required=True, help="Path to .ninja_log")
    parser.add_argument("--limit", type=int, default=25, help="Maximum rows to print")
    parser.add_argument("--filter", default="", help="Regex matched against output paths")
    parser.add_argument("--markdown", action="store_true", help="Print a Markdown table")
    return parser.parse_args()


def read_entries(path: pathlib.Path, pattern: re.Pattern[str] | None) -> list[Entry]:
    entries: list[Entry] = []
    with path.open("r", encoding="utf-8") as ninja_log:
        for line in ninja_log:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 4:
                continue
            start_ms, end_ms, _mtime, output = fields[:4]
            if pattern is not None and pattern.search(output) is None:
                continue
            entries.append(Entry(int(end_ms) - int(start_ms), output))
    return sorted(entries, key=lambda entry: entry.elapsed_ms, reverse=True)


def print_text(entries: list[Entry]) -> None:
    width = max((len(str(entry.elapsed_ms)) for entry in entries), default=1)
    for entry in entries:
        print(f"{entry.elapsed_ms:>{width}} ms  {entry.output}")


def print_markdown(entries: list[Entry]) -> None:
    print("| Rank | Time (ms) | Output |")
    print("| ---: | ---: | --- |")
    for rank, entry in enumerate(entries, start=1):
        print(f"| {rank} | {entry.elapsed_ms} | `{entry.output}` |")


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    pattern = re.compile(args.filter) if args.filter else None
    entries = read_entries(args.ninja_log, pattern)[: args.limit]
    if args.markdown:
        print_markdown(entries)
    else:
        print_text(entries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
