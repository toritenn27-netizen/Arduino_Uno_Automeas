#!/usr/bin/env python3
"""
Generate one 16-byte AES state mask per plaintext.

Usage:
  python gen_masks.py
  python gen_masks.py [path/to/sca_config.jsonc]
  python gen_masks.py --count 100000 --output Mask.txt [--seed 1234]

When a config is provided, total_plaintexts, mask_file_path, and
mask_random_seed are read from that JSONC file. If plaintext_file_path exists,
the number of non-empty plaintext rows is used as N.
When no arguments are provided, config/sca_config.jsonc is used.
"""
from __future__ import annotations

import argparse
import os
import random
import secrets
from pathlib import Path

from jsonc_loader import load_jsonc


BYTES_PER_RANDOM = 16


def count_plaintext_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def write_masks(path: Path, count: int, seed: int | None) -> None:
    if count <= 0:
        raise ValueError("count must be > 0")

    if seed is None:
        data = secrets.token_bytes(count * BYTES_PER_RANDOM)
    else:
        rng = random.Random(seed)
        data = bytes(rng.getrandbits(8) for _ in range(count * BYTES_PER_RANDOM))

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        for _ in range(count):
            start = _ * BYTES_PER_RANDOM
            block = data[start : start + BYTES_PER_RANDOM]
            f.write(block.hex() + "\n")
        f.flush()
        os.fsync(f.fileno())

    tmp_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate AES mask file")
    parser.add_argument("config", nargs="?", default="config/sca_config.jsonc")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.is_file():
        cfg = load_jsonc(str(config_path))
        plaintext_path = Path(str(cfg.get("plaintext_file_path", "")))
        if args.count is not None:
            count = args.count
        elif plaintext_path.is_file():
            count = count_plaintext_rows(plaintext_path)
        else:
            count = int(cfg["total_plaintexts"])
        output = args.output if args.output is not None else str(cfg.get("mask_file_path", "Mask.txt"))
        seed = args.seed if args.seed is not None else cfg.get("mask_random_seed")
        seed = None if seed is None else int(seed)
    else:
        if args.count is None or args.output is None:
            parser.error(
                "--count and --output are required when config is omitted or not found"
            )
        count = args.count
        output = args.output
        seed = args.seed

    out_path = Path(output)
    write_masks(out_path, count, seed)
    print(f"Saved masks: {out_path} ({count} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
