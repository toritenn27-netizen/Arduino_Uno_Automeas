#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect oscilloscope H5 waveform files without running averaging.

Examples:
    python inspect_h5.py
    python inspect_h5.py config/avg_config.jsonc --batch 0
    python inspect_h5.py path/to/Waveform_ch2_meas0.h5 --segments 0,1,2
    python inspect_h5.py --sample-start 1000 --sample-end 3000
    python inspect_h5.py --all-batches --no-plot
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    import h5py
except Exception:
    h5py = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from jsonc_loader import load_jsonc


_num_pat = re.compile(r"(\d+)")


def require_h5py():
    if h5py is None:
        raise RuntimeError("h5py is required. Install it in the Python environment.")
    return h5py


def require_numpy():
    if np is None:
        raise RuntimeError("numpy is required. Install it in the Python environment.")
    return np


def _num_key(path: str) -> int:
    m = None
    for m in _num_pat.finditer(path):
        pass
    return int(m.group(1)) if m else 1 << 30


def parse_index_list(text: str) -> List[int]:
    indices: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


def load_avg_config(path: str) -> dict:
    cfg = load_jsonc(path)
    if "past_config_filename" not in cfg:
        raise ValueError(f"avg_config missing past_config_filename: {path}")
    return cfg


def load_past_config(avg_cfg_path: str, past_filename: str) -> dict:
    past_path = Path(avg_cfg_path).parent / "past_config" / past_filename
    if not past_path.is_file():
        raise FileNotFoundError(f"past_config not found: {past_path}")
    return load_jsonc(str(past_path))


def resolve_h5_path(source: str, batch_index: int) -> Tuple[str, Optional[dict], Optional[dict]]:
    source_path = Path(source)
    if source_path.suffix.lower() in (".h5", ".hdf5"):
        return str(source_path), None, None

    avg_cfg = load_avg_config(source)
    past = load_past_config(source, avg_cfg["past_config_filename"])
    h5_path = f"{past['scope_save_dir']}\\{past['save_prefix']}{batch_index}.h5"
    return h5_path, avg_cfg, past


def list_numeric_1d_datasets(h5_path: str) -> Tuple[List[str], Counter, Counter, int, int]:
    h5 = require_h5py()
    np_mod = require_numpy()
    names: List[str] = []
    lengths: List[int] = []
    dtypes: List[str] = []
    numeric_other = 0
    non_numeric = 0

    with h5.File(h5_path, "r") as h:
        def _cb(name, obj):
            nonlocal numeric_other, non_numeric
            if not isinstance(obj, h5.Dataset):
                return
            if not np_mod.issubdtype(obj.dtype, np_mod.number):
                non_numeric += 1
                return
            if obj.ndim == 1:
                names.append(name)
                lengths.append(obj.size)
                dtypes.append(str(obj.dtype))
            else:
                numeric_other += 1

        h.visititems(_cb)

    names.sort(key=_num_key)
    return names, Counter(lengths), Counter(dtypes), numeric_other, non_numeric


def print_dataset_preview(h5_path: str, names: List[str], limit: int) -> None:
    if limit <= 0 or not names:
        return
    h5 = require_h5py()
    print()
    print(f"dataset preview        : first {min(limit, len(names))} numeric 1-D datasets")
    with h5.File(h5_path, "r") as h:
        for i, name in enumerate(names[:limit]):
            dset = h[name]
            print(f"  [{i}] {name}  shape={dset.shape}  dtype={dset.dtype}")


def print_structure(h5_path: str, past: Optional[dict], list_limit: int) -> Tuple[List[str], int]:
    names, length_counts, dtype_counts, numeric_other, non_numeric = list_numeric_1d_datasets(h5_path)
    common_len, common_count = length_counts.most_common(1)[0] if length_counts else (0, 0)

    print(f"H5 file                : {h5_path}")
    print(f"numeric 1-D datasets   : {len(names)}")
    print(f"common sample points   : {common_len}")
    print(f"same-length waveforms  : {common_count}")
    print(f"numeric non-1D datasets: {numeric_other}")
    print(f"non-numeric datasets   : {non_numeric}")

    if past is not None:
        osc_segments = int(past["osc_segments"])
        avg_count = int(past["avg_count"])
        plaintexts_per_batch = osc_segments // avg_count
        total_plaintexts = int(past["total_plaintexts"])
        num_batches = total_plaintexts // plaintexts_per_batch
        print(f"configured segments    : {osc_segments}")
        print(f"avg_count              : {avg_count}")
        print(f"plaintexts per batch   : {plaintexts_per_batch}")
        print(f"configured batches     : {num_batches}")
        print(f"configured shape       : ({osc_segments}, {common_len}) per H5")
        if common_count != osc_segments:
            print(f"[warn] same-length waveforms ({common_count}) != osc_segments ({osc_segments})")

    if dtype_counts:
        print("dtype counts           : " + ", ".join(f"{k}:{v}" for k, v in dtype_counts.most_common()))
    if length_counts:
        print("sample length counts   : " + ", ".join(f"{k}:{v}" for k, v in length_counts.most_common(10)))
    if names:
        print(f"first dataset          : {names[0]}")
        print(f"last dataset           : {names[-1]}")
        print_dataset_preview(h5_path, names, list_limit)

    return names, common_len


def read_waveforms(
    h5_path: str,
    dataset_names: List[str],
    segment_indices: Iterable[int],
    sample_start: int,
    sample_end: int,
    flat_sample_points: Optional[int] = None,
) -> List[Tuple[int, str, object]]:
    h5 = require_h5py()
    np_mod = require_numpy()
    out = []
    with h5.File(h5_path, "r") as h:
        if flat_sample_points is not None:
            if len(dataset_names) != 1:
                raise ValueError("flat layout expects exactly one waveform dataset")
            name = dataset_names[0]
            dset = h[name]
            segment_count = dset.size // flat_sample_points
            end = flat_sample_points - 1 if sample_end < 0 else min(sample_end, flat_sample_points - 1)
            for idx in segment_indices:
                if idx < 0 or idx >= segment_count:
                    print(f"[warn] segment index out of range, skipped: {idx}")
                    continue
                if sample_start < 0 or sample_start > end:
                    print(f"[warn] invalid sample range for segment {idx}, skipped")
                    continue
                start_pos = idx * flat_sample_points + sample_start
                end_pos = idx * flat_sample_points + end + 1
                data = np_mod.asarray(dset[start_pos:end_pos])
                out.append((idx, f"{name}[{idx}]", data))
            return out

        for idx in segment_indices:
            if idx < 0 or idx >= len(dataset_names):
                print(f"[warn] segment index out of range, skipped: {idx}")
                continue
            name = dataset_names[idx]
            dset = h[name]
            end = dset.size - 1 if sample_end < 0 else min(sample_end, dset.size - 1)
            if sample_start < 0 or sample_start > end:
                print(f"[warn] invalid sample range for segment {idx}, skipped")
                continue
            data = np_mod.asarray(dset[sample_start : end + 1])
            out.append((idx, name, data))
    return out


def infer_flat_sample_points(
    h5_path: str,
    dataset_names: List[str],
    past: Optional[dict],
    layout: str,
    samples_per_waveform: Optional[int],
) -> Optional[int]:
    if layout == "separate":
        return None
    if samples_per_waveform is not None:
        return samples_per_waveform
    if past is None:
        if layout == "flat":
            raise ValueError("--layout flat requires avg_config.jsonc so osc_segments is known")
        return None
    if len(dataset_names) != 1:
        if layout == "flat":
            raise ValueError("--layout flat requires exactly one numeric 1-D dataset")
        return None

    h5 = require_h5py()
    osc_segments = int(past["osc_segments"])
    with h5.File(h5_path, "r") as h:
        total_points = h[dataset_names[0]].size

    if total_points % osc_segments != 0:
        if layout == "flat":
            raise ValueError(
                f"flat dataset length {total_points} is not divisible by osc_segments {osc_segments}"
            )
        return None

    sample_points = total_points // osc_segments
    if layout == "flat":
        return sample_points
    if sample_points > 1:
        print()
        print("[note] one numeric 1-D dataset was found.")
        print(
            "[note] It could be one waveform, or a flat concatenation of multiple waveforms."
        )
        print(
            f"[note] Use --layout flat to force ({osc_segments}, {sample_points}), "
            "or --samples-per-waveform to state the true waveform length."
        )
    return None


def print_waveform_stats(waveforms: List[Tuple[int, str, object]], head: int) -> None:
    np_mod = require_numpy()
    for idx, name, data in waveforms:
        first_values = np_mod.asarray(data[:head])
        print()
        print(f"segment {idx} dataset   : {name}")
        print(f"shape / dtype           : {data.shape} / {data.dtype}")
        print(f"min / max               : {float(np_mod.min(data)):.9g} / {float(np_mod.max(data)):.9g}")
        print(f"mean / std              : {float(np_mod.mean(data)):.9g} / {float(np_mod.std(data)):.9g}")
        print(f"first {min(head, data.size)} samples     : {first_values}")


def plot_waveforms(
    waveforms: List[Tuple[int, str, object]],
    sample_start: int,
    out_path: str,
    show: bool,
) -> None:
    if plt is None:
        print("[warn] matplotlib is not available; skipping plot")
        return
    if not waveforms:
        print("[warn] no waveforms selected; skipping plot")
        return

    plt.figure(figsize=(12, 6))
    for idx, _name, data in waveforms:
        xs = range(sample_start, sample_start + len(data))
        plt.plot(xs, data, linewidth=1.0, label=f"seg {idx}")
    plt.xlabel("sample point")
    plt.ylabel("value")
    plt.title("H5 waveform preview")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_parent = Path(out_path).parent
    if str(out_parent) not in ("", "."):
        out_parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot             : {out_path}")
    if show:
        plt.show()
    plt.close()


def inspect_all_batches(source: str, past: dict) -> int:
    osc_segments = int(past["osc_segments"])
    avg_count = int(past["avg_count"])
    plaintexts_per_batch = osc_segments // avg_count
    num_batches = int(past["total_plaintexts"]) // plaintexts_per_batch
    ok = 0
    bad = 0
    ref_points = None

    for batch_index in range(num_batches):
        h5_path, _avg_cfg, _past = resolve_h5_path(source, batch_index)
        try:
            _names, length_counts, _dtype_counts, _numeric_other, _non_numeric = list_numeric_1d_datasets(h5_path)
            points, count = length_counts.most_common(1)[0] if length_counts else (0, 0)
            if ref_points is None:
                ref_points = points
            valid = count == osc_segments and points == ref_points
            ok += int(valid)
            bad += int(not valid)
            status = "OK" if valid else "WARN"
            print(f"batch {batch_index}: {status}  waveforms={count}/{osc_segments}, sample_points={points}")
        except Exception as e:
            bad += 1
            print(f"batch {batch_index}: ERROR  {h5_path}  {e}")

    print()
    print(f"checked batches        : {num_batches}")
    print(f"metadata OK            : {ok}")
    print(f"warnings/errors        : {bad}")
    if ref_points is not None:
        print(f"reference sample points: {ref_points}")
    return 0 if bad == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect H5 waveform contents, shape, and plot selected traces."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="config/avg_config.jsonc",
        help="avg_config.jsonc or direct .h5 path (default: config/avg_config.jsonc)",
    )
    parser.add_argument("--batch", type=int, default=0, help="batch index when source is avg_config")
    parser.add_argument("--all-batches", action="store_true", help="check metadata for every configured H5")
    parser.add_argument("--segments", default="0,1,2", help="segment indices, e.g. 0,1,2 or 0-9")
    parser.add_argument("--sample-start", type=int, default=0, help="first sample point to read")
    parser.add_argument("--sample-end", type=int, default=-1, help="last sample point to read (-1 = end)")
    parser.add_argument(
        "--samples-per-waveform",
        type=int,
        default=None,
        help="explicit sample count per waveform for one flat 1-D dataset",
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "separate", "flat"),
        default="auto",
        help="H5 layout: separate datasets or one flat concatenated dataset",
    )
    parser.add_argument("--head", type=int, default=10, help="number of first samples to print")
    parser.add_argument("--list-limit", type=int, default=20, help="number of dataset entries to list")
    parser.add_argument("--plot", default=None, help="PNG path for waveform plot")
    parser.add_argument("--no-plot", action="store_true", help="skip plot generation")
    parser.add_argument("--show", action="store_true", help="show interactive matplotlib window")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    h5_path, avg_cfg, past = resolve_h5_path(args.source, args.batch)

    if args.all_batches:
        if past is None:
            raise ValueError("--all-batches requires avg_config.jsonc as the source")
        return inspect_all_batches(args.source, past)

    if avg_cfg is not None:
        print(f"avg_config             : {args.source}")
        print(f"past_config            : {avg_cfg['past_config_filename']}")
        print(f"batch index            : {args.batch}")

    names, _common_len = print_structure(h5_path, past, args.list_limit)
    if not names:
        raise RuntimeError(f"no numeric 1-D waveform datasets found in {h5_path}")

    flat_sample_points = infer_flat_sample_points(
        h5_path,
        names,
        past,
        args.layout,
        args.samples_per_waveform,
    )
    if flat_sample_points is not None:
        segment_count = 1
        if len(names) == 1:
            h5 = require_h5py()
            with h5.File(h5_path, "r") as h:
                segment_count = h[names[0]].size // flat_sample_points
        print()
        print("inferred layout        : flat concatenated waveform dataset")
        print(f"inferred shape         : ({segment_count}, {flat_sample_points})")

    indices = parse_index_list(args.segments)
    waveforms = read_waveforms(
        h5_path,
        names,
        indices,
        args.sample_start,
        args.sample_end,
        flat_sample_points=flat_sample_points,
    )
    print_waveform_stats(waveforms, args.head)

    if not args.no_plot:
        plot_path = args.plot
        if plot_path is None:
            plot_path = f"h5_preview_batch{args.batch}.png"
        plot_waveforms(waveforms, args.sample_start, plot_path, args.show)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
