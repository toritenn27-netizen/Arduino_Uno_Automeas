#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trace averaging driver — reads measurement settings from a backed-up
sca_config.jsonc (in config/past_config/), then averages segmented
oscilloscope H5 waveforms per plaintext.

For each batch index in capture_logs:
  H5 path: {scope_save_dir}/{save_prefix}{batch_index}.h5
  H5 layout: osc_segments segments per file, grouped into
             plaintexts_per_batch = osc_segments // avg_count plaintexts.

Usage:
    python avg_meas.py [path/to/avg_config.jsonc]
    (default: config/avg_config.jsonc)
"""

from __future__ import annotations

import csv
import glob
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import h5py
    import numpy as np
except Exception:
    h5py = None
    np = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from jsonc_loader import load_jsonc


# ------------------------------------------------------------------
#  H5 reading helpers (mirror format_h5_average_save.py behavior)
# ------------------------------------------------------------------

_num_pat = re.compile(r"(\d+)")


def _num_key(path: str) -> int:
    m = None
    for m in _num_pat.finditer(path):
        pass
    return int(m.group(1)) if m else 1 << 30


def fast_stack_segments(
    h5_path: str,
    max_segments: Optional[int] = None,
    require_same_len: bool = True,
) -> Tuple[List[str], np.ndarray]:
    """Stack all 1-D numeric datasets in an H5 file into a 2-D array
    of shape (N_segments, sample_points)."""
    if h5py is None or np is None:
        raise RuntimeError("h5py and numpy are required to average H5 waveforms. Install them with: python -m pip install h5py numpy")
    with h5py.File(h5_path, "r") as h:
        cands: List[str] = []
        lengths: List[int] = []

        def _cb(name, obj):
            if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number):
                if obj.ndim == 1:
                    cands.append(name)
                    lengths.append(obj.size)

        h.visititems(_cb)

        if not cands:
            raise RuntimeError(f"no 1-D numeric datasets in {h5_path}")

        if require_same_len:
            from collections import Counter
            cnt = Counter(lengths)
            P, _ = cnt.most_common(1)[0]
            cands = [p for p, L in zip(cands, lengths) if L == P]
        else:
            P = lengths[0]

        cands.sort(key=_num_key)
        if max_segments is not None:
            cands = cands[:max_segments]

        if not cands:
            raise RuntimeError(f"no datasets matched in {h5_path}")

        dtype = h[cands[0]].dtype
        out = np.empty((len(cands), P), dtype=dtype)
        for i, name in enumerate(cands):
            d = h[name]
            if d.shape != (P,):
                tmp = np.empty(d.shape, dtype=d.dtype)
                d.read_direct(tmp)
                out[i, : min(P, tmp.size)] = np.ravel(tmp)[:P]
            else:
                d.read_direct(out[i])
        return cands, out


# ------------------------------------------------------------------
#  Config loading
# ------------------------------------------------------------------


def load_avg_config(path: str) -> dict:
    cfg = load_jsonc(path)
    required = ("past_config_filename", "avg_output_dir")
    for k in required:
        if k not in cfg:
            raise ValueError(f"avg_config: missing required key '{k}'")

    out = {
        "past_config_filename": cfg["past_config_filename"],
        "avg_output_dir": cfg["avg_output_dir"],
        "use_range": bool(cfg.get("use_range", False)),
        "plaintext_index_start": int(cfg.get("plaintext_index_start", 0)),
        "plaintext_index_end": int(cfg.get("plaintext_index_end", -1)),
        "sample_point_start": int(cfg.get("sample_point_start", 0)),
        "sample_point_end": int(cfg.get("sample_point_end", -1)),
        "output_format": cfg.get("output_format", "npy"),
        "save_preview_png": bool(cfg.get("save_preview_png", True)),
    }
    if out["output_format"] not in ("npy", "csv", "both"):
        raise ValueError("output_format must be one of: npy, csv, both")
    return out


def load_past_config(avg_cfg_path: str, past_filename: str) -> dict:
    past_path = Path(avg_cfg_path).parent / "past_config" / past_filename
    if not past_path.is_file():
        raise FileNotFoundError(f"past_config not found: {past_path}")
    data = load_jsonc(str(past_path))

    required = (
        "total_plaintexts", "output_dir",
        "scope_save_dir", "save_prefix",
        "osc_segments", "avg_count",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"past_config missing keys: {missing}")
    return data


def load_measurement_plan(
    scope_save_dir: str,
    save_prefix: str,
    capture_logs_dir: Path,
    num_batches: int,
) -> list[dict]:
    plan_path = Path(scope_save_dir) / f"{save_prefix}_measurement_plan.csv"
    if plan_path.is_file():
        with plan_path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    rows: list[dict] = []
    for batch_idx in range(num_batches):
        log_path = capture_logs_dir / f"plaintext_batch_{batch_idx:05d}.csv"
        if not log_path.is_file():
            return []
        with log_path.open("r", newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------


def main() -> int:
    if len(sys.argv) > 2:
        print("Usage: python avg_meas.py [path/to/config/avg_config.jsonc]")
        return 2

    avg_cfg_path = sys.argv[1] if len(sys.argv) == 2 else "config/avg_config.jsonc"
    avg_cfg = load_avg_config(avg_cfg_path)
    past = load_past_config(avg_cfg_path, avg_cfg["past_config_filename"])

    total_plaintexts = int(past["total_plaintexts"])
    osc_segments = int(past["osc_segments"])
    avg_count = int(past["avg_count"])
    scope_save_dir = str(past["scope_save_dir"])
    save_prefix = str(past["save_prefix"])
    capture_logs_dir = Path(past["output_dir"])

    if osc_segments % avg_count != 0:
        raise ValueError(
            f"past_config: osc_segments ({osc_segments}) not divisible by "
            f"avg_count ({avg_count})"
        )
    plaintexts_per_batch = osc_segments // avg_count
    if total_plaintexts % plaintexts_per_batch != 0:
        raise ValueError(
            f"past_config: total_plaintexts ({total_plaintexts}) not divisible by "
            f"plaintexts_per_batch ({plaintexts_per_batch})"
        )
    num_batches = total_plaintexts // plaintexts_per_batch

    # --- Resolve plaintext range ---
    if avg_cfg["use_range"]:
        pt_s = avg_cfg["plaintext_index_start"]
        pt_e = avg_cfg["plaintext_index_end"]
        if pt_e < 0 or pt_e >= total_plaintexts:
            pt_e = total_plaintexts - 1
        if pt_s < 0 or pt_s > pt_e:
            raise ValueError(f"invalid plaintext range: [{pt_s}, {pt_e}]")
    else:
        pt_s = 0
        pt_e = total_plaintexts - 1
    pt_count = pt_e - pt_s + 1

    batch_s = pt_s // plaintexts_per_batch
    batch_e = pt_e // plaintexts_per_batch
    pt_offset_in_first = pt_s - batch_s * plaintexts_per_batch
    pt_offset_in_last = pt_e - batch_e * plaintexts_per_batch  # inclusive

    # --- Capture-log integrity check (file presence + global_plaintext_index range) ---
    log_files = sorted(glob.glob(str(capture_logs_dir / "plaintext_batch_*.csv")))
    if len(log_files) < num_batches:
        print(
            f"[warn] capture_logs has {len(log_files)} files, "
            f"expected {num_batches} (continuing - H5 is the source of truth)"
        )

    # --- Detect sample count from the first H5 we will read ---
    first_h5 = f"{scope_save_dir}\\{save_prefix}{batch_s}.h5"
    _, probe = fast_stack_segments(first_h5, max_segments=osc_segments, require_same_len=True)
    if probe.shape[0] != osc_segments:
        raise RuntimeError(
            f"H5 segment count mismatch: file has {probe.shape[0]}, "
            f"past_config osc_segments={osc_segments} ({first_h5})"
        )
    sample_total = probe.shape[1]

    # --- Resolve sample-point range ---
    if avg_cfg["use_range"]:
        sp_s = avg_cfg["sample_point_start"]
        sp_e_raw = avg_cfg["sample_point_end"]
        sp_e = sample_total - 1 if sp_e_raw < 0 else sp_e_raw
        if sp_s < 0 or sp_s > sp_e or sp_e >= sample_total:
            raise ValueError(
                f"invalid sample-point range: [{sp_s}, {sp_e}] (sample_total={sample_total})"
            )
    else:
        sp_s = 0
        sp_e = sample_total - 1
    sp_slice = slice(sp_s, sp_e + 1)
    out_sample_num = sp_e - sp_s + 1

    print(f"avg_config             : {avg_cfg_path}")
    print(f"past_config            : {avg_cfg['past_config_filename']}")
    print(f"total_plaintexts       : {total_plaintexts}")
    print(f"osc_segments           : {osc_segments}")
    print(f"avg_count              : {avg_count}")
    print(f"plaintexts_per_batch   : {plaintexts_per_batch}")
    print(f"num_batches            : {num_batches}")
    print(f"sample_total           : {sample_total}")
    print(f"plaintext range        : [{pt_s}, {pt_e}] ({pt_count} plaintexts)")
    print(f"batch range            : [{batch_s}, {batch_e}]")
    print(f"sample-point range     : [{sp_s}, {sp_e}] ({out_sample_num} points)")
    print(f"H5 source              : {scope_save_dir}\\{save_prefix}*.h5")
    output_bytes = pt_count * out_sample_num * np.dtype(np.float64).itemsize
    print(f"output dtype           : float64")
    print(f"output size on disk    : {output_bytes / (1024 ** 3):.1f} GiB")

    measurement_plan = load_measurement_plan(
        scope_save_dir, save_prefix, capture_logs_dir, num_batches
    )
    if measurement_plan:
        if len(measurement_plan) < total_plaintexts:
            raise RuntimeError(
                f"measurement plan has {len(measurement_plan)} rows, "
                f"expected at least {total_plaintexts}"
            )
        print("measurement metadata   : available")
    else:
        print("[warn] measurement metadata not found; averaged_metadata.csv will be skipped")

    # --- Prepare streaming outputs and average per-batch ---
    out_dir = Path(avg_cfg["avg_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "traces"
    fmt = avg_cfg["output_format"]

    npy_path = Path(str(base) + ".npy")
    npy_tmp_path = Path(str(base) + ".tmp.npy")
    csv_path = Path(str(base) + ".csv")
    csv_tmp_path = Path(str(base) + ".tmp.csv")

    ave = None
    if fmt in ("npy", "both"):
        ave = np.lib.format.open_memmap(
            str(npy_tmp_path),
            mode="w+",
            dtype=np.float64,
            shape=(pt_count, out_sample_num),
        )

    csv_file = None
    if fmt in ("csv", "both"):
        csv_file = csv_tmp_path.open("w", newline="")

    preview_trace = None
    write_idx = 0
    t_start = time.time()

    try:
        for b_offset, batch_idx in enumerate(range(batch_s, batch_e + 1)):
            h5_path = first_h5 if batch_idx == batch_s else f"{scope_save_dir}\\{save_prefix}{batch_idx}.h5"
            if batch_idx == batch_s:
                traces = probe
            else:
                _, traces = fast_stack_segments(
                    h5_path, max_segments=osc_segments, require_same_len=True
                )
                if traces.shape != (osc_segments, sample_total):
                    raise RuntimeError(
                        f"H5 shape mismatch at batch {batch_idx}: got {traces.shape}, "
                        f"expected ({osc_segments}, {sample_total})"
                    )

            sliced = traces[:, sp_slice].reshape(plaintexts_per_batch, avg_count, out_sample_num)
            batch_ave = sliced.mean(axis=1)

            if batch_idx == batch_s and batch_idx == batch_e:
                take = batch_ave[pt_offset_in_first : pt_offset_in_last + 1]
            elif batch_idx == batch_s:
                take = batch_ave[pt_offset_in_first:]
            elif batch_idx == batch_e:
                take = batch_ave[: pt_offset_in_last + 1]
            else:
                take = batch_ave

            take64 = np.asarray(take, dtype=np.float64)

            if ave is not None:
                ave[write_idx : write_idx + take64.shape[0]] = take64
            if csv_file is not None:
                np.savetxt(csv_file, take64, delimiter=",")
            if preview_trace is None:
                preview_trace = take64[0].copy()

            write_idx += take64.shape[0]

            done = b_offset + 1
            total = batch_e - batch_s + 1
            elapsed = time.time() - t_start
            eta = elapsed / done * (total - done)
            print(
                f"\r  [{done}/{total}] {done*100/total:5.1f}%  "
                f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s",
                end="", flush=True,
            )
            # release reference to the probe so memory frees on next iter
            if batch_idx == batch_s:
                probe = None  # noqa: F841
    finally:
        if csv_file is not None:
            csv_file.close()
    print()

    if write_idx != pt_count:
        raise RuntimeError(f"averaged {write_idx} plaintexts, expected {pt_count}")

    # --- Save outputs ---
    if fmt in ("npy", "both"):
        ave.flush()
        del ave
        npy_tmp_path.replace(npy_path)
        print(f"Saved NPY: {npy_path}")
    if fmt in ("csv", "both"):
        csv_tmp_path.replace(csv_path)
        print(f"Saved CSV: {csv_path}")

    if measurement_plan:
        metadata_path = out_dir / "averaged_metadata.csv"
        selected_rows = measurement_plan[pt_s : pt_e + 1]
        with metadata_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(selected_rows[0].keys()))
            writer.writeheader()
            writer.writerows(selected_rows)
        print(f"Saved metadata CSV: {metadata_path}")

    if avg_cfg["save_preview_png"]:
        if plt is None:
            print("[warn] matplotlib not available; skipping preview PNG")
        else:
            plt.figure(figsize=(12, 6))
            plt.plot(preview_trace)
            plt.xlabel("sampling point")
            plt.ylabel("voltage")
            plt.title(f"averaged trace [pt={pt_s}]")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(str(base) + ".png", dpi=200)
            plt.close()
            print(f"Saved PNG: {base}.png")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
