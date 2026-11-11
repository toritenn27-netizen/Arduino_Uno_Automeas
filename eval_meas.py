#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random-delay evaluation / visualization driver.

Workflow:
  1. ファイル選択ダイアログで avg_meas.py の出力フォルダ
     (traces.npy が入っているフォルダ) を選択
  2. ファイル選択ダイアログで対応する delay フォルダを選択
     (フォルダ名から推定した候補が初期表示される。違っていれば自由に変更可能)
  3. 結果は <選択フォルダ>/eval_result/ に保存
        - wave_pt{N}_{label}.png    : 単発平文の平均波形
        - wave_overlay.png          : min/median/max delay などの重ね描き
        - wave_heatmap_sorted.png   : delay 昇順ヒートマップ (スライド向け)
        - wave_mean_vs_single.png   : 全平文平均 vs 単発 (delay smearing 効果)
        - delay_histogram.png       : 適用 delay のヒストグラム
        - eval_summary.txt          : 数値サマリ + 整合性チェック結果

Usage:
    python eval_meas.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"matplotlib が必要です: {e}")
    sys.exit(1)


# ------------------------------------------------------------------
#  Folder selection (Tk dialogs)
# ------------------------------------------------------------------

def _ask_directory(title: str, initialdir: Optional[str] = None) -> str:
    """Wrap tkinter askdirectory so callers don't touch Tk directly."""
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askdirectory(title=title, initialdir=initialdir or "")
    finally:
        root.destroy()
    return path


def select_traces_folder() -> Path:
    p = _ask_directory("平均化結果フォルダを選択 (traces.npy が入っているフォルダ)")
    if not p:
        print("キャンセルされました")
        sys.exit(0)
    return Path(p)


def select_delay_folder(initial: Optional[Path]) -> Path:
    p = _ask_directory(
        "対応する delay フォルダを選択 (generated_delay_values.csv が入っているフォルダ)",
        initialdir=str(initial) if initial else None,
    )
    if not p:
        print("キャンセルされました")
        sys.exit(0)
    return Path(p)


# ------------------------------------------------------------------
#  Delay folder auto-detection
# ------------------------------------------------------------------

# 例: "traces_ArduinoUno_ave50_1e5_delay-uniform" → "uniform"
#     "...delay_normal..."                       → "normal"
_DELAY_NAME_PAT = re.compile(r"delay[-_]([A-Za-z]+)", re.IGNORECASE)


def guess_delay_folder(traces_dir: Path, script_dir: Path) -> Optional[Path]:
    m = _DELAY_NAME_PAT.search(traces_dir.name)
    if not m:
        return None
    mode = m.group(1).lower()

    candidates = [
        script_dir / "delay" / f"delay_{mode}",
        script_dir.parent / "delay" / f"delay_{mode}",
        Path.cwd() / "delay" / f"delay_{mode}",
    ]
    for c in candidates:
        if (c / "generated_delay_values.csv").is_file():
            return c
    return None


# ------------------------------------------------------------------
#  Loaders
# ------------------------------------------------------------------

def load_delay_values(delay_dir: Path) -> Tuple[np.ndarray, str]:
    csv_path = delay_dir / "generated_delay_values.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"generated_delay_values.csv が見つかりません: {csv_path}")

    vals: List[float] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = header[0] if header else "delay_us"
        for row in reader:
            if not row:
                continue
            vals.append(float(row[0]))
    return np.asarray(vals, dtype=np.float64), col


def load_traces(npy_path: Path) -> np.ndarray:
    arr = np.load(str(npy_path), mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"traces.npy must be 2-D, got shape {arr.shape}")
    return arr


# ------------------------------------------------------------------
#  Plot helpers
# ------------------------------------------------------------------

def _save_single_wave(
    out_dir: Path, traces: np.ndarray, idx: int, delay_us: float,
    label: str, pt_offset: int,
) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(traces[idx], linewidth=0.6)
    plt.title(f"averaged trace  pt={pt_offset + idx}  ({label}, delay = {delay_us:g} us)")
    plt.xlabel("sample point")
    plt.ylabel("voltage")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / f"wave_pt{pt_offset + idx}_{label}.png", dpi=200)
    plt.close()


def _save_overlay(
    out_dir: Path, traces: np.ndarray,
    indices: List[Tuple[str, int, float]], pt_offset: int,
) -> None:
    plt.figure(figsize=(12, 5))
    for label, idx, d in indices:
        plt.plot(
            traces[idx], linewidth=0.7, alpha=0.85,
            label=f"{label}: pt={pt_offset+idx}, delay={d:g} us",
        )
    plt.title("averaged traces — overlay (delay-induced shift)")
    plt.xlabel("sample point")
    plt.ylabel("voltage")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "wave_overlay.png", dpi=200)
    plt.close()


def _save_heatmap_sorted(
    out_dir: Path, traces: np.ndarray, delays: np.ndarray,
    max_rows: int = 1500,
) -> None:
    n_traces, n_samples = traces.shape
    sorted_idx = np.argsort(delays)
    if n_traces > max_rows:
        step = max(1, n_traces // max_rows)
        sel = sorted_idx[::step][:max_rows]
    else:
        sel = sorted_idx

    sub = np.asarray(traces[sel])  # mmap → in-memory copy
    sub_delays = delays[sel]

    plt.figure(figsize=(12, 6))
    plt.imshow(
        sub, aspect="auto", origin="lower", cmap="viridis",
        extent=[0, n_samples, float(sub_delays[0]), float(sub_delays[-1])],
    )
    plt.colorbar(label="voltage")
    plt.title(
        f"averaged traces sorted by delay  (rows shown: {len(sel)}/{n_traces})"
    )
    plt.xlabel("sample point")
    plt.ylabel("delay [us]  (low → high)")
    plt.tight_layout()
    plt.savefig(out_dir / "wave_heatmap_sorted.png", dpi=200)
    plt.close()


def _save_mean_vs_single(
    out_dir: Path, traces: np.ndarray, delays: np.ndarray, pt_offset: int,
) -> None:
    n_match, n_samples = traces.shape

    # 大きい traces にも耐えられるよう chunk 平均
    chunk = 1000
    acc = np.zeros(n_samples, dtype=np.float64)
    for s in range(0, n_match, chunk):
        e = min(n_match, s + chunk)
        acc += np.asarray(traces[s:e], dtype=np.float64).sum(axis=0)
    mean_trace = acc / n_match

    plt.figure(figsize=(12, 5))
    plt.plot(np.asarray(traces[0]), linewidth=0.6, alpha=0.55,
             label=f"single (pt={pt_offset}, delay={delays[0]:g} us)")
    plt.plot(mean_trace, linewidth=0.9, color="crimson",
             label=f"mean over {n_match} plaintexts")
    plt.title("smearing effect by random delay  (single trace vs grand mean)")
    plt.xlabel("sample point")
    plt.ylabel("voltage")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "wave_mean_vs_single.png", dpi=200)
    plt.close()


def _save_delay_hist(out_dir: Path, delays: np.ndarray) -> None:
    plt.figure(figsize=(10, 5))
    bins = min(51, max(10, int(delays.max() - delays.min()) + 1))
    plt.hist(delays, bins=bins, edgecolor="black", alpha=0.75)
    plt.title(f"applied delay distribution  (n={len(delays)})")
    plt.xlabel("delay [us]")
    plt.ylabel("frequency")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "delay_histogram.png", dpi=200)
    plt.close()


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------

def parse_extra_indices(s: str, n_max: int) -> List[int]:
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            print(f"[warn] '{tok}' は整数として解釈できないのでスキップ")
            continue
        if 0 <= v < n_max:
            out.append(v)
        else:
            print(f"[warn] index {v} は範囲外 (0..{n_max-1}) のためスキップ")
    return out


def main() -> int:
    traces_dir = select_traces_folder()
    npy_path = traces_dir / "traces.npy"
    if not npy_path.is_file():
        print(f"[error] traces.npy が見つかりません: {npy_path}")
        return 1

    script_dir = Path(__file__).resolve().parent
    guess = guess_delay_folder(traces_dir, script_dir)
    if guess is not None:
        print(f"[info] delay フォルダ候補 (初期表示): {guess}")
        initial_dir = guess
    else:
        print("[info] delay フォルダの候補なし → delay/ を初期表示します")
        fallback = script_dir / "delay"
        initial_dir = fallback if fallback.is_dir() else None
    delay_dir = select_delay_folder(initial_dir)

    delays_all, col = load_delay_values(delay_dir)
    traces = load_traces(npy_path)
    n_traces = traces.shape[0]

    print(f"traces.npy        : {npy_path}  shape={traces.shape}")
    print(f"delay CSV         : {delay_dir / 'generated_delay_values.csv'}  "
          f"({col}, n={len(delays_all)})")

    # --- 行数アライン ---
    if len(delays_all) < n_traces:
        print(f"[error] traces ({n_traces}) > delays ({len(delays_all)}). "
              f"delay フォルダが正しいか確認してください。")
        return 1
    if len(delays_all) > n_traces:
        print(f"[warn] traces ({n_traces}) < delays ({len(delays_all)}). "
              f"先頭 {n_traces} 個の delay と対応付けます。")
    delays = delays_all[:n_traces]

    # --- 任意平文 index の追加プロット ---
    extra: List[int] = []
    try:
        s = input(
            f"追加でプロットしたい平文 index (0..{n_traces-1}, カンマ区切り、空 Enter でスキップ): "
        ).strip()
        if s:
            extra = parse_extra_indices(s, n_traces)
    except EOFError:
        pass

    out_dir = traces_dir / "eval_result"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[info] 結果出力先: {out_dir}")

    # --- 代表 index (min / mean / max delay) ---
    sorted_idx = np.argsort(delays)
    mean_delay = float(delays.mean())
    mean_idx = int(np.argmin(np.abs(delays - mean_delay)))
    rep: List[Tuple[str, int]] = [
        ("min_delay", int(sorted_idx[0])),
        ("mean_delay", mean_idx),
        ("max_delay", int(sorted_idx[-1])),
    ]
    print(
        f"[info] representative delays: "
        f"min={float(delays[sorted_idx[0]]):g} us, "
        f"mean(target)={mean_delay:.4f} us "
        f"-> nearest trace delay={float(delays[mean_idx]):g} us, "
        f"max={float(delays[sorted_idx[-1]]):g} us"
    )

    # --- 単発平均波形 (代表 + 追加) ---
    pt_offset = 0  # avg_meas.py が use_range=false 想定。必要なら手動補正。
    saved_singles: List[Tuple[str, int, float]] = []
    for label, idx in rep:
        d = float(delays[idx])
        _save_single_wave(out_dir, traces, idx, d, label, pt_offset)
        saved_singles.append((label, idx, d))

    for i in extra:
        d = float(delays[i])
        _save_single_wave(out_dir, traces, i, d, f"user", pt_offset)
        saved_singles.append((f"pt{i}", i, d))

    # --- overlay ---
    _save_overlay(out_dir, traces, saved_singles, pt_offset)

    # --- heatmap (delay 昇順) ---
    _save_heatmap_sorted(out_dir, traces, delays)

    # --- mean vs single ---
    _save_mean_vs_single(out_dir, traces, delays, pt_offset)

    # --- delay histogram ---
    _save_delay_hist(out_dir, delays)

    # --- summary ---
    summary_lines = [
        "Evaluation Summary",
        "==================",
        f"traces folder       : {traces_dir}",
        f"delay folder        : {delay_dir}",
        f"traces.npy shape    : {traces.shape}",
        f"delay column        : {col}",
        f"delay length        : {len(delays_all)}",
        f"matched count       : {n_traces}",
        "",
        f"delay min           : {float(delays.min()):g} us",
        f"delay max           : {float(delays.max()):g} us",
        f"delay mean          : {float(delays.mean()):.4f} us",
        f"delay stddev        : {float(delays.std()):.4f} us",
        "",
        "representative plaintext indices:",
    ]
    for label, idx, d in saved_singles:
        summary_lines.append(
            f"  {label:<14s} : pt index = {idx:>8d}  delay = {d:g} us"
        )
    summary_lines += [
        "",
        "outputs:",
        "  wave_pt*_*.png             single-plaintext averaged waveforms",
        "  wave_overlay.png           overlay of representatives",
        "  wave_heatmap_sorted.png    delay-sorted heatmap",
        "  wave_mean_vs_single.png    grand mean vs single (smearing)",
        "  delay_histogram.png        applied delay distribution",
    ]
    summary_path = out_dir / "eval_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("\n".join(summary_lines))
    print(f"\nDone. saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
