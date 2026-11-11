#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Delay value generator for SCA experiments.

Supports 4 distribution modes:
  - uniform  : Uniform distribution
  - normal   : Truncated normal distribution
  - mixture  : Mixture of normal distributions (truncated)
  - cauchy   : Truncated Cauchy distribution

Settings are loaded from delay_config.jsonc.

Usage:
    python gen_delay.py [path/to/delay_config.jsonc]
    (default: config/delay_config.jsonc)
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from jsonc_loader import load_jsonc


VALID_MODES = ("uniform", "normal", "mixture", "cauchy")


def load_settings(config_path: str) -> dict:
    cfg = load_jsonc(config_path)

    # num_values を sca_config.jsonc の total_plaintexts から取得
    sca_config_rel = cfg.get("sca_config_path", "sca_config.jsonc")
    sca_config_abs = Path(config_path).parent / sca_config_rel
    sca_cfg = load_jsonc(str(sca_config_abs))
    num_values = int(sca_cfg["total_plaintexts"])

    mode = cfg["distribution_mode"]
    suffix = cfg.get("folder_name_suffix", "") or ""
    folder_name = f"delay_{mode}" if not suffix else f"delay_{mode}_{suffix}"
    base_dir = Path("delay") / folder_name

    settings = {
        "num_values": num_values,
        "csv_column_name": cfg.get("csv_column_name", "delay_us"),
        "distribution_mode": mode,
        "round_to_int": bool(cfg.get("round_to_int", True)),
        "use_random_seed": bool(cfg.get("use_random_seed", True)),
        "random_seed": int(cfg.get("random_seed", 12345)),
        "histogram_bins": int(cfg.get("histogram_bins", 51)),
        "folder_name_suffix": suffix,
        "delay_folder_name": folder_name,
        "delay_base_dir": str(base_dir),
        "csv_output_path": str(base_dir / "generated_delay_values.csv"),
        "histogram_output_path": str(base_dir / "generated_delay_histogram.png"),
        "summary_output_path": str(base_dir / "generated_delay_summary.txt"),
        "config_snapshot_path": str(base_dir / "delay_config_snapshot.json"),
    }

    # Distribution-specific params
    u = cfg.get("uniform", {})
    settings["uniform_min"] = int(u.get("min", 0))
    settings["uniform_max"] = int(u.get("max", 50))

    n = cfg.get("normal", {})
    settings["normal_mean"] = float(n.get("mean", 25.0))
    settings["normal_stddev"] = float(n.get("stddev", 8.0))
    settings["normal_min"] = int(n.get("min", 0))
    settings["normal_max"] = int(n.get("max", 50))

    m = cfg.get("mixture", {})
    settings["mixture_components"] = m.get("components", [
        {"weight": 0.5, "mean": 15.0, "stddev": 5.0},
        {"weight": 0.5, "mean": 35.0, "stddev": 5.0},
    ])
    settings["mixture_min"] = int(m.get("min", 0))
    settings["mixture_max"] = int(m.get("max", 50))

    c = cfg.get("cauchy", {})
    settings["cauchy_x0"] = float(c.get("x0", 25.0))
    settings["cauchy_gamma"] = float(c.get("gamma", 5.0))
    settings["cauchy_min"] = int(c.get("min", 0))
    settings["cauchy_max"] = int(c.get("max", 50))

    return settings


def validate_settings(s: dict) -> None:
    if s["num_values"] <= 0:
        raise ValueError("num_values must be > 0")
    if s["distribution_mode"] not in VALID_MODES:
        raise ValueError(f'distribution_mode must be one of: {VALID_MODES}')

    mode = s["distribution_mode"]

    if mode == "uniform":
        if s["uniform_min"] > s["uniform_max"]:
            raise ValueError("uniform.min must be <= uniform.max")
        if s["uniform_min"] < 0 or s["uniform_max"] > 65535:
            raise ValueError("uniform range must fit in 0..65535 us")

    elif mode == "normal":
        if s["normal_min"] > s["normal_max"]:
            raise ValueError("normal.min must be <= normal.max")
        if s["normal_stddev"] <= 0:
            raise ValueError("normal.stddev must be > 0")
        if not (s["normal_min"] <= s["normal_mean"] <= s["normal_max"]):
            raise ValueError("normal.mean must lie inside [min, max]")
        if s["normal_min"] < 0 or s["normal_max"] > 65535:
            raise ValueError("normal range must fit in 0..65535 us")

    elif mode == "mixture":
        comps = s["mixture_components"]
        if len(comps) == 0:
            raise ValueError("mixture.components must not be empty")
        for i, c in enumerate(comps):
            if c.get("weight", 0) <= 0:
                raise ValueError(f"mixture.components[{i}].weight must be > 0")
            if c.get("stddev", 0) <= 0:
                raise ValueError(f"mixture.components[{i}].stddev must be > 0")
        if s["mixture_min"] < 0 or s["mixture_max"] > 65535:
            raise ValueError("mixture range must fit in 0..65535 us")
        if s["mixture_min"] > s["mixture_max"]:
            raise ValueError("mixture.min must be <= mixture.max")

    elif mode == "cauchy":
        if s["cauchy_gamma"] <= 0:
            raise ValueError("cauchy.gamma must be > 0")
        if s["cauchy_min"] > s["cauchy_max"]:
            raise ValueError("cauchy.min must be <= cauchy.max")
        if s["cauchy_min"] < 0 or s["cauchy_max"] > 65535:
            raise ValueError("cauchy range must fit in 0..65535 us")

    Path(s["delay_base_dir"]).mkdir(parents=True, exist_ok=True)


# ---- Distribution generators ----

def generate_uniform(s: dict) -> list:
    n = s["num_values"]
    lo, hi = s["uniform_min"], s["uniform_max"]
    if s["round_to_int"]:
        return [random.randint(lo, hi) for _ in range(n)]
    return [random.uniform(lo, hi) for _ in range(n)]


def generate_normal(s: dict) -> list:
    values = []
    n = s["num_values"]
    mu, sigma = s["normal_mean"], s["normal_stddev"]
    lo, hi = s["normal_min"], s["normal_max"]
    round_int = s["round_to_int"]
    while len(values) < n:
        v = random.gauss(mu, sigma)
        if lo <= v <= hi:
            values.append(int(round(v)) if round_int else v)
    return values


def generate_mixture(s: dict) -> list:
    values = []
    n = s["num_values"]
    comps = s["mixture_components"]
    lo, hi = s["mixture_min"], s["mixture_max"]
    round_int = s["round_to_int"]

    # Normalize weights
    total_w = sum(c["weight"] for c in comps)
    cum_weights = []
    cumsum = 0.0
    for c in comps:
        cumsum += c["weight"] / total_w
        cum_weights.append(cumsum)

    while len(values) < n:
        # Select component
        r = random.random()
        idx = 0
        for i, cw in enumerate(cum_weights):
            if r <= cw:
                idx = i
                break
        comp = comps[idx]

        v = random.gauss(comp["mean"], comp["stddev"])
        if lo <= v <= hi:
            values.append(int(round(v)) if round_int else v)
    return values


def generate_cauchy(s: dict) -> list:
    values = []
    n = s["num_values"]
    x0 = s["cauchy_x0"]
    gamma = s["cauchy_gamma"]
    lo, hi = s["cauchy_min"], s["cauchy_max"]
    round_int = s["round_to_int"]

    while len(values) < n:
        # Cauchy: x0 + gamma * tan(pi * (U - 0.5))
        u = random.random()
        v = x0 + gamma * math.tan(math.pi * (u - 0.5))
        if lo <= v <= hi:
            values.append(int(round(v)) if round_int else v)
    return values


GENERATORS = {
    "uniform": generate_uniform,
    "normal": generate_normal,
    "mixture": generate_mixture,
    "cauchy": generate_cauchy,
}


def save_csv(values: list, path: str, column_name: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([column_name])
        for v in values:
            writer.writerow([v])


def save_histogram(values: list, s: dict) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=s["histogram_bins"], edgecolor="black", alpha=0.7)
    plt.title(f"Delay Value Distribution ({s['distribution_mode']})")
    plt.xlabel("Delay [us]")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(s["histogram_output_path"], dpi=200)
    plt.close()


def summarize(values: list, s: dict) -> str:
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(var)

    lines = [
        "Delay Value Generation Summary",
        "==============================",
        f"num_values         : {s['num_values']}",
        f"distribution_mode  : {s['distribution_mode']}",
        f"round_to_int       : {s['round_to_int']}",
        "",
    ]

    mode = s["distribution_mode"]
    if mode == "uniform":
        lines += [
            f"uniform.min        : {s['uniform_min']}",
            f"uniform.max        : {s['uniform_max']}",
        ]
    elif mode == "normal":
        lines += [
            f"normal.mean        : {s['normal_mean']}",
            f"normal.stddev      : {s['normal_stddev']}",
            f"normal.min         : {s['normal_min']}",
            f"normal.max         : {s['normal_max']}",
        ]
    elif mode == "mixture":
        for i, c in enumerate(s["mixture_components"]):
            lines += [
                f"mixture[{i}].weight  : {c['weight']}",
                f"mixture[{i}].mean    : {c['mean']}",
                f"mixture[{i}].stddev  : {c['stddev']}",
            ]
        lines += [
            f"mixture.min        : {s['mixture_min']}",
            f"mixture.max        : {s['mixture_max']}",
        ]
    elif mode == "cauchy":
        lines += [
            f"cauchy.x0          : {s['cauchy_x0']}",
            f"cauchy.gamma       : {s['cauchy_gamma']}",
            f"cauchy.min         : {s['cauchy_min']}",
            f"cauchy.max         : {s['cauchy_max']}",
        ]

    lines += [
        "",
        f"generated count    : {n}",
        f"observed min       : {min(values)}",
        f"observed max       : {max(values)}",
        f"observed mean      : {mean:.6f}",
        f"observed stddev    : {std:.6f}",
        "",
        f"csv_output_path    : {s['csv_output_path']}",
        f"histogram_output   : {s['histogram_output_path']}",
    ]
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) > 2:
        print("Usage: python gen_delay.py [path/to/config/delay_config.jsonc]")
        return 2

    config_path = sys.argv[1] if len(sys.argv) == 2 else "config/delay_config.jsonc"
    s = load_settings(config_path)
    validate_settings(s)

    if s["use_random_seed"]:
        random.seed(s["random_seed"])

    generator = GENERATORS[s["distribution_mode"]]
    values = generator(s)

    save_csv(values, s["csv_output_path"], s["csv_column_name"])
    save_histogram(values, s)

    summary = summarize(values, s)
    with open(s["summary_output_path"], "w", encoding="utf-8") as f:
        f.write(summary)

    snapshot_keys_prefix = (
        "uniform", "normal", "mixture", "cauchy",
        "distribution_mode", "round_to_int",
        "use_random_seed", "random_seed",
        "num_values", "folder_name_suffix",
        "delay_folder_name",
    )
    snapshot = {k: v for k, v in s.items() if k.startswith(snapshot_keys_prefix)}
    with open(s["config_snapshot_path"], "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    print(summary)
    print(f"\ndelay_folder_name      : {s['delay_folder_name']}")
    print(f"delay_base_dir         : {s['delay_base_dir']}")
    print(f"config_snapshot_path   : {s['config_snapshot_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
