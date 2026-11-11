#!/usr/bin/env python3
"""
PC-side controller for Arduino UNO AES side-channel acquisition.

JSONC configuration version:
- All runtime settings are loaded from sca_config.jsonc.
- plaintexts_per_batch is auto-calculated: osc_segments // avg_count
- Delay is either disabled (all 0us) or loaded from a pre-generated CSV.

Protocol:
  SET_MASK_CONST payload = R_IN(1) + R_OUT(1)
  LOAD_PT payload = plaintext(16) + mask(16) + delay_us(2)
  For AES-128_TI_* modes, mask_source="none" sends a zero mask placeholder;
  the TI firmware ignores those 16 bytes and uses its internal PRNG shares.

Usage:
  python auto_meas.py [path/to/sca_config.jsonc]
  (default: config/sca_config.jsonc)
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import secrets
import shutil
import struct
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

try:
    import serial
except Exception:
    serial = None

try:
    import h5py
    import numpy as np
except Exception:
    h5py = None
    np = None

try:
    import pyvisa
except Exception:
    pyvisa = None

from jsonc_loader import load_jsonc

time.sleep(2)


MAGIC = 0xA55A
CMD_HELLO = 0x01
CMD_SET_KEY = 0x02
CMD_SET_MODE = 0x03
CMD_SET_MASK_CONST = 0x04
CMD_BEGIN_BATCH = 0x10
CMD_LOAD_PT = 0x11
CMD_RUN_CURRENT = 0x12
CMD_END_BATCH = 0x13
CMD_GET_RESULT = 0x14
CMD_ABORT = 0x7E
CMD_PING = 0x7F

STATUS_OK = 0x00
STATUS_ERR = 0xFF

HEADER_FMT = "<HBBH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

TI_AES_IMPL_FILES = {
    "AES-128_TI.h",
    "AES-128_TI_1Round.h",
    "AES-128_TI_FullRound.h",
}


def is_ti_aes_impl(aes_impl_file: str) -> bool:
    return aes_impl_file in TI_AES_IMPL_FILES


@dataclass
class CaptureConfig:
    # Paths
    plaintext_file_path: str
    scope_save_dir: str
    save_prefix: str
    output_dir: str

    # Oscilloscope
    scope_resource: str
    scope_channel: str
    osc_segments: int
    scope_arm_wait_s: float
    scope_stop_wait_s: float
    scope_timeout_ms: int

    # Arduino
    serial_port: str
    baudrate: int
    serial_timeout_s: float
    aes_key_hex: str
    aes_impl_file: str
    avg_count: int

    # Options
    total_plaintexts: int
    plaintext_source: str          # "random" or "file"
    mask_source: str               # "fixed", "file", "random", or "none" (TI only)
    mask_rin_hex: str
    mask_rout_hex: str
    mask_file_path: str
    mask_fixed_hex: str
    save_mask_plan: bool
    mask_output_path: str
    use_delay: bool                # true = use CSV, false = all 0us
    delay_csv_column_name: str
    save_ciphertexts: bool
    use_dummy_scope: bool

    # Delay folder (delay/<folder>/generated_delay_values.csv が読まれる)
    delay_folder_name: str = ""

    # Post-process options
    delete_capture_logs_after_meas: bool = False
    save_verify_timeout_s: float = 0.0
    save_verify_poll_s: float = 2.0
    save_stable_checks: int = 2
    past_config_name: str = ""
    mask_random_seed: Optional[int] = None

    # Auto-calculated
    plaintexts_per_batch: int = 0

    def __post_init__(self):
        self.plaintexts_per_batch = self.osc_segments // self.avg_count

    def validate(self) -> None:
        if self.osc_segments % self.avg_count != 0:
            raise ValueError("osc_segments must be divisible by avg_count")

        if self.total_plaintexts % self.plaintexts_per_batch != 0:
            raise ValueError(
                f"total_plaintexts ({self.total_plaintexts}) must be divisible by "
                f"plaintexts_per_batch ({self.plaintexts_per_batch})"
            )

        key = bytes.fromhex(self.aes_key_hex)
        if len(key) != 16:
            raise ValueError("aes_key_hex must be exactly 16 bytes (32 hex chars)")

        if not self.aes_impl_file:
            raise ValueError("aes_impl_file must not be empty")
        if "/" in self.aes_impl_file or "\\" in self.aes_impl_file:
            raise ValueError("aes_impl_file must be a file name, not a path")
        if not self.aes_impl_file.endswith(".h"):
            raise ValueError("aes_impl_file must be a header file name ending with .h")
        try:
            aes_impl_file_bytes = self.aes_impl_file.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("aes_impl_file must contain ASCII characters only") from exc
        if len(aes_impl_file_bytes) > 31:
            raise ValueError("aes_impl_file must be 31 ASCII bytes or fewer")

        if self.plaintext_source not in {"random", "file"}:
            raise ValueError("plaintext_source must be one of: random, file")
        if self.plaintext_source == "file" and not self.plaintext_file_path:
            raise ValueError("plaintext_file_path must be set when plaintext_source == 'file'")
        if self.mask_source not in {"fixed", "file", "random", "none"}:
            raise ValueError("mask_source must be one of: fixed, file, random, none")
        parse_hex_byte(self.mask_rin_hex, "mask_rin_hex")
        parse_hex_byte(self.mask_rout_hex, "mask_rout_hex")
        if self.mask_source == "none" and self.aes_impl_file != "AES-128.h" and not is_ti_aes_impl(self.aes_impl_file):
            raise ValueError("mask_source == 'none' is only supported for AES-128.h or AES-128_TI_* modes")
        if self.mask_source == "file" and not self.mask_file_path:
            raise ValueError("mask_file_path must be set when mask_source == 'file'")
        if self.mask_source == "fixed":
            parse_hex16_string(self.mask_fixed_hex, "mask_fixed_hex")

        if self.use_delay and not self.delay_folder_name:
            raise ValueError("delay_folder_name must be set when use_delay is true")

        if not self.scope_channel:
            raise ValueError("scope_channel must not be empty")
        if self.save_verify_timeout_s < 0:
            raise ValueError("save_verify_timeout_s must be >= 0 (0 means wait forever)")
        if self.save_verify_poll_s <= 0:
            raise ValueError("save_verify_poll_s must be > 0")
        if self.save_stable_checks < 0:
            raise ValueError("save_stable_checks must be >= 0")
        if self.past_config_name:
            invalid_chars = set('<>:"/\\|?*')
            if (
                any(ch in invalid_chars or ord(ch) < 32 for ch in self.past_config_name)
                or self.past_config_name in {".", ".."}
            ):
                raise ValueError(
                    'past_config_name must be a file base name without path separators '
                    'or Windows-invalid characters: <>:"/\\|?*'
                )
        if not self.use_dummy_scope and (h5py is None or np is None):
            raise RuntimeError(
                "h5py and numpy are required to verify saved H5 waveforms. "
                "Install them with: python -m pip install h5py numpy"
            )


class ProtocolError(RuntimeError):
    def __init__(self, message: str, error_payload: bytes | None = None):
        super().__init__(message)
        self.error_payload = error_payload


class ArduinoLink:
    def __init__(self, port: str, baudrate: int, timeout_s: float):
        if serial is None:
            raise RuntimeError("pyserial is required to communicate with Arduino. Install it with: python -m pip install pyserial")
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout_s)
        time.sleep(2)
        self.seq = 0

    def close(self) -> None:
        self.ser.close()

    def _build_packet(self, cmd: int, payload: bytes) -> bytes:
        pkt = struct.pack(HEADER_FMT, MAGIC, cmd, self.seq, len(payload)) + payload
        self.seq = (self.seq + 1) & 0xFF
        return pkt

    def _read_exact(self, n: int) -> bytes:
        data = self.ser.read(n)
        if len(data) != n:
            raise ProtocolError(f"serial timeout: expected {n} bytes, got {len(data)}")
        return data

    def transact(self, cmd: int, payload: bytes = b"") -> bytes:
        self.ser.write(self._build_packet(cmd, payload))
        self.ser.flush()

        hdr = self._read_exact(HEADER_SIZE)
        magic, resp_cmd, _seq, plen = struct.unpack(HEADER_FMT, hdr)
        if magic != MAGIC:
            raise ProtocolError(f"invalid magic in response: 0x{magic:04X}")

        resp = self._read_exact(plen)
        if resp_cmd == STATUS_ERR:
            raise ProtocolError(f"arduino returned error payload: {resp!r}", resp)
        if resp_cmd != STATUS_OK:
            raise ProtocolError(f"unexpected response cmd: 0x{resp_cmd:02X}")
        return resp

    def hello(self) -> str:
        return self.transact(CMD_HELLO).decode("ascii", errors="replace")

    def set_key(self, key16: bytes) -> None:
        self.transact(CMD_SET_KEY, key16)

    def set_aes_impl(self, aes_impl_file: str) -> None:
        payload = aes_impl_file.encode("ascii")
        legacy_modes = {
            "AES-128.h": 0,
            "AES-128_Masked.h": 1,
            "Another_AES-128.h": 2,
            "Sakagami_AES-128_Masked.h": 3,
            "AES-128_TI_1Round.h": 0,
            "AES-128_TI.h": 0,
            "AES-128_TI_FullRound.h": 1,
        }
        try:
            resp = self.transact(CMD_SET_MODE, payload)
        except ProtocolError as exc:
            if exc.error_payload != b"SET_MODE val" or aes_impl_file not in legacy_modes:
                raise

            legacy_payload = bytes([legacy_modes[aes_impl_file]])
            try:
                resp = self.transact(CMD_SET_MODE, legacy_payload)
            except ProtocolError as legacy_exc:
                raise ProtocolError(
                    "SET_MODE failed with both supported payload formats: "
                    f"header-name payload={payload!r} error={exc.error_payload!r}; "
                    f"legacy-byte payload={legacy_payload!r} error={legacy_exc.error_payload!r}. "
                    "The Arduino firmware is probably not compatible with the requested "
                    "AES implementation; upload AES-128_Selectable_ArduinoUno/ or "
                    "AES-128_TI_ArduinoUno/ as appropriate."
                ) from legacy_exc

        selected = resp.decode("ascii", errors="replace")
        if selected == aes_impl_file:
            return
        if aes_impl_file == "AES-128_TI.h" and selected == "AES-128_TI_1Round.h":
            return
        if resp == payload or resp == bytes([legacy_modes.get(aes_impl_file, 255)]):
            return
        if not selected and aes_impl_file in legacy_modes:
            return
        if selected != aes_impl_file:
            raise ProtocolError(f"unexpected AES implementation response: {selected!r}")

    def set_mask_constants(self, r_in: int, r_out: int) -> None:
        resp = self.transact(CMD_SET_MASK_CONST, bytes([r_in, r_out]))
        if resp != bytes([r_in, r_out]):
            raise ProtocolError(f"unexpected mask constant response: {resp!r}")

    def begin_batch(self, batch_index: int, batch_plaintexts: int, avg_count: int) -> None:
        self.transact(CMD_BEGIN_BATCH, struct.pack("<IHH", batch_index, batch_plaintexts, avg_count))

    def load_plaintext(self, pt16: bytes, mask16: bytes, delay_us: int) -> None:
        if len(pt16) != 16:
            raise ValueError("plaintext must be 16 bytes")
        if len(mask16) != 16:
            raise ValueError("mask must be 16 bytes")
        if not (0 <= delay_us <= 65535):
            raise ValueError("delay_us must be in 0..65535")
        self.transact(CMD_LOAD_PT, pt16 + mask16 + struct.pack("<H", delay_us))

    def run_current(self) -> bytes:
        resp = self.transact(CMD_RUN_CURRENT)
        if len(resp) != 16:
            raise ProtocolError(f"unexpected ciphertext length: {len(resp)}")
        return resp

    def end_batch(self) -> None:
        self.transact(CMD_END_BATCH)

    def abort(self) -> None:
        self.transact(CMD_ABORT)


class ScopeController:
    def __init__(self, resource_name: str, timeout_ms: int):
        if pyvisa is None:
            raise RuntimeError("pyvisa is not installed. Install pyvisa and pyvisa-py.")
        self.rm = pyvisa.ResourceManager()
        self.inst = self.rm.open_resource(resource_name)
        self.inst.timeout = timeout_ms
        self.inst.write_termination = "\n"
        self.inst.read_termination = "\n"

    def close(self) -> None:
        try:
            self.inst.close()
        finally:
            self.rm.close()

    def arm_for_batch(self, segments: int, wait_s: float) -> None:
        self.inst.write(":CDISplay")
        self.inst.write(":ACQuire:MODE SEGMented")
        self.inst.write(f":ACQuire:SEGMented:COUNt {segments}")
        self.inst.write("SINGle")
        time.sleep(wait_s)

    def finalize_and_save(self, channel: str, path: str, stop_wait_s: float) -> None:
        self.inst.write("STOP")
        time.sleep(stop_wait_s)
        self.inst.write(":DISK:SEGMented ALL")
        old_timeout = self.inst.timeout
        self.inst.write(f':DISK:SAVE:WAVEFORM {channel},"{path}",H5,OFF')
        try:
            # Saving segmented H5 files can take much longer than normal SCPI
            # traffic.  Use an infinite timeout for the save-completion query
            # so a slow file write does not interrupt the measurement sequence.
            self.inst.timeout = None
            resp = self.inst.query("*OPC?").strip()
            if resp != "1":
                raise RuntimeError(f"scope did not finish operation, got: {resp!r}")
        finally:
            self.inst.timeout = old_timeout


class DummyScopeController:
    def __init__(self, resource_name: str, timeout_ms: int):
        self.resource_name = resource_name
        self.timeout_ms = timeout_ms

    def close(self) -> None:
        pass

    def arm_for_batch(self, segments: int, wait_s: float) -> None:
        print(f"[dummy-scope] arm {self.resource_name}; segmented count {segments}; wait {wait_s}s")

    def finalize_and_save(self, channel: str, path: str, stop_wait_s: float) -> None:
        print(f"[dummy-scope] stop; wait {stop_wait_s}s; save all segments {channel} -> {path}")


def random_plaintext() -> bytes:
    return bytes(random.getrandbits(8) for _ in range(16))


def random_block(rng: random.Random | None = None) -> bytes:
    source = rng if rng is not None else random
    return bytes(source.getrandbits(8) for _ in range(16))


def parse_hex_byte(value: str, label: str) -> int:
    text = str(value).strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    if len(text) != 2:
        raise ValueError(f"{label} must be one hex byte")
    try:
        return int(text, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be one hex byte") from exc


def parse_hex16_string(value: str, label: str) -> bytes:
    text = str(value).strip()
    tokens = text.split()
    if len(tokens) == 16:
        return bytes(parse_hex_byte(tok, label) for tok in tokens)
    compact = text.replace("0x", "").replace("0X", "").replace(",", "").replace("_", "")
    compact = "".join(compact.split())
    try:
        block = bytes.fromhex(compact)
    except ValueError as exc:
        raise ValueError(f"{label} must contain exactly 16 hex bytes") from exc
    if len(block) != 16:
        raise ValueError(f"{label} must contain exactly 16 hex bytes")
    return block


def load_hex16_file(path: str, label: str) -> list[bytes]:
    """Load 16-byte hex rows from a text file.

    Supported formats:
      - 16 hex bytes separated by spaces/tabs, e.g.:
        06 12 d9 01 df 5c 2d 27 02 3a 93 03 04 10 7d 47
      - compact 128-bit hex, e.g.:
        0612d901df5c2d27023a930304107d47
    """
    blocks: list[bytes] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) == 1:
                blocks.append(parse_hex16_string(tokens[0], f"{label} line {line_no}"))
            elif len(tokens) == 16:
                blocks.append(bytes(parse_hex_byte(tok, f"{label} line {line_no}") for tok in tokens))
            else:
                raise ValueError(
                    f"{label} line {line_no}: expected compact 32-hex-char value "
                    f"or 16 separated hex bytes, got {len(tokens)} fields"
                )
    return blocks


def load_plaintexts_from_file(path: str) -> list[bytes]:
    return load_hex16_file(path, "plaintext file")


def load_masks_from_file(path: str) -> list[bytes]:
    return load_hex16_file(path, "mask file")


def hex16(b: bytes) -> str:
    return b.hex()


def hex16_spaced(b: bytes) -> str:
    return " ".join(f"{x:02x}" for x in b)


def save_hex16_plan(path: str, blocks: list[bytes]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        for block in blocks:
            f.write(hex16_spaced(block) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(out_path)


def save_batch_log(out_csv: Path, rows: list[dict]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")
    with tmp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    tmp_csv.replace(out_csv)


_num_pat = re.compile(r"(\d+)")


def _num_key(path: str) -> int:
    m = None
    for m in _num_pat.finditer(path):
        pass
    return int(m.group(1)) if m else 1 << 30


def ensure_output_dirs(cfg: CaptureConfig) -> None:
    for label, path in (
        ("output_dir", cfg.output_dir),
        ("scope_save_dir", cfg.scope_save_dir),
    ):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if not p.is_dir():
            raise RuntimeError(f"{label} is not a directory after mkdir: {p}")
        print(f"Ensured {label}: {p}")


def ensure_waveform_save_parent(path: str) -> None:
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir():
        raise RuntimeError(f"waveform save directory is not a directory after mkdir: {parent}")


def past_config_path(config_path: str, cfg: CaptureConfig) -> Path:
    base_name = cfg.past_config_name.strip() if cfg.past_config_name else Path(config_path).stem
    if not base_name:
        raise ValueError("past_config_name must not be empty")
    file_name = base_name if base_name.endswith(".jsonc") else f"{base_name}.jsonc"
    return Path(config_path).parent / "past_config" / file_name


def count_h5_waveform_segments(h5_path: Path) -> tuple[int, Optional[int]]:
    if h5py is None or np is None:
        raise RuntimeError("h5py and numpy are required to verify saved H5 waveforms")

    with h5py.File(h5_path, "r") as h:
        cands: list[str] = []
        lengths: list[int] = []

        def _cb(name, obj):
            if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number):
                if obj.ndim == 1:
                    cands.append(name)
                    lengths.append(obj.size)

        h.visititems(_cb)

    if not cands:
        return 0, None

    from collections import Counter

    common_len, _ = Counter(lengths).most_common(1)[0]
    same_len = [name for name, length in zip(cands, lengths) if length == common_len]
    same_len.sort(key=_num_key)
    return len(same_len), common_len


def wait_for_saved_waveform(
    path: str,
    expected_segments: int,
    timeout_s: float,
    poll_s: float,
    stable_checks: int,
) -> tuple[int, Optional[int]]:
    h5_path = Path(path)
    deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
    last_size = -1
    stable_count = 0
    last_error: Exception | None = None

    while True:
        if h5_path.is_file():
            size = h5_path.stat().st_size
            if size > 0 and size == last_size:
                stable_count += 1
            else:
                stable_count = 0
                last_size = size

            if stable_count >= stable_checks:
                try:
                    segment_count, sample_points = count_h5_waveform_segments(h5_path)
                    if segment_count == expected_segments:
                        return segment_count, sample_points
                    last_error = RuntimeError(
                        f"segment count {segment_count}, expected {expected_segments}"
                    )
                except Exception as exc:
                    last_error = exc

        if deadline is not None and time.monotonic() >= deadline:
            detail = f"; last error: {last_error}" if last_error else ""
            raise RuntimeError(
                f"waveform save verification timed out: {h5_path}{detail}"
            )

        detail = f" ({last_error})" if last_error else ""
        print(f"  waiting for waveform save verification: {h5_path}{detail}", flush=True)
        time.sleep(poll_s)


def verify_batch_log(out_csv: Path, expected_rows: int) -> None:
    if not out_csv.is_file():
        raise RuntimeError(f"plaintext log was not saved: {out_csv}")
    with out_csv.open("r", newline="", encoding="utf-8") as f:
        row_count = sum(1 for _ in csv.DictReader(f))
    if row_count != expected_rows:
        raise RuntimeError(
            f"plaintext log row count mismatch: {out_csv} has {row_count}, "
            f"expected {expected_rows}"
        )


def verify_measurement_outputs(
    cfg: CaptureConfig,
    output_dir: Path,
    num_batches: int,
) -> tuple[list[int], list[int]]:
    missing_waveforms: list[int] = []
    missing_logs: list[int] = []

    for batch_idx in range(num_batches):
        waveform_path = Path(cfg.scope_save_dir) / f"{cfg.save_prefix}{batch_idx}.h5"
        log_path = output_dir / f"plaintext_batch_{batch_idx:05d}.csv"
        if not waveform_path.is_file():
            missing_waveforms.append(batch_idx)
            continue
        try:
            segment_count, _ = count_h5_waveform_segments(waveform_path)
            if segment_count != cfg.osc_segments:
                missing_waveforms.append(batch_idx)
        except Exception:
            missing_waveforms.append(batch_idx)
        if not log_path.is_file():
            missing_logs.append(batch_idx)
        else:
            try:
                verify_batch_log(log_path, cfg.plaintexts_per_batch)
            except Exception:
                missing_logs.append(batch_idx)

    return missing_waveforms, missing_logs


def load_delay_list_from_csv(path: str, column_name: str) -> list[int]:
    delays: list[int] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if column_name not in (reader.fieldnames or []):
            raise ValueError(f"CSV must contain column '{column_name}'")
        for row in reader:
            d = int(row[column_name])
            if not (0 <= d <= 65535):
                raise ValueError(f"delay value out of range: {d}")
            delays.append(d)
    return delays


def build_delay_plan(cfg: CaptureConfig) -> list[int]:
    if not cfg.use_delay:
        return [0] * cfg.total_plaintexts

    delay_csv = Path("delay") / cfg.delay_folder_name / "generated_delay_values.csv"
    delays = load_delay_list_from_csv(str(delay_csv), cfg.delay_csv_column_name)
    if len(delays) < cfg.total_plaintexts:
        raise ValueError(
            f"delay CSV rows ({len(delays)}) are fewer than total_plaintexts ({cfg.total_plaintexts})"
        )
    return delays[: cfg.total_plaintexts]


def build_mask_plan(cfg: CaptureConfig) -> list[bytes]:
    if cfg.mask_source == "none":
        return [b"\x00" * 16] * cfg.total_plaintexts

    if cfg.mask_source == "fixed":
        fixed = parse_hex16_string(cfg.mask_fixed_hex, "mask_fixed_hex")
        return [fixed] * cfg.total_plaintexts

    if cfg.mask_source == "file":
        masks = load_masks_from_file(cfg.mask_file_path)
        if len(masks) < cfg.total_plaintexts:
            raise ValueError(
                f"mask file has {len(masks)} entries, "
                f"but total_plaintexts requires {cfg.total_plaintexts}"
            )
        return masks[: cfg.total_plaintexts]

    byte_count = cfg.total_plaintexts * 16
    if cfg.mask_random_seed is None:
        data = secrets.token_bytes(byte_count)
    else:
        rng = random.Random(cfg.mask_random_seed)
        data = bytes(rng.getrandbits(8) for _ in range(byte_count))
    return [data[i * 16 : (i + 1) * 16] for i in range(cfg.total_plaintexts)]


def load_config(path: str) -> CaptureConfig:
    data = load_jsonc(path)
    allowed = {f.name for f in CaptureConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in allowed}
    return CaptureConfig(**filtered)


def main() -> int:
    if len(sys.argv) > 2:
        print("Usage: python auto_meas.py [path/to/config/sca_config.jsonc]")
        return 2

    config_path = sys.argv[1] if len(sys.argv) == 2 else "config/sca_config.jsonc"
    cfg = load_config(config_path)
    cfg.validate()
    ensure_output_dirs(cfg)

    past_config = past_config_path(config_path, cfg)
    past_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, past_config)
    print(f"Config backed up: {past_config}")

    output_dir = Path(cfg.output_dir)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    key = bytes.fromhex(cfg.aes_key_hex)
    num_batches = cfg.total_plaintexts // cfg.plaintexts_per_batch
    delay_plan = build_delay_plan(cfg)
    mask_plan = build_mask_plan(cfg)
    mask_rin = parse_hex_byte(cfg.mask_rin_hex, "mask_rin_hex")
    mask_rout = parse_hex_byte(cfg.mask_rout_hex, "mask_rout_hex")
    if cfg.save_mask_plan:
        save_hex16_plan(cfg.mask_output_path, mask_plan)
        print(f"Saved mask plan: {cfg.mask_output_path}")

    # Load or prepare plaintext plan
    if cfg.plaintext_source == "file":
        plaintext_plan = load_plaintexts_from_file(cfg.plaintext_file_path)
        if len(plaintext_plan) < cfg.total_plaintexts:
            raise ValueError(
                f"plaintext file has {len(plaintext_plan)} entries, "
                f"but total_plaintexts requires {cfg.total_plaintexts}"
            )
    else:
        plaintext_plan = None

    print(f"config_path            : {config_path}")
    print(f"serial_port            : {cfg.serial_port}")
    print(f"plaintexts_per_batch   : {cfg.plaintexts_per_batch} (auto: {cfg.osc_segments} / {cfg.avg_count})")
    print(f"num_batches            : {num_batches}")
    print(f"aes_impl_file          : {cfg.aes_impl_file}")
    print(f"mask_source            : {cfg.mask_source}")
    print(f"mask_rin/rout          : 0x{mask_rin:02x} / 0x{mask_rout:02x}")
    if is_ti_aes_impl(cfg.aes_impl_file):
        print("ti_randomness          : internal fixed-seed Xorshift128 streams")
        if cfg.mask_source == "none":
            print("load_pt_mask           : zero placeholder; ignored by TI firmware")
    print(f"use_delay              : {cfg.use_delay}")
    print(f"plaintext_source       : {cfg.plaintext_source}")
    print(f"use_dummy_scope        : {cfg.use_dummy_scope}")
    print(f"scope_save_dir         : {cfg.scope_save_dir}")
    print(f"output_dir             : {output_dir}")
    print(f"save_verify_timeout_s  : {cfg.save_verify_timeout_s}")

    scope_cls = DummyScopeController if cfg.use_dummy_scope else ScopeController
    link = ArduinoLink(cfg.serial_port, cfg.baudrate, cfg.serial_timeout_s)
    scope = scope_cls(cfg.scope_resource, cfg.scope_timeout_ms)

    try:
        print(f"Arduino banner: {link.hello()}")
        link.set_key(key)
        link.set_aes_impl(cfg.aes_impl_file)
        link.set_mask_constants(mask_rin, mask_rout)

        global_pt_idx = 0
        all_rows: list[dict] = []
        for batch_idx in range(num_batches):
            print(f"\n=== batch {batch_idx + 1}/{num_batches} ===")
            batch_rows: list[dict] = []

            scope.arm_for_batch(cfg.osc_segments, cfg.scope_arm_wait_s)
            link.begin_batch(batch_idx, cfg.plaintexts_per_batch, cfg.avg_count)

            for pt_idx in range(cfg.plaintexts_per_batch):
                pt = plaintext_plan[global_pt_idx] if plaintext_plan is not None else random_plaintext()
                mask = mask_plan[global_pt_idx]
                delay_us = delay_plan[global_pt_idx]

                link.load_plaintext(pt, mask, delay_us)
                ct = link.run_current()

                row = {
                    "batch_index": batch_idx,
                    "plaintext_index": pt_idx,
                    "global_plaintext_index": global_pt_idx,
                    "aes_impl_file": cfg.aes_impl_file,
                    "avg_count": cfg.avg_count,
                    "delay_us": delay_us,
                    "plaintext_hex": hex16(pt),
                    "mask_hex": hex16(mask),
                    "mask_source": cfg.mask_source,
                    "mask_rin_hex": f"{mask_rin:02x}",
                    "mask_rout_hex": f"{mask_rout:02x}",
                    "ti_internal_prng": "xorshift128_fixed_seed" if is_ti_aes_impl(cfg.aes_impl_file) else "",
                }
                if cfg.save_ciphertexts:
                    row["ciphertext_hex"] = hex16(ct)
                batch_rows.append(row)
                all_rows.append(row.copy())
                global_pt_idx += 1

                if pt_idx % 20 == 0 or pt_idx == cfg.plaintexts_per_batch - 1:
                    print(f"  plaintext {pt_idx + 1}/{cfg.plaintexts_per_batch}  delay={delay_us} us")

            link.end_batch()

            waveform_path = str(Path(cfg.scope_save_dir) / f"{cfg.save_prefix}{batch_idx}.h5")
            ensure_waveform_save_parent(waveform_path)
            scope.finalize_and_save(cfg.scope_channel, waveform_path, cfg.scope_stop_wait_s)
            segment_count, sample_points = wait_for_saved_waveform(
                waveform_path,
                cfg.osc_segments,
                cfg.save_verify_timeout_s,
                cfg.save_verify_poll_s,
                cfg.save_stable_checks,
            )

            csv_path = output_dir / f"plaintext_batch_{batch_idx:05d}.csv"
            save_batch_log(csv_path, batch_rows)
            verify_batch_log(csv_path, cfg.plaintexts_per_batch)

            print(f"Saved plaintext log: {csv_path}")
            print(
                f"Saved waveform path: {waveform_path} "
                f"(segments={segment_count}, sample_points={sample_points})"
            )

        measurement_plan_path = Path(cfg.scope_save_dir) / f"{cfg.save_prefix}_measurement_plan.csv"
        save_batch_log(measurement_plan_path, all_rows)
        print(f"Saved measurement plan: {measurement_plan_path}")

        missing_waveforms, missing_logs = verify_measurement_outputs(
            cfg, output_dir, num_batches
        )
        print("\n=== save verification summary ===")
        print(f"expected waveform files : {num_batches}")
        print(f"expected log files      : {num_batches}")
        if missing_waveforms:
            print(f"[NG] missing/invalid waveform batches: {missing_waveforms}")
        else:
            print("[OK] all waveform files exist and have expected segment count")
        if missing_logs:
            print(f"[NG] missing plaintext log batches: {missing_logs}")
        else:
            print("[OK] all plaintext log files exist")

        if missing_waveforms or missing_logs:
            raise RuntimeError("measurement output verification failed")

        if cfg.delete_capture_logs_after_meas:
            shutil.rmtree(str(output_dir), ignore_errors=True)
            print(f"Deleted capture_logs: {output_dir}")

    except KeyboardInterrupt:
        print("\nInterrupted by user; sending ABORT to Arduino...")
        try:
            link.abort()
        except Exception:
            pass
        return 1
    finally:
        try:
            scope.close()
        except Exception:
            pass
        link.close()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
