#!/usr/bin/env python3
"""Small helpers for VITA49 example capture and packet analysis."""

from __future__ import annotations

import argparse
import base64
import json
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np


EXPECTED_CLASS_ID = 0xFA52530001000101


class UdpCapture:
    def __init__(self, host: str, port: int, output: Path) -> None:
        self.host = host
        self.port = port
        self.output = output
        self.packet_count = 0
        self.byte_count = 0
        self.error: str | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(3.0):
            raise RuntimeError("UDP capture did not start")
        if self.error:
            raise RuntimeError(self.error)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(5.0)
        if self._thread.is_alive():
            raise RuntimeError("UDP capture did not stop")
        if self.error:
            raise RuntimeError(self.error)

    def _run(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((self.host, self.port))
                sock.settimeout(0.1)
                self._ready.set()
                with self.output.open("w", encoding="utf-8") as handle:
                    while not self._stop.is_set():
                        try:
                            payload, addr = sock.recvfrom(65535)
                        except socket.timeout:
                            continue
                        self.packet_count += 1
                        self.byte_count += len(payload)
                        record = {
                            "index": self.packet_count,
                            "length": len(payload),
                            "received_monotonic": time.monotonic(),
                            "source": f"{addr[0]}:{addr[1]}",
                            "data_b64": base64.b64encode(payload).decode("ascii"),
                        }
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as exc:  # pragma: no cover - example diagnostics
            self.error = f"{type(exc).__name__}: {exc}"
            self._ready.set()


def add_capture_args(parser: argparse.ArgumentParser, default_port: int, default_fullscale: str = "1.0") -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--fullscale", default=default_fullscale)
    parser.add_argument("--epoch", default="1700000000123456789")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--capture-file", default="vita49_capture.jsonl")
    parser.add_argument("--run-json", default="vita49_run.json")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--threads", default="2")
    parser.add_argument("--timeout", type=float, default=60.0)


def run_capture(
    *,
    scenario_dir: Path,
    scenario_file: str,
    host: str,
    port: int,
    fullscale: str,
    epoch: str,
    out_dir: str,
    capture_file: str,
    run_json: str,
    log_level: str,
    threads: str,
    timeout: float,
    prepare_command: list[str] | None = None,
) -> int:
    if prepare_command:
        subprocess.run(prepare_command, cwd=scenario_dir, check=True)

    capture_path = scenario_dir / capture_file
    run_path = scenario_dir / run_json
    output_dir = scenario_dir / out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = UdpCapture(host, port, capture_path)
    capture.start()

    command = [
        "fers-cli",
        scenario_file,
        f"--out-dir={output_dir}",
        "--vita49",
        f"{host}:{port}",
        "--vita49-fullscale",
        fullscale,
        "--vita49-epoch",
        epoch,
        f"--log-level={log_level}",
        f"-n={threads}",
    ]

    start = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=scenario_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(command, 124, exc.stdout or "", exc.stderr or "timeout")
    elapsed = time.time() - start
    time.sleep(0.25)
    capture.stop()

    result = {
        "command": command,
        "cwd": str(scenario_dir),
        "exit_code": completed.returncode,
        "duration_seconds": elapsed,
        "captured_packets": capture.packet_count,
        "captured_bytes": capture.byte_count,
        "capture_file": str(capture_path),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    run_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"exit={completed.returncode} packets={capture.packet_count} bytes={capture.byte_count}")
    return completed.returncode


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from(">I", data, offset)[0]


def _u64(data: bytes, offset: int) -> int:
    return struct.unpack_from(">Q", data, offset)[0]


def _f64(data: bytes, offset: int) -> float:
    return struct.unpack_from(">d", data, offset)[0]


def decode_packet(data: bytes, index: int) -> dict:
    header = _u32(data, 0)
    packet_type = (header >> 28) & 0xF
    packet_count = (header >> 16) & 0xF
    size_words = header & 0xFFFF
    stream_id = _u32(data, 4)
    class_id = _u64(data, 8)
    timestamp = {
        "integer_seconds": _u32(data, 16),
        "fractional_picoseconds": _u64(data, 20),
    }
    issues: list[str] = []
    if size_words * 4 != len(data):
        issues.append("packet word count does not match datagram length")
    if class_id != EXPECTED_CLASS_ID:
        issues.append(f"unexpected class id 0x{class_id:016X}")

    packet = {
        "index": index,
        "packet_type": packet_type,
        "stream_id": stream_id,
        "stream_id_hex": f"0x{stream_id:08X}",
        "packet_count": packet_count,
        "timestamp": timestamp,
        "length": len(data),
        "issues": issues,
    }

    if packet_type == 1:
        payload = data[28:-4]
        iq = np.frombuffer(payload, dtype=">i2").astype(np.int16)
        trailer = _u32(data, len(data) - 4)
        packet.update(
            {
                "kind": "data",
                "sample_count": len(iq) // 2,
                "iq": iq.reshape(-1, 2) if len(iq) else np.empty((0, 2), dtype=np.int16),
                "trailer": trailer,
                "over_range": bool(trailer & (1 << 12)),
                "sample_loss": bool(trailer & (1 << 11)),
            }
        )
    elif packet_type == 4:
        metadata_text = data[92:].split(b"\0", 1)[0].decode("ascii", errors="replace")
        metadata = json.loads(metadata_text) if metadata_text else {}
        packet.update(
            {
                "kind": "context",
                "cif0": _u32(data, 28),
                "sample_rate": _f64(data, 44),
                "reference_frequency": _f64(data, 52),
                "adc_fullscale": _f64(data, 76),
                "receiver_id": _u64(data, 84),
                "metadata": metadata,
            }
        )
    else:
        packet["kind"] = "unknown"
    return packet


def load_capture(path: Path) -> list[dict]:
    packets: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            data = base64.b64decode(record["data_b64"])
            packets.append(decode_packet(data, int(record["index"])))
    return packets


def stream_summaries(packets: list[dict]) -> dict[int, dict]:
    summaries: dict[int, dict] = {}
    for packet in packets:
        stream_id = packet["stream_id"]
        summary = summaries.setdefault(
            stream_id,
            {
                "stream_id": stream_id,
                "stream_id_hex": f"0x{stream_id:08X}",
                "receiver_name": "unknown",
                "mode": "unknown",
                "sample_rate": None,
                "reference_frequency": None,
                "data_packets": 0,
                "context_packets": 0,
                "samples": 0,
                "packet_bytes": [],
                "over_range_packets": 0,
                "sample_loss_packets": 0,
                "iq_chunks": [],
                "issues": [],
            },
        )
        summary["packet_bytes"].append(packet["length"])
        summary["issues"].extend(packet.get("issues", []))
        if packet["kind"] == "context":
            metadata = packet.get("metadata", {})
            receiver = metadata.get("receiver", {})
            summary["context_packets"] += 1
            summary["receiver_name"] = receiver.get("name", summary["receiver_name"])
            summary["mode"] = receiver.get("mode", summary["mode"])
            summary["sample_rate"] = packet.get("sample_rate")
            summary["reference_frequency"] = packet.get("reference_frequency")
        elif packet["kind"] == "data":
            summary["data_packets"] += 1
            summary["samples"] += packet["sample_count"]
            summary["over_range_packets"] += int(packet["over_range"])
            summary["sample_loss_packets"] += int(packet["sample_loss"])
            summary["iq_chunks"].append(packet["iq"])

    for summary in summaries.values():
        if summary["iq_chunks"]:
            iq_pairs = np.vstack(summary["iq_chunks"])
            summary["iq_pairs"] = iq_pairs
            complex_iq = iq_pairs[:, 0].astype(np.float64) + 1j * iq_pairs[:, 1].astype(np.float64)
            summary["iq_complex"] = complex_iq / 32767.0
            summary["iq_min"] = int(iq_pairs.min())
            summary["iq_max"] = int(iq_pairs.max())
            summary["iq_rms"] = float(np.sqrt(np.mean(np.abs(summary["iq_complex"]) ** 2)))
        else:
            summary["iq_pairs"] = np.empty((0, 2), dtype=np.int16)
            summary["iq_complex"] = np.empty(0, dtype=np.complex128)
            summary["iq_min"] = 0
            summary["iq_max"] = 0
            summary["iq_rms"] = 0.0
        summary["min_packet_bytes"] = min(summary["packet_bytes"]) if summary["packet_bytes"] else 0
        summary["max_packet_bytes"] = max(summary["packet_bytes"]) if summary["packet_bytes"] else 0
    return summaries


def json_ready_summary(summary: dict) -> dict:
    return {
        key: value
        for key, value in summary.items()
        if key not in {"iq_chunks", "iq_pairs", "iq_complex", "packet_bytes"}
    } | {
        "min_packet_bytes": summary["min_packet_bytes"],
        "max_packet_bytes": summary["max_packet_bytes"],
    }
