#!/usr/bin/env python3
"""Validate the mixed-mode VITA49 capture against the radar scenario geometry."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vita49_common import load_capture, stream_summaries

C = 299_792_458.0
DURATION = 0.004

TARGET_START = np.array([120.0, 120.0, 50.0])
TARGET_VELOCITY = np.array([50.0, 0.0, 0.0])

PULSED_POS = np.array([0.0, 0.0, 0.0])
PULSED_FC = 10.0e9
PULSED_PRF = 10_000.0
PULSED_WINDOW_SKIP = 0.0
PULSED_WINDOW_LENGTH = 20.0e-6
PULSE_WIDTH = 1.0e-6

CW_POS = np.array([250.0, 0.0, 0.0])
CW_FC = 5.0e9

FMCW_POS = np.array([0.0, 250.0, 0.0])
FMCW_FC = 9.0e9
FMCW_BANDWIDTH = 5.0e4
FMCW_DURATION = 1.0e-3
FMCW_PERIOD = 1.0e-3
FMCW_START_OFFSET = -2.5e4
FMCW_CHIRP_COUNT = 4
FMCW_RATE = FMCW_BANDWIDTH / FMCW_DURATION


def target_position(t: float | np.ndarray) -> np.ndarray:
    return TARGET_START + np.asarray(t)[..., None] * TARGET_VELOCITY


def monostatic_range_doppler(t: float, radar_pos: np.ndarray, carrier_hz: float) -> tuple[float, float]:
    vec = target_position(float(t)) - radar_pos
    rng = float(np.linalg.norm(vec))
    radial_velocity = float(np.dot(TARGET_VELOCITY, vec / rng))
    doppler_hz = -2.0 * radial_velocity / (C / carrier_hz)
    return rng, doppler_hz


def alias_frequency(freq_hz: float, sample_hz: float) -> float:
    return float((freq_hz + sample_hz / 2.0) % sample_hz - sample_hz / 2.0)


def fit_frequency(times: np.ndarray, values: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    phase = np.unwrap(np.angle(values)) if np.iscomplexobj(values) else np.unwrap(np.asarray(values))
    x = times - np.mean(times)
    y = phase - np.mean(phase)
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y) / denom / (2.0 * np.pi))


def spectrum_peak(iq: np.ndarray, sample_rate: float, nfft_limit: int = 262_144) -> tuple[float, np.ndarray, np.ndarray]:
    if len(iq) < 8 or sample_rate <= 0.0:
        return 0.0, np.empty(0), np.empty(0)
    nfft = min(nfft_limit, 1 << (len(iq).bit_length() - 1))
    segment = iq[:nfft] * np.hanning(nfft)
    spectrum = np.fft.fftshift(np.fft.fft(segment))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    spectrum_db = 20.0 * np.log10(np.abs(spectrum) + 1.0e-30)
    spectrum_db -= float(np.max(spectrum_db))
    peak_hz = float(freqs[int(np.argmax(spectrum_db))])
    return peak_hz, freqs, spectrum_db


def stream_for_mode(summaries: dict[int, dict], mode: str) -> dict:
    matches = [summary for summary in summaries.values() if summary["mode"] == mode]
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one {mode} stream, decoded {len(matches)}")
    return matches[0]


def validate_pulsed(summary: dict) -> dict:
    fs = float(summary["sample_rate"] or 0.0)
    iq = summary["iq_complex"]
    fast_samples = int(round(PULSED_WINDOW_LENGTH * fs)) + 1
    pulse_count = min(int(round(DURATION * PULSED_PRF)) + 1, len(iq) // fast_samples)
    if fs <= 0.0 or fast_samples <= 0 or pulse_count < 8:
        raise SystemExit("Pulsed stream does not contain enough samples for range-Doppler validation")

    pulses = iq[: pulse_count * fast_samples].reshape(pulse_count, fast_samples)
    pulse_samples = max(1, int(round(PULSE_WIDTH * fs)))
    reference = np.ones(pulse_samples, dtype=np.complex128)
    compressed = np.array([np.convolve(row, np.conj(reference[::-1]), mode="full") for row in pulses])

    slow_window = np.hanning(pulse_count)[:, None]
    rd_map = np.fft.fftshift(np.fft.fft(compressed * slow_window, axis=0), axes=0)
    rd_mag = np.abs(rd_map)
    rd_db = 20.0 * np.log10(rd_mag + 1.0e-30)
    rd_db -= float(np.max(rd_db))

    range_axis = (PULSED_WINDOW_SKIP + (np.arange(compressed.shape[1]) - (pulse_samples - 1)) / fs) * C / 2.0
    doppler_axis = np.fft.fftshift(np.fft.fftfreq(pulse_count, d=1.0 / PULSED_PRF))

    valid_range = range_axis >= 0.0
    search_mag = rd_mag.copy()
    search_mag[:, ~valid_range] = 0.0
    peak_doppler_idx, peak_range_idx = np.unravel_index(int(np.argmax(search_mag)), search_mag.shape)
    measured_range = float(range_axis[peak_range_idx])
    measured_doppler = float(doppler_axis[peak_doppler_idx])

    t_mid = 0.5 * (pulse_count - 1) / PULSED_PRF
    expected_range, expected_doppler = monostatic_range_doppler(t_mid, PULSED_POS, PULSED_FC)
    expected_doppler = alias_frequency(expected_doppler, PULSED_PRF)

    range_bin = C / (2.0 * fs)
    doppler_bin = PULSED_PRF / pulse_count
    range_tolerance = max(2.0 * range_bin, 10.0)
    doppler_tolerance = max(2.0 * doppler_bin, 75.0)
    range_error = abs(measured_range - expected_range)
    doppler_error = abs(measured_doppler - expected_doppler)

    return {
        "receiver": summary["receiver_name"],
        "samples": int(summary["samples"]),
        "sample_rate_hz": fs,
        "pulse_count": int(pulse_count),
        "fast_samples_per_pulse": int(fast_samples),
        "expected_range_m": expected_range,
        "measured_range_m": measured_range,
        "range_error_m": float(range_error),
        "range_tolerance_m": float(range_tolerance),
        "expected_doppler_hz": expected_doppler,
        "measured_doppler_hz": measured_doppler,
        "doppler_error_hz": float(doppler_error),
        "doppler_tolerance_hz": float(doppler_tolerance),
        "passed": bool(range_error <= range_tolerance and doppler_error <= doppler_tolerance),
        "range_axis": range_axis,
        "doppler_axis": doppler_axis,
        "rd_db": rd_db,
    }


def validate_cw(summary: dict) -> dict:
    fs = float(summary["sample_rate"] or 0.0)
    iq = summary["iq_complex"]
    if fs <= 0.0 or len(iq) < 32:
        raise SystemExit("CW stream does not contain enough samples for Doppler validation")

    peak_hz, freqs, spectrum_db = spectrum_peak(iq, fs)
    expected_range, expected_doppler = monostatic_range_doppler(0.5 * DURATION, CW_POS, CW_FC)
    expected_doppler = alias_frequency(expected_doppler, fs)
    peak_error = abs(peak_hz - expected_doppler)
    bin_hz = fs / min(262_144, 1 << (len(iq).bit_length() - 1))
    tolerance = max(2.0 * bin_hz, 150.0)

    return {
        "receiver": summary["receiver_name"],
        "samples": int(summary["samples"]),
        "sample_rate_hz": fs,
        "expected_range_m": expected_range,
        "expected_doppler_hz": expected_doppler,
        "measured_doppler_hz": peak_hz,
        "spectrum_peak_hz": peak_hz,
        "doppler_error_hz": float(peak_error),
        "spectrum_peak_error_hz": float(peak_error),
        "doppler_tolerance_hz": tolerance,
        "passed": bool(peak_error <= tolerance),
        "freqs": freqs,
        "spectrum_db": spectrum_db,
    }


def fmcw_delay(times: np.ndarray) -> np.ndarray:
    ranges = np.linalg.norm(target_position(times) - FMCW_POS, axis=1)
    return 2.0 * ranges / C


def fmcw_reference_phase(times: np.ndarray) -> np.ndarray:
    local_time = times - np.floor(times / FMCW_PERIOD) * FMCW_PERIOD
    return 2.0 * np.pi * FMCW_START_OFFSET * local_time + np.pi * FMCW_RATE * local_time * local_time


def fmcw_received_phase(times: np.ndarray, delay: np.ndarray) -> np.ndarray:
    retarded_time = times - delay
    local_retarded = retarded_time - np.floor(retarded_time / FMCW_PERIOD) * FMCW_PERIOD
    return (
        -2.0 * np.pi * FMCW_FC * delay
        + 2.0 * np.pi * FMCW_START_OFFSET * local_retarded
        + np.pi * FMCW_RATE * local_retarded * local_retarded
    )


def fmcw_expected_if_phase(times: np.ndarray) -> np.ndarray:
    delay = fmcw_delay(times)
    return fmcw_reference_phase(times) - fmcw_received_phase(times, delay)


def validate_fmcw(summary: dict) -> dict:
    fs = float(summary["sample_rate"] or 0.0)
    iq = summary["iq_complex"]
    if fs <= 0.0 or len(iq) < 64:
        raise SystemExit("FMCW stream does not contain enough samples for IF validation")

    target_end = target_position(DURATION)
    max_range = max(float(np.linalg.norm(TARGET_START - FMCW_POS)), float(np.linalg.norm(target_end - FMCW_POS)))
    guard = max(10.0e-6, 4.0 * (2.0 * max_range / C))
    chirp_times = []
    measured = []
    expected = []

    for chirp in range(FMCW_CHIRP_COUNT):
        t0 = chirp * FMCW_PERIOD + guard
        t1 = chirp * FMCW_PERIOD + FMCW_DURATION - guard
        i0 = int(np.ceil(t0 * fs))
        i1 = int(np.floor(t1 * fs))
        if i1 <= i0 + 32 or i0 < 0 or i1 > len(iq):
            continue
        times = np.arange(i0, i1) / fs
        measured.append(fit_frequency(times, iq[i0:i1]))
        expected.append(fit_frequency(times, fmcw_expected_if_phase(times)))
        chirp_times.append(0.5 * (t0 + t1))

    if len(measured) < 4:
        raise SystemExit("FMCW stream does not contain enough complete chirps for IF validation")

    chirp_times_array = np.array(chirp_times)
    measured_array = np.array(measured)
    expected_array = np.array(expected)
    errors = measured_array - expected_array
    mae = float(np.mean(np.abs(errors)))
    max_error = float(np.max(np.abs(errors)))
    mae_tolerance = 75.0
    max_tolerance = 200.0

    return {
        "receiver": summary["receiver_name"],
        "samples": int(summary["samples"]),
        "sample_rate_hz": fs,
        "chirps_used": int(len(measured_array)),
        "expected_if_start_hz": float(expected_array[0]),
        "expected_if_end_hz": float(expected_array[-1]),
        "measured_if_start_hz": float(measured_array[0]),
        "measured_if_end_hz": float(measured_array[-1]),
        "if_mae_hz": mae,
        "if_max_error_hz": max_error,
        "if_mae_tolerance_hz": mae_tolerance,
        "if_max_tolerance_hz": max_tolerance,
        "passed": bool(mae <= mae_tolerance and max_error <= max_tolerance),
        "chirp_times": chirp_times_array,
        "measured_if_hz": measured_array,
        "expected_if_hz": expected_array,
    }


def json_ready(results: dict) -> dict:
    plot_only = {
        "range_axis",
        "doppler_axis",
        "rd_db",
        "freqs",
        "spectrum_db",
        "chirp_times",
        "measured_if_hz",
        "expected_if_hz",
    }
    cleaned = {}
    for key, value in results.items():
        if key in plot_only:
            continue
        if isinstance(value, dict):
            cleaned[key] = json_ready(value)
        elif isinstance(value, np.ndarray):
            cleaned[key] = value.tolist()
        elif isinstance(value, np.generic):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned


def write_text_report(path: Path, report: dict) -> None:
    pulsed = report["pulsed"]
    cw = report["cw"]
    fmcw = report["fmcw"]
    lines = [
        "VITA49 Mixed Modes Radar Simulation Analysis",
        "============================================",
        f"Overall: {report['overall_status']}",
        "",
        "Pulsed range-Doppler",
        f"  samples: {pulsed['samples']} at {pulsed['sample_rate_hz']:.1f} Hz",
        f"  range: expected {pulsed['expected_range_m']:.2f} m, measured {pulsed['measured_range_m']:.2f} m, error {pulsed['range_error_m']:.2f} m",
        f"  Doppler: expected {pulsed['expected_doppler_hz']:.2f} Hz, measured {pulsed['measured_doppler_hz']:.2f} Hz, error {pulsed['doppler_error_hz']:.2f} Hz",
        f"  result: {'PASS' if pulsed['passed'] else 'FAIL'}",
        "",
        "CW Doppler",
        f"  samples: {cw['samples']} at {cw['sample_rate_hz']:.1f} Hz",
        f"  target range at mid-run: {cw['expected_range_m']:.2f} m",
        f"  Doppler: expected {cw['expected_doppler_hz']:.2f} Hz, FFT peak {cw['spectrum_peak_hz']:.2f} Hz, error {cw['doppler_error_hz']:.2f} Hz",
        f"  result: {'PASS' if cw['passed'] else 'FAIL'}",
        "",
        "FMCW physical dechirp",
        f"  samples: {fmcw['samples']} at {fmcw['sample_rate_hz']:.1f} Hz",
        f"  IF start/end: expected {fmcw['expected_if_start_hz']:.2f} -> {fmcw['expected_if_end_hz']:.2f} Hz, measured {fmcw['measured_if_start_hz']:.2f} -> {fmcw['measured_if_end_hz']:.2f} Hz",
        f"  IF error: MAE {fmcw['if_mae_hz']:.2f} Hz, max {fmcw['if_max_error_hz']:.2f} Hz",
        f"  result: {'PASS' if fmcw['passed'] else 'FAIL'}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def make_plot(path: Path, pulsed: dict, cw: dict, fmcw: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    track_times = np.linspace(0.0, DURATION, 50)
    track = target_position(track_times)
    axes[0, 0].plot(track[:, 0], track[:, 1], "k-", label="target track")
    axes[0, 0].scatter([track[0, 0]], [track[0, 1]], marker="o", color="k", label="target start")
    axes[0, 0].scatter([track[-1, 0]], [track[-1, 1]], marker="x", color="k", label="target end")
    for label, pos, color in [
        ("Pulsed", PULSED_POS, "tab:blue"),
        ("CW", CW_POS, "tab:orange"),
        ("FMCW", FMCW_POS, "tab:green"),
    ]:
        axes[0, 0].scatter([pos[0]], [pos[1]], marker="^", s=80, label=label, color=color)
        axes[0, 0].text(pos[0] + 4.0, pos[1] + 4.0, label)
    axes[0, 0].set_title("Scenario geometry")
    axes[0, 0].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True, alpha=0.35)
    axes[0, 0].legend(fontsize="small")

    range_axis = pulsed["range_axis"]
    doppler_axis = pulsed["doppler_axis"]
    range_mask = (range_axis >= 0.0) & (range_axis <= 500.0)
    rd_plot = pulsed["rd_db"][:, range_mask]
    extent = [range_axis[range_mask][0], range_axis[range_mask][-1], doppler_axis[0], doppler_axis[-1]]
    image = axes[0, 1].imshow(rd_plot, aspect="auto", origin="lower", extent=extent, cmap="viridis", vmin=-60.0, vmax=0.0)
    fig.colorbar(image, ax=axes[0, 1], label="relative magnitude (dB)")
    axes[0, 1].scatter([pulsed["expected_range_m"]], [pulsed["expected_doppler_hz"]], marker="x", color="white", s=80, label="expected")
    axes[0, 1].scatter([pulsed["measured_range_m"]], [pulsed["measured_doppler_hz"]], marker="o", facecolors="none", edgecolors="red", s=80, label="measured")
    axes[0, 1].set_title("Pulsed range-Doppler")
    axes[0, 1].set_xlabel("range (m)")
    axes[0, 1].set_ylabel("Doppler (Hz)")
    axes[0, 1].legend(fontsize="small")

    freqs = cw["freqs"]
    spectrum_db = cw["spectrum_db"]
    axes[1, 0].plot(freqs, spectrum_db)
    axes[1, 0].axvline(cw["expected_doppler_hz"], color="k", linestyle="--", label="expected")
    axes[1, 0].axvline(cw["spectrum_peak_hz"], color="tab:red", linestyle=":", label="measured")
    axes[1, 0].set_xlim(cw["expected_doppler_hz"] - 3_000.0, cw["expected_doppler_hz"] + 3_000.0)
    axes[1, 0].set_ylim(-90.0, 3.0)
    axes[1, 0].set_title("CW Doppler spectrum")
    axes[1, 0].set_xlabel("frequency (Hz)")
    axes[1, 0].set_ylabel("relative magnitude (dB)")
    axes[1, 0].grid(True, alpha=0.35)
    axes[1, 0].legend(fontsize="small")

    axes[1, 1].plot(fmcw["chirp_times"] * 1e3, fmcw["expected_if_hz"], "k--", label="expected")
    axes[1, 1].plot(fmcw["chirp_times"] * 1e3, fmcw["measured_if_hz"], "o", ms=4, label="measured")
    axes[1, 1].set_title("FMCW dechirped IF by chirp")
    axes[1, 1].set_xlabel("time (ms)")
    axes[1, 1].set_ylabel("IF frequency (Hz)")
    axes[1, 1].grid(True, alpha=0.35)
    axes[1, 1].legend(fontsize="small")

    fig.savefig(path, dpi=170)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-file", default="vita49_capture.jsonl")
    parser.add_argument("--summary-json", default="vita49_analysis_summary.json")
    parser.add_argument("--summary-text", default="vita49_analysis.txt")
    parser.add_argument("--plot", default="vita49_mixed_analysis.png")
    args = parser.parse_args()

    scenario_dir = Path(__file__).resolve().parent
    packets = load_capture(scenario_dir / args.capture_file)
    summaries = stream_summaries(packets)

    decoded_modes = {summary["mode"] for summary in summaries.values()}
    expected_modes = {"cw", "fmcw", "pulsed"}
    if decoded_modes != expected_modes:
        raise SystemExit(f"Expected modes {sorted(expected_modes)}, decoded {sorted(decoded_modes)}")

    pulsed = validate_pulsed(stream_for_mode(summaries, "pulsed"))
    cw = validate_cw(stream_for_mode(summaries, "cw"))
    fmcw = validate_fmcw(stream_for_mode(summaries, "fmcw"))
    overall_pass = bool(pulsed["passed"] and cw["passed"] and fmcw["passed"])

    report = {
        "overall_status": "PASS" if overall_pass else "FAIL",
        "pulsed": pulsed,
        "cw": cw,
        "fmcw": fmcw,
    }

    (scenario_dir / args.summary_json).write_text(json.dumps(json_ready(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_text_report(scenario_dir / args.summary_text, report)
    make_plot(scenario_dir / args.plot, pulsed, cw, fmcw)
    print(f"Saved {args.plot}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
