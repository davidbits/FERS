#!/usr/bin/env python3
"""Validate the CW VITA49 capture as a static zero-Doppler radar signal."""

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

TX_POS = np.array([0.0, 0.0, 10.0])
RX_POS = np.array([100.0, 0.0, 10.0])
EXPECTED_DOPPLER_HZ = 0.0


def fit_phase(times: np.ndarray, iq: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    magnitude = np.abs(iq)
    threshold = max(1.0e-12, 0.1 * float(np.percentile(magnitude, 95.0)))
    mask = magnitude > threshold
    if int(np.count_nonzero(mask)) < 32:
        mask = np.ones(len(iq), dtype=bool)

    fit_times = times[mask]
    phase = np.unwrap(np.angle(iq[mask]))
    x = fit_times - np.mean(fit_times)
    y = phase - np.mean(phase)
    slope = float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) else 0.0
    fitted = np.mean(phase) + slope * x
    residual = phase - fitted
    return slope / (2.0 * np.pi), float(np.sqrt(np.mean(residual * residual))), fit_times, residual


def estimate_peak_frequency(iq: np.ndarray, sample_rate: float) -> tuple[float, np.ndarray, np.ndarray]:
    if len(iq) < 8 or sample_rate <= 0.0:
        return 0.0, np.empty(0), np.empty(0)
    nfft = min(262_144, 1 << (len(iq).bit_length() - 1))
    segment = iq[:nfft] * np.hanning(nfft)
    spectrum = np.fft.fftshift(np.fft.fft(segment))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    spectrum_db = 20.0 * np.log10(np.abs(spectrum) + 1.0e-30)
    spectrum_db -= float(np.max(spectrum_db))
    peak_hz = float(freqs[int(np.argmax(spectrum_db))])
    return peak_hz, freqs, spectrum_db


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-file", default="vita49_capture.jsonl")
    parser.add_argument("--summary-json", default="vita49_analysis_summary.json")
    parser.add_argument("--summary-text", default="vita49_analysis.txt")
    parser.add_argument("--plot", default="vita49_cw_analysis.png")
    args = parser.parse_args()

    scenario_dir = Path(__file__).resolve().parent
    packets = load_capture(scenario_dir / args.capture_file)
    summaries = stream_summaries(packets)
    if len(summaries) != 1:
        raise SystemExit(f"Expected one CW stream, decoded {len(summaries)}")

    summary = next(iter(summaries.values()))
    iq = summary["iq_complex"]
    sample_rate = float(summary["sample_rate"] or 0.0)
    if summary["mode"] != "cw":
        raise SystemExit(f"Expected CW stream, decoded {summary['mode']}")
    if sample_rate <= 0.0 or len(iq) < 32:
        raise SystemExit("CW stream does not contain enough samples for validation")

    times = np.arange(len(iq)) / sample_rate
    phase_frequency_hz, phase_residual_rms_rad, fit_times, phase_residual = fit_phase(times, iq)
    peak_hz, freqs, spectrum_db = estimate_peak_frequency(iq, sample_rate)

    magnitude = np.abs(iq)
    active = magnitude > max(1.0e-12, 0.1 * float(np.percentile(magnitude, 95.0)))
    active_magnitude = magnitude[active] if np.any(active) else magnitude
    amplitude_mean = float(np.mean(active_magnitude))
    amplitude_cv = float(np.std(active_magnitude) / amplitude_mean) if amplitude_mean else 0.0
    path_length = float(np.linalg.norm(RX_POS - TX_POS))
    frequency_tolerance_hz = 1.0
    passed = bool(
        abs(phase_frequency_hz - EXPECTED_DOPPLER_HZ) <= frequency_tolerance_hz
        and abs(peak_hz - EXPECTED_DOPPLER_HZ) <= frequency_tolerance_hz
        and summary["sample_loss_packets"] == 0
        and summary["over_range_packets"] == 0
    )

    report = {
        "overall_status": "PASS" if passed else "FAIL",
        "receiver": summary["receiver_name"],
        "mode": summary["mode"],
        "samples": int(summary["samples"]),
        "sample_rate_hz": sample_rate,
        "path_length_m": path_length,
        "expected_doppler_hz": EXPECTED_DOPPLER_HZ,
        "phase_fit_frequency_hz": phase_frequency_hz,
        "spectrum_peak_hz": peak_hz,
        "frequency_tolerance_hz": frequency_tolerance_hz,
        "phase_residual_rms_rad": phase_residual_rms_rad,
        "normalized_amplitude_mean": amplitude_mean,
        "normalized_amplitude_cv": amplitude_cv,
        "over_range_packets": int(summary["over_range_packets"]),
        "sample_loss_packets": int(summary["sample_loss_packets"]),
    }
    (scenario_dir / args.summary_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "VITA49 CW Streaming Radar Simulation Analysis",
        "=============================================",
        f"Overall: {report['overall_status']}",
        f"Receiver: {summary['receiver_name']}",
        f"Samples: {summary['samples']} at {sample_rate:.1f} Hz",
        f"Static path length: {path_length:.2f} m",
        f"Doppler: expected {EXPECTED_DOPPLER_HZ:.3f} Hz, phase-fit {phase_frequency_hz:.3f} Hz, FFT peak {peak_hz:.3f} Hz",
        f"Phase residual RMS: {phase_residual_rms_rad:.3e} rad",
        f"Normalized amplitude mean/CV: {amplitude_mean:.6g} / {amplitude_cv:.3e}",
        f"Over-range/sample-loss packets: {summary['over_range_packets']} / {summary['sample_loss_packets']}",
    ]
    (scenario_dir / args.summary_text).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    plot_count = min(len(iq), 1000)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    if plot_count:
        t_ms = times[:plot_count] * 1000.0
        axes[0].plot(t_ms, iq.real[:plot_count], label="I")
        axes[0].plot(t_ms, iq.imag[:plot_count], label="Q", alpha=0.8)
    axes[0].set_title("Static CW IQ")
    axes[0].set_xlabel("time (ms)")
    axes[0].set_ylabel("normalized int16")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend()

    axes[1].plot(freqs, spectrum_db)
    axes[1].axvline(EXPECTED_DOPPLER_HZ, color="k", linestyle="--", label="expected")
    axes[1].axvline(phase_frequency_hz, color="tab:red", linestyle=":", label="phase fit")
    axes[1].set_xlim(-50.0, 50.0)
    axes[1].set_ylim(-90.0, 3.0)
    axes[1].set_title("CW Doppler spectrum")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("relative magnitude (dB)")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend()

    residual_count = min(len(phase_residual), 20_000)
    axes[2].plot(fit_times[:residual_count], phase_residual[:residual_count])
    axes[2].set_title("Phase residual after zero-Doppler fit")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("phase residual (rad)")
    axes[2].grid(True, alpha=0.35)

    fig.savefig(scenario_dir / args.plot, dpi=160)
    print(f"Saved {args.plot}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
