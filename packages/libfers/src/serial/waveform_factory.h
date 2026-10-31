// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2006-2008 Marc Brooker and Michael Inggs
// Copyright (c) 2008-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

/**
 * @file waveform_factory.h
 * @brief Interface for loading waveform data into RadarSignal objects.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "core/config.h"
#include "core/sim_id.h"
#include "signal/radar_signal.h"

namespace serial
{
	/// Radar mode requested for a file-loaded waveform. Distinct from
	/// `fers_signal::FileWaveformKind` (which only distinguishes Cw vs Fmcw on the
	/// `FileWaveform` type itself): this factory-level enum additionally identifies
	/// the Pulsed case, which routes to `fers_signal::PulseWaveform` instead.
	enum class FileWaveformRequestKind : std::uint8_t
	{
		Pulsed,
		Cw,
		Fmcw
	};

	/**
	 * @brief Loads a radar waveform from a file and returns a RadarSignal object.
	 *
	 * @param name The name of the radar signal.
	 * @param filename The path to the file containing the waveform data.
	 * @param power The power of the radar signal in the waveform.
	 * @param carrierFreq The carrier frequency of the radar signal.
	 * @param id Optional explicit waveform identifier.
	 * @param kind Radar mode assigned to the loaded samples. Cw/Fmcw require an HDF5 (.h5) file.
	 * @return A unique pointer to a RadarSignal object loaded with the waveform data.
	 * @throws std::runtime_error If the file cannot be opened, the file format is unrecognized,
	 * or a non-pulsed kind is requested for a non-HDF5 file.
	 */
	[[nodiscard]] std::unique_ptr<fers_signal::RadarSignal>
	loadWaveformFromFile(const std::string& name, const std::string& filename, RealType power, RealType carrierFreq,
						 const SimId id = 0, FileWaveformRequestKind kind = FileWaveformRequestKind::Pulsed);
}
