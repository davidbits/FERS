// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2006-2008 Marc Brooker and Michael Inggs
// Copyright (c) 2008-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

/**
 * @file waveform_factory.cpp
 * @brief Implementation for loading waveform data into RadarSignal objects.
 */

#include "waveform_factory.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "core/config.h"
#include "core/parameters.h"
#include "core/sim_id.h"
#include "hdf5_handler.h"
#include "signal/radar_signal.h"

using fers_signal::FileWaveform;
using fers_signal::PulseWaveform;
using fers_signal::RadarSignal;
using serial::FileWaveformRequestKind;

namespace
{
	/// Converts a sample count to unsigned after validating the supported range.
	[[nodiscard]] unsigned checked_sample_count(const std::size_t sample_count, const std::string_view source)
	{
		if (sample_count == 0)
		{
			throw std::runtime_error(std::format("Waveform '{}' contains no samples.", source));
		}
		if (sample_count > static_cast<std::size_t>(std::numeric_limits<unsigned>::max()))
		{
			throw std::runtime_error(std::format("Waveform '{}' has too many samples to load into Signal.", source));
		}
		return static_cast<unsigned>(sample_count);
	}

	/// Parses and validates a CSV waveform sample count header.
	[[nodiscard]] unsigned parse_csv_sample_count(const RealType sample_count, const std::filesystem::path& filepath)
	{
		if (!std::isfinite(sample_count) || sample_count < 0.0 || std::trunc(sample_count) != sample_count)
		{
			throw std::runtime_error("Waveform file '" + filepath.string() + "' has an invalid sample count header.");
		}

		if (sample_count > static_cast<RealType>(std::numeric_limits<unsigned>::max()))
		{
			throw std::runtime_error("Waveform file '" + filepath.string() +
									 "' declares more samples than Signal can represent.");
		}

		return static_cast<unsigned>(sample_count);
	}

	/// Builds the RadarSignal for a loaded sample buffer, routing to PulseWaveform for
	/// the pulsed kind and to FileWaveform (classified Cw/Fmcw) otherwise.
	std::unique_ptr<RadarSignal> makeFileBackedWaveform(const std::string& name, const std::string& source_filename,
														const RealType power, const RealType carrierFreq,
														const SimId id, const FileWaveformRequestKind kind,
														std::span<const ComplexType> data, const unsigned sample_count,
														const RealType sampleRate)
	{
		if (kind == FileWaveformRequestKind::Pulsed)
		{
			PulseWaveform wave{};
			wave.setFilename(source_filename);
			wave.load(data, sample_count, sampleRate);
			return std::make_unique<RadarSignal>(name, power, carrierFreq, std::move(wave), id);
		}

		FileWaveform wave{};
		wave.setFilename(source_filename);
		wave.setKind(kind == FileWaveformRequestKind::Cw ? fers_signal::FileWaveformKind::Cw
														 : fers_signal::FileWaveformKind::Fmcw);
		wave.load(data, sample_count, sampleRate);
		return std::make_unique<RadarSignal>(name, power, carrierFreq, std::move(wave), id);
	}

	/**
	 * @brief Loads a radar waveform from an HDF5 file and returns a RadarSignal object.
	 *
	 * @param name The name of the radar signal.
	 * @param filepath The path to the HDF5 file containing the waveform data.
	 * @param power The power of the radar signal in the waveform.
	 * @param carrierFreq The carrier frequency of the radar signal.
	 * @param kind Radar mode assigned to the loaded samples.
	 * @return A unique pointer to a RadarSignal object loaded with the waveform data.
	 * @throws std::runtime_error If the file cannot be opened or the file format is unrecognized.
	 */
	std::unique_ptr<RadarSignal> loadWaveformFromHdf5File(const std::string& name,
														  const std::filesystem::path& filepath, const RealType power,
														  const RealType carrierFreq, const SimId id,
														  const FileWaveformRequestKind kind)
	{
		std::vector<ComplexType> data;
		serial::readPulseData(filepath.string(), data);
		const unsigned sample_count = checked_sample_count(data.size(), filepath.string());

		return makeFileBackedWaveform(name, filepath.string(), power, carrierFreq, id, kind, data, sample_count,
									  params::rate());
	}

	/**
	 * @brief Loads a radar waveform from a CSV file and returns a RadarSignal object.
	 *
	 * CSV is pulsed-only; the caller guarantees `kind == Pulsed` before reaching here
	 * (see `loadWaveformFromFile`'s HDF5-only guard for Cw/Fmcw).
	 *
	 * @param name The name of the radar signal.
	 * @param filepath The path to the CSV file containing the waveform data.
	 * @param power The power of the radar signal in the waveform.
	 * @param carrierFreq The carrier frequency of the radar signal.
	 * @return A unique pointer to a RadarSignal object loaded with the waveform data.
	 * @throws std::runtime_error If the file cannot be opened or the file format is unrecognized.
	 */
	std::unique_ptr<RadarSignal> loadWaveformFromCsvFile(const std::string& name, const std::filesystem::path& filepath,
														 const RealType power, const RealType carrierFreq,
														 const SimId id)
	{
		std::ifstream ifile(filepath);
		if (!ifile)
		{
			LOG(logging::Level::FATAL, "Could not open file '{}' to read waveform", filepath.string());
			throw std::runtime_error("Could not open file '" + filepath.string() + "' to read waveform");
		}

		RealType rlength = 0.0;
		RealType rate = 0.0;
		if (!(ifile >> rlength >> rate))
		{
			LOG(logging::Level::FATAL, "Could not read waveform header from file '{}'", filepath.string());
			throw std::runtime_error("Could not read waveform header from file '" + filepath.string() + "'");
		}
		if (!std::isfinite(rate) || rate <= 0.0)
		{
			LOG(logging::Level::FATAL, "Waveform file '{}' has invalid sample rate {}", filepath.string(), rate);
			throw std::runtime_error("Waveform file '" + filepath.string() + "' has an invalid sample rate");
		}

		const unsigned length = parse_csv_sample_count(rlength, filepath);
		if (length == 0)
		{
			throw std::runtime_error("Waveform file '" + filepath.string() + "' contains no samples.");
		}
		std::vector<ComplexType> data(length);

		// Read the file data
		for (std::size_t done = 0; done < length && ifile >> data[done]; ++done)
		{
		}

		if (ifile.fail() || data.size() != length)
		{
			LOG(logging::Level::FATAL, "Could not read full waveform from file '{}'", filepath.string());
			throw std::runtime_error("Could not read full waveform from file '" + filepath.string() + "'");
		}

		PulseWaveform wave{};
		wave.setFilename(filepath.string());
		wave.load(data, length, rate);
		return std::make_unique<RadarSignal>(name, power, carrierFreq, std::move(wave), id);
	}

	/**
	 * @brief Checks if a filename has a specific extension.
	 *
	 * @param filename The filename to check.
	 * @param ext The extension to check for.
	 * @return True if the filename has the specified extension, false otherwise.
	 */
	constexpr bool hasExtension(const std::string_view filename, const std::string_view ext) noexcept
	{
		return filename.ends_with(ext);
	}
}

namespace serial
{
	std::unique_ptr<RadarSignal> loadWaveformFromFile(const std::string& name, const std::string& filename,
													  const RealType power, const RealType carrierFreq, const SimId id,
													  const FileWaveformRequestKind kind)
	{
		const std::filesystem::path filepath = filename;
		const auto extension = filepath.extension().string();

		if (kind != FileWaveformRequestKind::Pulsed && !hasExtension(extension, ".h5"))
		{
			throw std::runtime_error("File-backed CW and FMCW waveforms require the pulsed-compatible HDF5 (.h5) "
									 "format: " +
									 filename);
		}

		if (hasExtension(extension, ".csv"))
		{
			return loadWaveformFromCsvFile(name, filepath, power, carrierFreq, id);
		}
		if (hasExtension(extension, ".h5"))
		{
			return loadWaveformFromHdf5File(name, filepath, power, carrierFreq, id, kind);
		}

		LOG(logging::Level::FATAL, "Unrecognized file extension '{}' for file: '{}'", extension, filename);
		throw std::runtime_error("Unrecognized file extension '" + extension + "' for file: " + filename);
	}
}
