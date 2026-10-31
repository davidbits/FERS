// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2006-2008 Marc Brooker and Michael Inggs
// Copyright (c) 2008-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

/**
 * @file radar_signal.h
 * @brief Classes for handling radar waveforms and signals.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

#include "core/config.h"
#include "core/parameters.h"
#include "core/sim_id.h"
#include "interpolation/interpolation_filter.h"
#include "interpolation/interpolation_point.h"

namespace fers_signal
{
	/// Sweep direction for a linear FMCW chirp.
	enum class FmcwChirpDirection : std::uint8_t
	{
		Up, ///< Instantaneous baseband frequency increases over the chirp.
		Down ///< Instantaneous baseband frequency decreases over the chirp.
	};

	/// Converts a chirp direction to the schema token.
	[[nodiscard]] std::string_view fmcwChirpDirectionToken(FmcwChirpDirection direction) noexcept;

	/// Parses a schema chirp direction token.
	[[nodiscard]] FmcwChirpDirection parseFmcwChirpDirection(std::string_view direction);

	/// Owns raw complex sample data loaded from a waveform file, plus basic
	/// single-time-sample interpolation. Shared by PulseWaveform (which additionally
	/// renders against interpolation-point delay/gain/phase tracks for pulsed radar)
	/// and FileWaveform (which only needs a fixed-time envelope query).
	class WaveformSampleBuffer
	{
	public:
		WaveformSampleBuffer() = default;
		~WaveformSampleBuffer() = default;

		WaveformSampleBuffer(const WaveformSampleBuffer&) noexcept = delete;
		WaveformSampleBuffer& operator=(const WaveformSampleBuffer&) noexcept = delete;
		WaveformSampleBuffer(WaveformSampleBuffer&&) noexcept = default;
		WaveformSampleBuffer& operator=(WaveformSampleBuffer&&) noexcept = default;

		/**
		 * @brief Sets the filename associated with this buffer.
		 * @param filename The source filename.
		 */
		void setFilename(const std::string& filename) noexcept { _filename = filename; }

		/**
		 * @brief Gets the filename associated with this buffer.
		 * @return The source filename, if one was set.
		 */
		[[nodiscard]] const std::optional<std::string>& getFilename() const noexcept { return _filename; }

		/**
		 * @brief Clears the internal sample data.
		 */
		void clear() noexcept;

		/**
		 * @brief Loads complex waveform sample data.
		 *
		 * @param inData The input span of complex signal data.
		 * @param samples The number of samples in the input data.
		 * @param sampleRate The sample rate of the input data.
		 */
		void load(std::span<const ComplexType> inData, unsigned samples, RealType sampleRate);

		/**
		 * @brief Gets the sample rate of the buffer.
		 *
		 * @return The sample rate of the buffer.
		 */
		[[nodiscard]] RealType getRate() const noexcept { return _rate; }

		/// Gets the number of native samples held by this buffer.
		[[nodiscard]] size_t getSampleCount() const noexcept { return _data.size(); }

		/**
		 * @brief Gets the duration of the buffer.
		 *
		 * @return The duration of the buffer.
		 */
		[[nodiscard]] RealType getDuration() const noexcept
		{
			return getRate() > 0.0 ? static_cast<RealType>(getSampleCount()) / getRate() : 0.0;
		}

		/// Samples the finite stored complex envelope using the render interpolation filter.
		/// Times outside [0, duration) return zero; file-backed buffers never wrap implicitly.
		[[nodiscard]] ComplexType sampleAt(RealType time_since_start) const noexcept;

		/// Direct access to the native sample buffer, for consumers (PulseWaveform) that
		/// render against interpolation point tracks rather than a single time query.
		[[nodiscard]] std::span<const ComplexType> data() const noexcept { return _data; }

	private:
		std::vector<ComplexType> _data; ///< The complex sample data.
		RealType _rate{0}; ///< The sample rate of the buffer.
		std::optional<std::string> _filename; ///< The original filename, if loaded from a file.
	};

	/// Pulsed radar waveform backed by a finite sample buffer, rendered via
	/// interpolation-filter convolution against delay/gain/phase tracks.
	class PulseWaveform
	{
	public:
		PulseWaveform() = default;
		~PulseWaveform() = default;

		PulseWaveform(const PulseWaveform&) noexcept = delete;
		PulseWaveform& operator=(const PulseWaveform&) noexcept = delete;
		PulseWaveform(PulseWaveform&&) noexcept = default;
		PulseWaveform& operator=(PulseWaveform&&) noexcept = default;

		/**
		 * @brief Sets the filename associated with this signal.
		 * @param filename The source filename.
		 */
		void setFilename(const std::string& filename) noexcept { _buffer.setFilename(filename); }

		/**
		 * @brief Gets the filename associated with this signal.
		 * @return The source filename, if one was set.
		 */
		[[nodiscard]] const std::optional<std::string>& getFilename() const noexcept { return _buffer.getFilename(); }

		/**
		 * @brief Clears the internal signal data.
		 */
		void clear() noexcept { _buffer.clear(); }

		/**
		 * @brief Loads complex radar waveform data.
		 *
		 * @param inData The input span of complex signal data.
		 * @param samples The number of samples in the input data.
		 * @param sampleRate The sample rate of the input data.
		 */
		void load(std::span<const ComplexType> inData, unsigned samples, RealType sampleRate)
		{
			_buffer.load(inData, samples, sampleRate);
		}

		/**
		 * @brief Gets the sample rate of the signal.
		 *
		 * @return The sample rate of the signal.
		 */
		[[nodiscard]] RealType getRate() const noexcept { return _buffer.getRate(); }

		/// Gets the number of native samples held by this signal.
		[[nodiscard]] size_t getSampleCount() const noexcept { return _buffer.getSampleCount(); }

		/**
		 * @brief Gets the duration of the radar signal.
		 *
		 * @return The duration of the radar signal.
		 */
		[[nodiscard]] RealType getDuration() const noexcept { return _buffer.getDuration(); }

		/**
		 * @brief Renders the signal data based on interpolation points.
		 *
		 * @param points A forward range of interp::InterpPoint used to render the signal.
		 * Any type exposing begin()/end() over InterpPoint-like elements works
		 * (a std::vector<InterpPoint>, or a Response's tapered point view).
		 * @param nativeAnchorTime The time at which native sample 0 of this signal's own
		 * sample buffer arrives. Differs from the first point's rx_time where `points`
		 * includes synthetic points (e.g. edge tapering) that don't correspond to native
		 * sample buffer positions.
		 * @param fracWinDelay Fractional window delay to apply during rendering.
		 * @return A vector of rendered complex signal data.
		 */
		template <typename PointRange>
		[[nodiscard]] std::vector<ComplexType> render(const PointRange& points, RealType nativeAnchorTime,
													  double fracWinDelay, RealType amplitudeScale) const;

		/// Renders a bounded absolute-time slice on the requested output grid.
		template <typename PointRange>
		[[nodiscard]] std::vector<ComplexType> renderSlice(const PointRange& points, RealType nativeAnchorTime,
														   RealType outputStartTime, RealType outputSampleRate,
														   std::size_t sampleCount, RealType fracWinDelay,
														   RealType amplitudeScale) const;

	private:
		WaveformSampleBuffer _buffer; ///< The underlying complex sample data.

		/**
		 * @brief Calculates weights and delays for rendering.
		 *
		 * @param iter The current interpolation point.
		 * @param next The next interpolation point (equal to iter's own rx_time if there
		 * isn't a distinct next point to interpolate towards).
		 * @param sampleTime Current sample time.
		 * @param idelay Integer delay value.
		 * @param fracWinDelay Fractional window delay to apply.
		 * @param amplitudeScale The amplitude scaling to apply (typically from `RadarSignal`).
		 * @return A tuple containing amplitude, phase, fractional delay, and sample unwrap index.
		 */
		[[nodiscard]] std::tuple<RealType, RealType, RealType, long>
		calculateWeightsAndDelays(const interp::InterpPoint& iter, const interp::InterpPoint& next, RealType sampleTime,
								  RealType idelay, RealType fracWinDelay, RealType amplitudeScale) const noexcept;

		/**
		 * @brief Performs convolution with a filter.
		 *
		 * @param i Index of the current sample.
		 * @param filt Pointer to the filter coefficients.
		 * @param filtLength Length of the filter.
		 * @param amplitude Amplitude scaling factor.
		 * @param iSampleUnwrap Unwrapped sample index for the convolution.
		 * @return The result of the convolution for the given sample.
		 */
		ComplexType performConvolution(long i, const RealType* filt, long filtLength, RealType amplitude,
									   long iSampleUnwrap) const noexcept;
	};

	template <typename PointRange>
	std::vector<ComplexType> PulseWaveform::render(const PointRange& points, const RealType nativeAnchorTime,
												   const double fracWinDelay, const RealType amplitudeScale) const
	{
		return renderSlice(points, nativeAnchorTime, nativeAnchorTime, _buffer.getRate(), getSampleCount(),
						   fracWinDelay, amplitudeScale);
	}

	template <typename PointRange>
	std::vector<ComplexType> PulseWaveform::renderSlice(const PointRange& points, const RealType nativeAnchorTime,
														const RealType outputStartTime, const RealType outputSampleRate,
														const std::size_t sampleCount, const RealType fracWinDelay,
														const RealType amplitudeScale) const
	{
		auto iter = points.begin();
		const auto points_end = points.end();

		auto out = std::vector<ComplexType>(sampleCount);
		if (getSampleCount() == 0 || _buffer.getRate() <= 0.0 || outputSampleRate <= 0.0 || iter == points_end)
		{
			return out;
		}

		const RealType timestep = 1.0 / outputSampleRate;
		const int filt_length = static_cast<int>(params::renderFilterLength());
		const auto& interp_filter = interp::InterpFilter::getInstance();

		auto next = std::next(iter);
		if (next == points_end)
		{
			next = iter;
		}
		const RealType idelay = std::round(_buffer.getRate() * iter->delay);
		RealType sample_time = outputStartTime;

		for (std::size_t i = 0; i < sampleCount; ++i)
		{
			while (sample_time > next->rx_time && next != iter)
			{
				iter = next;
				if (std::next(next) != points_end)
				{
					++next;
				}
				else
				{
					break;
				}
			}

			auto [amplitude, phase, fdelay, i_sample_unwrap] =
				calculateWeightsAndDelays(*iter, *next, sample_time, idelay, fracWinDelay, amplitudeScale);
			const RealType native_position = (sample_time - nativeAnchorTime) * _buffer.getRate();
			const auto source_index = static_cast<long>(std::floor(native_position));
			const RealType source_fraction = native_position - static_cast<RealType>(source_index);

			const RealType combined_delay = fdelay + source_fraction;
			const auto delay_unwrap = static_cast<int>(std::floor(combined_delay));
			fdelay = combined_delay - static_cast<RealType>(delay_unwrap);
			i_sample_unwrap += delay_unwrap;

			const auto& filt = interp_filter.getFilter(fdelay);
			const ComplexType accum =
				performConvolution(source_index, filt.data(), filt_length, amplitude, i_sample_unwrap);
			out[i] = std::exp(ComplexType(0.0, 1.0) * phase) * accum;

			sample_time += timestep;
		}

		return out;
	}

	/// Continuous-wave waveform implementation.
	class CwWaveform final
	{
	public:
		CwWaveform() = default;
		~CwWaveform() = default;
	};

	/// Stepped-frequency continuous-wave waveform implementation.
	class SfcwWaveform final
	{
	public:
		/// Active SFCW dwell selected for one local waveform time.
		struct StepState
		{
			std::size_t step_index = 0; ///< Zero-based step index inside a sweep.
			std::size_t sweep_index = 0; ///< Zero-based sweep index.
			RealType step_start_time = 0.0; ///< Local step period start time in seconds.
			RealType dwell_end_time = 0.0; ///< Local active dwell end time in seconds.
			RealType step_end_time = 0.0; ///< Local step period end time in seconds.
			RealType rf_frequency = 0.0; ///< RF frequency for the active step in hertz.
		};

		/// Constructs a uniform stepped-frequency CW waveform.
		SfcwWaveform(RealType start_frequency_offset, RealType step_size, std::size_t step_count, RealType dwell_time,
					 RealType step_period, std::optional<std::size_t> sweep_count = std::nullopt);

		~SfcwWaveform() = default;

		/// Gets the first-step offset from carrier in hertz.
		[[nodiscard]] RealType getStartFrequencyOffset() const noexcept { return _start_frequency_offset; }

		/// Gets the uniform frequency step in hertz.
		[[nodiscard]] RealType getStepSize() const noexcept { return _step_size; }

		/// Gets the number of steps per sweep.
		[[nodiscard]] std::size_t getStepCount() const noexcept { return _step_count; }

		/// Gets active dwell time per step in seconds.
		[[nodiscard]] RealType getDwellTime() const noexcept { return _dwell_time; }

		/// Gets step repetition period in seconds.
		[[nodiscard]] RealType getStepPeriod() const noexcept { return _step_period; }

		/// Gets optional finite sweep count.
		[[nodiscard]] const std::optional<std::size_t>& getSweepCount() const noexcept { return _sweep_count; }

		/// Gets full sweep period in seconds.
		[[nodiscard]] RealType getSweepPeriod() const noexcept
		{
			return static_cast<RealType>(_step_count) * _step_period;
		}

		/// Gets first-step RF frequency in hertz.
		[[nodiscard]] RealType firstFrequency(RealType carrier_frequency) const noexcept;

		/// Gets final-step RF frequency in hertz.
		[[nodiscard]] RealType lastFrequency(RealType carrier_frequency) const noexcept;

		/// Gets first-to-last absolute span in hertz.
		[[nodiscard]] RealType frequencySpan() const noexcept;

		/// Gets DFT-convention effective bandwidth in hertz.
		[[nodiscard]] RealType effectiveBandwidth() const noexcept;

		/// Gets finite waveform duration, if sweep_count is configured.
		[[nodiscard]] std::optional<RealType> totalDuration() const noexcept;

		/// Returns the active step for a time since the segment start.
		[[nodiscard]] std::optional<StepState> activeStepAt(RealType time_since_segment_start,
															RealType carrier_frequency) const noexcept;

	private:
		RealType _start_frequency_offset{}; ///< First-step frequency offset relative to carrier in hertz.
		RealType _step_size{}; ///< Uniform frequency step in hertz.
		std::size_t _step_count{}; ///< Steps per sweep.
		RealType _dwell_time{}; ///< Active dwell time per step in seconds.
		RealType _step_period{}; ///< Step repetition period in seconds.
		std::optional<std::size_t> _sweep_count; ///< Optional finite sweep count.
	};

	/// FMCW linear chirp waveform implementation.
	class FmcwChirpWaveform final
	{
	public:
		/// Constructs an FMCW chirp waveform with timing and sweep parameters.
		FmcwChirpWaveform(RealType chirp_bandwidth, RealType chirp_duration, RealType chirp_period,
						  RealType start_frequency_offset = 0.0, std::optional<std::size_t> chirp_count = std::nullopt,
						  FmcwChirpDirection direction = FmcwChirpDirection::Up);

		~FmcwChirpWaveform() = default;

		/// Gets the chirp bandwidth in hertz.
		[[nodiscard]] RealType getChirpBandwidth() const noexcept { return _chirp_bandwidth; }

		/// Gets the chirp duration in seconds.
		[[nodiscard]] RealType getChirpDuration() const noexcept { return _chirp_duration; }

		/// Gets the chirp period in seconds.
		[[nodiscard]] RealType getChirpPeriod() const noexcept { return _chirp_period; }

		/// Gets the start frequency offset relative to carrier in hertz.
		[[nodiscard]] RealType getStartFrequencyOffset() const noexcept { return _start_frequency_offset; }

		/// Gets the optional finite chirp count.
		[[nodiscard]] const std::optional<std::size_t>& getChirpCount() const noexcept { return _chirp_count; }

		/// Gets the chirp rate in hertz per second.
		[[nodiscard]] RealType getChirpRate() const noexcept { return _chirp_rate; }

		/// Gets the signed chirp rate in hertz per second.
		[[nodiscard]] RealType getSignedChirpRate() const noexcept
		{
			return isDownChirp() ? -_chirp_rate : _chirp_rate;
		}

		/// Gets the FMCW sweep direction.
		[[nodiscard]] FmcwChirpDirection getDirection() const noexcept { return _direction; }

		/// Returns true when this chirp sweeps downward.
		[[nodiscard]] bool isDownChirp() const noexcept { return _direction == FmcwChirpDirection::Down; }

		/// Returns the active chirp index for a time since the segment start.
		[[nodiscard]] std::optional<std::size_t> activeChirpIndexAt(RealType time_since_segment_start) const noexcept;

		/// Returns true when the signal is inside an active chirp at the specified time.
		[[nodiscard]] bool isActiveAt(RealType time_since_segment_start) const noexcept
		{
			return activeChirpIndexAt(time_since_segment_start).has_value();
		}

		/// Computes baseband phase for a time inside a chirp.
		[[nodiscard]] RealType basebandPhaseForChirpTime(RealType chirp_time) const noexcept;

		/// Computes instantaneous baseband phase at a time since segment start.
		[[nodiscard]] std::optional<RealType>
		instantaneousBasebandPhase(RealType time_since_segment_start) const noexcept;

	private:
		RealType _chirp_bandwidth{}; ///< Chirp bandwidth in hertz.
		RealType _chirp_duration{}; ///< Active chirp duration in seconds.
		RealType _chirp_period{}; ///< Chirp repetition period in seconds.
		RealType _start_frequency_offset{}; ///< Start frequency offset relative to carrier in hertz.
		std::optional<std::size_t> _chirp_count; ///< Optional finite chirp count.
		RealType _chirp_rate{}; ///< Frequency sweep rate in hertz per second.
		FmcwChirpDirection _direction{FmcwChirpDirection::Up}; ///< Frequency sweep direction.
	};

	/// FMCW symmetric triangular modulation waveform implementation.
	class FmcwTriangleWaveform final
	{
	public:
		/// Constructs an FMCW triangular modulation waveform.
		FmcwTriangleWaveform(RealType chirp_bandwidth, RealType chirp_duration, RealType start_frequency_offset = 0.0,
							 std::optional<std::size_t> triangle_count = std::nullopt);

		~FmcwTriangleWaveform() = default;

		/// Gets the chirp bandwidth in hertz.
		[[nodiscard]] RealType getChirpBandwidth() const noexcept { return _chirp_bandwidth; }

		/// Gets the per-leg chirp duration in seconds.
		[[nodiscard]] RealType getChirpDuration() const noexcept { return _chirp_duration; }

		/// Gets the start frequency offset relative to carrier in hertz.
		[[nodiscard]] RealType getStartFrequencyOffset() const noexcept { return _start_frequency_offset; }

		/// Gets the optional finite triangle count.
		[[nodiscard]] const std::optional<std::size_t>& getTriangleCount() const noexcept { return _triangle_count; }

		/// Gets the chirp rate magnitude in hertz per second.
		[[nodiscard]] RealType getChirpRate() const noexcept { return _chirp_rate; }

		/// Gets the full up/down triangle period in seconds.
		[[nodiscard]] RealType getTrianglePeriod() const noexcept { return _triangle_period; }

		/// Gets the full, unreduced phase accumulated by one leg.
		[[nodiscard]] RealType getDeltaPhiUp() const noexcept { return _delta_phi_up; }

		/// Computes baseband phase at a time since the triangle train start.
		[[nodiscard]] RealType basebandPhaseForTriangleTime(RealType triangle_time) const noexcept;

		/// Computes instantaneous baseband phase at a time since segment start.
		[[nodiscard]] std::optional<RealType>
		instantaneousBasebandPhase(RealType time_since_segment_start) const noexcept;

	private:
		RealType _chirp_bandwidth{}; ///< Chirp bandwidth in hertz.
		RealType _chirp_duration{}; ///< Per-leg chirp duration in seconds.
		RealType _start_frequency_offset{}; ///< Start frequency offset relative to carrier in hertz.
		std::optional<std::size_t> _triangle_count; ///< Optional finite triangle count.
		RealType _chirp_rate{}; ///< Frequency sweep rate magnitude in hertz per second.
		RealType _triangle_period{}; ///< Full triangle period in seconds.
		RealType _delta_phi_up{}; ///< Full, unreduced phase accumulated by one leg.
	};

	/// Classification of a file-backed streaming waveform. File-backed pulsed waveforms
	/// use PulseWaveform directly (see `waveform_factory`); this only distinguishes CW vs
	/// FMCW envelope playback, since both are evaluated identically at runtime by the
	/// streaming channel model - the file's own samples already encode whatever modulation
	/// shape they have, so the tag exists purely for mode validation and output metadata.
	enum class FileWaveformKind : std::uint8_t
	{
		Cw,
		Fmcw
	};

	/// File-backed continuous-wave or FMCW waveform, played back via direct sample lookup
	/// rather than pulsed convolution. Represents a sampled envelope, not pulsed data: it
	/// deliberately has no render()/renderSlice(), since it never participates in
	/// PropagationModel pulse rendering - only the streaming channel model consumes it,
	/// via sampleAt().
	class FileWaveform final
	{
	public:
		FileWaveform() = default;
		~FileWaveform() = default;

		FileWaveform(const FileWaveform&) noexcept = delete;
		FileWaveform& operator=(const FileWaveform&) noexcept = delete;
		FileWaveform(FileWaveform&&) noexcept = default;
		FileWaveform& operator=(FileWaveform&&) noexcept = default;

		/// Sets the CW/FMCW classification of this file-backed waveform.
		void setKind(FileWaveformKind kind) noexcept { _kind = kind; }

		/// Gets the CW/FMCW classification of this file-backed waveform.
		[[nodiscard]] FileWaveformKind getKind() const noexcept { return _kind; }

		/// Returns true when this file-backed waveform is classified as CW.
		[[nodiscard]] bool isCw() const noexcept { return _kind == FileWaveformKind::Cw; }

		/// Returns true when this file-backed waveform is classified as FMCW.
		[[nodiscard]] bool isFmcw() const noexcept { return _kind == FileWaveformKind::Fmcw; }

		/**
		 * @brief Sets the filename associated with this waveform.
		 * @param filename The source filename.
		 */
		void setFilename(const std::string& filename) noexcept { _buffer.setFilename(filename); }

		/**
		 * @brief Gets the filename associated with this waveform.
		 * @return The source filename, if one was set.
		 */
		[[nodiscard]] const std::optional<std::string>& getFilename() const noexcept { return _buffer.getFilename(); }

		/**
		 * @brief Loads the complex envelope sample data.
		 *
		 * @param inData The input span of complex signal data.
		 * @param samples The number of samples in the input data.
		 * @param sampleRate The sample rate of the input data.
		 */
		void load(std::span<const ComplexType> inData, unsigned samples, RealType sampleRate)
		{
			_buffer.load(inData, samples, sampleRate);
		}

		/// Gets the sample rate of the stored envelope.
		[[nodiscard]] RealType getRate() const noexcept { return _buffer.getRate(); }

		/// Gets the number of native samples held by this waveform.
		[[nodiscard]] size_t getSampleCount() const noexcept { return _buffer.getSampleCount(); }

		/// Gets the total duration of the stored envelope.
		[[nodiscard]] RealType getDuration() const noexcept { return _buffer.getDuration(); }

		/// Samples the finite stored complex envelope. Times outside [0, duration) return
		/// zero; file-backed waveforms never wrap implicitly.
		[[nodiscard]] ComplexType sampleAt(RealType time_since_start) const noexcept
		{
			return _buffer.sampleAt(time_since_start);
		}

	private:
		WaveformSampleBuffer _buffer; ///< The underlying complex envelope sample data.
		FileWaveformKind _kind{FileWaveformKind::Cw}; ///< CW/FMCW classification.
	};


	using Waveform =
		std::variant<PulseWaveform, CwWaveform, SfcwWaveform, FmcwChirpWaveform, FmcwTriangleWaveform, FileWaveform>;

	/**
	 * @class RadarSignal
	 * @brief Class representing a radar signal with associated properties.
	 */
	class RadarSignal
	{
	public:
		/**
		 * @brief Constructs a RadarSignal object.
		 *
		 * @param name The name of the radar signal.
		 * @param power The power of the radar signal.
		 * @param carrierfreq The carrier frequency of the radar signal.
		 * @param wave The `Waveform` variant containing the waveform data.
		 * @param id The unique SimId to assign, or 0 to generate one.
		 */
		RadarSignal(std::string name, RealType power, RealType carrierfreq, Waveform wave, const SimId id = 0);

		~RadarSignal() = default;

		RadarSignal(const RadarSignal&) noexcept = delete;

		RadarSignal& operator=(const RadarSignal&) noexcept = delete;

		RadarSignal(RadarSignal&&) noexcept = delete;

		RadarSignal& operator=(RadarSignal&&) noexcept = delete;

		/**
		 * @brief Gets the power of the radar signal.
		 *
		 * @return The power of the radar signal.
		 */
		[[nodiscard]] RealType getPower() const noexcept { return _power; }

		/**
		 * @brief Gets the carrier frequency of the radar signal.
		 *
		 * @return The carrier frequency of the radar signal.
		 */
		[[nodiscard]] RealType getCarrier() const noexcept { return _carrierfreq; }

		/**
		 * @brief Gets the name of the radar signal.
		 *
		 * @return The name of the radar signal.
		 */
		[[nodiscard]] const std::string& getName() const noexcept { return _name; }

		/**
		 * @brief Gets the unique ID of the radar signal.
		 *
		 * @return The radar signal SimId.
		 */
		[[nodiscard]] SimId getId() const noexcept { return _id; }


		/// Returns true when this signal is a pulsed waveform, used by pulsed radar.
		[[nodiscard]] bool isPulsed() const noexcept;

		/// Returns true when this signal is a continuous-wave waveform, whether
		/// generated (CwWaveform) or file-backed (FileWaveform classified as Cw).
		[[nodiscard]] bool isCw() const noexcept;

		/// Returns true when this signal is an FMCW linear chirp waveform.
		[[nodiscard]] bool isFmcwChirp() const noexcept;

		/// Returns true when this signal is an FMCW triangular modulation waveform.
		[[nodiscard]] bool isFmcwTriangle() const noexcept;

		/// Returns true when this signal belongs to the FMCW waveform family, whether
		/// generated (chirp/triangle) or file-backed (FileWaveform classified as Fmcw).
		[[nodiscard]] bool isFmcwFamily() const noexcept;

		/// Returns true when this signal is a stepped-frequency CW waveform.
		[[nodiscard]] bool isSfcw() const noexcept;

		/// Returns true when this signal is a file-backed CW/FMCW waveform.
		[[nodiscard]] bool isFileWaveform() const noexcept;

		/// Returns true when this signal is a file-backed continuous-wave waveform.
		[[nodiscard]] bool isFileCw() const noexcept;

		/// Returns true when this signal is a file-backed FMCW waveform.
		[[nodiscard]] bool isFileFmcw() const noexcept;

		/// Gets the pulsed waveform, if this signal owns one.
		[[nodiscard]] const PulseWaveform* getPulseWaveform() const noexcept;

		/// Gets the FMCW chirp implementation, if this signal owns one.
		[[nodiscard]] const FmcwChirpWaveform* getFmcwChirpWaveform() const noexcept;

		/// Gets the FMCW triangle implementation, if this signal owns one.
		[[nodiscard]] const FmcwTriangleWaveform* getFmcwTriangleWaveform() const noexcept;

		/// Gets the stepped-frequency implementation, if this signal owns one.
		[[nodiscard]] const SfcwWaveform* getSfcwWaveform() const noexcept;

		/// Gets the file-backed waveform implementation, if this signal owns one.
		[[nodiscard]] const FileWaveform* getFileWaveform() const noexcept;

		[[nodiscard]] const Waveform& getWaveform() const noexcept;

	private:
		std::string _name; ///< The name of the radar signal.
		SimId _id; ///< Unique ID for this radar signal.
		RealType _power; ///< The power of the radar signal.
		RealType _carrierfreq; ///< The carrier frequency of the radar signal.
		Waveform _wave; ///< The waveform data.
	};

	namespace util
	{
		template <class... Ts>
		struct Overloaded : Ts...
		{
			using Ts::operator()...;
		};
		template <class... Ts>
		Overloaded(Ts...) -> Overloaded<Ts...>;
	}
}
