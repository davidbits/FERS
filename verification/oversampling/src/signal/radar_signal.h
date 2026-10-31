// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2006-2008 Marc Brooker and Michael Inggs
// Copyright (c) 2008-present FERS Contributors (see AUTHORS.md).
//
// Upstream reference:
//   packages/libfers/src/signal/radar_signal.h
//
// Local simplification:
//   only SampledSignal is modeled; the Waveform variant (CW/FMCW/stepped-frequency) and
//   SimId support are removed because the isolation harness only audits sampled-signal
//   render/oversampling behavior.

#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "core/config.h"
#include "core/parameters.h"
#include "interpolation/interpolation_filter.h"
#include "interpolation/interpolation_point.h"

namespace fers_signal
{
	/// Digital signal implementation.
	class SampledSignal
	{
	public:
		SampledSignal() = default;
		~SampledSignal() = default;

		SampledSignal(const SampledSignal&) noexcept = delete;
		SampledSignal& operator=(const SampledSignal&) noexcept = delete;
		SampledSignal(SampledSignal&&) noexcept = default;
		SampledSignal& operator=(SampledSignal&&) noexcept = default;

		void clear() noexcept;
		void load(std::span<const ComplexType> inData, unsigned samples, RealType sampleRate);
		[[nodiscard]] RealType getRate() const noexcept { return _rate; }
		[[nodiscard]] size_t getSampleCount() const noexcept { return _data.size(); }

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
		std::vector<ComplexType> _data; ///< The complex signal data.
		RealType _rate{0}; ///< The sample rate of the signal.

		[[nodiscard]] std::tuple<RealType, RealType, RealType, long>
		calculateWeightsAndDelays(const interp::InterpPoint& iter, const interp::InterpPoint& next, RealType sampleTime,
								  RealType idelay, RealType fracWinDelay, RealType amplitudeScale) const noexcept;

		ComplexType performConvolution(long i, const RealType* filt, long filtLength, RealType amplitude,
									   long iSampleUnwrap) const noexcept;
	};

	template <typename PointRange>
	std::vector<ComplexType> SampledSignal::render(const PointRange& points, const RealType nativeAnchorTime,
												   const double fracWinDelay, const RealType amplitudeScale) const
	{
		return renderSlice(points, nativeAnchorTime, nativeAnchorTime, _rate, getSampleCount(), fracWinDelay,
						   amplitudeScale);
	}

	template <typename PointRange>
	std::vector<ComplexType> SampledSignal::renderSlice(const PointRange& points, const RealType nativeAnchorTime,
														const RealType outputStartTime, const RealType outputSampleRate,
														const std::size_t sampleCount, const RealType fracWinDelay,
														const RealType amplitudeScale) const
	{
		auto iter = points.begin();
		const auto points_end = points.end();

		auto out = std::vector<ComplexType>(sampleCount);
		if (getSampleCount() == 0 || _rate <= 0.0 || outputSampleRate <= 0.0 || iter == points_end)
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
		const RealType idelay = std::round(_rate * iter->delay);
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
			const RealType native_position = (sample_time - nativeAnchorTime) * _rate;
			const auto source_index = static_cast<long>(std::floor(native_position));
			RealType source_fraction = native_position - static_cast<RealType>(source_index);

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

	class RadarSignal
	{
	public:
		RadarSignal(std::string name, RealType power, RealType carrierfreq, std::unique_ptr<SampledSignal> wave);

		~RadarSignal() = default;
		RadarSignal(const RadarSignal&) = delete;
		RadarSignal& operator=(const RadarSignal&) = delete;
		RadarSignal(RadarSignal&&) noexcept = delete;
		RadarSignal& operator=(RadarSignal&&) noexcept = delete;

		void setFilename(const std::string& filename) noexcept { _filename = filename; }
		[[nodiscard]] const std::optional<std::string>& getFilename() const noexcept { return _filename; }
		[[nodiscard]] RealType getPower() const noexcept { return _power; }
		[[nodiscard]] RealType getCarrier() const noexcept { return _carrierfreq; }
		[[nodiscard]] const std::string& getName() const noexcept { return _name; }

		/// Returns true when this signal owns a SampledSignal (always true in this harness).
		[[nodiscard]] bool isSampled() const noexcept { return static_cast<bool>(_wave); }

		/// Gets the sampled waveform owned by this signal.
		[[nodiscard]] const SampledSignal* getSampledSignal() const noexcept { return _wave.get(); }

	private:
		std::string _name;
		RealType _power;
		RealType _carrierfreq;
		std::unique_ptr<SampledSignal> _wave;
		std::optional<std::string> _filename;
	};
}
