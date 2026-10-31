// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2006-2008 Marc Brooker and Michael Inggs
// Copyright (c) 2008-present FERS Contributors (see AUTHORS.md).
//
// Upstream reference:
//   packages/libfers/src/signal/radar_signal.cpp

#include "radar_signal.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <utility>

#include "core/parameters.h"
#include "dsp_filters.h"
#include "interpolation/interpolation_point.h"

namespace fers_signal
{
	void SampledSignal::clear() noexcept
	{
		_data.clear();
		_rate = 0;
	}

	void SampledSignal::load(std::span<const ComplexType> inData, const unsigned samples, const RealType sampleRate)
	{
		clear();
		const unsigned ratio = params::oversampleRatio();
		_data.resize(static_cast<size_t>(samples) * ratio);
		_rate = sampleRate * ratio;

		if (ratio == 1)
		{
			std::ranges::copy(inData, _data.begin());
		}
		else
		{
			upsample(inData, samples, _data);
		}
	}

	std::tuple<RealType, RealType, RealType, long>
	SampledSignal::calculateWeightsAndDelays(const interp::InterpPoint& iter, const interp::InterpPoint& next,
											 const RealType sampleTime, const RealType idelay,
											 const RealType fracWinDelay, const RealType amplitudeScale) const noexcept
	{
		const RealType bw =
			iter.rx_time < next.rx_time ? (sampleTime - iter.rx_time) / (next.rx_time - iter.rx_time) : 0.0;

		const RealType amplitude = amplitudeScale * std::lerp(std::sqrt(iter.gain), std::sqrt(next.gain), bw);
		const RealType phase = std::lerp(iter.phase_delay, next.phase_delay, bw);
		RealType fdelay = -(std::lerp(iter.delay, next.delay, bw) * _rate - idelay + fracWinDelay);
		const RealType int_part = std::floor(fdelay);
		fdelay -= int_part;

		const long i_sample_unwrap = static_cast<long>(int_part);

		return {amplitude, phase, fdelay, i_sample_unwrap};
	}

	ComplexType SampledSignal::performConvolution(const long i, const RealType* filt, const long filtLength,
												  const RealType amplitude, const long iSampleUnwrap) const noexcept
	{
		const long start = std::max(-filtLength / 2, -i);
		const long end = std::min(filtLength / 2, static_cast<long>(getSampleCount()) - i);

		ComplexType accum(0.0, 0.0);

		for (long j = start; j < end; ++j)
		{
			const long sample_idx = i + j + iSampleUnwrap;
			const long filt_idx = j + filtLength / 2;
			if (sample_idx >= 0 && sample_idx < static_cast<long>(getSampleCount()) && filt_idx >= 0 &&
				filt_idx < filtLength)
			{
				accum += amplitude * _data[static_cast<std::size_t>(sample_idx)] * filt[filt_idx];
			}
		}

		return accum;
	}

	RadarSignal::RadarSignal(std::string name, const RealType power, const RealType carrierfreq,
							 std::unique_ptr<SampledSignal> wave) :
		_name(std::move(name)), _power(power), _carrierfreq(carrierfreq), _wave(std::move(wave))
	{
		if (!_wave)
		{
			throw std::runtime_error("Signal is empty");
		}
	}
}
