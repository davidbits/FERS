// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2006-2008 Marc Brooker and Michael Inggs
// Copyright (c) 2008-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

/**
 * @file response.cpp
 * @brief Implementation of the ResponseBuilder and Response classes.
 */

#include "response.h"

#include <cmath>
#include <cstddef>

#include "radar/transmitter.h"
#include "signal/radar_signal.h"

using interp::InterpPoint;

namespace serial
{
	namespace
	{
		/**
		 * @brief Extrapolates a taper point by a Doppler-adjusted tx-time `interval` in seconds
		 * beyond `edge`, using the Doppler rate observed between `edge` and `neighbor`
		 * (the next interp point toward the response's interior).
		 */
		InterpPoint extrapolateTaperPoint(const InterpPoint& edge, const InterpPoint& neighbor,
										  const RealType interval, const bool forward)
		{
			InterpPoint point = edge;

			// Because the response was not detected at the previous/next sample, we know the taper
			// point should have zero gain here.
			point.gain = 0.0;

			// `interval` is a tx-time duration. The corresponding rx-time interval is affected by
			// Doppler rate, estimated here from the edge and neighbour's delays over their own
			// transmission gap, then linearly extrapolated to the taper's own delay and phase delay.
			const RealType tx_gap = (neighbor.rx_time - neighbor.delay) - (edge.rx_time - edge.delay);
			const RealType scale = interval / tx_gap;
			const RealType delay_delta = (neighbor.delay - edge.delay) * scale;
			const RealType phase_delta = (neighbor.phase_delay - edge.phase_delay) * scale;
			const RealType rx_width = interval + delay_delta;

			const RealType sign = forward ? 1.0 : -1.0;
			point.rx_time += sign * rx_width;
			point.delay += sign * delay_delta;
			point.phase_delay += sign * phase_delta;
			return point;
		}

		/// A zero-gain taper point offset by `interval` with no Doppler extrapolation, used when
		/// `edge` has no second real point on its side to estimate a Doppler rate from.
		InterpPoint simpleTaperPoint(const InterpPoint& edge, const RealType interval, const bool forward)
		{
			InterpPoint point = edge;
			point.gain = 0.0;
			point.rx_time += (forward ? 1.0 : -1.0) * interval;
			return point;
		}
	}

	ResponseBuilder::ResponseBuilder(const fers_signal::RadarSignal* signal, const InterpPoint& first_point,
									 const RealType left_interval, const RealType right_interval) :
		_signal(signal), _left_interval(left_interval), _right_interval(right_interval)
	{
		// Reserve the lead taper's slot up front: it's overwritten in place once the builder is
		// finalized into a Response and the number of real points (needed to decide between
		// Doppler extrapolation and a plain zero-gain point) is known.
		if (left_interval > 0.0)
		{
			_points.push_back(first_point);
		}
		_points.push_back(first_point);
	}

	void ResponseBuilder::addInterpPoint(const InterpPoint& point, const RealType right_interval)
	{
		if (point.rx_time < _points.back().rx_time)
		{
			throw std::logic_error("Attempted to add an InterpPoint earlier than the Response's last point");
		}
		_points.push_back(point);
		_right_interval = right_interval;
	}

	Response::Response(ResponseBuilder builder) :
		_signal(builder._signal), _points(std::move(builder._points))
	{
		if (!_signal->isPulsed())
		{
			throw std::logic_error(
				"Attempted to create Response with RadarSignal that didn't contain a PulseWaveform");
		}
		_wave = _signal->getPulseWaveform();

		// The lead taper reserved a slot at _points[0]; real points start right after it.
		const bool has_lead = builder._left_interval > 0.0;
		const std::size_t real_count = _points.size() - (has_lead ? 1 : 0);

		if (has_lead)
		{
			_points.front() = real_count > 1
								   ? extrapolateTaperPoint(_points[1], _points[2], builder._left_interval,
														   /*forward=*/false)
								   : simpleTaperPoint(_points[1], builder._left_interval, /*forward=*/false);
		}

		if (builder._right_interval > 0.0)
		{
			const auto tail = real_count > 1
								   ? extrapolateTaperPoint(_points.back(), _points[_points.size() - 2],
														   builder._right_interval, /*forward=*/true)
								   : simpleTaperPoint(_points.back(), builder._right_interval, /*forward=*/true);
			_points.push_back(tail);
		}

		// Native sample 0 lines up with the true first real point's rx_time, not a synthetic
		// lead taper point that may now sit in front of it.
		_native_anchor_time = _points[has_lead ? 1 : 0].rx_time;
	}

	std::vector<ComplexType> Response::render(const RealType fracWinDelay) const
	{
		const auto amplitude = std::sqrt(_signal->getPower());
		return _wave->render(_points, _native_anchor_time, fracWinDelay, amplitude);
	}

	std::vector<ComplexType> Response::renderSlice(const RealType outputRate, const RealType outputStartTime,
												   const std::size_t sampleCount, const RealType fracWinDelay) const
	{
		const auto amplitude = std::sqrt(_signal->getPower());
		return _wave->renderSlice(_points, _native_anchor_time, outputStartTime, outputRate, sampleCount,
								  fracWinDelay, amplitude);
	}

	RealType Response::sampleRate() const noexcept { return _wave->getRate(); }

	size_t Response::sampleCount() const noexcept { return _wave->getSampleCount(); }
}
