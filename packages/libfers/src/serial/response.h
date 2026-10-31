// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2006-2008 Marc Brooker and Michael Inggs
// Copyright (c) 2008-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

/**
 * @file response.h
 * @brief Classes for managing radar signal responses.
 */

#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "core/config.h"
#include "interpolation/interpolation_point.h"
#include "signal/radar_signal.h"

class XmlElement;

namespace fers_signal
{
	class RadarSignal;
}

namespace radar
{
	class Transmitter;
}

namespace serial
{
	/**
	 * @class ResponseBuilder
	 * @brief Accumulates a radar response's interpolation points sample by sample as they're
	 * detected. Finalize it into an immutable, tapered `Response` once the path is done.
	 */
	class ResponseBuilder
	{
	public:
		/**
		 * @brief Starts a response at its first detected sample.
		 *
		 * @param signal Pointer to the radar signal object.
		 * @param first_point The response's first sample.
		 * @param left_interval The tx-time sampling gap immediately before `first_point`
		 * (0 if `first_point` sits at the transmission's start).
		 * @param right_interval This response's tx-time sampling gap after `first_point`,
		 * used only if no further point is ever added.
		 */
		ResponseBuilder(const fers_signal::RadarSignal* signal, const interp::InterpPoint& first_point,
						RealType left_interval, RealType right_interval);

		/**
		 * @brief Adds the next detected sample to the response.
		 *
		 * @param point The interpolation point to be added.
		 * @param right_interval This response's tx-time gap after `point`, used when no
		 * further point is detected and added.
		 * @throws std::logic_error If the new point has a time earlier than the last point.
		 */
		void addInterpPoint(const interp::InterpPoint& point, RealType right_interval);

	private:
		friend class Response;

		const fers_signal::RadarSignal* _signal; ///< Pointer to the radar signal object.
		std::vector<interp::InterpPoint> _points; ///< Real points, plus a reserved lead slot if `_left_interval > 0`.
		RealType _left_interval; ///< True tx-time gap before the first real point (0 if untapered).
		RealType _right_interval; ///< Assumed tx-time gap after the last real point (0 if untapered).
	};

	/**
	 * @class Response
	 * @brief An immutable, tapered radar signal response, ready to render. Built by
	 * finalizing a `ResponseBuilder`.
	 */
	class Response
	{
	public:
		/// Finalizes `builder` into a tapered `Response`: bakes in a zero-gain lead/tail
		/// point on each side that still has something to taper into (a nonzero interval).
		explicit Response(ResponseBuilder builder);

		~Response() = default;
		Response(const Response&) = delete;
		Response& operator=(const Response&) = delete;
		Response(Response&&) = delete;
		Response& operator=(Response&&) = delete;

		/// Start time of the response, including its leading fade-in taper if it has one.
		[[nodiscard]] RealType startTime() const noexcept { return _points.front().rx_time; }

		/// End time of the response, including its trailing fade-out taper if it has one.
		[[nodiscard]] RealType endTime() const noexcept { return _points.back().rx_time; }

		/// The receiving duration of the response, in seconds.
		[[nodiscard]] RealType getRxDuration() const noexcept { return endTime() - startTime(); }

		/// The response's points in rx-time order, including any baked-in taper points.
		[[nodiscard]] std::span<const interp::InterpPoint> points() const noexcept { return _points; }

		/// Renders the response in binary format.
		[[nodiscard]] std::vector<ComplexType> render(RealType fracWinDelay) const;

		/// Renders a bounded absolute-time response slice on the requested output grid.
		[[nodiscard]] std::vector<ComplexType> renderSlice(RealType outputRate, RealType outputStartTime,
														   std::size_t sampleCount, RealType fracWinDelay) const;

		/// Returns the waveform native sample rate.
		[[nodiscard]] RealType sampleRate() const noexcept;

		/// Returns the waveform native sample count.
		[[nodiscard]] size_t sampleCount() const noexcept;

	private:
		const fers_signal::RadarSignal* _signal; ///< Pointer to the radar signal object.
		const fers_signal::PulseWaveform* _wave{nullptr}; ///< Pointer to the radar signal object's PulseWaveform.
		std::vector<interp::InterpPoint> _points; ///< Channel property interpolation points, taper points included.
		RealType _native_anchor_time; ///< rx_time of the true first sample (not a synthetic lead taper point).
	};
}
