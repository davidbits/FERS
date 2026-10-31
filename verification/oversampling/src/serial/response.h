// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2006-2008 Marc Brooker and Michael Inggs
// Copyright (c) 2008-present FERS Contributors (see AUTHORS.md).
//
// Upstream reference:
//   packages/libfers/src/serial/response.h

#pragma once

#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "core/config.h"
#include "core/parameters.h"
#include "interpolation/interpolation_point.h"
#include "signal/radar_signal.h"

namespace serial
{
	inline RealType controlPointEdgePeriod() noexcept { return 1.0 / params::simSamplingRate(); }

	/**
	 * @class Response
	 * @brief Manages radar signal responses from a transmitter.
	 */
	class Response
	{
	public:
		/**
		 * @brief Constructor for the `Response` class.
		 *
		 * @param signal Pointer to the radar signal object.
		 * @param first_point The first interpolation point of the response.
		 */
		Response(const fers_signal::RadarSignal* signal, const interp::InterpPoint first_point) : _signal(signal)
		{
			if (!signal->isSampled())
			{
				throw std::logic_error(
					"Attempted to create Reponse with RadarSignal that didn't contain a SampledSignal");
			}

			_wave = signal->getSampledSignal();
			addInterpPoint(first_point);
		}

		~Response() = default;
		Response(const Response&) = delete;
		Response& operator=(const Response&) = delete;
		Response(Response&&) = delete;
		Response& operator=(Response&&) = delete;

		/**
		 * @brief Retrieves the start time of the response, including its leading fade-in taper.
		 *
		 * @return Start time as a `RealType`. Returns 0.0 if no points are present.
		 */
		[[nodiscard]] RealType startTime() const noexcept { return _points.empty() ? 0.0 : taperEdges().lead.rx_time; }

		/**
		 * @brief Retrieves the end time of the response, including its trailing fade-out taper.
		 *
		 * @return End time as a `RealType`. Returns 0.0 if no points are present.
		 */
		[[nodiscard]] RealType endTime() const noexcept { return _points.empty() ? 0.0 : taperEdges().tail.rx_time; }

		/**
		 * @brief Adds an interpolation point to the response.
		 *
		 * @param point The interpolation point to be added.
		 */
		void addInterpPoint(const interp::InterpPoint& point);

		/**
		 * @brief Renders the response at the waveform's native sample rate.
		 *
		 * @param fracWinDelay Delay factor applied during windowing.
		 * @return A vector of `ComplexType` representing the rendered signal with a sample rate of `sampleRate()`.
		 */
		[[nodiscard]] std::vector<ComplexType> render(RealType fracWinDelay) const;

		/// Renders a bounded absolute-time response slice on the requested output grid.
		[[nodiscard]] std::vector<ComplexType> renderSlice(RealType outputRate, RealType outputStartTime,
														   std::size_t sampleCount, RealType fracWinDelay) const;

		/// Returns the waveform native sample rate.
		[[nodiscard]] RealType sampleRate() const noexcept;

		/// Returns the waveform native sample count.
		[[nodiscard]] size_t sampleCount() const noexcept;

		/**
		 * @brief Retrieves the receiving duration of the response.
		 *
		 * @return The receiving of the response, in seconds.
		 */
		[[nodiscard]] RealType getRxDuration() const noexcept { return endTime() - startTime(); }

		/**
		 * @brief Forward iterator over a Response's tapered point sequence:
		 * a synthetic zero-gain lead point, the real points, and a synthetic zero-gain tail point.
		 */
		class TaperedPointIterator
		{
		public:
			using iterator_category = std::forward_iterator_tag;
			using value_type = interp::InterpPoint;
			using difference_type = std::ptrdiff_t;
			using pointer = const interp::InterpPoint*;
			using reference = const interp::InterpPoint&;

			TaperedPointIterator() noexcept = default;

			[[nodiscard]] reference operator*() const noexcept
			{
				if (_index == 0)
				{
					return _lead;
				}
				if (_index == _real_count + 1)
				{
					return _tail;
				}
				return _real[_index - 1];
			}

			[[nodiscard]] pointer operator->() const noexcept { return &**this; }

			TaperedPointIterator& operator++() noexcept
			{
				++_index;
				return *this;
			}

			TaperedPointIterator operator++(int) noexcept
			{
				auto copy = *this;
				++*this;
				return copy;
			}

			[[nodiscard]] friend bool operator==(const TaperedPointIterator& lhs,
												 const TaperedPointIterator& rhs) noexcept
			{
				return lhs._index == rhs._index;
			}

		private:
			friend class Response;

			TaperedPointIterator(const interp::InterpPoint* real, const std::size_t real_count,
								 const interp::InterpPoint& lead, const interp::InterpPoint& tail,
								 const std::size_t index) noexcept :
				_real(real), _lead(lead), _tail(tail), _real_count(real_count), _index(index)
			{
			}

			const interp::InterpPoint* _real = nullptr;
			interp::InterpPoint _lead{};
			interp::InterpPoint _tail{};
			std::size_t _real_count = 0;
			std::size_t _index = 0;
		};

		/// A view of a Response's tapered point sequence (see TaperedPointIterator).
		class TaperedPoints
		{
		public:
			[[nodiscard]] TaperedPointIterator begin() const noexcept { return {_real, _real_count, _lead, _tail, 0}; }

			[[nodiscard]] TaperedPointIterator end() const noexcept
			{
				return {_real, _real_count, _lead, _tail, _real_count + 2};
			}

		private:
			friend class Response;

			TaperedPoints(const interp::InterpPoint* real, const std::size_t real_count,
						  const interp::InterpPoint& lead, const interp::InterpPoint& tail) noexcept :
				_real(real), _lead(lead), _tail(tail), _real_count(real_count)
			{
			}

			const interp::InterpPoint* _real;
			interp::InterpPoint _lead;
			interp::InterpPoint _tail;
			std::size_t _real_count;
		};

		/// Returns a view over this response's real points bracketed by its synthetic
		/// lead/tail taper points (see taperEdges()).
		[[nodiscard]] TaperedPoints taperedPoints() const
		{
			const auto [lead, tail] = taperEdges();
			return {_points.data(), _points.size(), lead, tail};
		}

	private:
		struct TaperEdges
		{
			interp::InterpPoint lead;
			interp::InterpPoint tail;
		};

		/**
		 * @brief Computes this response's synthetic zero-gain lead and tail points.
		 *
		 * With at least two real points, rx_time, delay and phase_delay are all linearly
		 * extrapolated from the nearest two real points, using the same time step observed
		 * there, not a fixed controlPointEdgePeriod() width.
		 * With only one real point, lead/tail fall back to a fixed controlPointEdgePeriod()
		 * width with delay/phase_delay copied from that point.
		 */
		[[nodiscard]] TaperEdges taperEdges() const
		{
			interp::InterpPoint lead = _points.front();
			interp::InterpPoint tail = _points.back();
			lead.gain = 0.0;
			tail.gain = 0.0;

			if (_points.size() > 1)
			{
				const auto& second = _points[1];
				const RealType lead_dt = second.rx_time - _points.front().rx_time;
				lead.rx_time -= lead_dt;
				lead.delay -= second.delay - _points.front().delay;
				lead.phase_delay -= second.phase_delay - _points.front().phase_delay;

				const auto& second_last = _points[_points.size() - 2];
				const RealType tail_dt = _points.back().rx_time - second_last.rx_time;
				tail.rx_time += tail_dt;
				tail.delay += _points.back().delay - second_last.delay;
				tail.phase_delay += _points.back().phase_delay - second_last.phase_delay;
			}
			else
			{
				lead.rx_time -= controlPointEdgePeriod();
				tail.rx_time += controlPointEdgePeriod();
			}

			return {lead, tail};
		}

		const fers_signal::RadarSignal* _signal; ///< Pointer to the radar signal object.
		const fers_signal::SampledSignal* _wave; ///< Pointer to the radar signal object's SampledSignal.
		std::vector<interp::InterpPoint> _points; ///< Vector of simulated InterpPoints.
	};
}
