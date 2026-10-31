// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "core/config.h"
#include "core/simulation_state.h"

namespace radar
{
	class Receiver;
	class Transmitter;
}
namespace propagation
{
	/**
	 * @struct PropagationPath
	 * @brief Data type containing a valid radar path identified by the path finder.
	 */
	struct PropagationPath
	{
		/// Total path delay, in seconds.
		///
		/// This is never zero. No path is created if this would be zero.
		RealType delay;

		/// The ratio of received to transmitted voltage (Vr / Vt) at unit impedance.
		/// This is phase-coherent: path-dependent phase modulation is included,
		/// however bulk phase is not (e^{-jkR}).
		///
		/// The magnitude is equivalent to the square root of the linear power gain.
		ComplexType gain;

		/// A unique, stable ID for a continuously-varying channel over time.
		uint64_t path_id;

		/// The streaaming source index for the path's transmitter, if `traceFromReceivers` is used.
		/// Otherwise undefined.
		size_t source_index;
		/// The receiver of the path.
		radar::Receiver* receiver;
	};

	struct ThreadContext
	{
		virtual ~ThreadContext() = default;
	};

	struct PathsAtTime
	{
		/// All propagation channel found from the transmitter to any receiver at tx_time.
		std::vector<PropagationPath> paths;
		/// The transmission time of each path in `paths`.
		RealType tx_time{};
	};

	class PropagationModel
	{
	public:
		virtual ~PropagationModel();

		/// Called once per worker thread that will call the methods below.
		/// Default: no context needed.
		[[nodiscard]] virtual std::unique_ptr<ThreadContext> makeThreadContext() const { return nullptr; }

		/// Traces forward from one transmitter, at a sequence of times, to all receivers.
		///
		/// Used for pulsed-mode transmitters, to receivers.
		[[nodiscard]] virtual std::vector<PathsAtTime>
		traceFromTransmitter(ThreadContext* ctx, const radar::Transmitter& transmitter,
							 const std::vector<RealType>& times) const = 0;

		/// Traces backward from a batch of receivers, at one time, to all currently active streaming sources.
		///
		/// Used for receivers, from CW-family streaming sources.
		[[nodiscard]] virtual std::unordered_map<radar::Receiver*, std::vector<PropagationPath>>
		traceFromReceivers(ThreadContext* ctx, const std::vector<radar::Receiver*>& receivers,
						   const std::vector<core::ActiveStreamingSource>& sources, RealType rx_time) const = 0;

		/// Conservative upper bound on the number of tx-to-rx path legs this model can produce for a
		/// single path (direct = 1 leg, each additional bounce/reflection adds 1 leg). Used to bound
		/// how long a stopped streaming source's propagation tail could still take to arrive.
		[[nodiscard]] virtual unsigned int maxPropagationLegs() const noexcept = 0;
	};
}
