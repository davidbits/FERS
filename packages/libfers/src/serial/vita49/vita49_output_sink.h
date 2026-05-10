// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "core/receiver_output.h"
#include "serial/vita49/paced_sender.h"
#include "serial/vita49/stream_registry.h"
#include "serial/vita49/vita49_context_builder.h"
#include "serial/vita49/vita49_packetizer.h"

namespace serial::vita49
{
	class Vita49OutputSink final : public core::ReceiverOutputSink
	{
	public:
		explicit Vita49OutputSink(std::unique_ptr<DatagramSender> sender = nullptr);
		~Vita49OutputSink() override;

		void initializeRun(const core::OutputConfig& config, std::string simulation_name) override;
		std::uint32_t registerStream(const core::ReceiverStreamDescriptor& stream) override;
		void openStream(std::uint32_t stream_id, RealType first_sample_time) override;
		void submitBlock(const core::ReceiverSampleBlock& block) override;
		void emitContextHeartbeat(RealType simulation_time) override;
		void closeStream(std::uint32_t stream_id) override;
		core::OutputStats finalize() override;

	private:
		struct StreamState
		{
			core::ReceiverStreamDescriptor descriptor;
			core::ReceiverStreamStats stats;
			PacketCountSequencer packet_counts;
			bool opened = false;
			bool closed = false;
			bool sample_loss_pending = false;
			bool over_range_pending = false;
			RealType last_context_time = -1.0e300;
		};

		[[nodiscard]] StreamState& stateFor(std::uint32_t stream_id);
		[[nodiscard]] const StreamState& stateFor(std::uint32_t stream_id) const;
		[[nodiscard]] bool enqueuePacket(SerializedPacket&& packet);
		void emitContext(std::uint32_t stream_id, RealType simulation_time, bool stream_open, bool stream_close);
		void applyDropped(const DroppedDatagram& dropped);

		core::OutputConfig _config;
		std::string _simulation_name;
		std::unique_ptr<DatagramSender> _provided_sender;
		StreamRegistry _registry;
		std::unique_ptr<Vita49Packetizer> _packetizer;
		std::unique_ptr<PacedSender> _sender;
		std::unordered_map<std::uint32_t, StreamState> _streams;
		mutable std::recursive_mutex _mutex;
		bool _initialized = false;
		bool _finalized = false;
	};

	[[nodiscard]] std::unique_ptr<core::ReceiverOutputSink> makeVita49OutputSink();
}
