// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>

#include "serial/vita49/udp_sender.h"
#include "serial/vita49/vita49_types.h"

namespace serial::vita49
{
	struct DroppedDatagram
	{
		std::uint32_t stream_id = 0;
		std::uint64_t sample_count = 0;
		bool data_packet = false;
		bool context_packet = false;
	};

	struct EnqueueResult
	{
		bool enqueued = false;
		std::optional<DroppedDatagram> dropped;
	};

	class PacedSender
	{
	public:
		PacedSender(std::unique_ptr<DatagramSender> sender, std::size_t queue_depth);
		~PacedSender();

		PacedSender(const PacedSender&) = delete;
		PacedSender& operator=(const PacedSender&) = delete;

		void open(const std::string& host, std::uint16_t port);
		void start(RealType simulation_epoch_time = 0.0);
		[[nodiscard]] EnqueueResult enqueue(SerializedPacket packet);
		void flush();
		void stop();

		[[nodiscard]] std::uint64_t latePacketCount(std::uint32_t stream_id) const;
		[[nodiscard]] std::uint64_t sentPacketCount(std::uint32_t stream_id) const;
		[[nodiscard]] std::uint64_t sendFailureCount(std::uint32_t stream_id) const;
		[[nodiscard]] std::uint64_t droppedDataPacketCount(std::uint32_t stream_id) const;
		[[nodiscard]] std::uint64_t droppedContextPacketCount(std::uint32_t stream_id) const;
		[[nodiscard]] std::uint64_t droppedSampleCount(std::uint32_t stream_id) const;

	private:
		void run();
		[[nodiscard]] bool waitUntilDue(std::unique_lock<std::mutex>& lock, std::chrono::steady_clock::time_point due);
		[[nodiscard]] bool waitUntilDueOrStopping(std::chrono::steady_clock::time_point due);
		void sendOneUnlocked(SerializedPacket packet, std::chrono::steady_clock::time_point now);
		void recordDropped(SerializedPacket packet);
		void recordDroppedUnlocked(const SerializedPacket& packet);
		void clearQueuedPacketsAsDroppedUnlocked();
		void drainQueuedPackets();
		[[nodiscard]] std::chrono::steady_clock::time_point dueTime(const SerializedPacket& packet) const;
		[[nodiscard]] std::optional<DroppedDatagram> dropQueuedDataPacketUnlocked();

		std::unique_ptr<DatagramSender> _sender;
		std::size_t _queue_depth = 0;
		mutable std::mutex _mutex;
		std::condition_variable _cv;
		std::list<SerializedPacket> _queue;
		std::deque<std::list<SerializedPacket>::iterator> _queued_data_packets;
		std::unordered_map<std::uint32_t, std::uint64_t> _late_packets;
		std::unordered_map<std::uint32_t, std::uint64_t> _sent_packets;
		std::unordered_map<std::uint32_t, std::uint64_t> _send_failures;
		std::unordered_map<std::uint32_t, std::uint64_t> _dropped_data_packets;
		std::unordered_map<std::uint32_t, std::uint64_t> _dropped_context_packets;
		std::unordered_map<std::uint32_t, std::uint64_t> _dropped_samples;
		std::chrono::steady_clock::time_point _steady_epoch = std::chrono::steady_clock::now();
		RealType _simulation_epoch_time = 0.0;
		bool _started = false;
		bool _stopping = false;
		std::thread _thread;
	};
}
