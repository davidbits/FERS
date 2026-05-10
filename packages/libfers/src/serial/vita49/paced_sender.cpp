// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#include "serial/vita49/paced_sender.h"

#include <algorithm>
#include <stdexcept>

namespace serial::vita49
{
	PacedSender::PacedSender(std::unique_ptr<DatagramSender> sender, const std::size_t queue_depth) :
		_sender(std::move(sender)), _queue_depth(queue_depth)
	{
		if (!_sender)
		{
			throw std::invalid_argument("PacedSender requires a datagram sender");
		}
		if (queue_depth == 0)
		{
			throw std::invalid_argument("PacedSender queue depth must be positive");
		}
	}

	PacedSender::~PacedSender() { stop(); }

	void PacedSender::open(const std::string& host, const std::uint16_t port) { _sender->open(host, port); }

	void PacedSender::start(const RealType simulation_epoch_time)
	{
		std::lock_guard lock(_mutex);
		if (_started)
		{
			return;
		}
		_simulation_epoch_time = simulation_epoch_time;
		_steady_epoch = std::chrono::steady_clock::now();
		_stopping = false;
		_started = true;
		_thread = std::thread([this] { run(); });
	}

	EnqueueResult PacedSender::enqueue(SerializedPacket packet)
	{
		std::lock_guard lock(_mutex);
		if (_stopping)
		{
			return EnqueueResult{
				.enqueued = false,
				.dropped = DroppedDatagram{.stream_id = packet.stream_id,
										   .sample_count = packet.sample_count,
										   .data_packet = packet.data_packet,
										   .context_packet = packet.context_packet},
			};
		}

		if (_queue.size() >= _queue_depth)
		{
			if (packet.data_packet)
			{
				return EnqueueResult{
					.enqueued = false,
					.dropped = DroppedDatagram{.stream_id = packet.stream_id,
											   .sample_count = packet.sample_count,
											   .data_packet = packet.data_packet,
											   .context_packet = packet.context_packet},
				};
			}

			auto dropped = dropQueuedDataPacketUnlocked();
			if (!dropped)
			{
				return EnqueueResult{
					.enqueued = false,
					.dropped = DroppedDatagram{.stream_id = packet.stream_id,
											   .sample_count = packet.sample_count,
											   .data_packet = packet.data_packet,
											   .context_packet = packet.context_packet},
				};
			}

			_queue.push_back(std::move(packet));
			_cv.notify_one();
			return EnqueueResult{.enqueued = true, .dropped = dropped};
		}

		_queue.push_back(std::move(packet));
		_cv.notify_one();
		return EnqueueResult{.enqueued = true, .dropped = std::nullopt};
	}

	void PacedSender::flush()
	{
		std::unique_lock lock(_mutex);
		while (!_queue.empty())
		{
			auto packet = std::move(_queue.front());
			_queue.pop_front();
			const auto now = std::chrono::steady_clock::now();
			lock.unlock();
			if (dueTime(packet) > now)
			{
				std::this_thread::sleep_until(dueTime(packet));
			}
			sendOneUnlocked(std::move(packet), std::chrono::steady_clock::now());
			lock.lock();
		}
	}

	void PacedSender::stop()
	{
		{
			std::lock_guard lock(_mutex);
			if (!_started && !_thread.joinable())
			{
				_sender->close();
				return;
			}
			_stopping = true;
			_cv.notify_all();
		}

		if (_thread.joinable())
		{
			_thread.join();
		}

		std::deque<SerializedPacket> due_packets;
		{
			std::lock_guard lock(_mutex);
			const auto now = std::chrono::steady_clock::now();
			for (auto& packet : _queue)
			{
				if (dueTime(packet) <= now)
				{
					due_packets.push_back(std::move(packet));
				}
			}
			_queue.clear();
			_started = false;
		}

		for (auto& packet : due_packets)
		{
			sendOneUnlocked(std::move(packet), std::chrono::steady_clock::now());
		}
		_sender->close();
	}

	std::uint64_t PacedSender::latePacketCount(const std::uint32_t stream_id) const
	{
		std::lock_guard lock(_mutex);
		const auto found = _late_packets.find(stream_id);
		return found == _late_packets.end() ? 0 : found->second;
	}

	std::uint64_t PacedSender::sentPacketCount(const std::uint32_t stream_id) const
	{
		std::lock_guard lock(_mutex);
		const auto found = _sent_packets.find(stream_id);
		return found == _sent_packets.end() ? 0 : found->second;
	}

	void PacedSender::run()
	{
		std::unique_lock lock(_mutex);
		while (true)
		{
			if (_stopping)
			{
				return;
			}
			if (_queue.empty())
			{
				_cv.wait(lock, [this] { return _stopping || !_queue.empty(); });
				continue;
			}

			const auto due = dueTime(_queue.front());
			if (_cv.wait_until(lock, due, [this] { return _stopping; }))
			{
				return;
			}

			auto packet = std::move(_queue.front());
			_queue.pop_front();
			const auto now = std::chrono::steady_clock::now();
			lock.unlock();
			sendOneUnlocked(std::move(packet), now);
			lock.lock();
		}
	}

	void PacedSender::sendOneUnlocked(SerializedPacket packet, const std::chrono::steady_clock::time_point now)
	{
		const auto due = dueTime(packet);
		_sender->send(packet.bytes);
		std::lock_guard lock(_mutex);
		++_sent_packets[packet.stream_id];
		if (now > due + std::chrono::milliseconds(1))
		{
			++_late_packets[packet.stream_id];
		}
	}

	std::chrono::steady_clock::time_point PacedSender::dueTime(const SerializedPacket& packet) const
	{
		const auto seconds = packet.first_sample_time - _simulation_epoch_time;
		const auto nanos = static_cast<std::int64_t>(seconds * 1'000'000'000.0);
		return _steady_epoch + std::chrono::nanoseconds(nanos);
	}

	std::optional<DroppedDatagram> PacedSender::dropQueuedDataPacketUnlocked()
	{
		const auto found = std::find_if(_queue.begin(), _queue.end(),
										[](const SerializedPacket& packet) { return packet.data_packet; });
		if (found == _queue.end())
		{
			return std::nullopt;
		}

		DroppedDatagram dropped{.stream_id = found->stream_id,
								.sample_count = found->sample_count,
								.data_packet = found->data_packet,
								.context_packet = found->context_packet};
		_queue.erase(found);
		return dropped;
	}
}
