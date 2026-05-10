// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#include "serial/vita49/paced_sender.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iterator>
#include <stdexcept>

#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace serial::vita49
{
	namespace
	{
		constexpr auto kCoarsePacingSleep = std::chrono::milliseconds(1);
		constexpr auto kFinePacingSpin = std::chrono::microseconds(200);

		void cpuPause() noexcept
		{
#if defined(__i386__) || defined(__x86_64__)
			_mm_pause();
#elif defined(__aarch64__) || defined(__arm__)
			__builtin_arm_yield();
#else
			std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
		}
	}

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
			if (_queue.back().data_packet)
			{
				_queued_data_packets.push_back(std::prev(_queue.end()));
			}
			_cv.notify_one();
			return EnqueueResult{.enqueued = true, .dropped = dropped};
		}

		_queue.push_back(std::move(packet));
		if (_queue.back().data_packet)
		{
			_queued_data_packets.push_back(std::prev(_queue.end()));
		}
		_cv.notify_one();
		return EnqueueResult{.enqueued = true, .dropped = std::nullopt};
	}

	void PacedSender::flush()
	{
		std::unique_lock lock(_mutex);
		while (!_queue.empty())
		{
			auto packet = std::move(_queue.front());
			if (packet.data_packet && !_queued_data_packets.empty() && _queued_data_packets.front() == _queue.begin())
			{
				_queued_data_packets.pop_front();
			}
			_queue.pop_front();
			const auto due = dueTime(packet);
			lock.unlock();
			if (std::chrono::steady_clock::now() < due && waitUntilDueOrStopping(due))
			{
				lock.lock();
				continue;
			}
			sendOneUnlocked(std::move(packet), std::chrono::steady_clock::now());
			lock.lock();
		}
	}

	void PacedSender::stop()
	{
		bool drain_after_join = false;
		{
			std::lock_guard lock(_mutex);
			if (!_started && !_thread.joinable())
			{
				_queue.clear();
				_queued_data_packets.clear();
				_sender->close();
				return;
			}
			drain_after_join = _started;
			_stopping = true;
			_cv.notify_all();
		}

		if (_thread.joinable())
		{
			_thread.join();
		}

		if (drain_after_join)
		{
			drainQueuedPackets();
		}
		else
		{
			std::lock_guard lock(_mutex);
			_queue.clear();
			_queued_data_packets.clear();
		}
		{
			std::lock_guard lock(_mutex);
			_started = false;
			_stopping = false;
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

	std::uint64_t PacedSender::sendFailureCount(const std::uint32_t stream_id) const
	{
		std::lock_guard lock(_mutex);
		const auto found = _send_failures.find(stream_id);
		return found == _send_failures.end() ? 0 : found->second;
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
			if (waitUntilDue(lock, due))
			{
				return;
			}

			auto packet = std::move(_queue.front());
			if (packet.data_packet && !_queued_data_packets.empty() && _queued_data_packets.front() == _queue.begin())
			{
				_queued_data_packets.pop_front();
			}
			_queue.pop_front();
			const auto now = std::chrono::steady_clock::now();
			lock.unlock();
			sendOneUnlocked(std::move(packet), now);
			lock.lock();
		}
	}

	bool PacedSender::waitUntilDue(std::unique_lock<std::mutex>& lock, const std::chrono::steady_clock::time_point due)
	{
		while (true)
		{
			if (_stopping)
			{
				return true;
			}

			const auto now = std::chrono::steady_clock::now();
			if (now >= due)
			{
				return false;
			}

			const auto remaining = due - now;
			if (remaining > kCoarsePacingSleep)
			{
				const auto coarse_sleep =
					std::chrono::duration_cast<std::chrono::steady_clock::duration>(kCoarsePacingSleep);
				const auto fine_spin = std::chrono::duration_cast<std::chrono::steady_clock::duration>(kFinePacingSpin);
				const auto wait_duration = std::min(remaining - fine_spin, coarse_sleep);
				_cv.wait_for(lock, wait_duration, [this] { return _stopping; });
				continue;
			}

			lock.unlock();
			while (std::chrono::steady_clock::now() < due)
			{
				cpuPause();
			}
			lock.lock();
		}
	}

	bool PacedSender::waitUntilDueOrStopping(const std::chrono::steady_clock::time_point due)
	{
		std::unique_lock lock(_mutex);
		while (!_stopping)
		{
			const auto now = std::chrono::steady_clock::now();
			if (now >= due)
			{
				return false;
			}
			_cv.wait_until(lock, due, [this] { return _stopping; });
		}
		return true;
	}

	void PacedSender::sendOneUnlocked(SerializedPacket packet, const std::chrono::steady_clock::time_point now)
	{
		const auto due = dueTime(packet);
		try
		{
			_sender->send(packet.bytes);
		}
		catch (...)
		{
			std::lock_guard lock(_mutex);
			++_send_failures[packet.stream_id];
			return;
		}
		std::lock_guard lock(_mutex);
		++_sent_packets[packet.stream_id];
		if (now > due + std::chrono::milliseconds(1))
		{
			++_late_packets[packet.stream_id];
		}
	}

	void PacedSender::drainQueuedPackets()
	{
		while (true)
		{
			SerializedPacket packet;
			{
				std::lock_guard lock(_mutex);
				if (_queue.empty())
				{
					_queued_data_packets.clear();
					return;
				}
				packet = std::move(_queue.front());
				_queue.pop_front();
				if (packet.data_packet && !_queued_data_packets.empty())
				{
					_queued_data_packets.pop_front();
				}
			}

			const auto due = dueTime(packet);
			if (std::chrono::steady_clock::now() < due && waitUntilDueOrStopping(due))
			{
				continue;
			}
			sendOneUnlocked(std::move(packet), std::chrono::steady_clock::now());
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
		while (!_queued_data_packets.empty() && !_queued_data_packets.front()->data_packet)
		{
			_queued_data_packets.pop_front();
		}

		if (_queued_data_packets.empty())
		{
			return std::nullopt;
		}

		const auto found = _queued_data_packets.front();
		_queued_data_packets.pop_front();
		DroppedDatagram dropped{.stream_id = found->stream_id,
								.sample_count = found->sample_count,
								.data_packet = found->data_packet,
								.context_packet = found->context_packet};
		_queue.erase(found);
		return dropped;
	}
}
