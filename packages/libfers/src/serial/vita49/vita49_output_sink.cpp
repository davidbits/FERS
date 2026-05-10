// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#include "serial/vita49/vita49_output_sink.h"

#include <chrono>
#include <stdexcept>

#include "core/parameters.h"
#include "serial/vita49/udp_sender.h"

namespace serial::vita49
{
	namespace
	{
		[[nodiscard]] std::uint64_t defaultEpochNanoseconds()
		{
			const auto now = std::chrono::system_clock::now().time_since_epoch();
			return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
		}
	}

	Vita49OutputSink::Vita49OutputSink(std::unique_ptr<DatagramSender> sender) : _provided_sender(std::move(sender)) {}

	Vita49OutputSink::~Vita49OutputSink()
	{
		if (!_finalized)
		{
			try
			{
				(void)finalize();
			}
			catch (...)
			{
			}
		}
	}

	void Vita49OutputSink::initializeRun(const core::OutputConfig& config, std::string simulation_name)
	{
		std::scoped_lock lock(_mutex);
		if (config.mode != core::OutputMode::Vita49Udp)
		{
			throw std::invalid_argument("Vita49OutputSink requires VITA output mode");
		}
		if (config.vita49.host.empty() || config.vita49.port == 0)
		{
			throw std::invalid_argument("VITA output requires destination host and port");
		}
		if (config.vita49.queue_depth == 0)
		{
			throw std::invalid_argument("VITA output queue depth must be positive");
		}

		_config = config;
		_simulation_name = std::move(simulation_name);
		const auto epoch_ns = config.vita49.epoch_unix_nanoseconds.value_or(defaultEpochNanoseconds());
		_packetizer =
			std::make_unique<Vita49Packetizer>(epoch_ns, config.vita49.adc_fullscale, config.vita49.max_udp_payload);
		_sender = std::make_unique<PacedSender>(
			_provided_sender ? std::move(_provided_sender) : std::make_unique<UdpSender>(), config.vita49.queue_depth);
		_sender->open(config.vita49.host, config.vita49.port);
		_sender->start(params::startTime());
		_initialized = true;
		_finalized = false;
	}

	std::uint32_t Vita49OutputSink::registerStream(const core::ReceiverStreamDescriptor& stream)
	{
		std::scoped_lock lock(_mutex);
		const auto stream_id = _registry.registerStream(stream);
		if (!_streams.contains(stream_id))
		{
			StreamState state;
			state.descriptor = stream;
			state.stats.receiver_id = stream.receiver_id;
			state.stats.receiver_name = stream.receiver_name;
			state.stats.stream_id = stream_id;
			state.stats.sample_rate = stream.sample_rate;
			state.stats.reference_frequency = stream.reference_frequency;
			_streams.emplace(stream_id, std::move(state));
		}
		return stream_id;
	}

	void Vita49OutputSink::openStream(const std::uint32_t stream_id, const RealType first_sample_time)
	{
		std::scoped_lock lock(_mutex);
		emitContext(stream_id, first_sample_time, true, false);
		stateFor(stream_id).opened = true;
	}

	void Vita49OutputSink::submitBlock(const core::ReceiverSampleBlock& block)
	{
		std::scoped_lock lock(_mutex);
		if (!_initialized || !_packetizer)
		{
			throw std::logic_error("VITA output sink has not been initialized");
		}
		const auto stream_id = registerStream(block.stream);
		auto& state = stateFor(stream_id);
		if (!state.opened)
		{
			openStream(stream_id, block.first_sample_time);
		}

		auto result = _packetizer->packetize(block, stream_id, state.packet_counts, state.sample_loss_pending);
		state.sample_loss_pending = false;
		for (auto& packet : result.packets)
		{
			const auto packet_sample_count = packet.sample_count;
			const auto packet_first_sample_time = packet.first_sample_time;
			const auto packet_over_range = packet.over_range;
			if (!enqueuePacket(std::move(packet)))
			{
				continue;
			}
			state.stats.samples_emitted += packet_sample_count;
			++state.stats.packets_emitted;
			const auto packet_ps =
				saturatedUnixPicoseconds(_packetizer->epochUnixNanoseconds(), packet_first_sample_time);
			if (state.stats.first_timestamp_unix_ps == 0)
			{
				state.stats.first_timestamp_unix_ps = packet_ps;
			}
			state.stats.last_timestamp_unix_ps = packet_ps;
			if (packet_over_range)
			{
				state.over_range_pending = true;
			}
		}
		state.stats.over_range_count += result.over_range_count;
	}

	void Vita49OutputSink::emitContextHeartbeat(const RealType simulation_time)
	{
		std::scoped_lock lock(_mutex);
		for (auto& [stream_id, state] : _streams)
		{
			if (!state.closed && simulation_time - state.last_context_time >= 1.0)
			{
				emitContext(stream_id, simulation_time, false, false);
			}
		}
	}

	void Vita49OutputSink::closeStream(const std::uint32_t stream_id)
	{
		std::scoped_lock lock(_mutex);
		auto& state = stateFor(stream_id);
		if (!state.closed)
		{
			emitContext(stream_id, state.last_context_time, false, true);
			state.closed = true;
		}
	}

	core::OutputStats Vita49OutputSink::finalize()
	{
		std::scoped_lock lock(_mutex);
		if (_finalized)
		{
			core::OutputStats stats{.mode = core::OutputMode::Vita49Udp,
									.epoch_unix_nanoseconds = _packetizer
										? std::optional<std::uint64_t>(_packetizer->epochUnixNanoseconds())
										: std::nullopt,
									.streams = {}};
			for (auto& [stream_id, state] : _streams)
			{
				stats.streams.push_back(state.stats);
			}
			return stats;
		}

		for (auto& [stream_id, state] : _streams)
		{
			if (!state.closed)
			{
				emitContext(stream_id, state.last_context_time, false, true);
				state.closed = true;
			}
		}

		if (_sender)
		{
			_sender->stop();
		}

		core::OutputStats stats{.mode = core::OutputMode::Vita49Udp,
								.epoch_unix_nanoseconds = _packetizer
									? std::optional<std::uint64_t>(_packetizer->epochUnixNanoseconds())
									: std::nullopt,
								.streams = {}};
		for (auto& [stream_id, state] : _streams)
		{
			if (_sender)
			{
				state.stats.late_packet_count = _sender->latePacketCount(stream_id);
			}
			stats.streams.push_back(state.stats);
		}
		_finalized = true;
		return stats;
	}

	Vita49OutputSink::StreamState& Vita49OutputSink::stateFor(const std::uint32_t stream_id)
	{
		const auto found = _streams.find(stream_id);
		if (found == _streams.end())
		{
			throw std::out_of_range("Unknown VITA stream ID");
		}
		return found->second;
	}

	const Vita49OutputSink::StreamState& Vita49OutputSink::stateFor(const std::uint32_t stream_id) const
	{
		const auto found = _streams.find(stream_id);
		if (found == _streams.end())
		{
			throw std::out_of_range("Unknown VITA stream ID");
		}
		return found->second;
	}

	bool Vita49OutputSink::enqueuePacket(SerializedPacket&& packet)
	{
		if (!_sender)
		{
			throw std::logic_error("VITA paced sender is unavailable");
		}

		const auto result = _sender->enqueue(std::move(packet));
		if (result.dropped)
		{
			applyDropped(*result.dropped);
		}
		return result.enqueued;
	}

	void Vita49OutputSink::emitContext(const std::uint32_t stream_id, const RealType simulation_time,
									   const bool stream_open, const bool stream_close)
	{
		if (!_packetizer)
		{
			throw std::logic_error("VITA packetizer is unavailable");
		}
		auto& state = stateFor(stream_id);
		const RealType context_time = simulation_time <= -1.0e200 ? 0.0 : simulation_time;
		const auto timestamp = timestampFromEpoch(_packetizer->epochUnixNanoseconds(), context_time);
		const ContextBuildRequest request{.stream = state.descriptor,
										  .stream_id = stream_id,
										  .simulation_name = _simulation_name,
										  .adc_fullscale = _packetizer->adcFullscale(),
										  .timestamp = timestamp,
										  .packet_count = state.packet_counts.next(),
										  .valid_data = true,
										  .calibrated_time = true,
										  .reference_lock = true,
										  .over_range = state.over_range_pending,
										  .sample_loss = state.sample_loss_pending,
										  .stream_open = stream_open,
										  .stream_close = stream_close};
		const auto context = Vita49ContextBuilder::build(request);
		auto packet = _packetizer->makeContextPacket(context);
		packet.first_sample_time = context_time;
		if (enqueuePacket(std::move(packet)))
		{
			++state.stats.context_packets;
		}
		state.last_context_time = context_time;
		state.sample_loss_pending = false;
		state.over_range_pending = false;
	}

	void Vita49OutputSink::applyDropped(const DroppedDatagram& dropped)
	{
		if (dropped.stream_id == 0 || !_streams.contains(dropped.stream_id))
		{
			return;
		}
		auto& state = stateFor(dropped.stream_id);
		++state.stats.packets_dropped;
		state.stats.samples_dropped += dropped.sample_count;
		state.sample_loss_pending = true;
	}

	std::unique_ptr<core::ReceiverOutputSink> makeVita49OutputSink() { return std::make_unique<Vita49OutputSink>(); }
}
