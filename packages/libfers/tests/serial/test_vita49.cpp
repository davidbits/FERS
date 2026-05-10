// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).

#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "serial/vita49/paced_sender.h"
#include "serial/vita49/stream_registry.h"
#include "serial/vita49/udp_sender.h"
#include "serial/vita49/vita49_context_builder.h"
#include "serial/vita49/vita49_output_sink.h"
#include "serial/vita49/vita49_packetizer.h"
#include "serial/vita49/vita49_serializer.h"

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace
{
	[[nodiscard]] std::uint16_t readU16(const std::vector<std::uint8_t>& bytes, const std::size_t offset)
	{
		return static_cast<std::uint16_t>((static_cast<std::uint16_t>(bytes.at(offset)) << 8u) |
										  static_cast<std::uint16_t>(bytes.at(offset + 1u)));
	}

	[[nodiscard]] std::uint32_t readU32(const std::vector<std::uint8_t>& bytes, const std::size_t offset)
	{
		return (static_cast<std::uint32_t>(bytes.at(offset)) << 24u) |
			(static_cast<std::uint32_t>(bytes.at(offset + 1u)) << 16u) |
			(static_cast<std::uint32_t>(bytes.at(offset + 2u)) << 8u) |
			static_cast<std::uint32_t>(bytes.at(offset + 3u));
	}

	[[nodiscard]] std::uint64_t readU64(const std::vector<std::uint8_t>& bytes, const std::size_t offset)
	{
		return (static_cast<std::uint64_t>(readU32(bytes, offset)) << 32u) | readU32(bytes, offset + 4u);
	}

	[[nodiscard]] double readF64(const std::vector<std::uint8_t>& bytes, const std::size_t offset)
	{
		return std::bit_cast<double>(readU64(bytes, offset));
	}

	class RecordingSender final : public serial::vita49::DatagramSender
	{
	public:
		void open(const std::string&, std::uint16_t) override { opened = true; }
		void send(const std::span<const std::uint8_t> bytes) override { sent.emplace_back(bytes.begin(), bytes.end()); }
		void close() noexcept override { closed = true; }

		bool opened = false;
		bool closed = false;
		std::vector<std::vector<std::uint8_t>> sent;
	};
}

TEST_CASE("VITA signal data packet serializes golden header and IQ bytes", "[serial][vita49]")
{
	using namespace serial::vita49;

	const SignalDataPacket packet{
		.stream_id = 0x10203040u,
		.class_id = kFersVrtIqClassId,
		.timestamp = Timestamp{.integer_seconds = 0x01020304u, .fractional_picoseconds = 0x0102030405060708ull},
		.packet_count = 0x0Au,
		.iq_interleaved = {1, -2, 0x1234, static_cast<std::int16_t>(-0x1234)},
		.trailer = makeTrailer(true, true, true, true, false)};

	const auto bytes = Vita49Serializer::serializeSignalData(packet);

	REQUIRE(bytes.size() == 40u);
	REQUIRE(readU32(bytes, 0) == 0x1E6A000Au);
	REQUIRE(readU32(bytes, 4) == 0x10203040u);
	REQUIRE(readU64(bytes, 8) == kFersVrtIqClassId);
	REQUIRE(readU32(bytes, 16) == 0x01020304u);
	REQUIRE(readU64(bytes, 20) == 0x0102030405060708ull);
	REQUIRE(readU16(bytes, 28) == 0x0001u);
	REQUIRE(readU16(bytes, 30) == 0xFFFEu);
	REQUIRE(readU16(bytes, 32) == 0x1234u);
	REQUIRE(readU16(bytes, 34) == 0xEDCCu);
	REQUIRE(readU32(bytes, 36) ==
			(TrailerEnableValidData | TrailerEnableCalibratedTime | TrailerEnableReferenceLock |
			 TrailerEnableOverRange | TrailerEnableSampleLoss | TrailerValidData | TrailerCalibratedTime |
			 TrailerReferenceLock | TrailerOverRange));
}

TEST_CASE("VITA context packet serializes CIF and fields in profile order", "[serial][vita49]")
{
	using namespace serial::vita49;

	const core::ReceiverStreamDescriptor stream{.receiver_id = 0x1122334455667788ull,
												.receiver_name = "rx1",
												.mode = "cw",
												.sample_rate = 2.5e6,
												.reference_frequency = 9.6e9,
												.if_offset = -1.25e6,
												.bandwidth = 5.0e6,
												.dechirped = true,
												.if_resampled = false,
												.adc_bits = 14};
	const auto context = Vita49ContextBuilder::build(ContextBuildRequest{.stream = stream,
																		 .stream_id = 0x55667788u,
																		 .simulation_name = "sim",
																		 .adc_fullscale = 2.0,
																		 .timestamp = Timestamp{100u, 250u},
																		 .packet_count = 3,
																		 .sample_loss = true,
																		 .stream_open = true});

	const auto bytes = Vita49Serializer::serializeContext(context);

	REQUIRE(bytes.size() % 4u == 0u);
	REQUIRE((readU32(bytes, 0) & 0xFFFF0000u) == 0x4A630000u);
	REQUIRE((readU32(bytes, 0) & 0x0000FFFFu) == bytes.size() / 4u);
	REQUIRE(readU32(bytes, 4) == 0x55667788u);
	REQUIRE(readU64(bytes, 8) == kFersVrtIqClassId);
	REQUIRE(readU32(bytes, 16) == 100u);
	REQUIRE(readU64(bytes, 20) == 250u);
	REQUIRE(readU32(bytes, 28) == kFersContextCif0);
	REQUIRE((readU32(bytes, 32) & TrailerSampleLoss) == TrailerSampleLoss);
	REQUIRE(readU64(bytes, 36) == makeComplexInt16PayloadFormat());
	REQUIRE(readF64(bytes, 44) == 2.5e6);
	REQUIRE(readF64(bytes, 52) == 9.6e9);
	REQUIRE(readF64(bytes, 60) == -1.25e6);
	REQUIRE(readF64(bytes, 68) == 5.0e6);
	REQUIRE(readF64(bytes, 76) == 2.0);
	REQUIRE(readU64(bytes, 84) == 0x1122334455667788ull);
	REQUIRE(readU32(bytes, 92) == 14u);
	REQUIRE((readU32(bytes, 96) & (ContextFlagDechirped | ContextFlagSampleLoss | ContextFlagStreamOpen)) ==
			(ContextFlagDechirped | ContextFlagSampleLoss | ContextFlagStreamOpen));
}

TEST_CASE("VITA packetizer uses first sample timestamp, packet cap, and big-endian payload", "[serial][vita49]")
{
	using namespace serial::vita49;

	Vita49Packetizer packetizer(1'700'000'000'123'456'789ull, 1.0, 1400);
	REQUIRE(packetizer.maxComplexSamplesPerPacket() == 342u);

	std::vector<ComplexType> samples(1000, ComplexType(0.5, -0.5));
	const core::ReceiverStreamDescriptor stream{
		.receiver_id = 77, .receiver_name = "rx", .mode = "cw", .sample_rate = 1.0, .reference_frequency = 1.0e9};
	const core::ReceiverSampleBlock block{
		.stream = stream, .first_sample_time = 1.25, .sample_rate = 1.0, .samples = samples, .sample_start = 0};
	PacketCountSequencer counts;
	const auto result = packetizer.packetize(block, 0xABCDEF01u, counts, false);

	REQUIRE(result.packets.size() == 3u);
	REQUIRE(result.samples_emitted == 1000u);
	for (const auto& packet : result.packets)
	{
		REQUIRE(packet.bytes.size() <= 1400u);
		REQUIRE((readU32(packet.bytes, 0) & 0x0000FFFFu) == packet.bytes.size() / 4u);
	}
	REQUIRE(readU32(result.packets.front().bytes, 16) == 1'700'000'001u);
	REQUIRE(readU64(result.packets.front().bytes, 20) == 373'456'789'000ull);
	REQUIRE(readU16(result.packets.front().bytes, 28) == 0x4000u);
	REQUIRE(readU16(result.packets.front().bytes, 30) == 0xC000u);
}

TEST_CASE("VITA packet count rolls over modulo 16", "[serial][vita49]")
{
	serial::vita49::PacketCountSequencer counts;
	counts.setNext(14);
	REQUIRE(counts.next() == 14u);
	REQUIRE(counts.next() == 15u);
	REQUIRE(counts.next() == 0u);
	REQUIRE(counts.next() == 1u);
}

TEST_CASE("VITA stream registry allocates stable non-truncated stream IDs", "[serial][vita49]")
{
	serial::vita49::StreamRegistry registry;
	const core::ReceiverStreamDescriptor a{.receiver_id = 0x00030000FFFFFFFFull, .receiver_name = "rx-a", .mode = "cw"};
	const core::ReceiverStreamDescriptor b{.receiver_id = 0x00030001FFFFFFFFull, .receiver_name = "rx-b", .mode = "cw"};

	const auto stream_a = registry.registerStream(a);
	const auto stream_a_again = registry.registerStream(a);
	const auto stream_b = registry.registerStream(b);

	REQUIRE(stream_a != 0u);
	REQUIRE(stream_a == stream_a_again);
	REQUIRE(stream_a != stream_b);
	REQUIRE(stream_a != static_cast<std::uint32_t>(a.receiver_id));
}

TEST_CASE("VITA queue overflow drops data and loss flags appear in next packet/context", "[serial][vita49]")
{
	using namespace serial::vita49;

	auto recording = std::make_unique<RecordingSender>();
	auto* recording_raw = recording.get();
	PacedSender sender(std::move(recording), 1);
	sender.open("127.0.0.1", 1);

	const SerializedPacket first{.bytes = {0, 0, 0, 1},
								 .stream_id = 9,
								 .sample_count = 10,
								 .first_sample_time = 1000.0,
								 .data_packet = true,
								 .timestamp = Timestamp{}};
	const SerializedPacket second{.bytes = {0, 0, 0, 2},
								  .stream_id = 9,
								  .sample_count = 12,
								  .first_sample_time = 1000.1,
								  .data_packet = true,
								  .timestamp = Timestamp{}};

	const auto first_result = sender.enqueue(first);
	const auto second_result = sender.enqueue(second);

	REQUIRE(recording_raw->opened);
	REQUIRE(first_result.enqueued);
	REQUIRE_FALSE(first_result.dropped);
	REQUIRE_FALSE(second_result.enqueued);
	REQUIRE(second_result.dropped);
	REQUIRE(second_result.dropped->stream_id == 9u);
	REQUIRE(second_result.dropped->sample_count == 12u);

	Vita49Packetizer packetizer(1'700'000'000'000'000'000ull, 1.0, 1400);
	std::vector<ComplexType> samples{ComplexType(0.0, 0.0)};
	const core::ReceiverStreamDescriptor stream{
		.receiver_id = 9, .receiver_name = "rx", .mode = "cw", .sample_rate = 1.0};
	const core::ReceiverSampleBlock block{.stream = stream, .sample_rate = 1.0, .samples = samples};
	PacketCountSequencer counts;
	const auto packets = packetizer.packetize(block, 9, counts, true);

	REQUIRE((readU32(packets.packets.front().bytes, packets.packets.front().bytes.size() - 4u) & TrailerSampleLoss) ==
			TrailerSampleLoss);

	const auto context = Vita49ContextBuilder::build(ContextBuildRequest{.stream = stream,
																		 .stream_id = 9,
																		 .simulation_name = "sim",
																		 .adc_fullscale = 1.0,
																		 .timestamp = Timestamp{10u, 0u},
																		 .packet_count = counts.next(),
																		 .sample_loss = true});
	const auto context_bytes = Vita49Serializer::serializeContext(context);
	REQUIRE((readU32(context_bytes, 32) & TrailerSampleLoss) == TrailerSampleLoss);
	REQUIRE((readU32(context_bytes, 96) & ContextFlagSampleLoss) == ContextFlagSampleLoss);
}

TEST_CASE("VITA output sink emits context, signal data, and stats through injected sender", "[serial][vita49]")
{
	using namespace serial::vita49;

	auto recording = std::make_unique<RecordingSender>();
	auto* recording_raw = recording.get();
	Vita49OutputSink sink(std::move(recording));
	const core::OutputConfig config{.mode = core::OutputMode::Vita49Udp,
									.vita49 = {.host = "127.0.0.1",
											   .port = 1,
											   .adc_fullscale = 1.0,
											   .queue_depth = 8,
											   .epoch_unix_nanoseconds = 1'700'000'000'000'000'000ull,
											   .max_udp_payload = 1400}};
	sink.initializeRun(config, "sim");

	const core::ReceiverStreamDescriptor stream{
		.receiver_id = 42, .receiver_name = "rx", .mode = "cw", .sample_rate = 2.0, .reference_frequency = 9.0e8};
	std::vector<ComplexType> samples{ComplexType(0.25, -0.25), ComplexType(2.0, 0.0)};
	const core::ReceiverSampleBlock block{
		.stream = stream, .first_sample_time = 0.0, .sample_rate = 2.0, .samples = samples, .sample_start = 0};

	sink.submitBlock(block);
	const auto stats = sink.finalize();

	REQUIRE(recording_raw->opened);
	REQUIRE(recording_raw->closed);
	REQUIRE(recording_raw->sent.size() >= 3u);
	REQUIRE(stats.streams.size() == 1u);
	CHECK(stats.streams.front().samples_emitted == 2u);
	CHECK(stats.streams.front().packets_emitted == 1u);
	CHECK(stats.streams.front().context_packets >= 2u);
	CHECK(stats.streams.front().over_range_count == 1u);
}

#ifndef _WIN32
TEST_CASE("VITA UDP sender loopback carries stream ID and packet count", "[serial][vita49][udp]")
{
	using namespace serial::vita49;

	const int receiver = ::socket(AF_INET, SOCK_DGRAM, 0);
	if (receiver < 0)
	{
		SKIP("loopback socket unavailable");
	}

	sockaddr_in bind_addr{};
	bind_addr.sin_family = AF_INET;
	bind_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	bind_addr.sin_port = 0;
	if (::bind(receiver, reinterpret_cast<sockaddr*>(&bind_addr), sizeof(bind_addr)) != 0)
	{
		::close(receiver);
		SKIP("loopback bind unavailable");
	}

	socklen_t addr_len = sizeof(bind_addr);
	REQUIRE(::getsockname(receiver, reinterpret_cast<sockaddr*>(&bind_addr), &addr_len) == 0);

	timeval timeout{};
	timeout.tv_sec = 1;
	REQUIRE(::setsockopt(receiver, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0);

	const SignalDataPacket packet{.stream_id = 0x01020304u,
								  .timestamp = Timestamp{1u, 2u},
								  .packet_count = 15u,
								  .iq_interleaved = {1, 2},
								  .trailer = makeTrailer(true, true, true, false, false)};
	const auto bytes = Vita49Serializer::serializeSignalData(packet);

	UdpSender sender;
	sender.open("127.0.0.1", ntohs(bind_addr.sin_port));
	sender.send(bytes);

	std::vector<std::uint8_t> received(256);
	const auto received_count = ::recv(receiver, received.data(), received.size(), 0);
	::close(receiver);
	REQUIRE(received_count == static_cast<ssize_t>(bytes.size()));
	received.resize(static_cast<std::size_t>(received_count));

	REQUIRE(readU32(received, 4) == 0x01020304u);
	REQUIRE(((readU32(received, 0) >> 16u) & 0x0Fu) == 15u);
}
#endif
