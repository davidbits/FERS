// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#include "serial/vita49/vita49_serializer.h"

#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace serial::vita49
{
	namespace
	{
		[[nodiscard]] std::uint16_t checkedWordCount(const std::size_t byte_count)
		{
			if (byte_count % sizeof(std::uint32_t) != 0u)
			{
				throw std::logic_error("VITA packet size must be 32-bit aligned");
			}
			const auto words = byte_count / sizeof(std::uint32_t);
			if (words > std::numeric_limits<std::uint16_t>::max())
			{
				throw std::length_error("VITA packet exceeds 16-bit word count");
			}
			return static_cast<std::uint16_t>(words);
		}
	}

	void ByteWriter::writeU16(const std::uint16_t value)
	{
		_bytes.push_back(static_cast<std::uint8_t>((value >> 8u) & 0xFFu));
		_bytes.push_back(static_cast<std::uint8_t>(value & 0xFFu));
	}

	void ByteWriter::writeI16(const std::int16_t value) { writeU16(static_cast<std::uint16_t>(value)); }

	void ByteWriter::writeU32(const std::uint32_t value)
	{
		_bytes.push_back(static_cast<std::uint8_t>((value >> 24u) & 0xFFu));
		_bytes.push_back(static_cast<std::uint8_t>((value >> 16u) & 0xFFu));
		_bytes.push_back(static_cast<std::uint8_t>((value >> 8u) & 0xFFu));
		_bytes.push_back(static_cast<std::uint8_t>(value & 0xFFu));
	}

	void ByteWriter::writeU64(const std::uint64_t value)
	{
		writeU32(static_cast<std::uint32_t>((value >> 32u) & 0xFFFFFFFFull));
		writeU32(static_cast<std::uint32_t>(value & 0xFFFFFFFFull));
	}

	void ByteWriter::writeF64(const RealType value)
	{
		static_assert(sizeof(RealType) == sizeof(std::uint64_t));
		const auto bits = std::bit_cast<std::uint64_t>(value);
		writeU64(bits);
	}

	void ByteWriter::writeStringField(const std::string& value)
	{
		if (value.size() > std::numeric_limits<std::uint32_t>::max())
		{
			throw std::length_error("VITA context string field too large");
		}
		writeU32(static_cast<std::uint32_t>(value.size()));
		_bytes.insert(_bytes.end(), value.begin(), value.end());
		while (_bytes.size() % sizeof(std::uint32_t) != 0u)
		{
			_bytes.push_back(0);
		}
	}

	void ByteWriter::writeBytes(const std::span<const std::uint8_t> bytes)
	{
		_bytes.insert(_bytes.end(), bytes.begin(), bytes.end());
	}

	const std::vector<std::uint8_t>& ByteWriter::bytes() const noexcept { return _bytes; }

	std::vector<std::uint8_t> ByteWriter::takeBytes() noexcept { return std::move(_bytes); }

	std::vector<std::uint8_t> Vita49Serializer::serializeSignalData(const SignalDataPacket& packet)
	{
		if (packet.iq_interleaved.size() % 2u != 0u)
		{
			throw std::invalid_argument("VITA signal IQ payload must contain I/Q pairs");
		}

		const std::size_t byte_count = kSignalDataFixedBytes + packet.iq_interleaved.size() * sizeof(std::int16_t);
		const auto packet_size_words = checkedWordCount(byte_count);

		ByteWriter writer;
		writer.writeU32(makeHeader(PacketType::SignalDataWithStreamId, true, true, IntegerTimestampMode::Utc,
								   FractionalTimestampMode::RealTimePicoseconds, packet.packet_count,
								   packet_size_words));
		writer.writeU32(packet.stream_id);
		writer.writeU64(packet.class_id);
		writer.writeU32(packet.timestamp.integer_seconds);
		writer.writeU64(packet.timestamp.fractional_picoseconds);
		for (const auto item : packet.iq_interleaved)
		{
			writer.writeI16(item);
		}
		writer.writeU32(packet.trailer);

		if (writer.bytes().size() != byte_count)
		{
			throw std::logic_error("VITA signal packet byte count mismatch");
		}
		return writer.takeBytes();
	}

	std::vector<std::uint8_t> Vita49Serializer::serializeContext(const ContextPacket& packet)
	{
		ByteWriter payload;
		payload.writeU32(packet.cif0);
		payload.writeU32(packet.state_indicators);
		payload.writeU64(packet.payload_format);
		payload.writeF64(packet.sample_rate);
		payload.writeF64(packet.reference_frequency);
		payload.writeF64(packet.if_offset);
		payload.writeF64(packet.bandwidth);
		payload.writeF64(packet.adc_fullscale);
		payload.writeU64(packet.receiver_id);
		payload.writeU32(static_cast<std::uint32_t>(packet.adc_bits));
		payload.writeU32(packet.context_flags);
		payload.writeStringField(packet.receiver_name);
		payload.writeStringField(packet.simulation_name);
		payload.writeStringField(packet.receiver_mode);

		const std::size_t byte_count = 4u + 4u + 8u + 4u + 8u + payload.bytes().size();
		const auto packet_size_words = checkedWordCount(byte_count);

		ByteWriter writer;
		writer.writeU32(makeHeader(PacketType::Context, true, false, IntegerTimestampMode::Utc,
								   FractionalTimestampMode::RealTimePicoseconds, packet.packet_count,
								   packet_size_words));
		writer.writeU32(packet.stream_id);
		writer.writeU64(packet.class_id);
		writer.writeU32(packet.timestamp.integer_seconds);
		writer.writeU64(packet.timestamp.fractional_picoseconds);
		writer.writeBytes(payload.bytes());
		return writer.takeBytes();
	}
}
