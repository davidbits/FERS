// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#include "serial/vita49/vita49_serializer.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace serial::vita49
{
	namespace
	{
		[[nodiscard]] std::int16_t scaleComponentToInt16(const RealType value, const RealType fullscale,
														 bool& clipped) noexcept
		{
			if (value > fullscale)
			{
				clipped = true;
				return std::numeric_limits<std::int16_t>::max();
			}
			if (value < -fullscale)
			{
				clipped = true;
				return std::numeric_limits<std::int16_t>::min();
			}
			if (value == fullscale)
			{
				return std::numeric_limits<std::int16_t>::max();
			}
			if (value == -fullscale)
			{
				return std::numeric_limits<std::int16_t>::min();
			}

			const RealType scaled =
				std::round((value / fullscale) * static_cast<RealType>(std::numeric_limits<std::int16_t>::max()));
			return static_cast<std::int16_t>(
				std::clamp<RealType>(scaled, static_cast<RealType>(std::numeric_limits<std::int16_t>::min()),
									 static_cast<RealType>(std::numeric_limits<std::int16_t>::max())));
		}

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

	ByteWriter::ByteWriter(const std::size_t reserve_bytes)
	{
		if (reserve_bytes > 0u)
		{
			_bytes.reserve(reserve_bytes);
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

		ByteWriter writer(byte_count);
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

	SignalDataSerializationResult Vita49Serializer::serializeSignalDataFixedFullscale(
		const std::uint32_t stream_id, const std::uint64_t class_id, const Timestamp timestamp,
		const std::uint8_t packet_count, const bool valid_data, const bool calibrated_time, const bool reference_lock,
		const bool sample_loss, const std::span<const ComplexType> samples, const RealType fullscale)
	{
		if (!std::isfinite(fullscale) || fullscale <= 0.0)
		{
			throw std::invalid_argument("VITA signal full-scale must be positive and finite");
		}

		const std::size_t byte_count = kSignalDataFixedBytes + samples.size() * sizeof(std::int16_t) * 2u;
		const auto packet_size_words = checkedWordCount(byte_count);

		ByteWriter writer(byte_count);
		writer.writeU32(makeHeader(PacketType::SignalDataWithStreamId, true, true, IntegerTimestampMode::Utc,
								   FractionalTimestampMode::RealTimePicoseconds, packet_count, packet_size_words));
		writer.writeU32(stream_id);
		writer.writeU64(class_id);
		writer.writeU32(timestamp.integer_seconds);
		writer.writeU64(timestamp.fractional_picoseconds);

		std::uint64_t clipped_sample_count = 0;
		for (const auto& sample : samples)
		{
			bool clipped = false;
			writer.writeI16(scaleComponentToInt16(sample.real(), fullscale, clipped));
			writer.writeI16(scaleComponentToInt16(sample.imag(), fullscale, clipped));
			if (clipped)
			{
				++clipped_sample_count;
			}
		}
		writer.writeU32(
			makeTrailer(valid_data, calibrated_time, reference_lock, clipped_sample_count > 0, sample_loss));

		if (writer.bytes().size() != byte_count)
		{
			throw std::logic_error("VITA direct signal packet byte count mismatch");
		}
		return SignalDataSerializationResult{.bytes = writer.takeBytes(), .clipped_sample_count = clipped_sample_count};
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

		ByteWriter writer(byte_count);
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
