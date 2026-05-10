// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "serial/vita49/vita49_types.h"

namespace serial::vita49
{
	class ByteWriter
	{
	public:
		void writeU16(std::uint16_t value);
		void writeI16(std::int16_t value);
		void writeU32(std::uint32_t value);
		void writeU64(std::uint64_t value);
		void writeF64(RealType value);
		void writeStringField(const std::string& value);
		void writeBytes(std::span<const std::uint8_t> bytes);

		[[nodiscard]] const std::vector<std::uint8_t>& bytes() const noexcept;
		[[nodiscard]] std::vector<std::uint8_t> takeBytes() noexcept;

	private:
		std::vector<std::uint8_t> _bytes;
	};

	class Vita49Serializer
	{
	public:
		[[nodiscard]] static std::vector<std::uint8_t> serializeSignalData(const SignalDataPacket& packet);
		[[nodiscard]] static std::vector<std::uint8_t> serializeContext(const ContextPacket& packet);
	};
}
