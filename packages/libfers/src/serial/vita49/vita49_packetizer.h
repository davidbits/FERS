// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <cstdint>

#include "serial/vita49/vita49_types.h"

namespace serial::vita49
{
	class Vita49Packetizer
	{
	public:
		Vita49Packetizer(std::uint64_t epoch_unix_nanoseconds, RealType adc_fullscale,
						 std::uint16_t max_udp_payload_bytes = kDefaultMaxUdpPayloadBytes);

		[[nodiscard]] std::uint64_t epochUnixNanoseconds() const noexcept;
		[[nodiscard]] RealType adcFullscale() const noexcept;
		[[nodiscard]] std::uint16_t maxUdpPayloadBytes() const noexcept;
		[[nodiscard]] std::size_t maxComplexSamplesPerPacket() const noexcept;

		[[nodiscard]] PacketizerResult packetize(const core::ReceiverSampleBlock& block, std::uint32_t stream_id,
												 PacketCountSequencer& packet_counts, bool sample_loss_pending) const;

		[[nodiscard]] SerializedPacket makeContextPacket(const ContextPacket& context) const;

	private:
		[[nodiscard]] std::int16_t quantize(RealType value, bool& clipped) const noexcept;

		std::uint64_t _epoch_unix_nanoseconds = 0;
		RealType _adc_fullscale = 1.0;
		std::uint16_t _max_udp_payload_bytes = kDefaultMaxUdpPayloadBytes;
		std::size_t _max_complex_samples_per_packet = 0;
	};
}
