// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <string>

#include "serial/vita49/vita49_types.h"

namespace serial::vita49
{
	struct ContextBuildRequest
	{
		core::ReceiverStreamDescriptor stream;
		std::uint32_t stream_id = 0;
		std::string simulation_name;
		RealType adc_fullscale = 0.0;
		Timestamp timestamp;
		std::uint8_t packet_count = 0;
		bool valid_data = true;
		bool calibrated_time = true;
		bool reference_lock = true;
		bool over_range = false;
		bool sample_loss = false;
		bool stream_open = false;
		bool stream_close = false;
	};

	class Vita49ContextBuilder
	{
	public:
		[[nodiscard]] static ContextPacket build(const ContextBuildRequest& request);
	};
}
