// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include "core/receiver_output.h"

namespace serial::vita49
{
	class StreamRegistry
	{
	public:
		[[nodiscard]] std::uint32_t registerStream(const core::ReceiverStreamDescriptor& stream);
		[[nodiscard]] bool contains(std::uint32_t stream_id) const;
		[[nodiscard]] const core::ReceiverStreamDescriptor& descriptor(std::uint32_t stream_id) const;

	private:
		struct Key
		{
			SimId receiver_id = 0;
			std::string receiver_name;

			[[nodiscard]] bool operator==(const Key& other) const noexcept;
		};

		struct KeyHash
		{
			[[nodiscard]] std::size_t operator()(const Key& key) const noexcept;
		};

		[[nodiscard]] static std::uint32_t initialStreamId(const Key& key) noexcept;

		std::unordered_map<Key, std::uint32_t, KeyHash> _by_key;
		std::unordered_map<std::uint32_t, Key> _by_stream_id;
		std::unordered_map<std::uint32_t, core::ReceiverStreamDescriptor> _descriptors;
	};
}
