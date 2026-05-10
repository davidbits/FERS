// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#include "serial/vita49/stream_registry.h"

#include <stdexcept>

namespace serial::vita49
{
	namespace
	{
		[[nodiscard]] std::uint32_t fnv1a32(const std::string& value) noexcept
		{
			std::uint32_t hash = 2166136261u;
			for (const char ch : value)
			{
				hash ^= static_cast<std::uint8_t>(ch);
				hash *= 16777619u;
			}
			return hash;
		}
	}

	bool StreamRegistry::Key::operator==(const Key& other) const noexcept
	{
		return receiver_id == other.receiver_id && receiver_name == other.receiver_name;
	}

	std::size_t StreamRegistry::KeyHash::operator()(const Key& key) const noexcept
	{
		return static_cast<std::size_t>(initialStreamId(key));
	}

	std::uint32_t StreamRegistry::registerStream(const core::ReceiverStreamDescriptor& stream)
	{
		Key key{.receiver_id = stream.receiver_id, .receiver_name = stream.receiver_name};
		if (const auto found = _by_key.find(key); found != _by_key.end())
		{
			return found->second;
		}

		std::uint32_t stream_id = initialStreamId(key);
		for (std::uint32_t attempts = 0; attempts < 0x7FFFFFFFu; ++attempts)
		{
			if (stream_id == 0)
			{
				stream_id = 1;
			}
			if (!_by_stream_id.contains(stream_id))
			{
				_by_key.emplace(key, stream_id);
				_by_stream_id.emplace(stream_id, key);
				_descriptors.emplace(stream_id, stream);
				return stream_id;
			}
			stream_id = (stream_id & 0x7FFFFFFFu) + 1u;
		}

		throw std::runtime_error("Unable to allocate collision-free VITA stream ID");
	}

	bool StreamRegistry::contains(const std::uint32_t stream_id) const { return _by_stream_id.contains(stream_id); }

	const core::ReceiverStreamDescriptor& StreamRegistry::descriptor(const std::uint32_t stream_id) const
	{
		const auto found = _descriptors.find(stream_id);
		if (found == _descriptors.end())
		{
			throw std::out_of_range("Unknown VITA stream ID");
		}
		return found->second;
	}

	std::uint32_t StreamRegistry::initialStreamId(const Key& key) noexcept
	{
		const std::string material = std::to_string(key.receiver_id) + ":" + key.receiver_name;
		std::uint32_t stream_id = fnv1a32(material) & 0x7FFFFFFFu;
		if (stream_id == 0)
		{
			stream_id = 1;
		}
		return stream_id;
	}
}
