// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <cstdint>

namespace core
{
	struct Vita49Timestamp
	{
		std::uint32_t integer_seconds = 0;
		std::uint64_t fractional_picoseconds = 0;
	};
}
