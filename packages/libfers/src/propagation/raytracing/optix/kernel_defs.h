// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <cstdint>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <optix.h>

#include "propagation/raytracing/sbr_shared.h"

namespace propagation::raytracing::optix
{
	constexpr unsigned int RAY_TYPE_COUNT = 2;
	constexpr unsigned int RAY_TYPE_INDIRECT = 0;
	constexpr unsigned int RAY_TYPE_SHADOW = 1;

	using ContributionCountType = cuda::atomic<unsigned int, cuda::thread_scope_device>;

	struct ShaderParams
	{
		/// The IASs for this run, per time. Indexed by launch Z dimension (time index).
		OptixTraversableHandle* iass;
		// Output buffer of contributions.
		// Indexing this safely device-side requires platform-specific atomic counters.
		Contribution* contributions;
		/// The number of contributions already recorded. Indexes the contribution buffer.
		ContributionCountType* contribution_count;
		/// The capacity of the contribution buffer. Do not write beyond this.
		uint32_t contribution_capacity;
		/// Direction index offset added to the launch's X index.
		/// This allows for chunking over a sub-range of [0, sbr.directions_per_source).
		uint32_t direction_index_offset;

		/// The simulation engine's parameters.
		SbrParams sbr;
	};
}
