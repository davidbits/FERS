// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <cstdint>
#include <embree4/rtcore.h>
#include <vector>

#include "propagation/math.h"
#include "propagation/raytracing/sbr_impl.h"
#include "propagation/raytracing/sbr_shared.h"

namespace propagation::raytracing::embree
{
	/// Per-instance data mapping Embree hit indices (`instID`/`primID`) into the shared kernel's `HitInfo`.
	struct InstanceInfo
	{
		/// Added to `RTCHit::primID` to get a global triangle index into `SbrParams::indices`.
		/// Stands in for OptiX's `primitiveIndexOffset`.
		uint32_t mesh_triangle_offset{};
		uint32_t material_index{};
		Float3x4 obj_to_world;
		Float3x4 world_to_obj;
	};

	/// Finds the closest hit along a ray against `scene`.
	///
	/// Misses leave `hit.geomID == RTC_INVALID_GEOMETRY_ID` and `ray.tfar == MAX_RAY_DISTANCE`.
	[[nodiscard]] RTCRayHit castRay(RTCScene scene, const Float3& origin, const Float3& direction);

	/// Engine-supplied occlusion test for `facetHit`/`directPath`.
	///
	/// Return the hit distance along the ray, or `MAX_RAY_DISTANCE` if the ray misses.
	[[nodiscard]] inline float shadowTest(RTCScene scene, const Float3 origin, const Float3 direction)
	{
		return castRay(scene, origin, direction).ray.tfar;
	}

	/// Walks one indirect (GO/PO) path to completion against `scene`.
	/// Traces, passes each hit to `facetHit`, and keeps going while `facetHit` returns true.
	template <typename ContribIndex>
	void traceIndirectPath(RTCScene scene, const SbrParams& params, const std::vector<InstanceInfo>& instances,
						   const uint32_t path_idx, const uint32_t source_idx, const uint32_t time_idx,
						   ContribIndex get_contrib_ptr)
	{
		PathState path{};
		initIndirectPath(params, path_idx, source_idx, time_idx, path);

		const auto shadow_test = [scene](const Float3 origin, const Float3 segment)
		{ return shadowTest(scene, origin, segment); };

		for (;;)
		{
			// Past the launch, the ray leaves the facet just hit along its normal side, so its
			// origin is lifted off that facet to ensure against self-intersection.
			const Float3 origin = path.ray_count == 0
				? Float3{path.path_vertices.at(0)}
				: offsetSurfaceOrigin(Float3{path.path_vertices.at(path.ray_count)},
									  path.surface_normals.at(path.ray_count - 1));

			const RTCRayHit hit = castRay(scene, origin, path.trace_directions.at(path.ray_count));

			if (hit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
				return; // Miss. The radar energy left the scene.

			// Build the hit info struct.
			const InstanceInfo& inst = instances.at(hit.hit.instID[0]);
			HitInfo info{};
			info.triangle_index = hit.hit.primID + inst.mesh_triangle_offset;
			info.material_index = inst.material_index;
			info.bary_u = hit.hit.u;
			info.bary_v = hit.hit.v;
			info.world_to_obj = inst.world_to_obj;
			info.obj_to_world = inst.obj_to_world;

			if (!facetHit(params, &path, info, shadow_test, get_contrib_ptr))
				return;
		}
	}
}
