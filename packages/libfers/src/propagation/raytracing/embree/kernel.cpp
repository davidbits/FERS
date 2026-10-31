// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#include "propagation/raytracing/embree/kernel.h"

#include <embree4/rtcore.h>

#include "propagation/raytracing/sbr_shared.h"

namespace propagation::raytracing::embree
{
	RTCRayHit castRay(RTCScene scene, const Float3& origin, const Float3& direction)
	{
		RTCRayHit rh{};
		rh.ray.org_x = origin.x;
		rh.ray.org_y = origin.y;
		rh.ray.org_z = origin.z;
		rh.ray.dir_x = direction.x;
		rh.ray.dir_y = direction.y;
		rh.ray.dir_z = direction.z;
		rh.ray.tnear = MIN_RAY_DISTANCE;
		rh.ray.tfar = MAX_RAY_DISTANCE;
		rh.ray.mask = 0xFFFFFFFFu;
		rh.ray.flags = 0;
		rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
		for (auto& inst_id : rh.hit.instID)
			inst_id = RTC_INVALID_GEOMETRY_ID;

		RTCIntersectArguments args{};
		rtcInitIntersectArguments(&args);
		rtcIntersect1(scene, &rh, &args);

		return rh;
	}
}
