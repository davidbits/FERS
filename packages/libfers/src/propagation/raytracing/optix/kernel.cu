// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#include <cstdint>
#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_device.h>
#include <optix_types.h>
#include <vector_types.h>

#include "propagation/math.h"
#include "propagation/raytracing/optix/kernel_defs.h"
#include "propagation/raytracing/sbr_impl.h"
#include "propagation/raytracing/sbr_shared.h"

// Silence the warnings about the names starting with two underscores.
// This naming scheme comes from official NVIDIA OptiX tutorials and thus
// are unlikely to be aliased by new identifiers.
// NOLINTBEGIN(bugprone-reserved-identifier)

namespace propagation::raytracing::optix
{
	// -------------------------------------------------------------------------------- //
	// Utilities
	// -------------------------------------------------------------------------------- //

	template <typename T>
	static constexpr inline __device__ void packPointer(T* ptr, uint32_t* hi, uint32_t* lo)
	{
		// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): bitcasting pointers to uintptr_t is safe.
		auto bits = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
		*lo = static_cast<uint32_t>(bits);
		*hi = static_cast<uint32_t>(bits >> 32);
	}

	template <typename T>
	static constexpr inline __device__ T* unpackPointer(uint32_t hi, uint32_t lo)
	{
		auto bits = static_cast<uint64_t>(lo) + (static_cast<uint64_t>(hi) << 32);
		// Bitcasting uintptr_t to a pointer is safe:
		// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast, performance-no-int-to-ptr)
		return reinterpret_cast<T*>(static_cast<uintptr_t>(bits));
	}

	static constexpr inline __device__ float3 asF32(const Double3& v)
	{
		return float3{static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)};
	}
	static constexpr inline __device__ float3 asF32(const Float3& v) { return float3{v.x, v.y, v.z}; }

	/**
	 * @brief Build an affine 3x4 transform from an OptiX transform float array.
	 */
	static constexpr inline __device__ Float3x4 fromOptixTransform(const float t[12])
	{
		// OptiX dumps it's matrix as row-major list.
		// Float3x4 is currently column-major.
		return Float3x4{
			Float3{t[0], t[4], t[8]},
			Float3{t[1], t[5], t[9]},
			Float3{t[2], t[6], t[10]},
			Float3{t[3], t[7], t[11]},
		};
	}


	// -------------------------------------------------------------------------------- //
	// Shadow/Occlusion rays
	// -------------------------------------------------------------------------------- //

// Suppress "warning #20044-D: extern declaration of the entity shader_params is treated as a static definition"
// That's what we want to happen - this is the only device-side translation unit.
// NOLINTNEXTLINE: clangd incorrectly complains about the nvcc-specific pragma.
#pragma nv_diag_suppress 20044

	/// Launch parameters in constant memory, filled in by optix upon optixLaunch.
	extern "C" __constant__ ShaderParams shader_params;

	extern "C" __global__ void __closesthit__shadow()
	{
		unsigned int hi = optixGetPayload_0(), lo = optixGetPayload_1();
		auto* shadow_t = unpackPointer<float>(hi, lo);
		*shadow_t = optixGetRayTmax();
	}

	extern "C" __global__ void __anyhit__shadow() {}

	extern "C" __global__ void __miss__shadow() {}


	static __device__ inline float shadowTest(const Float3& origin, const Float3& direction, OptixTraversableHandle ias)
	{
		// Default to unoccluded. Upon hit, this is updated.
		float shadow_t = MAX_RAY_DISTANCE;

		unsigned int hi{}, lo{};
		packPointer(&shadow_t, &hi, &lo);
		optixTrace(ias, // AS handle
				   float3{origin.x, origin.y, origin.z}, // rayOrigin
				   float3{direction.x, direction.y, direction.z}, // rayDirection
				   MIN_RAY_DISTANCE, // tmin
				   MAX_RAY_DISTANCE, // tmax
				   0.0f, // rayTime
				   OptixVisibilityMask(0xFF), // visibilityMask TODO?
				   OPTIX_RAY_FLAG_DISABLE_ANYHIT, // rayFlags
				   RAY_TYPE_SHADOW, // SBT offset
				   RAY_TYPE_COUNT, // SBT stride
				   RAY_TYPE_SHADOW, // missSBTIndex
				   hi, lo); // payload; up to 32 unsigned ints

		return shadow_t;
	}

	static __device__ inline Contribution* getContributionPtr()
	{
		const auto index = min(shader_params.contribution_count->fetch_add(1), shader_params.contribution_capacity - 1);
		return &shader_params.contributions[index];
	}

	// -------------------------------------------------------------------------------- //
	// Geometric Optics (GO) rays
	// -------------------------------------------------------------------------------- //

	/// The Geometric Optics (GO) closest hit program.
	extern "C" __global__ void __closesthit__indirect()
	{
		const uint32_t material_index = optixGetInstanceId();
		const uint32_t triangle_index = optixGetPrimitiveIndex();
		const float2 o_bary = optixGetTriangleBarycentrics();

		HitInfo hit{};
		hit.triangle_index = triangle_index;
		hit.material_index = material_index;
		hit.bary_u = o_bary.x;
		hit.bary_v = o_bary.y;

		float world_to_obj_raw[12];
		float obj_to_world_raw[12];

		// These OptiX C API functions and helpers expect float[12] arrays. Passing them in as-is is appropriate.
		// NOLINTBEGIN(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
		optixGetWorldToObjectTransformMatrix(world_to_obj_raw);
		optixGetObjectToWorldTransformMatrix(obj_to_world_raw);
		hit.world_to_obj = fromOptixTransform(world_to_obj_raw);
		hit.obj_to_world = fromOptixTransform(obj_to_world_raw);
		// NOLINTEND(cppcoreguidelines-pro-bounds-array-to-pointer-decay)

		unsigned int hi = optixGetPayload_0(), lo = optixGetPayload_1();
		auto* path = unpackPointer<PathState>(hi, lo);

		const auto ias = shader_params.iass[path->time_index];
		const auto shadow_test = [ias](const Float3& origin, const Float3& dir)
		{ return shadowTest(origin, dir, ias); };

		auto should_reflect = facetHit(shader_params.sbr, path, hit, shadow_test, getContributionPtr);

		if (!should_reflect)
			return; // we're done with this path

		const auto origin = asF32(offsetSurfaceOrigin(Float3{path->path_vertices[path->ray_count]},
													  path->surface_normals[path->ray_count - 1]));
		const auto dir = asF32(path->trace_directions[path->ray_count]);

		// NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index): can't use throwing .at() in CUDA
		optixTrace(ias, // AS handle
				   origin, // rayOrigin
				   dir, // rayDirection
				   MIN_RAY_DISTANCE, // tmin
				   MAX_RAY_DISTANCE, // tmax
				   0.0f, // rayTime
				   OptixVisibilityMask(0xFF), // visibilityMask
				   OPTIX_RAY_FLAG_DISABLE_ANYHIT, // rayFlags
				   RAY_TYPE_INDIRECT, // SBT offset
				   RAY_TYPE_COUNT, // SBT stride
				   RAY_TYPE_INDIRECT, // missSBTIndex
				   hi, lo // payload; up to 32 unsigned ints
		);
		// NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
	}

	/// This is the Geometric Optics (GO) any-hit program.
	/// This does nothing because we only care about the closest hit.
	/// Any-hit checks get disabled where applicable.
	extern "C" __global__ void __anyhit__indirect() {}

	/// This is the indirect path ray miss program.
	/// The radar energy has zoomed off into space. Goodbye energy. Do nothing. Stop tracing.
	extern "C" __global__ void __miss__indirect() {}


	/// This function is where we're going to for each radar "source"
	/// (transmitter or receiver antenna, depending on traceFromTransmitter or traceFromReceiver)
	/// and each time step and generate lots of rays to trace through the scene and
	/// determine contributions to each of the "sinks" (receivers or transmitters, respectively).
	extern "C" __global__ void __raygen__paths()
	{
		uint3 idx = optixGetLaunchIndex();
		/// The launch direction index.
		unsigned int path_index = idx.x + shader_params.direction_index_offset;
		/// The "source" antenna to launch from. Could be a transmitter or receiver.
		unsigned int antenna_index = idx.y;
		/// The time-step index. Determines which IAS to use.
		unsigned int t_index = idx.z;

		const auto ias = shader_params.iass[t_index];
		const auto shadow_test = [ias](const Float3& origin, const Float3& dir)
		{ return shadowTest(origin, dir, ias); };

		// Zero-bounce direct-path check, piggybacked on spare X-lanes (dest_antenna_count is tiny).
		// Looped, not `if`, so it stays correct even if rays_per_source < dest_antenna_count.
		for (uint32_t d = path_index; d < shader_params.sbr.dest_antenna_count;
			 d += shader_params.sbr.directions_per_source)
		{
			directPath(shader_params.sbr, antenna_index, d, t_index, shadow_test, getContributionPtr);
		}

		PathState path{};
		initIndirectPath(shader_params.sbr, path_index, antenna_index, t_index, path);

		const auto origin = asF32(path.path_vertices[0]);

		unsigned int hi{}, lo{};
		packPointer(&path, &hi, &lo);
		optixTrace(ias, // AS handle
				   origin, // rayOrigin
				   asF32(path.trace_directions[0]), // rayDirection
				   MIN_RAY_DISTANCE, // tmin
				   MAX_RAY_DISTANCE, // tmax
				   0.0f, // rayTime
				   OptixVisibilityMask(0xFF), // visibilityMask
				   OPTIX_RAY_FLAG_DISABLE_ANYHIT, // rayFlags
				   RAY_TYPE_INDIRECT, // SBT offset
				   RAY_TYPE_COUNT, // SBT stride
				   RAY_TYPE_INDIRECT, // missSBTIndex
				   hi, lo // payload; up to 32 unsigned ints
		);
	}
}

// NOLINTEND(bugprone-reserved-identifier)
