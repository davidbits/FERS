// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <embree4/rtcore.h>
#include <unordered_map>
#include <vector>

#include "core/sim_id.h"
#include "core/thread_pool.h"
#include "propagation/propagation_model.h"
#include "propagation/raytracing/embree/kernel.h"
#include "propagation/raytracing/embree/utils.h"
#include "propagation/raytracing/raytracing_model.h"

namespace propagation::raytracing::embree
{
	struct EmbreeThreadContext : public ThreadContext
	{
		/// The top-level scene, equivalent to a TLAS or IAS.
		/// One `RTC_GEOMETRY_TYPE_INSTANCE` geometry per target with geometry, instancing that
		/// target's mesh-level scene. Built in `makeThreadContext` and updated in
		/// `EmbreeBackend::updateInstanceTransforms`..
		Scene tlas;

		std::vector<Geometry> instances;
		std::vector<const radar::Target*> instance_targets;
		/// Parallel to `instances`/`instance_targets`, indexed by `RTCHit::instID[0]` - see
		/// `InstanceInfo`'s doc comment on why that indexing is safe.
		std::vector<InstanceInfo> instance_info;

		explicit EmbreeThreadContext(RTCDevice device) : tlas(device) {}
	};

	class EmbreeBackend : public RayTracingBackend
	{
	public:
		/// `pool` is the simulation's shared CPU worker pool. `EmbreeBackend` uses it to
		/// parallelise ray batches across threads within a single `trace()` call; it must
		/// outlive this backend.
		EmbreeBackend(SceneData scene, pool::ThreadPool& pool);

		EmbreeBackend(const EmbreeBackend&) = delete;
		EmbreeBackend& operator=(const EmbreeBackend&) = delete;
		EmbreeBackend(EmbreeBackend&&) = delete;
		EmbreeBackend& operator=(EmbreeBackend&&) = delete;

		~EmbreeBackend() override = default;

		std::unique_ptr<ThreadContext> makeThreadContext(core::World* world) const override;

		std::vector<Contribution> trace(ThreadContext* thread_context, TraceJob& job) const override;

	private:
		/// Refreshes every instance's object-to-world transform (and its cached inverse) for time
		/// `t`, re-commits each changed instance geometry, and re-commits `ctx.tlas` (Embree's
		/// refit/rebuild, analogous to OptiX's `buildOrRefitIAS`).
		void updateInstanceTransforms(EmbreeThreadContext& ctx, RealType t) const;

		pool::ThreadPool& _pool;

		Device _device;

		// Host-side geometry data, owned for the backend's lifetime. Per-mesh scenes reference
		// `_vertices`/`_indices` directly via `rtcSetSharedGeometryBuffer` (zero-copy) - these must
		// outlive `_mesh_scenes`.
		std::vector<Float3> _vertices; ///< One extra zero-valued element appended at the end; see ctor.
		std::vector<Uint3> _indices;
		std::vector<TriangleMeshView> _meshes;
		std::unordered_map<SimId, uint32_t> _mesh_indices;

		std::unordered_map<SimId, uint32_t> _material_indices;
		std::vector<Material> _materials;

		std::vector<AntennaModel> _antenna_models;
		std::vector<float> _antenna_gains;

		std::vector<Scene> _mesh_scenes;
	};
}
