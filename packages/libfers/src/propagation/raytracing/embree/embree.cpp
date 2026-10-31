// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#include "propagation/raytracing/embree/embree.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <embree4/rtcore.h>
#include <embree4/rtcore_geometry.h>
#include <format>
#include <future>
#include <iterator>
#include <stdexcept>

#include "core/parameters.h"
#include "core/world.h"
#include "propagation/raytracing/embree/kernel.h"
#include "radar/target.h"

namespace propagation::raytracing::embree
{
	EmbreeBackend::EmbreeBackend(SceneData scene, pool::ThreadPool& pool) :
		_pool(pool), _device(std::format("threads={}", params::workerThreads()).c_str()),
		_vertices(std::move(scene.vertices)), _indices(std::move(scene.indices)), _meshes(std::move(scene.meshes)),
		_mesh_indices(std::move(scene.mesh_ids)), _material_indices(std::move(scene.mat_ids)),
		_materials(std::move(scene.materials)), _antenna_models(std::move(scene.antenna_models)),
		_antenna_gains(std::move(scene.antenna_gains))
	{
		// Embree reads a triangle mesh's vertex buffer up to 16 bytes past the last vertex actually
		// used (SIMD-widened reads), so pad the buffer and tell Embree the real (unpadded) count.
		_vertices.emplace_back();

		const auto mesh_count = _meshes.size();
		_mesh_scenes.reserve(mesh_count);

		for (const auto& mesh : _meshes)
		{
			auto* geom = rtcNewGeometry(_device.get(), RTC_GEOMETRY_TYPE_TRIANGLE);

			// Supply the full shared vertex buffer (indices index globally). This is zero-copy.
			rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, _vertices.data(), 0,
									   sizeof(Float3), _vertices.size() - 1);
			rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
									   _indices.data() + mesh.index_buffer_index, 0, sizeof(Uint3),
									   mesh.triangle_count);
			rtcCommitGeometry(geom);

			auto& mesh_scene = _mesh_scenes.emplace_back(_device.get());
			rtcSetSceneBuildQuality(mesh_scene.get(), RTC_BUILD_QUALITY_HIGH);
			rtcAttachGeometry(mesh_scene.get(), geom);
			rtcCommitScene(mesh_scene.get());

			// The Embree-internal data is ref-counted, and the scene holds onto what it needs.
			rtcReleaseGeometry(geom);
		}
	}

	std::unique_ptr<ThreadContext> EmbreeBackend::makeThreadContext(core::World* world) const
	{
		auto ctx = std::make_unique<EmbreeThreadContext>(_device.get());

		const auto max_instances = world->getTargets().size();
		ctx->instances.reserve(max_instances);
		ctx->instance_targets.reserve(max_instances);
		ctx->instance_info.reserve(max_instances);

		for (const auto& target : world->getTargets())
		{
			const auto& geom = target->getGeometry();
			if (!geom.has_value())
				continue;

			const auto mesh_idx = _mesh_indices.at(geom->mesh->id);
			const auto mat_idx = _material_indices.at(geom->material->id);

			auto& inst = ctx->instances.emplace_back(_device.get(), RTC_GEOMETRY_TYPE_INSTANCE);
			rtcSetGeometryInstancedScene(inst.get(), _mesh_scenes[mesh_idx].get());

			// Sequential attach order (never `rtcAttachGeometryByID`) - `hit.instID[0]` then matches
			// `instance_info`'s index directly. See `InstanceInfo`'s doc comment.
			[[maybe_unused]] const unsigned int geom_id = rtcAttachGeometry(ctx->tlas.get(), inst.get());
			assert(geom_id == ctx->instance_targets.size());

			ctx->instance_targets.push_back(target.get());
			ctx->instance_info.push_back(InstanceInfo{.mesh_triangle_offset = _meshes[mesh_idx].index_buffer_index,
													  .material_index = mat_idx,
													  .obj_to_world = Float3x4{},
													  .world_to_obj = Float3x4{}});
		}

		return ctx;
	}

	void EmbreeBackend::updateInstanceTransforms(EmbreeThreadContext& ctx, const RealType t) const
	{
		for (size_t i = 0; i < ctx.instance_targets.size(); ++i)
		{
			const Float3x4 obj_to_world = computeTargetTransform(*ctx.instance_targets[i], t);
			ctx.instance_info[i].obj_to_world = obj_to_world;
			ctx.instance_info[i].world_to_obj = affineInverse(obj_to_world);

			rtcSetGeometryTransform(ctx.instances[i].get(), 0, RTC_FORMAT_FLOAT3X4_COLUMN_MAJOR, &obj_to_world);
			rtcCommitGeometry(ctx.instances[i].get());
		}

		rtcCommitScene(ctx.tlas.get());
	}

	std::vector<Contribution> EmbreeBackend::trace(ThreadContext* thread_context, TraceJob& job) const
	{
		auto* ctx = dynamic_cast<EmbreeThreadContext*>(thread_context);
		if (ctx == nullptr)
			throw std::runtime_error("EmbreeBackend::trace did not receive a valid EmbreeThreadContext.");

		const auto rays_per_source = params::rayTracingParams().directionsPerSource();
		const auto time_count = static_cast<uint32_t>(job.times.size());
		const auto source_antenna_count = static_cast<uint32_t>(job.source_antennas.size() / job.times.size());
		const auto dest_antenna_count = static_cast<uint32_t>(job.dest_antennas.size() / job.times.size());

		const SbrParams sbr{
			.scatter_limit = maxScatterLimit(),
			.directions_per_source = rays_per_source,
			.boresight_rays = params::rayTracingParams().boresightDirections(),
			.boresight_fraction = float(params::rayTracingParams().boresightSolidAngle() / (4.0 * PI)),
			.vertices = _vertices.data(),
			.indices = _indices.data(),
			.materials = _materials.data(),
			.times = job.times.data(),
			.carrier_models = job.carriers.data(),
			.antenna_models = _antenna_models.data(),
			.antenna_gains = _antenna_gains.data(),
			.source_antennas = job.source_antennas.data(),
			.source_antenna_count = source_antenna_count,
			.dest_antennas = job.dest_antennas.data(),
			.dest_antenna_count = dest_antenna_count,
			.rx_flags = job.rx_flags.data(),
			.rx_to_tx = job.type == TraceJobType::FromRx,
		};

		// Multiple chunks per worker thread, so that varying workloads per chunk are load-balanced
		// across threads.
		constexpr uint32_t paths_per_chunk = 1024;

		std::vector<Contribution> contribs;

		for (uint32_t time_idx = 0; time_idx < time_count; ++time_idx)
		{
			updateInstanceTransforms(*ctx, job.times[time_idx]);
			RTCScene scene = ctx->tlas.get();

			// Direct paths: dest_antenna_count is typically small.
			// This is handled inline rather than splitting it across the pool.
			{
				const auto capacity = source_antenna_count * dest_antenna_count;
				auto contrib_index = static_cast<uint32_t>(contribs.size());
				contribs.resize(contribs.size() + capacity);

				const auto get_contrib_ptr = [&contrib_index, &contribs] { return contribs.data() + contrib_index++; };

				const auto shadow_test = [scene](const Float3 origin, const Float3 dir)
				{ return shadowTest(scene, origin, dir); };

				for (uint32_t s = 0; s < source_antenna_count; ++s)
					for (uint32_t d = 0; d < dest_antenna_count; ++d)
						directPath(sbr, s, d, time_idx, shadow_test, get_contrib_ptr);

				contribs.resize(contrib_index);
			}

			// Indirect paths: split across the pool.
			const uint32_t total_paths = source_antenna_count * rays_per_source;
			std::vector<std::future<std::vector<Contribution>>> futures;
			for (uint32_t start_ray_index = 0; start_ray_index < total_paths; start_ray_index += paths_per_chunk)
			{
				const uint32_t end_ray_index = std::min(start_ray_index + paths_per_chunk, total_paths);

				futures.push_back(_pool.enqueue(
					[scene, sbr, &instances = ctx->instance_info, time_idx, start_ray_index, end_ray_index,
					 rays_per_source, dest_antenna_count]() -> std::vector<Contribution>
					{
						// This is the maximum number of contributions the SBR engine will possibly write.
						const uint32_t capacity =
							(end_ray_index - start_ray_index) * dest_antenna_count * maxScatterLimit();
						std::vector<Contribution> local(capacity);
						uint32_t contrib_index = 0;

						const auto get_contrib_ptr = [&contrib_index, &local]
						{ return local.data() + contrib_index++; };

						for (uint32_t flat = start_ray_index; flat < end_ray_index; ++flat)
						{
							const uint32_t source_idx = flat / rays_per_source;
							const uint32_t path_idx = flat % rays_per_source;

							traceIndirectPath(scene, sbr, instances, path_idx, source_idx, time_idx, get_contrib_ptr);
						}

						// This would mean the SBR code changed and the documented assumption above broke.
						assert(contrib_index <= capacity);

						local.resize(contrib_index);
						return local;
					}));
			}

			for (auto& f : futures)
			{
				auto local = f.get();
				contribs.insert(contribs.end(), std::make_move_iterator(local.begin()),
								std::make_move_iterator(local.end()));
			}
		}

		return contribs;
	}
}
