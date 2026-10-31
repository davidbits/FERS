// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <cccl/cuda/__atomic/atomic.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <optix.h>
#include <optix_host.h>
#include <optix_types.h>
#include <unordered_map>
#include <vector>

#include "core/config.h"
#include "core/sim_id.h"
#include "propagation/raytracing/optix/kernel_defs.h"
#include "propagation/raytracing/optix/utils.h"
#include "propagation/raytracing/raytracing_model.h"
#include "propagation/raytracing/sbr_shared.h"
#include "radar/target.h"

namespace propagation::raytracing::optix
{
	/// SBT record for a raygen program. __raygen__
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	};

	/// SBT record for a ray hit-group program. __anyhit__ or __closesthit__
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	};

	/// SBT record for a ray miss program. __miss__
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	};

	/**
	 * @brief RAII handle for a CUDA context.
	 */
	class CudaContext
	{
	public:
		explicit CudaContext(int device);
		~CudaContext()
		{
			if (_context != nullptr)
				cuCtxDestroy(_context);
		}
		CudaContext(const CudaContext&) = delete;
		CudaContext& operator=(const CudaContext&) = delete;
		CudaContext(CudaContext&&) = delete;
		CudaContext& operator=(CudaContext&&) = delete;

		[[nodiscard]] CUcontext get() const { return _context; }

	private:
		CUcontext _context{};
	};

	/**
	 * @brief RAII handle for an OptiX device context.
	 */
	class OptixContext
	{
	public:
		explicit OptixContext(CUcontext context);
		~OptixContext()
		{
			if (_context != nullptr)
				optixDeviceContextDestroy(_context);
		}
		OptixContext(const OptixContext&) = delete;
		OptixContext& operator=(const OptixContext&) = delete;
		OptixContext(OptixContext&&) = delete;
		OptixContext& operator=(OptixContext&&) = delete;

		[[nodiscard]] OptixDeviceContext get() const { return _context; }

	private:
		OptixDeviceContext _context{nullptr};
	};

	/**
	 * @brief RAII handle for a CUDA stream.
	 */
	class CudaStream
	{
	public:
		CudaStream() { cuStreamCreate(&_stream, CU_STREAM_DEFAULT); }
		~CudaStream()
		{
			if (_stream != nullptr)
				cudaStreamDestroy(_stream);
		}
		CudaStream(const CudaStream&) = delete;
		CudaStream& operator=(const CudaStream&) = delete;
		CudaStream(CudaStream&&) = delete;
		CudaStream& operator=(CudaStream&&) = delete;

		[[nodiscard]] CUstream get() const { return _stream; }

	private:
		CUstream _stream{};
	};


	struct OptixThreadContext : ThreadContext
	{
		CudaStream stream;
		OptixPipeline pipeline{};

		/// OptiX launch parameters.
		DeviceBuffer<ShaderParams> params;
		DeviceBuffer<ContributionCountType> contribution_count;

		std::vector<OptixInstance> instances;
		std::vector<const radar::Target*> instance_targets;

		DeviceBuffer<OptixTraversableHandle> iass;

		/// Instance Acceleration Structure (IAS) build/refit state, scoped per-thread so that
		/// concurrent worker threads each build/refit their own IAS without synchronization.
		/// See OptixEngine::buildOrRefitIAS.
		DeviceBuffer<OptixInstance> ias_instances;
		DeviceBuffer<uint8_t> ias_build_temp;
		DeviceBuffer<uint8_t> ias_output;
		OptixTraversableHandle ias_handle{};
		OptixAccelBufferSizes ias_sizes{}; // cached from the first (full) build
		std::vector<uint32_t> ias_topology; // last-used sequence of mesh indices, for change detection


		OptixThreadContext() :
			stream(), params(stream.get()), contribution_count(stream.get()), iass(stream.get()),
			ias_instances(stream.get()), ias_build_temp(stream.get()), ias_output(stream.get())
		{
		}
	};

	class OptixBackend : public RayTracingBackend
	{
	public:
		OptixBackend(SceneData scene);

		OptixBackend(const OptixBackend&) = delete;
		OptixBackend& operator=(const OptixBackend&) = delete;
		OptixBackend(OptixBackend&&) = delete;
		OptixBackend& operator=(OptixBackend&&) = delete;

		~OptixBackend() override = default;

		std::unique_ptr<ThreadContext> makeThreadContext(core::World* world) const override;

		std::vector<Contribution> trace(ThreadContext* thread_context, TraceJob& job) const override;

	private:
		void createDevicePrograms();
		void processAssets(SceneData scene);
		void createSbt();
		OptixTraversableHandle buildOrRefitIAS(OptixThreadContext& ctx, RealType t) const;

		/// Picks how many rays-per-source to launch per chunk, so that the exact worst-case
		/// contribution buffer for one chunk's launch fits within currently-free GPU memory.
		static uint32_t chunkRaysPerSource(uint32_t rays_per_source, uint32_t source_antenna_count,
										   uint32_t dest_antenna_count, uint32_t time_count);

		/// The engine's CUDA context handle.
		CudaContext _cuda_ctx;
		/// The engine's OptiX device context handle.
		OptixContext _optix_ctx;
		/// The engine's CUDA stream. This is primarily used for setup and teardown.
		/// Simulation traces occur on per-thread streams.
		CudaStream _stream;

		// The device programs and associated data.

		OptixModuleCompileOptions _module_comp_options;
		OptixPipelineCompileOptions _pipeline_comp_options;
		OptixPipelineLinkOptions _pipeline_link_options;
		OptixModule _module{};
		OptixProgramGroup _raygen_program{};
		std::vector<OptixProgramGroup> _hitgroup_programs;
		std::vector<OptixProgramGroup> _miss_programs;
		DeviceBuffer<RaygenRecord> _raygen_record;
		DeviceBuffer<MissRecord> _miss_records;
		DeviceBuffer<HitgroupRecord> _hitgroup_records;
		OptixShaderBindingTable _sbt;

		// Host-side data tracking device-side geometry data.

		std::unordered_map<SimId, uint32_t> _material_indices;
		DeviceBuffer<Material> _materials;

		std::unordered_map<SimId, uint32_t> _mesh_indices;
		std::vector<TriangleMeshView> _meshes;
		DeviceBuffer<Float3> _vertices;
		DeviceBuffer<Uint3> _indeces;
		/// Per-mesh geometry acceleration structures (GASs).
		std::vector<OptixTraversableHandle> _gas_handles;
		/// Device-side mesh GAS data.
		std::vector<DeviceBuffer<uint8_t>> _gas_data;

		DeviceBuffer<AntennaModel> _antenna_models;
		DeviceBuffer<float> _antenna_gains;
	};
}
