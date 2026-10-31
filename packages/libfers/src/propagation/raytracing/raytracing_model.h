// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "core/config.h"
#include "core/simulation_state.h"
#include "core/world.h"
#include "propagation/math.h"
#include "propagation/propagation_model.h"
#include "propagation/raytracing/sbr_shared.h"
#include "radar/receiver.h"

namespace pool
{
	class ThreadPool;
}

namespace radar
{
	class Target;
}

namespace propagation::raytracing
{
	/**
	 * @brief Builds a target's object-to-world transform (world = R * local + T) from its position
	 * (translation) and az/el rotation, shared by every ray-tracing backend's instancing setup.
	 *
	 * `SVec3` only carries azimuth+elevation (a pointing direction, like antenna boresight) - there's
	 * no roll/bank component, so this assumes roll = 0. That's the best this data can give until the
	 * planned transform-hierarchy rework lands.
	 *
	 * Mesh local-axis convention: +Y is "forward" (nose direction), +Z is "up", +X is "right" -
	 * there's no other convention already established in this codebase to match against, so meshes
	 * must be authored this way or they'll appear rotated.
	 */
	[[nodiscard]] Float3x4 computeTargetTransform(const radar::Target& target, RealType t);

	/**
	 * @brief Inverse of an orthonormal affine transform (rotation + translation, no scale/shear),
	 * i.e. a transform produced by `computeTargetTransform`. `R_inv = R^T`, `T_inv = -R^T * T`.
	 */
	[[nodiscard]] Float3x4 affineInverse(const Float3x4& m) noexcept;

	/// Caps the user-requested scatter limit to `MAX_SCATTER_LIMIT`, the hard-coded bound every
	/// backend's path-state storage is sized for. `RayTracingModel`'s constructor warns once if this
	/// actually caps anything; backends should just call this rather than re-deriving/re-warning.
	[[nodiscard]] uint32_t maxScatterLimit() noexcept;

	enum class TraceJobType : uint8_t
	{
		FromTx = 0,
		FromRx = 1,
	};

	struct SceneData
	{
		std::unordered_map<SimId, uint32_t> mesh_ids;
		std::vector<Float3> vertices;
		std::vector<Uint3> indices;
		std::vector<TriangleMeshView> meshes;

		std::unordered_map<SimId, uint32_t> mat_ids;
		std::vector<Material> materials;

		std::vector<float> antenna_gains;
		std::vector<AntennaModel> antenna_models;
	};

	struct TraceJob
	{
		core::World* world;
		std::span<const RealType> times;

		std::vector<CarrierModel<float>> carriers;

		/// Buffer of ActiveAntenna indexed by `source_index + t_index * source_antenna_count`.
		std::vector<ActiveAntenna> source_antennas;
		/// Buffer of ActiveAntenna indexed by `dest_index + t_index * dest_antenna_count`.
		std::vector<ActiveAntenna> dest_antennas;

		std::vector<RxFlags> rx_flags;

		TraceJobType type;
	};

	/**
	 * @brief A `(source_times_index, dest_index, path_id)` key. Contributions sharing a key are
	 * ray-tube samples along the same TX->facet-sequence->RX path, at the same timestep.
	 */
	struct ContributionGroupKey
	{
		/// `(source_times_index << 32) | dest_index`.
		uint64_t coarse;
		uint64_t path_id;

		bool operator==(const ContributionGroupKey&) const = default;
	};

	struct ContributionGroupKeyHash
	{
		size_t operator()(const ContributionGroupKey& k) const
		{
			return std::hash<uint64_t>{}(k.coarse) ^ (std::hash<uint64_t>{}(k.path_id) << 1);
		}
	};

	using ContributionGroups =
		std::unordered_map<ContributionGroupKey, std::vector<const Contribution*>, ContributionGroupKeyHash>;

	/**
	 * @brief Buckets `contributions` by `(source_times_index, dest_index, path_id)`: ray-tube samples
	 * along the same TX->facet-sequence->RX path, at the same timestep.
	 */
	[[nodiscard]] ContributionGroups groupContributions(const std::vector<Contribution>& contributions);

	/**
	 * @brief The reference delay for a group: the minimum delay across its contributions, i.e. the
	 * most direct of the ray-tube samples making up the path.
	 *
	 * @param group Non-empty.
	 */
	[[nodiscard]] RealType contributionGroupDelay(const std::vector<const Contribution*>& group);

	/**
	 * @brief Coherently sums a group of `Contribution`s.
	 *
	 * @param group Non-empty. All contributions must share the same `source_times_index`/`dest_index`/`path_id`.
	 * @param group_delay The reference delay (typically `contributionGroupDelay(group)`) used to align phases.
	 * @param carrier_frequency The carrier frequency (Hz) shared by every contribution in `group`.
	 */
	[[nodiscard]] ComplexType aggregateContributionGroup(const std::vector<const Contribution*>& group,
														 RealType group_delay, RealType carrier_frequency);

	/**
	 * @brief Drops paths more than `below_max_db` weaker than the strongest path in `paths`.
	 * A no-op (returns `paths` unchanged) if `below_max_db` is `std::nullopt`.
	 *
	 * @param paths Typically all paths for one (transmit time, destination) pair.
	 */
	void pruneWeakPaths(std::vector<PropagationPath>& paths, std::optional<RealType> below_max_db);


	class RayTracingBackend
	{
	public:
		virtual ~RayTracingBackend() = default;

		/// Create a worker thread's mutable state object.
		[[nodiscard]] virtual std::unique_ptr<ThreadContext> makeThreadContext(core::World* world) const = 0;

		/// Execute the ray tracing batch job.
		[[nodiscard]] virtual std::vector<Contribution> trace(ThreadContext*, TraceJob&) const = 0;
	};


	class RayTracingModel : public PropagationModel
	{
	public:
		/**
		 * @brief Construct a RayTracingModel. This loads all required assets and initialises the RT engine.
		 * This isn't sufficient to trace against scene. A thread context is required first `makeThreadContext()`.
		 *
		 * @param world The world being simulated.
		 * @param engine The RT engine back-end to initialise.
		 * @param pool The simulations shared CPU worker pool, used by CPU-only backends (currently Embree).
		 */
		RayTracingModel(core::World* world, params::RayTracingEnginePreference engine, pool::ThreadPool& pool);

		[[nodiscard]] std::unique_ptr<ThreadContext> makeThreadContext() const override;

		[[nodiscard]] std::vector<PathsAtTime> traceFromTransmitter(ThreadContext* ctx,
																	const radar::Transmitter& transmitter,
																	const std::vector<RealType>& times) const override;

		[[nodiscard]] std::unordered_map<radar::Receiver*, std::vector<PropagationPath>>
		traceFromReceivers(ThreadContext* ctx, const std::vector<radar::Receiver*>& receivers,
						   const std::vector<core::ActiveStreamingSource>& sources, RealType rx_time) const override;

		[[nodiscard]] unsigned int maxPropagationLegs() const noexcept override;

	private:
		core::World* _world;

		std::unordered_map<SimId, uint32_t> _antenna_id_to_index;

		std::unique_ptr<RayTracingBackend> _engine;
	};
}
