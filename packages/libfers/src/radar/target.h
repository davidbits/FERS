// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2006-2008 Marc Brooker and Michael Inggs
// Copyright (c) 2008-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

/**
 * @file target.h
 * @brief Defines classes for radar targets and their Radar Cross-Section (RCS) models.
 */

#pragma once

#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>

#include "core/config.h"
#include "core/sim_id.h"
#include "interpolation/interpolation_set.h"
#include "noise/noise_generators.h"
#include "object.h"

namespace math
{
	class SVec3;
}

namespace core
{
	struct MeshAsset;
	struct MaterialAsset;
}

namespace radar
{
	class Platform;
	class Target;

	/**
	 * @struct TargetGeometry
	 * @brief Non-owning references to the mesh/material a target uses for ray-tracing propagation.
	 */
	struct TargetGeometry
	{
		const core::MeshAsset* mesh; ///< The target's mesh asset, owned by the World.
		const core::MaterialAsset* material; ///< The target's material asset, owned by the World.
	};

	/**
	 * @class RcsModel
	 * @brief Base class for RCS fluctuation models.
	 */
	class RcsModel
	{
	public:
		virtual ~RcsModel() = default;

		RcsModel() = default;

		RcsModel(const RcsModel&) = delete;

		RcsModel& operator=(const RcsModel&) = delete;

		RcsModel(RcsModel&&) = delete;

		RcsModel& operator=(RcsModel&&) = delete;

		/**
		 * @brief Samples the RCS model to produce a value.
		 *
		 * @return The sampled RCS value.
		 */
		virtual RealType sampleModel() = 0;
	};

	/**
	 * @class RcsConst
	 * @brief Constant RCS model.
	 */
	class RcsConst final : public RcsModel
	{
	public:
		/**
		 * @brief Samples the constant RCS model.
		 *
		 * @return The RCS value (always 1.0).
		 */
		RealType sampleModel() override { return 1.0; }
	};

	/**
	 * @class RcsChiSquare
	 * @brief Chi-square distributed RCS model.
	 */
	class RcsChiSquare final : public RcsModel
	{
	public:
		/**
		 * @brief Constructs an RcsChiSquare model.
		 *
		 * @param rngEngine The random number engine to use.
		 * @param k The degrees of freedom for the chi-square distribution.
		 */
		explicit RcsChiSquare(std::mt19937& rngEngine, RealType k) :
			_gen(std::make_unique<noise::GammaGenerator>(rngEngine, k)), _k(k)
		{
		}

		/**
		 * @brief Gets the 'k' parameter (degrees of freedom) of the distribution.
		 * @return The k value.
		 */
		[[nodiscard]] RealType getK() const noexcept { return _k; }

		/**
		 * @brief Samples the chi-square RCS model.
		 *
		 * @return The sampled RCS value.
		 */
		RealType sampleModel() override { return _gen->getSample(); }

	private:
		std::unique_ptr<noise::GammaGenerator> _gen; ///< The gamma generator for sampling the chi-square distribution.
		RealType _k; ///< The 'k' parameter (degrees of freedom).
	};

	/**
	 * @class TargetRcs
	 * @brief A target's analytic or file-based RCS value, plus its optional fluctuation model and
	 * the RNG that drives it.
	 *
	 * This is the optional RCS "spec" a Target may or may not have (parallel to TargetGeometry,
	 * though held behind a pointer since it's polymorphic — see Target::getRcsSpec()). RNG/seed
	 * live here, not on Target, because they only ever exist to drive fluctuation sampling.
	 */
	class TargetRcs
	{
	public:
		virtual ~TargetRcs() = default;

		TargetRcs(const TargetRcs&) = delete;

		TargetRcs& operator=(const TargetRcs&) = delete;

		TargetRcs(TargetRcs&&) = delete;

		TargetRcs& operator=(TargetRcs&&) = delete;

		/**
		 * @brief Gets the RCS value for the target, with any fluctuation model applied.
		 *
		 * Non-virtual so fluctuation is always applied consistently, exactly once, regardless of
		 * which concrete RCS kind this is; subclasses implement `computeRcs()` instead.
		 *
		 * @param inAngle The incoming angle of the radar wave.
		 * @param outAngle The outgoing angle of the reflected radar wave.
		 * @param time The current simulation time.
		 * @param owner The target this RCS belongs to. Passed through (rather than a precomputed
		 * rotation) so constant RCS never has to query the owner's rotation path at all — only
		 * file-based RCS, which actually needs it, calls `owner.getRotation(time)`.
		 * @return The RCS value.
		 */
		RealType getRcs(math::SVec3& inAngle, math::SVec3& outAngle, RealType time, const Target& owner) const
		{
			const RealType raw = computeRcs(inAngle, outAngle, time, owner);
			return _fluctuation ? raw * _fluctuation->sampleModel() : raw;
		}

		/**
		 * @brief Sets the RCS fluctuation model.
		 * @param in Unique pointer to the new RCS fluctuation model.
		 */
		void setFluctuationModel(std::unique_ptr<RcsModel> in) { _fluctuation = std::move(in); }

		/**
		 * @brief Gets the RCS fluctuation model.
		 * @return A const pointer to the RcsModel.
		 */
		[[nodiscard]] const RcsModel* getFluctuationModel() const noexcept { return _fluctuation.get(); }

		/**
		 * @brief Gets the RNG engine used to sample this RCS's fluctuation model.
		 * @return A mutable reference to the RNG engine.
		 */
		[[nodiscard]] std::mt19937& getRngEngine() noexcept { return _rng; }

		/**
		 * @brief Gets the initial seed used for the RNG.
		 * @return The initial seed value.
		 */
		[[nodiscard]] unsigned getSeed() const noexcept { return _seed; }

	protected:
		/**
		 * @brief Constructs a TargetRcs, seeding its RNG.
		 * @param seed The seed for the fluctuation-sampling RNG.
		 */
		explicit TargetRcs(const unsigned seed) : _rng(seed), _seed(seed) {}

	private:
		/**
		 * @brief Computes the raw RCS value (before fluctuation), for a specific concrete RCS kind.
		 */
		virtual RealType computeRcs(math::SVec3& inAngle, math::SVec3& outAngle, RealType time,
									const Target& owner) const = 0;

		std::unique_ptr<RcsModel> _fluctuation{nullptr}; ///< The RCS fluctuation model, if any.
		std::mt19937 _rng; ///< RNG used for fluctuation sampling, for statistical independence.
		unsigned _seed; ///< The initial seed for the RNG.
	};

	/**
	 * @class IsoTargetRcs
	 * @brief Constant (isotropic) RCS.
	 */
	class IsoTargetRcs final : public TargetRcs
	{
	public:
		/**
		 * @brief Constructs a constant RCS.
		 *
		 * @param rcs The constant RCS value.
		 * @param seed The seed for the fluctuation-sampling RNG.
		 */
		IsoTargetRcs(const RealType rcs, const unsigned seed) : TargetRcs(seed), _rcs(rcs) {}

		/**
		 * @brief Gets the constant RCS value (without fluctuation model applied).
		 * @return The constant RCS value.
		 */
		[[nodiscard]] RealType getConstRcs() const noexcept { return _rcs; }

	private:
		RealType computeRcs(math::SVec3& /*inAngle*/, math::SVec3& /*outAngle*/, RealType /*time*/,
							const Target& /*owner*/) const noexcept override
		{
			return _rcs;
		}

		RealType _rcs; ///< The constant RCS value.
	};

	/**
	 * @class FileTargetRcs
	 * @brief File-based, aspect-dependent RCS.
	 */
	class FileTargetRcs final : public TargetRcs
	{
	public:
		/**
		 * @brief Constructs a file-based RCS, loading its angular data immediately.
		 *
		 * @param filename The name of the file containing RCS data.
		 * @param seed The seed for the fluctuation-sampling RNG.
		 * @throws std::runtime_error If the file cannot be loaded or parsed.
		 */
		FileTargetRcs(const std::string& filename, unsigned seed);

		/**
		 * @brief Gets the filename associated with this RCS data.
		 * @return The source filename.
		 */
		[[nodiscard]] const std::string& getFilename() const noexcept { return _filename; }

	private:
		/**
		 * @brief Computes the raw, aspect-dependent RCS value (before fluctuation) for a specific
		 * bistatic geometry and time.
		 * @param inAngle The incoming angle of the radar wave in the global frame.
		 * @param outAngle The outgoing angle of the reflected radar wave in the global frame.
		 * @param time The current simulation time, passed through to `owner.getRotation(time)`.
		 * @param owner The target this RCS belongs to; its orientation at `time` is looked up here,
		 * lazily (unlike `IsoTargetRcs`, this is the one RCS kind that actually needs it).
		 * @return The Radar Cross Section (RCS) value in meters squared (m²).
		 * @throws std::runtime_error If RCS data cannot be retrieved.
		 *
		 * This function calculates the target's aspect-dependent RCS. The key steps are:
		 * 1.  Calculate the bistatic angle bisector in the global simulation coordinate system.
		 * 2.  Transform the global bistatic angle into the target's local, body-fixed frame by
		 *     subtracting the owner's rotation. This is critical, as RCS patterns are defined
		 *     relative to the target itself.
		 * 3.  Use this local aspect angle to look up the azimuthal and elevation RCS values from
		 *     the loaded data.
		 *
		 * NOTE: This function returns the raw RCS value (σ), which is linearly proportional to
		 * scattered power. The calling physics engine is responsible for converting this to a
		 * signal amplitude by taking the square root.
		 */
		RealType computeRcs(math::SVec3& inAngle, math::SVec3& outAngle, RealType time,
							const Target& owner) const override;

		std::unique_ptr<interp::InterpSet> _azi_samples; ///< The azimuthal RCS samples.
		std::unique_ptr<interp::InterpSet> _elev_samples; ///< The elevation RCS samples.
		std::string _filename; ///< The original filename for the RCS data.
	};

	/**
	 * @class Target
	 * @brief A radar target: a platform-attached object with optional RCS and/or geometry.
	 *
	 * A target with no RCS is simply invisible to the point-scatter propagation model; a target
	 * with no geometry is simply invisible to the ray-tracing propagation model. A target may have
	 * either, both, or (unusually) neither.
	 */
	class Target final : public Object
	{
	public:
		/**
		 * @brief Constructs a radar target.
		 *
		 * @param platform Pointer to the platform associated with the target.
		 * @param name The name of the target.
		 * @param id Optional explicit SimId.
		 */
		explicit Target(Platform* platform, std::string name, const SimId id = 0) :
			Object(platform, std::move(name), ObjectType::Target, id)
		{
		}

		/**
		 * @brief Sets the target's RCS.
		 * @param rcs The RCS to attach, or nullptr to remove any existing RCS.
		 */
		void setRcs(std::unique_ptr<TargetRcs> rcs) noexcept { _rcs = std::move(rcs); }

		/**
		 * @brief Gets the target's RCS spec, if any.
		 * @return A pointer to the TargetRcs, or nullptr if this target has no RCS.
		 */
		[[nodiscard]] TargetRcs* getRcsSpec() const noexcept { return _rcs.get(); }

		/**
		 * @brief Computes the RCS value for the target at a given bistatic geometry and time.
		 *
		 * @param inAngle The incoming angle of the radar wave.
		 * @param outAngle The outgoing angle of the reflected radar wave.
		 * @param time The current simulation time.
		 * @return The RCS value, or std::nullopt if this target has no RCS (ignored by point-scatter
		 * propagation).
		 */
		[[nodiscard]] std::optional<RealType> getRcs(math::SVec3& inAngle, math::SVec3& outAngle,
													 RealType time) const
		{
			if (!_rcs)
			{
				return std::nullopt;
			}
			return _rcs->getRcs(inAngle, outAngle, time, *this);
		}

		/**
		 * @brief Sets the target's mesh/material geometry, used by ray-tracing propagation.
		 *
		 * @param mesh Non-owning pointer to the mesh asset, owned by the World.
		 * @param material Non-owning pointer to the material asset, owned by the World.
		 */
		void setGeometry(const core::MeshAsset* mesh, const core::MaterialAsset* material) noexcept
		{
			_geometry = TargetGeometry{mesh, material};
		}

		/**
		 * @brief Gets the target's mesh/material geometry, if any.
		 *
		 * Targets without geometry are ignored by ray-tracing propagation.
		 * @return The TargetGeometry, or std::nullopt if this target has no mesh/material.
		 */
		[[nodiscard]] const std::optional<TargetGeometry>& getGeometry() const noexcept { return _geometry; }

	private:
		std::unique_ptr<TargetRcs> _rcs; ///< Optional RCS; nullptr if this target has none.
		std::optional<TargetGeometry> _geometry; ///< Optional mesh/material for ray-tracing propagation.
	};

	/**
	 * @brief Creates a target with a constant (isotropic) RCS.
	 *
	 * @param platform Pointer to the platform associated with the target.
	 * @param name The name of the target.
	 * @param rcs The constant RCS value for the target.
	 * @param seed The seed for the RCS fluctuation-sampling RNG.
	 * @return A unique pointer to the newly created Target.
	 */
	inline std::unique_ptr<Target> createIsoTarget(Platform* platform, std::string name, RealType rcs, unsigned seed,
													const SimId id = 0)
	{
		auto target = std::make_unique<Target>(platform, std::move(name), id);
		target->setRcs(std::make_unique<IsoTargetRcs>(rcs, seed));
		return target;
	}

	/**
	 * @brief Creates a target with a file-based RCS.
	 *
	 * @param platform Pointer to the platform associated with the target.
	 * @param name The name of the target.
	 * @param filename The name of the file containing RCS data.
	 * @param seed The seed for the RCS fluctuation-sampling RNG.
	 * @return A unique pointer to the newly created Target.
	 */
	inline std::unique_ptr<Target> createFileTarget(Platform* platform, std::string name, const std::string& filename,
													unsigned seed, const SimId id = 0)
	{
		auto target = std::make_unique<Target>(platform, std::move(name), id);
		target->setRcs(std::make_unique<FileTargetRcs>(filename, seed));
		return target;
	}
}
