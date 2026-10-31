// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2006-2008 Marc Brooker and Michael Inggs
// Copyright (c) 2008-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

/**
 * @file parameters.h
 * @brief Defines the Parameters struct and provides methods for managing simulation parameters.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>

#include "config.h"
#include "core/portable_utils.h"
#include "logging.h"

namespace params
{
	/**
	 * @enum CoordinateFrame
	 * @brief Defines the coordinate systems supported for KML/geospatial export.
	 */
	enum class CoordinateFrame : std::uint8_t
	{
		ENU, ///< East-North-Up local tangent plane (default)
		UTM, ///< Universal Transverse Mercator
		ECEF ///< Earth-Centered, Earth-Fixed
	};

	/**
	 * @enum RotationAngleUnit
	 * @brief Defines the units used at external rotation-path boundaries.
	 */
	enum class RotationAngleUnit : std::uint8_t
	{
		Degrees, ///< Compass azimuth and elevation expressed in degrees
		Radians ///< Compass azimuth and elevation expressed in radians
	};

	/**
	 * @enum PropagationModelKind
	 * @brief Selects which propagation model the simulation uses.
	 */
	enum class PropagationModelKind : std::uint8_t
	{
		RcsPointScatter = 1, ///< Analytic point-target RCS/radar-equation model (default).
		SbrRayTracing = 2, ///< GO+PO ray-tracing model against target meshes.
	};

	/**
	 * @enum RayTracingEnginePreference
	 * @brief Runtime override for which ray-tracing engine the SBR propagation model should use.
	 */
	enum class RayTracingEnginePreference : std::uint8_t
	{
		Auto = 0, ///< Auto-select: prefer OptiX, fall back to Embree (default).
		OptiX = 1, ///< Require the OptiX engine.
		Embree = 2, ///< Require the Embree engine.
	};

	inline const char* rayTracingEngineName(RayTracingEnginePreference engine)
	{
		switch (engine)
		{
		case RayTracingEnginePreference::Auto:
			return "Auto";
		case RayTracingEnginePreference::OptiX:
			return "OptiX";
		case RayTracingEnginePreference::Embree:
			return "Embree";
		}
		return "Unknown";
	}

	struct PointScatterParameters
	{
		/// Whether to enable polarimetric transport, including antenna polarisations and
		/// Target scattering matrices.
		bool polarimetric = false;
	};

	/// Simulation parameters for running the ray tracing propagation model.
	///
	/// Ray directions are sampled with two uniform densities: a finer one within a cone around
	/// each source antenna's boresight, and a coarser one over the rest of the sphere. Pick each
	/// density (`*_tube_solid_angle`) from the smallest facet area `A_facet` that needs resolving
	/// in that regime, at the shortest expected range `R` and most oblique incidence angle `theta`
	/// targets in that regime will be seen at: `S <= A_facet * cos(theta) / R^2`, where `S` is the
	/// ray tube's solid angle. A ray tube whose footprint (`R^2 * S / cos(theta)`) exceeds a hit
	/// facet's area is not fully resolving that facet.
	struct RayTracingParameters
	{
		/// Maximum number of bounces/scattering events per path. Max ray depth is scatter_limit+1.
		unsigned scatter_limit = 4;

		/// Full angular width (radians) of the boresight sampling cone, centered on each source
		/// antenna's boresight direction. The cone's half-angle is `boresight_width / 2`.
		RealType boresight_width = 0.3;
		/// Desired ray-tube solid angle (steradians) for rays sampled within the boresight cone.
		RealType boresight_tube_solid_angle = 2e-8;
		/// Desired ray-tube solid angle (steradians) for rays sampled outside the boresight cone.
		RealType off_boresight_tube_solid_angle = 1e-5;

		/// Optional dynamic-range cull, in dB of power. Within each
		/// (transmit time, destination) group, paths more than this many dB below that group's
		/// strongest path are dropped before being handed back to the rest of FERS. Unset (the
		/// default) disables pruning entirely, preserving prior FERS behaviour exactly.
		std::optional<RealType> prune_below_max_db;

		/// Solid angle (steradians) of the boresight sampling cone, derived from `boresight_width`.
		[[nodiscard]] RealType boresightSolidAngle() const noexcept
		{
			return 2.0 * PI * (1.0 - std::cos(boresight_width / 2.0));
		}

		/// Number of ray launch directions dedicated to the boresight cone,
		/// derived from `boresightSolidAngle()` and `boresight_tube_solid_angle`.
		[[nodiscard]] unsigned boresightDirections() const noexcept
		{
			return static_cast<unsigned>(std::llround(boresightSolidAngle() / boresight_tube_solid_angle));
		}

		/// Number of ray launch directions dedicated to outside the boresight cone,
		/// derived from the remaining solid angle and `off_boresight_tube_solid_angle`.
		[[nodiscard]] unsigned offBoresightDirections() const noexcept
		{
			const RealType off_solid_angle = 4.0 * PI - boresightSolidAngle();
			return static_cast<unsigned>(std::llround(off_solid_angle / off_boresight_tube_solid_angle));
		}

		/// Total number of rays to launch per source antenna.
		[[nodiscard]] unsigned directionsPerSource() const noexcept
		{
			return boresightDirections() + offBoresightDirections();
		}
	};

	/**
	 * @class Parameters
	 * @brief Struct to hold simulation parameters.
	 */
	struct Parameters
	{
		constexpr static RealType DEFAULT_C = 299792458.0; ///< Speed of light (m/s)
		constexpr static RealType DEFAULT_BOLTZMANN_K = 1.3806503e-23; ///< Boltzmann constant
		RealType c = DEFAULT_C; ///< Speed of light (modifiable)
		RealType boltzmann_k = DEFAULT_BOLTZMANN_K; ///< Boltzmann constant
		RealType start = 0; ///< Start time for the simulation.
		RealType end = 0; ///< End time for the simulation.
		RealType sim_sampling_rate = 1000; ///< Time-step sampling rate for radar pulse simulation.
		std::optional<RealType> cw_coarse_sample_rate; ///< CW/streaming coarse propagation sampling
													   ///< rate; unset means every output sample is
													   ///< propagated exactly.

		// KML/geospatial export defaults to the location of the University of Cape Town in South Africa.
		double origin_latitude = -33.957652; ///< KML ENU geodetic origin latitude
		double origin_longitude = 18.4611991; ///< KML ENU geodetic origin longitude
		double origin_altitude = 111.01; ///< KML ENU geodetic origin altitude (in meters)
		CoordinateFrame coordinate_frame = CoordinateFrame::ENU; ///< KML/geospatial export coordinate frame
		RotationAngleUnit rotation_angle_unit = RotationAngleUnit::Degrees; ///< External rotation angle unit
		int utm_zone = 0; ///< KML UTM zone (1-60), if applicable
		bool utm_north_hemisphere = true; ///< KML UTM hemisphere, if applicable
		RealType rate = 0; ///< Rendering sample rate.
		std::optional<unsigned> random_seed; ///< Random seed for simulation.
		unsigned adc_bits = 0; ///< ADC quantization bits.
		unsigned filter_length = 33; ///< Default render filter length.
		std::optional<unsigned> worker_threads = std::nullopt; ///< Number of worker threads to use for parallel tasks.
		std::string simulation_name; ///< The name of the simulation, from the XML.
		unsigned oversample_ratio = 1; ///< Oversampling ratio.

		PropagationModelKind propagation_model = PropagationModelKind::RcsPointScatter; ///< Active propagation model.
		PointScatterParameters point_scatterer_params{}; ///< Point-scatterer (RCS) model parameters.
		RayTracingParameters ray_tracing_params{}; ///< Ray-tracing (SBR) model parameters.
		RayTracingEnginePreference ray_tracing_engine =
			RayTracingEnginePreference::Auto; ///< Ray-tracing engine runtime override.

		/**
		 * @brief Resets the parameters to their default-constructed state.
		 * This ensures all members are restored to the values specified by their
		 * default member initializers.
		 */
		void reset() noexcept { *this = Parameters{}; }
	};

	inline Parameters params; ///< Global simulation parameter state.

	/**
	 * @brief Get the speed of light.
	 * @return The speed of light in meters per second.
	 */
	inline RealType c() noexcept { return params.c; }

	/**
	 * @brief Get the Boltzmann constant.
	 * @return The Boltzmann constant.
	 */
	inline RealType boltzmannK() noexcept { return params.boltzmann_k; }

	/**
	 * @brief Get the start time for the simulation.
	 * @return Start time for the simulation.
	 */
	inline RealType startTime() noexcept { return params.start; }

	/**
	 * @brief Get the end time for the simulation.
	 * @return End time for the simulation.
	 */
	inline RealType endTime() noexcept { return params.end; }

	/**
	 * @brief Get the simulation sampling rate.
	 * @return The simulation sampling rate.
	 */
	inline RealType simSamplingRate() noexcept { return params.sim_sampling_rate; }

	/**
	 * @brief Get the CW/streaming coarse propagation sampling rate, if enabled.
	 *
	 * When unset, CW/streaming propagation is evaluated exactly at every output sample.
	 * When set, the scene is only re-sampled at this rate, with each path's delay/gain
	 * linearly interpolated between samples.
	 *
	 * @return The coarse sampling rate in Hz, or `std::nullopt` if disabled.
	 */
	inline std::optional<RealType> cwSimSamplingRate() noexcept { return params.cw_coarse_sample_rate; }

	/**
	 * @brief Get the rendering sample rate.
	 * @return The rendering sample rate.
	 */
	inline RealType rate() noexcept { return params.rate; }

	/**
	 * @brief Get the random seed.
	 * @return The current random seed value.
	 */
	inline unsigned randomSeed() noexcept { return params.random_seed.value_or(0); }

	/**
	 * @brief Get the ADC quantization bits.
	 * @return Number of ADC quantization bits.
	 */
	inline unsigned adcBits() noexcept { return params.adc_bits; }

	/**
	 * @brief Get the render filter length.
	 * @return The length of the render filter.
	 */
	inline unsigned renderFilterLength() noexcept { return params.filter_length; }

	/**
	 * @brief Get the number of worker threads.
	 * @return The number of worker threads.
	 */
	inline unsigned workerThreads() noexcept { return params.worker_threads.value_or(core::countProcessors()); }

	/**
	 * @brief Get the oversampling ratio.
	 * @return The oversampling ratio.
	 */
	inline unsigned oversampleRatio() noexcept { return params.oversample_ratio; }

	/**
	 * @brief Get the active propagation model.
	 * @return The propagation model selected for this simulation.
	 */
	inline PropagationModelKind propagationModel() noexcept { return params.propagation_model; }

	/**
	 * @brief Get the ray-tracing engine runtime override.
	 * @return The ray-tracing engine preference selected for this simulation.
	 */
	inline RayTracingEnginePreference rayTracingEnginePreference() noexcept { return params.ray_tracing_engine; }

	/**
	 * @brief Get the point-scatterer modelling parameters.
	 * @return The point-scatterer modelling parameters.
	 */
	inline const PointScatterParameters& pointScatterParams() noexcept { return params.point_scatterer_params; }

	/**
	 * @brief Get the ray-tracing modelling parameters.
	 * @return The ray-tracing modelling parameters.
	 */
	inline const RayTracingParameters& rayTracingParams() noexcept { return params.ray_tracing_params; }

	/**
	 * @brief Gets the maximum supported oversampling ratio.
	 * @return The maximum supported oversampling ratio.
	 */
	inline constexpr unsigned maxSupportedOversampleRatio() noexcept { return 8u; }

	/**
	 * @brief Validates that an oversampling ratio is supported.
	 * @param ratio The oversampling ratio to validate.
	 * @throws std::runtime_error if the ratio is outside the supported range.
	 */
	inline void validateOversampleRatio(const unsigned ratio)
	{
		if (ratio == 0)
		{
			throw std::runtime_error("Oversample ratio must be >= 1");
		}
		if (ratio > maxSupportedOversampleRatio())
		{
			throw std::runtime_error("Oversampling ratios > 8 are not supported with the current fixed filter length "
									 "of 33. Signal attenuation will occur.");
		}
	}

	/**
	 * @brief Set the speed of light.
	 * @param cValue The new speed of light.
	 */
	inline void setC(RealType cValue) noexcept
	{
		params.c = cValue;
		LOG(logging::Level::INFO, "Propagation speed (c) set to: {:.5f}", cValue);
	}

	/**
	 * @brief Set the start and end times for the simulation.
	 * @param startTime Start time for the simulation.
	 * @param endTime End time for the simulation.
	 */
	inline void setTime(const RealType startTime, const RealType endTime) noexcept
	{
		params.start = startTime;
		params.end = endTime;
		LOG(logging::Level::INFO, "Simulation time set from {:.5f} to {:.5f} seconds", startTime, endTime);
	}

	/**
	 * @brief Set the simulation sampling rate.
	 * @param rate The new simulation sampling rate.
	 */
	inline void setSimSamplingRate(const RealType rate)
	{
		if (rate <= 0)
		{
			throw std::runtime_error("Simulation sampling rate must be > 0");
		}

		params.sim_sampling_rate = rate;
		LOG(logging::Level::DEBUG, "Simulation sampling rate set to: {:.5f} Hz", rate);
	}

	/**
	 * @brief Set the rendering sample rate.
	 * @param rateValue The new sample rate for rendering.
	 */
	inline void setRate(RealType rateValue)
	{
		if (rateValue <= 0)
		{
			throw std::runtime_error("Sampling rate must be > 0");
		}
		params.rate = rateValue;
		LOG(logging::Level::DEBUG, "Sample rate set to: {:.5f}", rateValue);
	}

	/**
	 * @brief Set the random seed.
	 * @param seed The new random seed value.
	 */
	inline void setRandomSeed(const unsigned seed) noexcept
	{
		params.random_seed = seed;
		LOG(logging::Level::DEBUG, "Random seed set to: {}", seed);
	}

	/**
	 * @brief Set the ADC quantization bits.
	 * @param bits The new ADC quantization bits.
	 */
	inline void setAdcBits(const unsigned bits) noexcept
	{
		params.adc_bits = bits;
		LOG(logging::Level::DEBUG, "ADC quantization bits set to: {}", bits);
	}

	/**
	 * @brief Set the oversampling ratio.
	 * @param ratio The new oversampling ratio.
	 * @throws std::runtime_error if the ratio is zero.
	 */
	inline void setOversampleRatio(unsigned ratio)
	{
		validateOversampleRatio(ratio);
		params.oversample_ratio = ratio;
		LOG(logging::Level::DEBUG, "Oversampling enabled with ratio: {}", ratio);
	}

	/**
	 * @brief Set the geodetic origin for ENU KML/geospatial export.
	 * @param lat The latitude of the KML origin.
	 * @param lon The longitude of the KML origin.
	 * @param alt The altitude of the KML origin (MSL).
	 */
	inline void setOrigin(const double lat, const double lon, const double alt) noexcept
	{
		params.origin_latitude = lat;
		params.origin_longitude = lon;
		params.origin_altitude = alt;
		LOG(logging::Level::INFO, "KML origin set to lat: {}, lon: {}, alt: {}", lat, lon, alt);
	}

	/**
	 * @brief Gets the KML/geospatial export origin latitude.
	 * @return The KML origin latitude in degrees.
	 */
	inline double originLatitude() noexcept { return params.origin_latitude; }

	/**
	 * @brief Gets the KML/geospatial export origin longitude.
	 * @return The KML origin longitude in degrees.
	 */
	inline double originLongitude() noexcept { return params.origin_longitude; }

	/**
	 * @brief Gets the KML/geospatial export origin altitude.
	 * @return The KML origin altitude in meters.
	 */
	inline double originAltitude() noexcept { return params.origin_altitude; }

	/**
	 * @brief Set the number of worker threads.
	 * @param threads The number of worker threads.
	 * @return A `std::expected<void, std::string>` indicating success or an error message if the number of threads is
	 * invalid.
	 */
	inline std::expected<void, std::string> setThreads(const unsigned threads) noexcept
	{
		if (threads == 0)
		{
			return std::unexpected("Thread count must be >= 1");
		}
		params.worker_threads = threads;
		LOG(logging::Level::INFO, "Number of worker threads set to: {}", threads);
		return {};
	}

	/**
	 * @brief Set the ray-tracing engine runtime override.
	 * @param engine The ray-tracing engine preference.
	 */
	inline void setRayTracingEnginePreference(const RayTracingEnginePreference engine) noexcept
	{
		params.ray_tracing_engine = engine;
		LOG(logging::Level::INFO, "Ray tracing engine preference set to: {}", rayTracingEngineName(engine));
	}

	/**
	 * @brief Set the coordinate system for KML/geospatial export.
	 * @param frame The KML coordinate frame (ENU, UTM, ECEF).
	 * @param zone The KML UTM zone, if applicable.
	 * @param north The KML UTM hemisphere (true for North), if applicable.
	 */
	inline void setCoordinateSystem(const CoordinateFrame frame, const int zone, const bool north) noexcept
	{
		params.coordinate_frame = frame;
		params.utm_zone = zone;
		params.utm_north_hemisphere = north;
	}

	/**
	 * @brief Gets the KML/geospatial export coordinate frame.
	 * @return The KML/geospatial export coordinate frame.
	 */
	inline CoordinateFrame coordinateFrame() noexcept { return params.coordinate_frame; }

	/**
	 * @brief Gets the external rotation angle unit.
	 * @return The active rotation angle unit.
	 */
	inline RotationAngleUnit rotationAngleUnit() noexcept { return params.rotation_angle_unit; }

	/**
	 * @brief Gets the configured KML UTM zone.
	 * @return The KML UTM zone, or zero when UTM KML export is not configured.
	 */
	inline int utmZone() noexcept { return params.utm_zone; }

	/**
	 * @brief Gets the configured KML UTM hemisphere.
	 * @return true for northern hemisphere, false for southern hemisphere.
	 */
	inline bool utmNorthHemisphere() noexcept { return params.utm_north_hemisphere; }

	/**
	 * @brief Sets the external rotation angle unit.
	 * @param unit The rotation angle unit to store.
	 */
	inline void setRotationAngleUnit(const RotationAngleUnit unit) noexcept { params.rotation_angle_unit = unit; }

	/**
	 * @brief Converts a rotation angle unit to its XML token.
	 * @param unit The rotation angle unit.
	 * @return The corresponding token string.
	 */
	inline constexpr std::string_view rotationAngleUnitToken(const RotationAngleUnit unit) noexcept
	{
		return unit == RotationAngleUnit::Radians ? "rad" : "deg";
	}

	/**
	 * @brief Parses a rotation angle unit from an XML token.
	 * @param token The token to parse.
	 * @return The parsed unit, or std::nullopt if the token is unknown.
	 */
	inline std::optional<RotationAngleUnit> rotationAngleUnitFromToken(const std::string_view token) noexcept
	{
		if (token == "deg")
		{
			return RotationAngleUnit::Degrees;
		}
		if (token == "rad")
		{
			return RotationAngleUnit::Radians;
		}
		return std::nullopt;
	}
}
