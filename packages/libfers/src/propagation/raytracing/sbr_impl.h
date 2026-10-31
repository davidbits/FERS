// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "antenna/antenna_gain.h"
#include "propagation/common.h"
#include "propagation/common_defs.h"
#include "propagation/math.h"
#include "propagation/raytracing/sbr_shared.h"

#ifndef HC_FN
#if defined(__CUDACC__) || defined(__HIPCC__)
#define HC_FN __host__ __device__ inline
#else
#define HC_FN inline
#endif
#endif

// This file defines most of the propagation logic for the RayTracingModel propagation model.
//
// ## On Precision
//
// Floating point precision is worth careful consideration in FERS.
// For example, the difference of a facet 1cm wide verses 2cm is important, even 200km away.
//
// A number of bounding facts are considered in the approach used:
// - GPU FLOPS differs greatly between single and double-precision on consumer cards. Typically 1:32 or 1:64.
// - OptiX and similar engines trace against 32-bit precision geometry (local mesh space).
// - FERS may be used for very long-distance radar, e.g. ground-to-space, i.e. 100s of kilometers between objects.
// - This means that global spacial data and geometry must be in double-precision.
// - Double-precision is valuable for cases where a very big number and a very small number are added together,
//      and the small number must retain a high degree of precision.
// - This geometry remains in local space (centered at its local origin), so this precision issue is minimal.
// - However, normals often don't need to be in double-precision - they're unit-magnitude.
// - Complex math operations (e.g. sqrt, trig) on GPUs have significant performance gains at 32-bit precision.
// - Similarly, most physical material properties are only known to within single-precision.
// - The carrier frequencies in use are often narrowband approximations to far worse accuracy than 32-bit precision.
// - Antenna gains are linear, often between 0-1, and are often approximated by interpolation anyway.
//
// As such, the following approach is taken to choices of precision:
// - Local geometry is in single-precision.
// - Global spatial positions and lengths is kept in double-precision.
// - Unit vectors' precision is generally chosen to match the precision of the other data in operations.
// - AzEl orientations are in 32-bit precision.
// - Carrier frequencies are in 32-bit precision.
// - Antenna gains are in 32-bit precision.
// - Material property calculations are in 32-bit precision, and hence so are electric fields.

namespace propagation::raytracing
{
	struct RadiationCache
	{
		/// Electric field-oriented unitless radiation vector. Carrier-specific.
		/// Perpendicular to direction of propagation.
		///
		/// Note that the units here are a bit interesting.
		/// far-field E(r-hat, R) = sqrt(Z0/4/pi * power) * sqrt(gain) * unit-polarisation(r-hat) * e^(-jkR) / R
		/// The radiation vector consist only of this   :   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		/// The sqrt(gain) is more of an opportunistic scalar factor here instead of storing it seperately.
		/// Think of this as the transformed unit-polarisation.
		///
		/// The point of it is to pick up all the voltage attenuation and phase effects
		/// of the path while being proportional to the electric field.
		/// All geometric optics and physical optics physics performed is linear with
		/// respect to the electric field, and thus the results remain physical.
		CFloat3 incident_field{};
		CFloat3 reflected_field{};
		float src_power_gain{};
		// /// This is the sum of the reflected and incident electric polarisations.
		// /// Same units as `electric_unit_reflected`.
		// CFloat3 electric_unit_total{};
		// /// This is the
		// /// `norm x ((u_inc x e_i) + (u_refl x e_r))`
		// /// which is
		// /// `norm x (Z0 * magnetic_unit_total)`
		// /// which is the surface current (unit-ified) multiplied by Z0.
		// ///
		// /// This remains in units of `electric_unit_reflected`.
		// CFloat3 n_cross_magnetic_unit_total{};

		/// The carrier frequency for which the cached radiation and surface current are valid.
		float carrier_frequency = -1.0f;
		/// The depth for which the cached radiation and surface current were computed for.
		uint32_t depth = 0;
	};

	/**
	 * @brief The ray state. This is updated at each step.
	 *
	 * All spacial properties are in FERS's global simulation space, cartesian, in meters.
	 */
	struct PathState
	{
		// Launch information.

		/// Which radar source antenna this ray originates from, by index.
		uint32_t source_index{};
		/// Which time this ray originates from, by index.
		uint32_t time_index{};
		/// The "weight" of the ray tube, in steradians. (solid
		/// angle/pdf)
		float weight{};
		/// The antenna-local far-field radiation direction. Used to
		/// sample the antenna gain/pol.
		Float3 local_emmission_direction;


		// Path information.

		/// The number of path segments found along the GO path thus far.
		uint32_t ray_count = 0;
		/// Path vertices or hit points, positions, in meters.
		///
		/// Contains `ray_count + 1` vertices.
		std::array<Double3, MAX_SCATTER_LIMIT + 1> path_vertices;
		/// Unit directions from path vertex i to i+1 in trace direction.
		///
		/// Contains `ray_count + 1` unit directions.
		std::array<Float3, MAX_SCATTER_LIMIT + 1> trace_directions;
		/// Unit normals at hit surfaces.
		///
		/// There are `ray_count` unit normals.
		/// Normal 0 corresponds to Path Vertex 1.
		std::array<Float3, MAX_SCATTER_LIMIT> surface_normals;
		/// Materials of hit surfaces.
		///
		/// There are `ray_count` materials.
		/// Material 0 corresponds to Path Vertex 1.
		std::array<const Material*, MAX_SCATTER_LIMIT> surface_materials{};
		/// The accumulative path length.
		double length = 0.0;
		/// The hash of the path so far, unique per facet sequence.
		uint64_t hash = HASH_SEED;

		///
		RadiationCache cache;
	};

	struct HitInfo
	{
		/// The triangle index that was hit.
		uint32_t triangle_index{};
		/// The hit objects surface material.
		uint32_t material_index{};
		/// The barycentric hit posititon u coordinate.
		float bary_u{};
		/// The barycentric hit posititon v coordinate.
		float bary_v{};
		/// The hit object's world to local transformation (affine 3D matrix with translation).
		Float3x4 world_to_obj;
		/// The hit object's local to world transformation (affine 3D matrix with translation).
		Float3x4 obj_to_world;
	};

	/// Lifts a point that lies on a facet clear of that facet along `u_norm`, far enough that a
	/// ray leaving it into `u_norm`'s half-space cannot intersect the facet it came from.
	///
	/// Every accelerated ray-tracing engine back-end accepts the ray origin in single precision.
	/// When determining the ray origin, care needs to be taken to avoid the computed value ending
	/// up on the wrong side of the nearby triangles, leading to a self-intersection/shadow acne.
	///
	/// This is the origin-biasing approach of Waechter and Binder (Ray Tracing Gems). Their
	/// integer formulation didn't work as well as this simple formulation in my testing,
	/// partly as their implementation is tuned to avoid self-intersection much more aggressively.
	/// This simulator needs much tighter bounds to accurately determine returns from distant,
	/// concave objects where over-displacing the origin results in missing the multi-hit continuation.
	///
	/// One ULP of the largest coordinate is the measured optimum between two failure modes.
	/// A convex, electrically smooth target at range self-occludes below it, while a concave
	/// target's multi-bounce return is over-filtered above it, the displacement being large enough
	/// to step past the neighbouring facet across a shared edge.
	///
	/// Only the traced ray's origin moves, while the double-precision hit point is used for
	/// path length and phase computations.
	HC_FN Float3 offsetSurfaceOrigin(const Float3& pos, const Float3& u_norm)
	{
		// 2^-23: a float's mantissa carries 24 significant bits, so consecutive representable
		// values at magnitude m are m * 2^-23 apart. The magnitude is floored at 10 m so that a
		// scene sitting at or near the world origin still gets a decent margin.
		// Testing against a ground plane at the origin showed 10m was better than 1m.
		constexpr float ULP_SCALE = 1.1920928955078125e-07f;
		const float argmax = std::max({std::abs(pos.x), std::abs(pos.y), std::abs(pos.z), 10.0f});
		return pos + (ULP_SCALE * argmax) * u_norm;
	}

	/// Bilinearly samples an `AntennaPatternView` at a local (azimuth, elevation), clamping
	/// both axes to the grid's covered range.
	///
	/// Clamping (rather than wrapping) azimuth is correct here because the grid already samples the
	/// full circle inclusive at both ends (az=-pi and az=+pi are both present, redundantly), so
	/// clamping at the edges reproduces the same behavior wrapping would.
	HC_FN float sampleAntennaPattern(const SbrParams& params, const AntennaPatternView& pattern, float azimuth,
									 float elevation)
	{
		assert(pattern.az_count >= 2);
		assert(pattern.el_count >= 2);
		assert(params.antenna_gains != nullptr);

		const auto az_max = static_cast<float>(pattern.az_count - 1);
		const auto el_max = static_cast<float>(pattern.el_count - 1);

		float az_frac = (azimuth + PI_V<float>) / (2.0f * PI_V<float>)*az_max;
		float el_frac = (elevation + PI_V<float> / 2.0f) / PI_V<float> * el_max;
		az_frac = az_frac < 0.0f ? 0.0f : (az_frac > az_max ? az_max : az_frac);
		el_frac = el_frac < 0.0f ? 0.0f : (el_frac > el_max ? el_max : el_frac);

		const auto az0 = static_cast<uint32_t>(az_frac);
		const auto el0 = static_cast<uint32_t>(el_frac);
		const uint32_t az1 = az0 + 1 < pattern.az_count ? az0 + 1 : az0;
		const uint32_t el1 = el0 + 1 < pattern.el_count ? el0 + 1 : el0;

		const auto taz = az_frac - static_cast<float>(az0);
		const auto tel = el_frac - static_cast<float>(el0);

		const float g00 = params.antenna_gains[pattern.gains_buffer_offset + el0 * pattern.az_count + az0];
		const float g01 = params.antenna_gains[pattern.gains_buffer_offset + el0 * pattern.az_count + az1];
		const float g10 = params.antenna_gains[pattern.gains_buffer_offset + el1 * pattern.az_count + az0];
		const float g11 = params.antenna_gains[pattern.gains_buffer_offset + el1 * pattern.az_count + az1];

		const float g0 = g00 + (g01 - g00) * taz;
		const float g1 = g10 + (g11 - g10) * taz;
		return g0 + (g1 - g0) * tel;
	}

	/**
	 * @brief Evaluates an `AntennaModel` at a local direction, returning linear gain (efficiency
	 * already applied).
	 *
	 * @param local_dir The local-frame direction (unit vector; see `AzEl`), used directly for the
	 * rotationally-symmetric analytic kinds (`Sinc`/`SquareHorn`/`Parabolic`: angle off boresight is
	 * `acos(local_dir.x)`, since boresight is local +X by construction - cheaper than round-tripping
	 * through `local_az`/`local_el`).
	 * @param local_az, local_el As used by `Gaussian` (directly, as the az/el deltas from boresight)
	 * and by the tabulated kinds (`Grid2D`).
	 * @param wavelength Only used by the wavelength-dependent kinds (`SquareHorn`/`Parabolic`).
	 */
	HC_FN float sampleAntennaModel(const SbrParams& params, const AntennaModel& model, const Float3& local_dir,
								   float local_az, float local_el, float wavelength)
	{
		const float boresight_cos = local_dir.x < -1.0f ? -1.0f : (local_dir.x > 1.0f ? 1.0f : local_dir.x);

		float pattern_gain = 0.0;
		switch (model.kind)
		{
		case AntennaKind::Isotropic:
			pattern_gain = 1.0;
			break;
		case AntennaKind::Sinc:
			pattern_gain = antenna::gain::sincGain<float>(std::acos(boresight_cos), model.params.sinc.alpha,
														  model.params.sinc.beta, model.params.sinc.gamma);
			break;
		case AntennaKind::Gaussian:
			pattern_gain = antenna::gain::gaussianGain<float>(local_az, local_el, model.params.gaussian.azimuth_scale,
															  model.params.gaussian.elevation_scale);
			break;
		case AntennaKind::SquareHorn:
			pattern_gain = antenna::gain::squareHornGain<float>(std::acos(boresight_cos),
																model.params.square_horn.dimension, wavelength);
			break;
		case AntennaKind::Parabolic:
			pattern_gain = antenna::gain::parabolicGain<float>(std::acos(boresight_cos),
															   model.params.parabolic.diameter, wavelength);
			break;
		case AntennaKind::Grid2D:
			pattern_gain = sampleAntennaPattern(params, model.params.grid_2d.grid, local_az, local_el);
			break;
		}

		return float(pattern_gain * model.efficiency);
	}

	[[nodiscard]] HC_FN Float3 fibonacciDirFromX(float one_minus_x, float one_plus_x, uint32_t index)
	{
		// In a fibonacci spiral, `theta = golden_angle * i`,
		// where `golden_angle = 2pi(1 - golden_ratio) = 2pi(1 - 0.618...)`.
		// This suffers from precision issues with millions of rays as the
		// inter-float spacing becomes high at large values.
		//
		// However,
		// `theta = 2pi ((1 - golden_ratio) * i)` or equivalently,
		// `theta = 2pi frac((1 - golden_ratio) * i)`
		// We can compute the fractional part in fixed precision using integer wrapping.

		// Fixed-precision golden ratio (0.618...) with a denominator of 2^32.
		// i.e. this is `round(2^32×(golden))`
		constexpr uint32_t one_minus_golden = 0x9E3779B9u;
		constexpr float two_raied_to_24 = 16777216.0f;

		// frac((1 - golden_ratio) * i) * 2^32
		const uint32_t u = index * one_minus_golden;
		// Put the most significant 24 bits into the mantissa.
		const float turns = float(u >> 8) * (1.0f / two_raied_to_24); // [0, 1)
		// Scale fractional to raidans.
		const float theta = 2.0f * PI_V<float> * turns;

		// This is often sqrt(1 - x * x) in implementations, but this results in
		// loss of accuracy due to cancellation near the poles.
		// So (1-x)(1+x) is used instead where the tiny one can be determined accurately at each pole.
		const float r = std::sqrt(one_minus_x * one_plus_x);
		return Float3{1.0f - one_minus_x, r * std::cos(theta), r * std::sin(theta)};
	}

	[[nodiscard]] HC_FN Float3 boresightWeightedSample(const SbrParams& params, uint32_t path_idx, float& weight_out)
	{
		const float f = params.boresight_fraction;

		if (path_idx < params.boresight_rays)
		{
			weight_out = 4 * PI_V<float> * f / float(params.boresight_rays);

			// the above `if` ensures boresight_rays != 0
			const float boresight_ray_frac = (float(path_idx) + 0.5f) / float(params.boresight_rays);
			// compute (1-x) directly to keep it accurate (avoid catastrophic cancellation)
			const float one_minus_x = 2.0f * f * boresight_ray_frac;
			return fibonacciDirFromX(one_minus_x, 2.0f - one_minus_x, path_idx);
		}

		const uint32_t off_index = path_idx - params.boresight_rays;
		// the above `if` ensures off_rays != 0, given that path_idx < rays_per_source
		const uint32_t off_rays = params.directions_per_source - params.boresight_rays;
		weight_out = 4 * PI_V<float> * (1 - f) / float(off_rays);

		// calculate from far pole so (1+x) stays accurate near x = -1
		const float off_ray_frac = (float(off_rays - off_index) - 0.5f) / float(off_rays);
		const float one_plus_x = 2.0f * (1.0f - f) * off_ray_frac;
		return fibonacciDirFromX(2.0f - one_plus_x, one_plus_x, path_idx);
	}

	// /**
	//  * @brief Direction sampling using the Fibonacci/Vogel spiral point set.
	//  * In local coordinates; `i=0` lands near local (1,0,0), sweeping to near (-1,0,0)
	//  * as `i` approaches `n`.
	//  */
	// [[nodiscard]] HC_FN Float3 sampleFibonacciSpiral(uint32_t index, uint32_t count)
	// {
	// 	// In a fibonacci spiral, `theta = golden_angle * i`,
	// 	// where `golden_angle = 2pi(1 - golden_ratio) = 2pi(1 - 0.618...)`.
	// 	// This suffers from precision issues with millions of rays as the
	// 	// inter-float spacing becomes high at large values.
	// 	//
	// 	// However,
	// 	// `theta = 2pi ((1 - golden_ratio) * i)` or equivalently,
	// 	// `theta = 2pi frac((1 - golden_ratio) * i)`
	// 	// We can compute the fractional part in fixed precision using integer wrapping.

	// 	// Fixed-precision golden ratio (0.618...) with a denominator of 2^32.
	// 	// i.e. this is `round(2^32×(golden))`
	// 	constexpr uint32_t one_minus_golden = 0x9E3779B9u;
	// 	constexpr float two_raied_to_24 = 16777216.0f;

	// 	// frac((1 - golden_ratio) * i) * 2^32
	// 	const uint32_t u = index * one_minus_golden;
	// 	// Put the most significant 24 bits into the mantissa.
	// 	const float turns = float(u >> 8) * (1.0f / two_raied_to_24); // [0, 1)
	// 	// Scale fractional to raidans.
	// 	const float theta = 2.0f * PI_V<float> * turns;

	// 	const float t = (float(index) + 0.5f) / float(count);
	// 	const float i = float(index) + 0.5f;
	// 	const float n = float(count);
	// 	const float x = 1.0f - (2.0f * i) / n; // x from 1 to -1
	// 	const float r = std::sqrt(1.0f - x * x); // radius at x

	// 	return Float3{x, r * std::cos(theta), r * std::sin(theta)};
	// }

	// // /**
	// //  * @brief Isotropic direction sampling using the Fibonacci/Vogel spiral point set.
	// //  */
	// // [[nodiscard]] HC_FN Float3 isotropicWeightedSample(const SbrParams& params, uint32_t path_idx, float&
	// weight_out)
	// // {
	// // 	weight_out = 4 * PI_V<float> / float(params.rays_per_source);
	// // 	return sampleFibonacciSpiral(float(path_idx), float(params.rays_per_source));
	// // }


	// /**
	//  * @brief Dual-weighted sphere sampling.
	//  * A (higher) weight around the boresight and a lower weight outsight that.
	//  */
	// [[nodiscard]] HC_FN Float3 boresightWeightedSample(const SbrParams& params, uint32_t path_idx, float& weight_out)
	// {
	// 	if (path_idx < params.boresight_rays)
	// 	{
	// 		// We're within the boresight regime.

	// 		// `params.boresight_rays == 0` is never true as the above check would fail
	// 		weight_out = 4 * PI_V<float> * params.boresight_fraction / float(params.boresight_rays);

	// 		const float spiral_fraction =
	// 			(float(path_idx) + 0.5f) / float(params.boresight_rays) * params.boresight_fraction;
	// 		return sampleFibonacciSpiral(spiral_fraction, 1.0);
	// 	}

	// 	// We're outside the boresight regime.

	// 	// `off_boresight_rays == 0` never occurs as `path_idx < params.boresight_rays` would always be true.
	// 	const auto off_boresight_rays = float(params.rays_per_source - params.boresight_rays);
	// 	weight_out = 4 * PI_V<float> * (1 - params.boresight_fraction) / off_boresight_rays;

	// 	const float spiral_fraction = params.boresight_fraction +
	// 		(1 - params.boresight_fraction) * (float(path_idx - params.boresight_rays) + 0.5f) / off_boresight_rays;
	// 	return sampleFibonacciSpiral(spiral_fraction, 1.0);
	// }


	HC_FN void initIndirectPath(const SbrParams& params, uint32_t path_idx, uint32_t source_idx, uint32_t time_idx,
								PathState& path_out)
	{
		const ActiveAntenna& antenna = params.source_antennas[source_idx + time_idx * params.source_antenna_count];

		const Float3 local_dir = boresightWeightedSample(params, path_idx, path_out.weight);
		const Float3 world_dir = rotateLocalToWorld(antenna.direction, local_dir);

		path_out.ray_count = 0;
		path_out.source_index = source_idx;
		path_out.time_index = time_idx;
		path_out.local_emmission_direction = local_dir;
		path_out.path_vertices[0] = antenna.position;
		path_out.trace_directions[0] = world_dir;
		path_out.length = 0.0;
		path_out.hash = hash_mix(HASH_SEED, source_idx);
	}

	namespace po
	{
		struct TubePatch
		{
			/// The vertices defining the hit ray tube patch triangle.
			/// The origin of the vertices is the parent triangle's centroid, thus these are at local scale.
			/// CCW winding order (right-hand-rule)
			std::array<Float3, 3> verts;
			/// The pre-computed area of the patch defined by `verts`.
			float area;
		};

		/**
		 * @brief Computes the ray tube's PO integration patch at a facet hit: the tube's footprint,
		 * clamped to the triangle's own area, expressed as a triangle scaled toward the triangle's centroid.
		 *
		 * Relies on the flat-facet assumption: a flat-mirror reflection of a spherical wave is exactly
		 * another spherical wave from the mirrored virtual source, so the wavefront keeps diverging as if
		 * from the *original* source over the full accumulated path length `dist`, not just the current
		 * segment. This breaks once curved facets are introduced (planned future work) - the
		 * virtual-source distance would then need tracking explicitly instead of being read off `dist`.
		 *
		 * Footprint area: for a tube of solid angle `solid_angle` steradians hitting the facet at
		 * incidence angle theta_i off its normal, the footprint projected onto the facet plane is
		 * `dist^2 * solid_angle / cos(theta_i)` (standard oblique-projection geometry). The returned
		 * patch is then clamped to `min(footprint, facet_area)` by scaling the triangle toward its own
		 * centroid - which, since it's a scale toward an interior point, stays inside the parent triangle
		 * automatically, without needing separate edge-clipping.
		 *
		 * @param tri_verts World-space triangle vertices.
		 * @param u_norm Outward triangle normal (unit length).
		 * @param u_inc Unit direction the ray is travelling in (towards the hit point).
		 * @param facet_area The triangle's own area.
		 * @param dist Total accumulated path length from the ray's original source to this hit.
		 * @param solid_angle The ray tube's solid angle (steradians).
		 */
		HC_FN TubePatch computeTubePatch(const std::array<const Double3, 3>& tri_verts, const Float3& u_norm,
										 const Float3& u_inc, float facet_area, float dist, float solid_angle)
		{
			const float cos_theta_i = std::max(dot(-u_inc, u_norm), 1e-3f);
			const float tube_area = dist * dist * solid_angle / cos_theta_i;
			const float scale = std::sqrt(std::min(tube_area, facet_area) / facet_area);

			Double3 centroid{}; // TODO_SHAUN optimize
			for (size_t vi = 0; vi < 3; vi++)
				centroid += tri_verts.at(vi);
			centroid = centroid / 3.0;

			TubePatch patch{};
			for (size_t vi = 0; vi < 3; vi++)
				patch.verts.at(vi) = scale * Float3(tri_verts.at(vi) - centroid);
			patch.area = scale * scale * facet_area;
			return patch;
		}

		/// Numerically stable unnormalised sinc = sin(x)/x with the limit at sin(0) = 1.
		HC_FN float sinc(float x, float tol = 1e-5f) { return std::abs(x) < tol ? 1.0f : std::sin(x) / x; }

		/// Gordon's formula for computing the Physical Optics phase integral.
		HC_FN CFloat phaseIntegral(const TubePatch& patch, const Float3& norm, const Float3& u_inc,
								   const Float3& u_scat, float wavenumber)
		{
			/// Scattering vector, representing phase difference across patch.
			const Float3 scatter = wavenumber * (u_scat - u_inc);
			/// In-plane component of the scattering vector.
			const Float3 scatter_inplane = scatter - dot(scatter, norm) * norm;
			const float scatter_inplane2 = length2(scatter_inplane);

			// Handle the near-specular case.
			// The normalized phase integral is I = A
			if (scatter_inplane2 < 1e-8f)
				return CFloat{patch.area};

			std::array<Float3, 3> delta_vs = {patch.verts[1] - patch.verts[0], patch.verts[2] - patch.verts[1],
											  patch.verts[0] - patch.verts[2]};
			std::array<Float3, 3> mid_vs = {0.5f * (patch.verts[1] + patch.verts[0]),
											0.5f * (patch.verts[2] + patch.verts[1]),
											0.5f * (patch.verts[0] + patch.verts[2])};

			CFloat acc{0.0f};
			for (size_t i = 0; i < 3; i++)
			{
				const float re = dot(cross(scatter, norm), delta_vs.at(i)) * sinc(0.5f * dot(scatter, delta_vs.at(i)));
				acc = acc + re * cexp_i(dot(scatter, mid_vs.at(i)));
			}

			return mul_i(acc) / scatter_inplane2;
		}

		HC_FN CFloat3 vectorScatteringAmplitude(const TubePatch& patch, const Float3& u_norm, const Float3& u_inc,
												const Float3& u_scat, const Float3& u_refl, const CFloat3& e_inc,
												const CFloat3& e_refl, const float wavenumber)
		{
			const CFloat phase_integral = phaseIntegral(patch, u_norm, u_inc, u_scat, wavenumber);

			const CFloat3 e_total = e_inc + e_refl;

			// Js ~ normal x (h_inc + h_refl)
			// h = k x e
			const CFloat3 m_inc = cross(u_inc, e_inc);
			const CFloat3 m_refl = cross(u_refl, e_refl);
			const CFloat3 n_x_m_total = cross(u_norm, m_inc + m_refl);

			const CFloat3 field_vector = cross(u_scat, cross(u_norm, e_total) - cross(u_scat, n_x_m_total));

			// -jk/4/pi
			const CFloat constants = mul_i(CFloat{wavenumber * (-1.0f / 4.0f / PI_V<float>)});
			return constants * phase_integral * field_vector;
		}
	} // namespace po

	namespace go
	{
		/// Compute the reflected ray direction given the incident direction (into the surface)
		/// and surface normal at the point of incidence.
		HC_FN Float3 reflect(const Float3& u_inc, const Float3& u_norm)
		{
			const Float3 back = -u_inc;
			const Float3 cent = dot(u_norm, back) * u_norm;
			const Float3 u_refl = 2.0f * cent - back;
			return u_refl;
		}

		HC_FN void complexSnellsLaw(const Float3& norm, const Float3& k_in, CFloat n2, CFloat& gamma_tm_out,
									CFloat& gamma_te_out)
		{
			// Index of refraction of free space.
			constexpr CFloat N1 = CFloat{1.0, 0.0};

			const float cos_theta_i = -dot(k_in, norm);
			const float sin2_theta_i = 1 - cos_theta_i * cos_theta_i;
			// Snell's law: sin(theta_t) = (n1/n2) sin(theta_i)
			// So sin^2(theta_t) = (n1/n2)^2 sin^2(theta_i)
			// cos(theta_t) = sqrt(1 - sin^2(theta_t))
			const CFloat ratio = N1 / n2;
			const CFloat cos_theta_t = csqrt(1.0f - (ratio * ratio) * sin2_theta_i);

			gamma_tm_out = (n2 * cos_theta_i - N1 * cos_theta_t) / (n2 * cos_theta_i + N1 * cos_theta_t);
			gamma_te_out = (N1 * cos_theta_i - n2 * cos_theta_t) / (N1 * cos_theta_i + n2 * cos_theta_t);
		}

		/// Determine the complex index of refraction of a Material at a given carrier frequency.
		HC_FN CFloat indexOfRefraction(float relative_permittivity, float conductivity, float frequency)
		{
			// 1/sqrt(2). Enough digits to saturate double-precision.
			constexpr float ISQRT2 = 0.70710678118654752440f;

			// Special-case the calculation for dielectrics to avoid a complex sqrt.
			if (conductivity == 0)
				return CFloat{std::sqrt(relative_permittivity), 0.0};

			const float im = conductivity / (E0 * 2 * PI_V<float> * frequency);

			// Special-case the calculation for PECs.
			if (conductivity == INFINITY)
				return CFloat{ISQRT2, -ISQRT2} * std::sqrt(im);

			// Handle the partially-conducting case.
			const CFloat permittivity{relative_permittivity, -im};
			return csqrt(permittivity);
		}


		/// Reflect Cartesian radiation vector off a facet specularly
		/// using geometric optics and Fresnel equations to enforce
		/// interface conditions.
		HC_FN CFloat3 fresnelGoReflection(const CFloat3& e_inc, const Float3& u_norm, const Float3& u_inc,
										  const Float3& u_refl, const Material* mat, float frequency)
		{
			// This implements Fresnel equations for solving the reflected
			// field given an incident electric plane wave on a planar surface.

			// Compute the indident basis.
			const auto basis = incidencePlaneBasis(u_norm, u_inc);
			const auto e_tm_out = cross(basis.e_te, u_refl);

			const auto relative_permittivity = mat->relative_permittivity;
			const auto conductivity = mat->conductivity;

			CFloat gamma_tm = 1, gamma_te = -1; // PEC values
			if (mat->conductivity != INFINITY) // if not a PEC
			{
				const Complex n2 = indexOfRefraction(relative_permittivity, conductivity, frequency);
				complexSnellsLaw(u_norm, u_inc, n2, gamma_tm, gamma_te);
			}

			const Complex elec_te = dot(e_inc, basis.e_te);
			const Complex elec_tm = dot(e_inc, basis.e_tm_in);
			return basis.e_te * (gamma_te * elec_te) + e_tm_out * (gamma_tm * elec_tm);
		}


		/// Computes the reflected GO radiation vector and surface current at the hit point.
		///
		HC_FN void computeGoRadiation(const SbrParams& params, PathState* path, float frequency)
		{
			// This computes the radiation vector along the ray path.
			//
			// This function is called on-demand when a GO path ray hit occurs (many rays miss, so pre-computing is
			// likely not worthwhile). It uses a cache to avoid recomputing the full GO traced path radiation where
			// possible. Paths are found in monotonically increasing length, so the cache is always the full path or a
			// prefix. The main condition for the radiation vector being re-usable is whether the carrier wavenumber
			// matches. If it doesn't, then the antenna gain and Fresnel coefficients may differ as a function of the
			// wavenumber, and so re-computing is necessary.
			//
			// TODO: Consider whether a tolerance on the wavenumber matching is acceptable, rather than exact-match.


			CFloat3 reflected_field = path->cache.reflected_field;
			uint32_t ray_index = path->cache.depth;

			if (path->cache.carrier_frequency != frequency)
			{
				// The radiation cache isn't for the correct carrier.
				// Re-compute the full radiation path, starting with sampling the source antenna properties.

				ray_index = 0;

				const ActiveAntenna& antenna =
					params.source_antennas[path->source_index + path->time_index * params.source_antenna_count];
				const AntennaModel& model = params.antenna_models[antenna.antenna_model_index];

				const Float3 local_dir = path->local_emmission_direction;
				const CFloat3 pol = propagation::antennaPolarizationVector(model.horizontal_pol, model.vertical_pol,
																		   antenna.direction, local_dir);
				reflected_field = pol;

				// Gain from the source antenna.
				const float src_local_az = atan2f(local_dir.y, local_dir.x);
				const float src_local_el = asinf(local_dir.z);
				const float wavelength = C<float> / frequency;
				const float power_gain =
					sampleAntennaModel(params, model, local_dir, src_local_az, src_local_el, wavelength);
				path->cache.src_power_gain = power_gain;
			}

			CFloat3 incident_field;
			for (; ray_index < path->ray_count; ray_index++)
			{
				if (ray_index + 1 == path->ray_count)
				{
					incident_field = reflected_field;
				}

				reflected_field = fresnelGoReflection(
					reflected_field, path->surface_normals.at(ray_index), path->trace_directions.at(ray_index),
					path->trace_directions.at(ray_index + 1), path->surface_materials.at(ray_index), frequency);
			}

			// If the radiation vector was updated for any reason, we need to recompute the surface current.
			if (path->cache.carrier_frequency != frequency || path->cache.depth != path->ray_count)
			{
				path->cache.incident_field = incident_field;
				path->cache.reflected_field = reflected_field;
				path->cache.depth = path->ray_count;
				path->cache.carrier_frequency = frequency;
			}
		}
	} // namespace go


	/**
	 * @brief TODO
	 *
	 * @param shadow_test Engine-supplied occlusion test: `float(Float3 origin, Float3 direction)`,
	 * returning the distance along the ray to the nearest hit.
	 * @param get_contrib_ptr Returns a location to write contribution data to. `Contribution*()`
	 */
	template <typename ShadowTest, typename GetContribPtr>
	[[nodiscard]] HC_FN bool facetHit(const SbrParams& params, PathState* path, const HitInfo& hit,
									  ShadowTest shadow_test, GetContribPtr get_contrib_ptr)
	{
		// Local geometry computations in 32-bit precision in object space.

		auto indices = params.indices[hit.triangle_index];
		auto l_v0 = params.vertices[indices.x];
		auto l_v1 = params.vertices[indices.y];
		auto l_v2 = params.vertices[indices.z];

		auto bary_w = 1.0f - hit.bary_u - hit.bary_v;
		auto l_pos = l_v0 * bary_w + l_v1 * hit.bary_u + l_v2 * hit.bary_v;

		// Simulation computations in 64-bit precision world space.

		const std::array<const Double3, 3> verts{
			amuld(hit.obj_to_world, l_v0),
			amuld(hit.obj_to_world, l_v1),
			amuld(hit.obj_to_world, l_v2),
		};

		const Double3 pos = amuld(hit.obj_to_world, l_pos);
		const Float3 u_inc = path->trace_directions.at(path->ray_count);

		// Outward-facing normal, per the CCW winding convention documented on `SbrParams.indices`.
		const Double3 dnormal = cross(verts[1] - verts[0], verts[2] - verts[0]);
		const double two_times_facet_area = length(dnormal);
		Float3 u_norm = Float3{dnormal / two_times_facet_area};
		const float facet_area = float(two_times_facet_area) / 2;
		// Flip the normal to face the incoming ray.
		u_norm = dot(u_inc, u_norm) <= 0 ? u_norm : -u_norm;

		// Compute the reflected ray direction.
		const Float3 u_refl = go::reflect(u_inc, u_norm);

		// Compute the additional propagation distance.
		const double ray_len = dot(pos - path->path_vertices.at(path->ray_count), Double3{u_inc});
		const double go_path_len = path->length + ray_len;
		const auto* material = &params.materials[hit.material_index];

		// Update path
		path->ray_count += 1;
		path->path_vertices.at(path->ray_count) = pos;
		path->trace_directions.at(path->ray_count) = u_refl;
		path->surface_normals.at(path->ray_count - 1) = u_norm;
		path->surface_materials.at(path->ray_count - 1) = material;
		path->hash = hash_mix(path->hash, hit.triangle_index); // Add the current facet to the path hash
		path->length = go_path_len;

		// Ray-tube PO integration patch.
		// The ray tube footprint is governed by the incident path geometry
		// (`u_inc`, `go_path_len`, `weight`) not the destination antenna position.
		const auto patch = po::computeTubePatch(verts, u_norm, u_inc, facet_area, float(go_path_len), path->weight);

		// Determine PO contributions from this path and this ray's tube patch on this facet.
		for (uint32_t dst_idx = 0; dst_idx < params.dest_antenna_count; dst_idx++)
		{
			const ActiveAntenna& dst = params.dest_antennas[dst_idx + params.dest_antenna_count * path->time_index];

			// The scattered field direction from the hit point to destination.
			const Float3 u_scat = Float3{normalize(dst.position - pos)};

			// Does the triangle face the destination?
			if (dot(u_scat, u_norm) <= 0)
				continue;

			// Cast shadow ray from pos in direction po_seg.
			// OptiX/Embree use floating-point precision here. Conversion is required.
			// Facet self-intersection must be avoided here.
			const float t_shadow = shadow_test(offsetSurfaceOrigin(Float3{pos}, u_norm), u_scat);
			const double po_len = length(dst.position - pos);

			// If t_shadow is less than po_len, the nearest object is less than po_len away,
			// in which case the destination is occuded: skip.
			if (t_shadow < float(po_len))
				continue;

			// Perform the PO computation for the scattered field from the facet, over the ray
			// tube's patch (clipped to the triangle - see above), not the whole triangle.

			const double path_len = go_path_len + po_len;
			const double prop_time = path_len / C<double>;

			// We have a full path now.
			// Therefore we know the transmitter, TX time, and thus also the carrier.
			const uint32_t tx_index = params.rx_to_tx ? dst_idx : path->source_index;
			const double tx_time = params.times[path->time_index] - (params.rx_to_tx ? prop_time : 0);
			const CarrierModel<float>& carrier = params.carrier_models[tx_index];
			const float frequency = sampleCarrierFrequency(carrier, tx_time);
			const float wavenumber = 2 * PI_V<float> / C<float> * frequency;
			const float wavelength = C<float> / frequency;

			// Compute the destination antenna gain and polarisation into the path.
			const AntennaModel& dst_model = params.antenna_models[dst.antenna_model_index];
			// Note that this points from the antenna outward. Negative of the scatter direction to the destination.
			const Float3 dst_local_dir = rotateWorldToLocal(dst.direction, -u_scat);
			const float dst_local_az = atan2f(dst_local_dir.y, dst_local_dir.x);
			const float dst_local_el = asinf(std::clamp(dst_local_dir.z, -1.0f, 1.0f));
			const float dst_gain =
				sampleAntennaModel(params, dst_model, dst_local_dir, dst_local_az, dst_local_el, wavelength);
			const CFloat3 dst_pol = antennaPolarizationVector(dst_model.horizontal_pol, dst_model.vertical_pol,
															  dst.direction, dst_local_dir);

			// Updates the path's field cache data.
			go::computeGoRadiation(params, path, frequency);

			// The PO bit is tricky, because some inputs aren't reciprocol by tracing direction.
			const Float3 po_u_inc = params.rx_to_tx ? -u_scat : u_inc;
			const Float3 po_u_scat = params.rx_to_tx ? -u_inc : u_scat;
			const Float3 po_u_refl = params.rx_to_tx ? go::reflect(-u_scat, u_norm) : u_refl;
			const CFloat3 po_e_inc = params.rx_to_tx ? dst_pol : path->cache.incident_field;
			const CFloat3 po_e_refl = params.rx_to_tx
				? go::fresnelGoReflection(po_e_inc, u_norm, po_u_inc, po_u_refl, material, frequency)
				: path->cache.reflected_field;

			const CFloat3 po_e_scat = po::vectorScatteringAmplitude(patch, u_norm, po_u_inc, po_u_scat, po_u_refl,
																	po_e_inc, po_e_refl, wavenumber);

			// We need to compute the path gain now using the PO scatter and GO fields.

			// The normalised field/polarisation vector to link up with. This complements `po_e_inc`.
			const CFloat3 po_e_linkup = params.rx_to_tx ? path->cache.incident_field : dst_pol;
			// Note that these vector point into each other, so they should not be conjugated.
			const CFloat linkage = dot_no_conj(po_e_scat, po_e_linkup);

			// a_p = sqrt(G_t G_r) * lambda / (4 pi) * linkage * (1 / (R_GO * R_PO))
			CFloat path_gain =
				(std::sqrt(path->cache.src_power_gain * dst_gain) * wavelength / (4.0f * PI_V<float>)) * linkage;

			// Apply GO sphere spreading loss and PO distance loss unless requested not to.
			const uint32_t rx_index = params.rx_to_tx ? path->source_index : dst_idx;
			if ((params.rx_flags[rx_index] & RxFlags::FLAG_NOSPREADING) == 0)
			{
				path_gain = path_gain / (float(go_path_len) * float(po_len));
			}

			// Add a contribution to the sink.
			Contribution* contrib = get_contrib_ptr();
			contrib->path_gain = path_gain;
			contrib->delay = prop_time;
			contrib->dest_index = dst_idx;
			contrib->path_id = hash_mix(path->hash, dst_idx);
			contrib->source_times_index = path->source_index + params.source_antenna_count * path->time_index;
		}

		return path->ray_count < params.scatter_limit;
	} // facetHit

	template <typename ShadowTest, typename GetContribPtr>
	HC_FN void directPath(const SbrParams& params, uint32_t src_idx, uint32_t dst_idx, uint32_t time_idx,
						  ShadowTest shadow_test, GetContribPtr get_contrib_ptr)
	{
		const auto rx_idx = params.rx_to_tx ? src_idx : dst_idx;
		const auto rx_flags = params.rx_flags[rx_idx];

		if ((rx_flags & RxFlags::FLAG_NODIRECT) != 0)
		{
			return;
		}

		const auto& src_antenna = params.source_antennas[src_idx];
		const auto& dst_antenna = params.dest_antennas[dst_idx];
		const double path_len = length(dst_antenna.position - src_antenna.position);
		const double prop_time = path_len / C<double>;
		const Float3 u_dir = Float3{normalize(dst_antenna.position - src_antenna.position)};

		if (path_len < 1e-2)
		{
			// The antennas are co-located to within 1cm.
			// FERS does not handle such near-field calculations within its propagation model. Skipping.
			return;
		}

		const float hitlen = shadow_test(Float3{src_antenna.position}, u_dir);

		if (hitlen < float(path_len))
		{
			// Direct path is occluded. Skip.
			return;
		}

		// Get the carrier.
		const auto tx_index = params.rx_to_tx ? dst_idx : src_idx;
		const double tx_time = params.times[time_idx] - (params.rx_to_tx ? double(prop_time) : 0);
		const CarrierModel<float>& carrier = params.carrier_models[tx_index];
		const float frequency = sampleCarrierFrequency(carrier, tx_time);
		const float wavelength = C<float> / frequency;

		// Compute the antenna gains.
		const AntennaModel& src_model = params.antenna_models[src_antenna.antenna_model_index];
		const AntennaModel& dst_model = params.antenna_models[dst_antenna.antenna_model_index];

		const Float3 src_local_dir = rotateWorldToLocal(src_antenna.direction, u_dir);
		const float src_local_az = atan2f(src_local_dir.y, src_local_dir.x);
		const float src_local_el = asinf(src_local_dir.z);
		const float src_gain =
			sampleAntennaModel(params, src_model, src_local_dir, src_local_az, src_local_el, wavelength);

		const Float3 dst_local_dir = rotateWorldToLocal(dst_antenna.direction, -u_dir);
		const float dst_local_az = atan2f(dst_local_dir.y, dst_local_dir.x);
		const float dst_local_el = asinf(dst_local_dir.z);
		const float dst_gain =
			sampleAntennaModel(params, dst_model, dst_local_dir, dst_local_az, dst_local_el, wavelength);

		// Compute antenna polarisation vectors.
		const CFloat3 src_pol = propagation::antennaPolarizationVector(src_model.horizontal_pol, src_model.vertical_pol,
																	   src_antenna.direction, src_local_dir);
		const CFloat3 dst_pol = propagation::antennaPolarizationVector(dst_model.horizontal_pol, dst_model.vertical_pol,
																	   dst_antenna.direction, dst_local_dir);

		const bool no_prop_loss = (rx_flags & RxFlags::FLAG_NOSPREADING) != 0;
		const float scalar_gain = computeDirectPathGain(src_gain, dst_gain, wavelength, float(path_len), no_prop_loss);
		const CFloat voltage = scalar_gain * dot_no_conj(src_pol, dst_pol);

		Contribution* contrib = get_contrib_ptr();
		contrib->path_gain = voltage;
		contrib->delay = prop_time;
		contrib->source_times_index = src_idx + time_idx * params.source_antenna_count;
		contrib->dest_index = dst_idx;
		contrib->path_id = hash_mix(hash_mix(HASH_SEED, src_idx), dst_idx);
	}

} // namespace propagation::raytracing
