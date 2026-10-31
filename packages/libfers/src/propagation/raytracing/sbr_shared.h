// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <cassert>
#include <cstdint>

#include "propagation/common_defs.h"
#include "propagation/math.h"

// This file defines shared definitions for types used by the SBR-like engine
// for the RayTracingModel propagation model.
//
// ## On Precision
//
// See sbr_impl.h's section of the same name.

namespace propagation::raytracing
{
	/// The number of recursive ray-casts allowed.
	///
	/// OptiX interpretation:
	/// Zero means the ray-launch program can run, but can't cast any rays.
	/// One means the ray-launch program can do direct shadow tests and single-step GO rays, but no PO shadow returns.
	/// Two is the first functional option, direct shadow rays, single-step GO rays, and PO shadow returns (bistatic
	/// paths).
	///     This is the "single GO bounce" case. Thus ray depth is (go_bounces+1).
	///
	/// This number can be adjusted as needed. This caps the depth users can request.
	constexpr uint32_t MAX_SCATTER_LIMIT = 7;

	/// This is around the scale of the Milky Way.
	/// Should be sufficient to avoid far-clipping valid geometry.
	constexpr float MAX_RAY_DISTANCE = 1e16f;

	/// A ray's minimum hit distance. Self-intersection is addressed by displacing the ray origin
	/// off the surface with `offsetSurfaceOrigin`, which handles grazing launch angles better
	/// than minimum ray intersection distance. Results are fairly insensitive to this value
	/// per my testing, but it does need to be quite small.
	constexpr float MIN_RAY_DISTANCE = 1e-5f;

	/// SBR engine triangle mesh view.
	struct TriangleMeshView
	{
		/// The number of vertices in this mesh's buffer.
		uint32_t vertex_count;
		/// The offset of this mesh's index buffer within the SbrParams::indices buffer.
		///
		/// Note that the indices are not offset from the start of the vertex buffer.
		uint32_t index_buffer_index;
		/// The number of triangles in this mesh's buffer. Equal to the number of Uint3 index sets.
		uint32_t triangle_count;
	};

	/// SBR engine material.
	struct Material
	{
		/// The relative permittivity of the material.
		///
		/// This is used in conjuction with the `conductivity` to compute the
		/// complex index of refraction: n = sqrt(ε_r' - j·σ/(ω·ε0))
		float relative_permittivity;

		/// The conductivity of the material, Siemens per meter.
		///
		/// Zero for dielectrics, infinity for perfect electric conductors.
		float conductivity;
	};

	/// SBR engine view of baked antenna gain pattern.
	struct AntennaPatternView
	{
		/// Dense az/el grid of raw linear gain, in the antenna's local frame.
		/// Boresight = local +X = local azimuth=0, elevation=0.
		///
		/// Index as `gains[el_idx * az_count + az_idx]`.
		/// There are `az_count` samples per elevation spanning `[-pi, pi]`.
		/// There are `el_count` samples per azimuth spanning `[-pi/2, pi/2]`.
		/// Inclusive at both ends, so index 0 and index count-1 land exactly on the boundary.
		uint32_t gains_buffer_offset;
		uint32_t az_count;
		uint32_t el_count;
	};

	/// Discriminates `AntennaModel`'s payload.
	enum class AntennaKind : uint8_t
	{
		Isotropic,
		Sinc,
		Gaussian,
		SquareHorn,
		Parabolic,
		Grid2D,
	};

	/// Device-side antenna gain model.
	///
	/// A tagged union over the parameterised and tabulated representations.
	struct AntennaModel
	{
		AntennaKind kind = AntennaKind::Isotropic;
		float efficiency = 1.0;
		union
		{
			struct
			{
				float alpha, beta, gamma;
			} sinc;

			struct
			{
				float azimuth_scale, elevation_scale;
			} gaussian;

			struct
			{
				float dimension;
			} square_horn;

			struct
			{
				float diameter;
			} parabolic;

			struct
			{
				AntennaPatternView grid;
			} grid_2d;
		} params{};

		CFloat horizontal_pol;
		CFloat vertical_pol;
	};

	struct ActiveAntenna
	{
		Double3 position;
		FloatAzEl direction;
		uint32_t antenna_model_index{};
	};


	struct Contribution
	{
		CFloat path_gain;
		double delay{};
		uint64_t path_id{};
		/// source_index + time_index * source_count
		uint32_t source_times_index{};
		/// Which destination antenna was reached.
		///
		/// forward tracing (traceFromTransmitter) traces from one transmitter to all receivers, so this is an
		/// index into receivers; backward tracing (traceFromReceiver) traces from one receiver to all
		/// active streaming sources, so this is an index into those sources.
		uint32_t dest_index{};
	};

	enum RxFlags : uint8_t
	{
		/// Disable 1/R sphere spreading.
		FLAG_NOSPREADING = 1,
		/// Disable direct paths to the receiver.
		FLAG_NODIRECT = 2,
	};

	struct SbrParams
	{
		/// The number of GO steps allowed.
		/// (Reflections/transmissions/diffractions.)
		///
		/// Note that PO is performed at each hit point, so zero GO bounces
		/// still means that radar effectively bounces once.
		uint32_t scatter_limit;

		/// Number of ray tubes launched per source antenna (the launch's X dimension).
		uint32_t directions_per_source;
		/// The number of rays dedicated to the boresight fraction of the sphere.
		/// This should be less than or equal to `rays_per_source`.
		uint32_t boresight_rays;
		/// The ratio of the area or solid angle of the boresight cap to the launch sphere.
		/// This is `boresight_solid_angle / 4pi`.
		float boresight_fraction;

		/// Triangle mesh vertex buffer.
		const Float3* vertices;
		/// Triangle mesh index buffer.
		///
		/// Winding order convention: vertices v0=vertices[x], v1=vertices[y], v2=vertices[z]
		/// are wound counter-clockwise from the outside of the mesh, inward (front face).
		/// Thus the outward facing normal thus obeys the right-hand rule.
		/// OptiX and Embree use the same convention.
		const Uint3* indices;
		/// Materials buffer.
		const Material* materials;
		/// Per-index timestamp buffer.
		const double* times;
		/// Transmitting carrier model buffer.
		/// Indexed by transmitter ID.
		const CarrierModel<float>* carrier_models;
		/// Antenna model buffer.
		const AntennaModel* antenna_models;
		/// Antenna gains buffer.
		const float* antenna_gains;

		/// Destination antennas positions and orientations.
		/// This is `src_antenna_count * times` in length.
		/// Indexed by `src_antennas[src_antenna_idx + src_antenna_count * time_idx]`.
		const ActiveAntenna* source_antennas;
		/// The number of source antennas to trace from.
		uint32_t source_antenna_count;
		/// Destination antennas positions and orientations.
		/// This is `src_antenna_count * times` in length.
		/// Indexed by `src_antennas[src_antenna_idx + src_antenna_count * time_idx]`.
		const ActiveAntenna* dest_antennas;
		/// The number of destination antennas to trace to.
		uint32_t dest_antenna_count;

		/// Receiver flags.
		RxFlags* rx_flags;


		/// Whether we're tracing backwards from receivers to transmitters.
		///
		/// forward/traceFromTransmitter
		/// - one transmitter
		/// - at a sequence of times
		/// - to all receivers
		///
		/// backward/traceFromReceiver
		/// - currently one receiver, but that should probably change
		/// - currently one time, but that should probably change
		/// - to all active streaming sources
		bool rx_to_tx;

		// Currently the per-instance data is limited to a material index.
		// If there's a need to go beyond that, turn that into an instance
		// index which indexes into per-instance data here.
		// Instance* instances;
	};

} // namespace propagation::raytracing
