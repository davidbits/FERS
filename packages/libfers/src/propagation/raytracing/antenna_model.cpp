// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#include "antenna_model.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "core/config.h"
#include "math/geometry_ops.h"

namespace propagation::raytracing
{
	namespace
	{
		/// Resamples an antenna's gain pattern onto a dense az/el grid via its own `getGain()`.
		/// The output buffer must match `AntennaPatternView`'s convention.
		uint32_t bakeGrid(const antenna::Antenna& antenna, const uint32_t az_count, const uint32_t el_count,
						  std::vector<float>& gains_buffer)
		{
			const auto buffer_offset = static_cast<uint32_t>(gains_buffer.size());
			gains_buffer.resize(buffer_offset + az_count * el_count);

			// The reference angle (boresight) is implicitly the local X-axis in the FERS engine (see
			// `Antenna::getAngle`/callers), matching `AzEl`'s convention (sbr.h) and
			// `api.cpp::fers_get_antenna_pattern`'s UI preview bake.
			const math::SVec3 ref_angle(1.0, 0.0, 0.0);

			const auto az_denominator = static_cast<RealType>(az_count - 1);
			const auto el_denominator = static_cast<RealType>(el_count - 1);

			for (uint32_t i = 0; i < el_count; ++i)
			{
				const RealType elevation = (static_cast<RealType>(i) / el_denominator) * PI - (PI / 2.0);
				for (uint32_t j = 0; j < az_count; ++j)
				{
					const RealType azimuth = (static_cast<RealType>(j) / az_denominator) * 2.0 * PI - PI;
					const math::SVec3 sample_angle(1.0, azimuth, elevation);
					// H5Antenna::isWavelengthDependent() is false; the wavelength argument is unused.
					// getGain() already folds in the antenna's efficiency factor, but so does
					// sampleAntennaModel (via AntennaModel::efficiency) for every kind uniformly -
					// divide it back out here so it isn't applied twice for the baked Grid2D kind.
					const RealType gain = antenna.getGain(sample_angle, ref_angle, 1.0) / antenna.getEfficiencyFactor();
					gains_buffer[buffer_offset + i * az_count + j] = static_cast<float>(gain);
				}
			}

			return buffer_offset;
		}
	}

	AntennaModel buildAntennaModel(const antenna::Antenna& antenna, std::vector<float>& gains_buffer_out,
								   uint32_t grid_az_count, uint32_t grid_el_count)
	{
		AntennaModel result{};
		result.efficiency = float(antenna.getEfficiencyFactor());

		const auto pol = antenna.getPolarisation();
		result.horizontal_pol = CFloat{float(pol.horizontal.real()), float(pol.horizontal.imag())};
		result.vertical_pol = CFloat{float(pol.vertical.real()), float(pol.vertical.imag())};

		if (dynamic_cast<const antenna::Isotropic*>(&antenna) != nullptr)
		{
			result.kind = AntennaKind::Isotropic;
			return result;
		}
		if (const auto* sinc = dynamic_cast<const antenna::Sinc*>(&antenna))
		{
			result.kind = AntennaKind::Sinc;
			result.params.sinc = {float(sinc->getAlpha()), float(sinc->getBeta()), float(sinc->getGamma())};
			return result;
		}
		if (const auto* gaussian = dynamic_cast<const antenna::Gaussian*>(&antenna))
		{
			result.kind = AntennaKind::Gaussian;
			result.params.gaussian = {float(gaussian->getAzimuthScale()), float(gaussian->getElevationScale())};
			return result;
		}
		if (const auto* horn = dynamic_cast<const antenna::SquareHorn*>(&antenna))
		{
			result.kind = AntennaKind::SquareHorn;
			result.params.square_horn = {float(horn->getDimension())};
			return result;
		}
		if (const auto* parabolic = dynamic_cast<const antenna::Parabolic*>(&antenna))
		{
			result.kind = AntennaKind::Parabolic;
			result.params.parabolic = {float(parabolic->getDiameter())};
			return result;
		}
		if (dynamic_cast<const antenna::H5Antenna*>(&antenna) != nullptr ||
			dynamic_cast<const antenna::XmlAntenna*>(&antenna) != nullptr)
		{
			if (grid_az_count < 2 || grid_el_count < 2)
			{
				throw std::invalid_argument("buildAntennaModel: grid_az_count and grid_el_count must both be >= 2.");
			}
			result.kind = AntennaKind::Grid2D;
			result.params.grid_2d.grid.gains_buffer_offset =
				bakeGrid(antenna, grid_az_count, grid_el_count, gains_buffer_out);
			result.params.grid_2d.grid.az_count = grid_az_count;
			result.params.grid_2d.grid.el_count = grid_el_count;
			return result;
		}

		throw std::invalid_argument("buildAntennaModel: unrecognized antenna type for '" + antenna.getName() + "'.");
	}
}
