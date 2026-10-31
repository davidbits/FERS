// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <cstdint>
#include <vector>

#include "antenna/antenna_factory.h"
#include "propagation/raytracing/sbr_shared.h"

namespace propagation::raytracing
{
	/**
	 * @brief Builds a device-ready `AntennaModel` from a CPU `antenna::Antenna`.
	 *
	 * No wavelength parameter: analytic antenna kinds (`Sinc`/`Gaussian`/`SquareHorn`/`Parabolic`/`Isotropic`)
	 * are modelled parametrically, and thus the gains are computed on-device where the wavelength becomes known.
	 *
	 * @param antenna The antenna to build a model for.
	 * @param grid_az_count, grid_el_count Grid resolution used for `XmlAntenna` and `H5Antenna`.
	 */
	[[nodiscard]] AntennaModel buildAntennaModel(const antenna::Antenna& antenna, std::vector<float>& gains_buffer_out,
												 uint32_t grid_az_count = 180, uint32_t grid_el_count = 90);
}
