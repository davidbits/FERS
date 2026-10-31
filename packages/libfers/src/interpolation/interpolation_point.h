// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2006-2008 Marc Brooker and Michael Inggs
// Copyright (c) 2008-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

/**
 * @file interpolation_point.h
 * @brief Defines a structure to store interpolation point data for signal processing.
 */

#pragma once

#include "core/config.h"

namespace interp
{
	/**
	 * @struct InterpPoint
	 * @brief Stores channel properties to interpolate between.
	 */
	struct InterpPoint
	{
		RealType gain{}; ///< Amplitude gain magnitude of the channel, as a ratio of voltage into a unit impedance.
		RealType rx_time{}; ///< RX time at which the channel exists, in seconds.
		RealType delay{}; ///< Propagation delay of the channel, in seconds.
		RealType phase_delay{}; ///< Phase delay of the carrier frequency through the channel plus path ampltide phase
								///< modulation, in radians.
	};
}
