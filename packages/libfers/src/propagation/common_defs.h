// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <cstdint>

namespace propagation
{

	// --- Carrier Modelling ----------------------------------------------------------------------------- //

	enum class CarrierModelKind : uint8_t
	{
		/// A constant carrier wavenumber.
		Constant,
		/// A repeatedly uniformly stepping and looping carrier.
		Stairs,
	};

	template <typename Real>
	struct CarrierModel
	{
		CarrierModelKind kind = CarrierModelKind::Constant;

		union
		{
			struct
			{
				Real frequency = Real(-1);
			} constant{};

			struct
			{
				/// A time offset for one of the staircase start times.
				/// Expected to be near the simulation time such that
				/// `float(t - start_time)` doesn't lose accuracy.
				double start_time;
				/// The stair starting wavenumber.
				Real start_frequency;
				/// The stair step size, in wavenumbers.
				Real step_frequency;
				/// The duration between steps.
				Real step_duration;
				/// The number of steps per stair.
				Real step_count;
			} stairs;
		} params;
	};
};
