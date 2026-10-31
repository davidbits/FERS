// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include "core/simulation_state.h"
#include "propagation/common_defs.h"

// This file defines host-only utilities used by both the point-scatterer and ray-tracing
// propagation models. (Meanwhile `common.h` and `common_defs` must be GPU/CUDA compatible.)

namespace propagation
{
	/// Build an instantaneous carrier frequency model defined for all time used by propagation models.
	///
	/// This does not make a narrowband approximation for parametric carrier models, it is exact.
	///
	/// This is defined for all time, unlike actual FERS carrier models, which are undefined at
	/// certain times for various reasons. Instead, a reasonable extension is made here to extend
	/// the carrier to all time. This ensures interpolation of coursely-simulated returned paths
	/// doesn't suffer from noise due to carriers being undefined/zero in some signal samples.
	/// This isn't currently leveraged as the CW simulation occurs at sample rate, not a course
	/// simulation rate, which should be addressed someday, while the pulsed-mode simulation
	/// assumed a constant carrier frequency.
	template <typename Real>
	[[nodiscard]] CarrierModel<Real> buildPropagationCarrier(const core::ActiveStreamingSource& source)
	{
		CarrierModel<Real> model;

		switch (source.kind)
		{
		case core::StreamingWaveformKind::Cw:
		case core::StreamingWaveformKind::FileCw:
		case core::StreamingWaveformKind::FileFmcw:
		case core::StreamingWaveformKind::FmcwLinear:
		case core::StreamingWaveformKind::FmcwTriangle:
			{
				model.kind = CarrierModelKind::Constant;
				model.params.constant.frequency = Real(source.carrier_freq);
				break;
			}
		case core::StreamingWaveformKind::Sfcw:
			{
				model.kind = CarrierModelKind::Stairs;
				model.params.stairs.start_time = source.segment_start;
				model.params.stairs.step_duration = Real(source.sfcw_step_period);
				model.params.stairs.step_count = Real(source.sfcw_step_count);
				model.params.stairs.start_frequency = Real(source.carrier_freq + source.start_freq_off);
				model.params.stairs.step_frequency = Real(source.sfcw_step_size);
				break;
			}
		}

		return model;
	}
}
