// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#pragma once

#include <unordered_map>
#include <vector>

#include "core/config.h"
#include "core/simulation_state.h"
#include "core/world.h"
#include "math/geometry_ops.h"
#include "propagation/math.h"
#include "propagation/propagation_model.h"
#include "radar/receiver.h"

namespace radar
{
	class Radar;
}

namespace propagation::pointscatter
{
	/**
	 * @brief Computes an antenna's polarisation vector, in world space, for a given direction and time.
	 * @param radar The radar object (Transmitter or Receiver).
	 * @param direction The vector pointing away from the antenna towards the target/receiver (need not be unit).
	 * @param time The simulation time for rotation lookup.
	 * @return The antenna's polarisation vector, in world space.
	 */
	[[nodiscard]] Complex3<RealType> computeAntennaPolarisation(const radar::Radar& radar, const math::Vec3& direction,
																RealType time);

	/**
	 * @brief Computes the voltage scaling factor for a reflected path (Bistatic Radar Range Equation), ignoring
	 * antenna polarisation and target scattering matrices (treats the path as perfectly polarisation-matched).
	 * This is also the RCS-scaled base amplitude that `computePolarimetricReflectedLinkage` multiplies into.
	 * @param tx_gain Transmitter gain (linear).
	 * @param rx_gain Receiver gain (linear).
	 * @param rcs Target Radar Cross Section (m^2).
	 * @param lambda Wavelength (meters).
	 * @param r_tx Distance from Transmitter to Target.
	 * @param r_rx Distance from Target to Receiver.
	 * @param no_prop_loss If true, distance-based attenuation is ignored.
	 * @return The amplitude/voltage scaling factor (Vr / Vt). Equivalent to sqrt(Pr / Pt).
	 */
	[[nodiscard]] RealType computeReflectedPathGain(RealType tx_gain, RealType rx_gain, RealType rcs, RealType lambda,
													 RealType r_tx, RealType r_rx, bool no_prop_loss);

	/**
	 * @brief Computes the dimensionless polarimetric coupling factor for a reflected path: how much of
	 * `computeReflectedPathGain`'s amplitude actually couples through given the Tx/Rx antenna polarisations
	 * and the target's scattering matrix. Unit magnitude for a perfectly co-polarised path; does not itself
	 * scale by RCS, since `computeReflectedPathGain` already does.
	 * @param tx Transmitter.
	 * @param rx Receiver.
	 * @param tx_to_tgt Vector from the transmitter to the target.
	 * @param tgt_to_rx Vector from the target to the receiver.
	 * @param r_tx Distance from Transmitter to Target.
	 * @param r_rx Distance from Target to Receiver.
	 * @param t_tx Simulation time for the Transmitter's antenna rotation lookup.
	 * @param t_rx Simulation time for the Receiver's antenna rotation lookup.
	 * @return The dimensionless complex coupling factor.
	 * @pre `tx_to_tgt` and `tgt_to_rx` must not point in exactly the same direction (forward scatter).
	 */
	[[nodiscard]] ComplexType computePolarimetricReflectedLinkage(const radar::Transmitter& tx,
																	const radar::Receiver& rx,
																	const math::Vec3& tx_to_tgt,
																	const math::Vec3& tgt_to_rx, RealType r_tx,
																	RealType r_rx, RealType t_tx, RealType t_rx);

	class PointScatterModel : public PropagationModel
	{
	public:
		PointScatterModel(core::World* world);

		[[nodiscard]] std::vector<PathsAtTime> traceFromTransmitter(ThreadContext* ctx,
															   const radar::Transmitter& transmitter,
															   const std::vector<RealType>& times) const override;

		[[nodiscard]] std::unordered_map<radar::Receiver*, std::vector<PropagationPath>>
		traceFromReceivers(ThreadContext* ctx, const std::vector<radar::Receiver*>& receivers,
						   const std::vector<core::ActiveStreamingSource>& sources, RealType rx_time) const override;

		[[nodiscard]] unsigned int maxPropagationLegs() const noexcept override
		{
			// directPath (1 leg) and bistaticPath (2 legs)
			return 2;
		}

	private:
		core::World* _world;
	};
}
