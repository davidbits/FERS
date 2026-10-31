// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2026-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

#include "pointscatter.h"

#include <cassert>
#include <cmath>
#include <optional>
#include <vector>

#include "core/config.h"
#include "core/logging.h"
#include "core/parameters.h"
#include "core/simulation_state.h"
#include "core/world.h"
#include "math/geometry_ops.h"
#include "propagation/common.h"
#include "propagation/common_defs.h"
#include "propagation/math.h"
#include "propagation/propagation_model.h"
#include "propagation/utils.h"
#include "radar/radar_obj.h"
#include "radar/receiver.h"
#include "radar/transmitter.h"

namespace propagation::pointscatter
{
	/**
	 * @brief Calculates the antenna gain for a specific direction and time.
	 * @param radar The radar object (Transmitter or Receiver).
	 * @param direction_vec The unit vector pointing AWAY from the antenna towards the target/receiver.
	 * @param time The simulation time for rotation lookup.
	 * @param lambda The signal wavelength.
	 * @return The linear gain value.
	 */
	static RealType computeAntennaGain(const radar::Radar& radar, const math::Vec3& direction_vec, RealType time,
									   RealType lambda)
	{
		return radar.getGain(math::SVec3(direction_vec), radar.getRotation(time), lambda);
	}

	Complex3<RealType> computeAntennaPolarisation(const radar::Radar& radar, const math::Vec3& direction, RealType time)
	{
		const auto pol = radar.getAntenna()->getPolarisation();
		const auto h = Complex{pol.horizontal.real(), pol.horizontal.imag()};
		const auto v = Complex{pol.vertical.real(), pol.vertical.imag()};
		const auto fers_orient = radar.getRotation(time);
		const AzEl orientation{fers_orient.azimuth, fers_orient.elevation};
		const RealType inv_len = 1.0 / direction.length();
		const Real3<RealType> world_dir{direction.x * inv_len, direction.y * inv_len, direction.z * inv_len};
		const Real3<RealType> local_dir = rotateWorldToLocal(orientation, world_dir);
		return antennaPolarizationVector(h, v, orientation, local_dir);
	}


	/**
	 * @brief Determine the PropagationPath along a direct tx->rx path
	 * @param time The time at which to find propagation paths. May be at reception or transmission time.
	 * @param tx_time True if time is a transmission time, otherwise it's considered to be the receiving time.
	 */
	static std::optional<PropagationPath> directPath(const radar::Transmitter& tx, radar::Receiver& rx,
													 const RealType time, const bool is_tx_time,
													 const CarrierModel<RealType>& carrier,
													 const RealType segment_start)
	{

		// Calculate the direct path contribution.
		//
		// Handle the co-location cases:
		// 1. If explicitly attached (monostatic), skip (internal leakage handled elsewhere).
		// 2. If independent but on the same platform, distance is 0. Far-field logic (1/R^2)
		//    diverges. We skip calculation, assuming no direct coupling/interference for
		//    co-located far-field antennas.
		if (tx.getPlatform() == rx.getPlatform())
		{
			return std::nullopt;
		}

		const auto p_tx = tx.getPosition(time);
		const auto p_rx = rx.getPosition(time); // Assumption: v_rx << c
		const auto tx_to_rx = p_rx - p_tx;
		const auto tx_to_rx_dist = tx_to_rx.length();

		// Handle co-location of independent TX/RX on distinct platforms.
		// Similar to case 2 above, far-field logic diverges and assume no direct interference
		// for co-located antennas.
		if (tx_to_rx_dist <= EPSILON)
		{
			return std::nullopt;
		}

		const auto delay = tx_to_rx_dist / params::c();
		const RealType tx_time = is_tx_time ? time : time - delay;
		const RealType rx_time = is_tx_time ? time + delay : time;

		const RealType frequency = sampleCarrierFrequency(carrier, tx_time - segment_start);
		const RealType lambda = params::c() / frequency;

		// Tx Gain: Direction Tx -> Rx
		const auto tx_gain = computeAntennaGain(tx, tx_to_rx, tx_time, lambda);
		// Rx Gain: Direction Rx -> Tx
		const auto rx_gain = computeAntennaGain(rx, -tx_to_rx, rx_time, lambda);

		const bool no_loss = rx.checkFlag(radar::Receiver::RecvFlag::FLAG_NOPROPLOSS);
		ComplexType gain = computeDirectPathGain(tx_gain, rx_gain, lambda, tx_to_rx_dist, no_loss);

		if (params::pointScatterParams().polarimetric)
		{
			const auto tx_pol = computeAntennaPolarisation(tx, tx_to_rx, tx_time);
			const auto rx_pol = computeAntennaPolarisation(rx, -tx_to_rx, tx_time);
			const auto linkage = dot_no_conj(tx_pol, rx_pol);

			// TODO_SHAUN remove
			LOG(logging::Level::DEBUG, "Direct Polarimetric Linkage: {}+j{}", linkage.re, linkage.im);

			gain *= ComplexType(linkage.re, linkage.im);
		}

		return PropagationPath{
			.delay = delay,
			.gain = gain,
			.path_id = hash_mix(hash_mix(HASH_SEED, tx.getId()), rx.getId()),
			.source_index{},
			.receiver = &rx,
		};
	}

	RealType computeReflectedPathGain(RealType tx_gain, RealType rx_gain, RealType rcs, RealType lambda, RealType r_tx,
									  RealType r_rx, bool no_prop_loss)
	{
		const RealType numerator = std::sqrt(tx_gain * rx_gain * rcs) * lambda;
		RealType denominator = std::sqrt(64.0 * PI * PI * PI); // (4 * PI)^3/2

		if (!no_prop_loss)
		{
			denominator *= r_tx * r_rx;
		}

		return numerator / denominator;
	}

	// Computes the dimensionless polarimetric coupling factor for a reflected path: how much of
	// computeReflectedPathGain's (polarisation-independent, RCS-scaled) amplitude actually couples
	// through given the Tx/Rx antenna polarisations and the target's scattering matrix. Unit
	// magnitude for a perfectly co-polarised path; must not itself scale by RCS, since
	// computeReflectedPathGain already does.
	ComplexType computePolarimetricReflectedLinkage(const radar::Transmitter& tx, const radar::Receiver& rx,
													const math::Vec3& tx_to_tgt, const math::Vec3& tgt_to_rx,
													RealType r_tx, RealType r_rx, RealType t_tx, RealType t_rx)
	{

		const auto tx_pol = computeAntennaPolarisation(tx, tx_to_tgt, t_tx);
		const auto rx_pol = computeAntennaPolarisation(rx, -tgt_to_rx, t_rx);

		const auto k_in = Real3<RealType>{tx_to_tgt.x, tx_to_tgt.y, tx_to_tgt.z} / r_tx;
		const auto k_scat = Real3<RealType>{tgt_to_rx.x, tgt_to_rx.y, tgt_to_rx.z} / r_rx;

		// The facet normal bisects the incidence and scatter directions.
		const auto bisector_sum = -k_in + k_scat;
		const auto bisector_len = length(bisector_sum);
		// Assert against the k_in ~= k_scat case, aka forward scatter.
		// This function contractually doesn't handle this case. No reflection occurs.
		assert(bisector_len > RealType(1e-9));
		const auto bisection = bisector_sum / bisector_len;

		const auto basis = incidencePlaneBasis(bisection, k_in);

		const auto e_te = dot(tx_pol, basis.e_te);
		const auto e_tm = dot(tx_pol, basis.e_tm_in);

		// Assume the FSA Jones scattering matrix for a PEC for now, normalised to unit magnitude -
		// computeReflectedPathGain supplies the RCS scaling.
		// TODO add polarimetric scattering matrices to fersxml targets.
		const auto scatter = Real2x2<RealType>{{-1, 0}, {0, 1}};
		const auto e_scat = scatter * Complex2{e_te, e_tm};

		const auto e_tm_scat = cross(basis.e_te, k_scat);
		const auto refl = basis.e_te * e_scat.x + e_tm_scat * e_scat.y;
		const auto linkage = dot_no_conj(refl, rx_pol);

		return ComplexType{linkage.re, linkage.im};
	}

	static std::optional<PropagationPath> reflectedPath(const radar::Transmitter& tx, radar::Receiver& rx,
														const radar::Target& target, const RealType time,
														const bool is_tx_time, const CarrierModel<RealType>& carrier,
														const RealType segment_start)
	{

		// If calculating reflected path and target is co-located with either Tx or Rx:
		// Skip to avoid singularity. Simulating a radar tracking its own platform
		// requires near-field clutter models, not point-target RCS models.
		if (target.getPlatform() == tx.getPlatform() || target.getPlatform() == rx.getPlatform())
		{
			return std::nullopt;
		}

		const auto p_tx = tx.getPosition(time);
		const auto p_tgt = target.getPosition(time);
		const auto p_rx = rx.getPosition(time);
		const auto tx_to_tgt = p_tgt - p_tx;
		const auto tx_to_tgt_dist = tx_to_tgt.length();
		const auto tgt_to_rx = p_rx - p_tgt;
		const auto tgt_to_rx_dist = tgt_to_rx.length();

		// Handle co-location of independent TX/TGT/RX on distinct platforms.
		// Similar to the above case, skip to avoid a signularity.
		if (tx_to_tgt_dist <= EPSILON || tgt_to_rx_dist <= EPSILON)
		{
			LOG(logging::Level::TRACE,
				"Skipping reflected path calculation for Target {} co-located with Transmitter {} or Receiver {}",
				target.getName(), tx.getName(), rx.getName());

			return std::nullopt;
		}

		const auto path_length = tx_to_tgt_dist + tgt_to_rx_dist;
		const auto delay = path_length / params::c();
		const RealType tx_time = is_tx_time ? time : time - delay;
		const RealType rx_time = is_tx_time ? time + delay : time;
		const auto target_delay = tx_to_tgt_dist / params::c();
		const RealType tgt_time = tx_time + target_delay;

		const RealType frequency = sampleCarrierFrequency(carrier, tx_time - segment_start);
		const RealType lambda = params::c() / frequency;

		// Calculate RCS
		// InAngle: Tx -> Tgt (tx_to_tgt)
		// OutAngle: Rx -> Tgt (Opposite of Tgt->Rx, so -tgt_to_rx)
		math::SVec3 in_angle(tx_to_tgt);
		math::SVec3 out_angle(-tgt_to_rx);
		const auto rcs_opt = target.getRcs(in_angle, out_angle, tgt_time);
		if (!rcs_opt)
		{
			return std::nullopt;
		}
		const RealType rcs = *rcs_opt;

		// Tx Gain: Direction Tx -> Tgt
		const auto tx_gain = computeAntennaGain(tx, tx_to_tgt, tx_time, lambda);
		// Rx Gain: Direction Rx -> Tgt
		const auto rx_gain = computeAntennaGain(rx, -tgt_to_rx, rx_time, lambda);

		const bool no_prop_loss = rx.checkFlag(radar::Receiver::RecvFlag::FLAG_NOPROPLOSS);
		ComplexType gain =
			computeReflectedPathGain(tx_gain, rx_gain, rcs, lambda, tx_to_tgt_dist, tgt_to_rx_dist, no_prop_loss);

		if (params::pointScatterParams().polarimetric)
		{
			// Skip near-exact forward scatter (Rx directly behind the target as seen from Tx).
			// This isn't handled accurately by specular reflection physics, it's dominated by diffraction
			// and other effects. The direct path is already computed separately, this skips "reflecting"
			// off of a in-between target for now.
			// This special case only applies to the polarimetric scattering physics below; it must not be
			// applied to the non-polarimetric (incoherent) reflected-path gain.
			if (math::dotProduct(tx_to_tgt / tx_to_tgt_dist, tgt_to_rx / tgt_to_rx_dist) > 0.995)
			{
				LOG(logging::Level::TRACE,
					"Skipping reflected path calculation for Target between Tx {} and Rx {}. Rx is in the "
					"forward-scatter direction from Tx through the target, which is not handled as a bistatic "
					"reflection path for RCS point targets",
					tx.getName(), rx.getName());

				return std::nullopt;
			}

			const ComplexType linkage = computePolarimetricReflectedLinkage(
				tx, rx, tx_to_tgt, tgt_to_rx, tx_to_tgt_dist, tgt_to_rx_dist, tx_time, rx_time);

			gain *= linkage;
		}

		return PropagationPath{
			.delay = delay,
			.gain = gain,
			.path_id = hash_mix(hash_mix(hash_mix(HASH_SEED, tx.getId()), target.getId()), rx.getId()),
			.source_index{},
			.receiver = &rx,
		};
	}

	std::vector<PathsAtTime> PointScatterModel::traceFromTransmitter(ThreadContext* /* ctx */,
																const radar::Transmitter& transmitter,
																const std::vector<RealType>& times) const
	{
		CarrierModel<RealType> carrier;
		carrier.kind = CarrierModelKind::Constant;
		carrier.params.constant.frequency = transmitter.getSignal()->getCarrier();

		std::vector<PathsAtTime> timesteps;

		for (const auto current_time : times)
		{
			std::vector<PropagationPath> paths;

			for (const auto& receiver : _world->getReceivers())
			{
				// Note, we're not setting the PropagationPath source index as the caller of traceFromTransmitter
				// only gives us a Transmitter, not an ActiveStreamingSource.
				// traceFromTransmitter is used for pulsed radar instead of continuous, so this works.

				if (!receiver->checkFlag(radar::Receiver::RecvFlag::FLAG_NODIRECT))
				{
					// Calculate the direct path contribution, if any.
					const auto opt = directPath(transmitter, *receiver, current_time, true, carrier, current_time);

					if (opt != std::nullopt)
					{
						auto path = opt.value();
						paths.push_back(path);
					}
				}

				for (const auto& target : _world->getTargets())
				{
					if (target->getRcsSpec() == nullptr)
					{
						continue;
					}

					// Calculate bistatic reflection contribution, if any.
					const auto opt =
						reflectedPath(transmitter, *receiver, *target, current_time, true, carrier, current_time);

					if (opt != std::nullopt)
					{
						auto path = opt.value();
						paths.push_back(path);
					}
				}
			}

			timesteps.push_back({paths, current_time});
		}

		return timesteps;
	};

	std::unordered_map<radar::Receiver*, std::vector<PropagationPath>>
	PointScatterModel::traceFromReceivers(ThreadContext* /* ctx */, const std::vector<radar::Receiver*>& receivers,
										  const std::vector<core::ActiveStreamingSource>& sources,
										  RealType rx_time) const
	{
		std::unordered_map<radar::Receiver*, std::vector<PropagationPath>> paths_by_receiver;

		for (const auto& receiver : receivers)
		{
			auto& paths = paths_by_receiver[receiver]; // Ensures an entry even if no paths are found.

			for (size_t s = 0; s < sources.size(); s++)
			{
				const auto& source = sources[s];
				const auto carrier = buildPropagationCarrier<RealType>(source);

				if (!receiver->checkFlag(radar::Receiver::RecvFlag::FLAG_NODIRECT))
				{
					// Calculate the direct path contribution, if any.
					const auto opt =
						directPath(*source.transmitter, *receiver, rx_time, false, carrier, source.segment_start);

					if (opt != std::nullopt)
					{
						auto path = opt.value();
						path.source_index = s;
						paths.push_back(path);
					}
				}

				for (const auto& target : _world->getTargets())
				{
					if (target->getRcsSpec() == nullptr)
					{
						continue;
					}

					// Calculate bistatic reflection contribution, if any.
					const auto opt = reflectedPath(*source.transmitter, *receiver, *target, rx_time, false, carrier,
												   source.segment_start);

					if (opt != std::nullopt)
					{
						auto path = opt.value();
						path.source_index = s;
						paths.push_back(path);
					}
				}
			}
		}

		return paths_by_receiver;
	}

	PointScatterModel::PointScatterModel(core::World* world) : _world(world) {}
}
