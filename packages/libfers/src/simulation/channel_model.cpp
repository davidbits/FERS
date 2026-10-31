// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2006-2008 Marc Brooker and Michael Inggs
// Copyright (c) 2008-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

/**
 * @file channel_model.cpp
 * @brief Implementation of radar channel propagation and interaction models.
 *
 * This file provides the implementations for the functions that model the radar
 * channel, as declared in channel_model.h. It contains the core physics calculations
 * that determine signal properties based on geometry, velocity, and object characteristics.
 */

#include "channel_model.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/config.h"
#include "core/logging.h"
#include "core/parameters.h"
#include "core/world.h"
#include "interpolation/interpolation_point.h"
#include "math/geometry_ops.h"
#include "propagation/common.h"
#include "propagation/pointscatter/pointscatter.h"
#include "propagation/propagation_model.h"
#include "radar/radar_obj.h"
#include "radar/receiver.h"
#include "radar/target.h"
#include "radar/transmitter.h"
#include "serial/response.h"
#include "signal/radar_signal.h"
#include "timing/timing.h"

using fers_signal::RadarSignal;
using logging::Level;
using math::SVec3;
using math::Vec3;
using radar::Receiver;
using radar::Target;
using radar::Transmitter;

namespace
{
	/// Initializes an FMCW chirp tracker from a retarded time.
	void initializeFmcwTracker(const core::ActiveStreamingSource& source, const RealType t_ret,
							   core::FmcwChirpBoundaryTracker& tracker)
	{
		tracker.initialized = true;
		if (t_ret >= source.segment_start)
		{
			const RealType time_since_segment_start = t_ret - source.segment_start;
			tracker.n_current = static_cast<std::size_t>(std::floor(time_since_segment_start / source.chirp_period));
			tracker.t_n = source.segment_start + static_cast<RealType>(tracker.n_current) * source.chirp_period;
			return;
		}

		tracker.n_current = 0;
		tracker.t_n = source.segment_start;
	}

	/// Initializes an FMCW triangle tracker from a retarded time.
	void initializeFmcwTriangleTracker(const core::ActiveStreamingSource& source, const RealType t_ret,
									   core::FmcwChirpBoundaryTracker& tracker)
	{
		tracker.triangle_initialized = true;
		if (t_ret < source.segment_start)
		{
			tracker.triangle_index = 0;
			tracker.triangle_leg = 0;
			tracker.triangle_t_leg = source.segment_start;
			tracker.triangle_phi_base = 0.0;
			return;
		}

		const RealType delta = t_ret - source.segment_start;
		tracker.triangle_index = static_cast<std::size_t>(std::floor(delta / source.triangle_period));
		const RealType local_triangle_time =
			delta - static_cast<RealType>(tracker.triangle_index) * source.triangle_period;
		tracker.triangle_leg = local_triangle_time < source.chirp_duration ? 0U : 1U;
		tracker.triangle_t_leg = source.segment_start +
			static_cast<RealType>(tracker.triangle_index) * source.triangle_period +
			(tracker.triangle_leg == 1U ? source.chirp_duration : 0.0);
		tracker.triangle_phi_base =
			std::fmod(static_cast<RealType>(tracker.triangle_index) * source.mod_phi_tri, 2.0 * PI) +
			(tracker.triangle_leg == 1U ? source.mod_phi_up : 0.0);
		if (tracker.triangle_phi_base >= 2.0 * PI)
		{
			tracker.triangle_phi_base -= 2.0 * PI;
		}
		if (tracker.triangle_phi_base < 0.0)
		{
			tracker.triangle_phi_base += 2.0 * PI;
		}
	}

	struct StreamingWaveformEvaluation
	{
		RealType phase = 0.0;
		ComplexType envelope{1.0, 0.0};
	};

	/**
	 * @brief Computes the non-coherent phase shift due to timing offsets.
	 *
	 * Used for CW simulation where LO effects are modeled analytically.
	 *
	 * @param tx The transmitter.
	 * @param rx The receiver.
	 * @param time The current simulation time.
	 * @return The phase shift in radians.
	 */
	RealType computeTimingPhase(const Transmitter* tx, const Receiver* rx, RealType time)
	{
		const auto tx_timing = tx->getTiming();
		const auto rx_timing = rx->getTiming();
		const RealType delta_f = tx_timing->getFreqOffset() - rx_timing->getFreqOffset();
		const RealType delta_phi = tx_timing->getPhaseOffset() - rx_timing->getPhaseOffset();
		return 2 * PI * delta_f * time + delta_phi;
	}

	/// Computes deterministic timing phase for one timing source when no lookup is available.
	RealType computeSingleTimingPhase(const timing::Timing* const timing, const RealType time)
	{
		if (timing == nullptr)
		{
			return 0.0;
		}
		return 2.0 * PI * timing->getFreqOffset() * time + timing->getPhaseOffset();
	}

	/// Computes timing phase with an optional lookup and streaming phase-application mode.
	RealType computeTimingPhase(const Transmitter* tx, const Receiver* rx, const RealType rx_time,
								const RealType tx_time, const simulation::CwPhaseNoiseLookup* phase_noise_lookup,
								const simulation::StreamingTimingPhaseMode mode)
	{
		if (mode == simulation::StreamingTimingPhaseMode::None)
		{
			return 0.0;
		}
		if (mode == simulation::StreamingTimingPhaseMode::TransmitterOnly)
		{
			if (phase_noise_lookup == nullptr)
			{
				return computeSingleTimingPhase(tx->getTiming().get(), tx_time);
			}
			return phase_noise_lookup->sample(tx->getTiming().get(), tx_time);
		}
		if (phase_noise_lookup == nullptr)
		{
			return computeTimingPhase(tx, rx, rx_time);
		}
		return phase_noise_lookup->phaseDifference(rx->getTiming().get(), rx_time, tx->getTiming().get(), tx_time);
	}

	/// Checks whether received power is above the thermal noise floor.
	bool isSignalStrong(RealType power_watts, RealType temp_kelvin)
	{
		// Use configured rate or default to 1Hz if unconfigured to prevent divide-by-zero or silly values
		const RealType bw = params::rate() > 0 ? params::rate() : 1.0;
		const RealType noise_floor = params::boltzmannK() * (temp_kelvin > 0 ? temp_kelvin : 290.0) * bw;
		return power_watts > noise_floor;
	}

	/**
	 * @brief Converts power in watts to decibels milliwatts (dBm).
	 *
	 * @param watts Power in watts.
	 * @return Power in dBm. Returns -999.0 dBm for non-positive input.
	 */
	RealType wattsToDbm(RealType watts)
	{
		if (watts <= 0)
		{
			return -999.0;
		}
		return 10.0 * std::log10(watts * 1000.0);
	}

	/**
	 * @brief Converts power in watts to decibels (dB).
	 *
	 * @param watts Power in watts.
	 * @return Power in decibels (dB). Returns -999.0 dB for non-positive input.
	 */
	RealType wattsToDb(RealType watts)
	{
		if (watts <= 0)
		{
			return -999.0;
		}
		return 10.0 * std::log10(watts);
	}

	/**
	 * @brief Checks if a component is active at the given time based on its schedule.
	 *
	 * @param schedule The component's operating schedule.
	 * @param time The current simulation time.
	 * @return true If the schedule is empty (implied always on) or if time is within a period.
	 * @return false If the schedule is populated but the time is outside all periods.
	 */
	bool isComponentActive(const std::vector<radar::SchedulePeriod>& schedule, RealType time)
	{
		if (schedule.empty())
		{
			return true;
		}
		for (const auto& period : schedule)
		{
			if (time >= period.start && time <= period.end)
			{
				return true;
			}
		}
		return false;
	}

	bool computeLinearFmcwPhaseWithoutTracker(const core::ActiveStreamingSource& source, const RealType t_ret,
											  const RealType tau, RealType& phase_out)
	{
		const RealType time_since_segment_start = t_ret - source.segment_start;
		const auto chirp_index = static_cast<std::size_t>(std::floor(time_since_segment_start / source.chirp_period));
		if (source.chirp_count.has_value() && chirp_index >= *source.chirp_count)
		{
			return false;
		}
		const RealType chirp_time = time_since_segment_start - static_cast<RealType>(chirp_index) * source.chirp_period;
		if (chirp_time < 0.0 || chirp_time >= source.chirp_duration)
		{
			return false;
		}
		phase_out = -2.0 * PI * source.carrier_freq * tau + source.two_pi_f0 * chirp_time +
			source.s_pi_alpha * chirp_time * chirp_time;
		return true;
	}

	bool computeLinearFmcwPhaseWithTracker(const core::ActiveStreamingSource& source, const RealType t_ret,
										   const RealType tau, core::FmcwChirpBoundaryTracker& chirp_tracker,
										   RealType& phase_out)
	{
		if (!chirp_tracker.initialized)
		{
			initializeFmcwTracker(source, t_ret, chirp_tracker);
		}

		while (t_ret >= chirp_tracker.t_n + source.chirp_period)
		{
			chirp_tracker.t_n += source.chirp_period;
			++chirp_tracker.n_current;
		}

		if (source.chirp_count.has_value() && chirp_tracker.n_current >= *source.chirp_count)
		{
			return false;
		}

		const RealType u_ret = t_ret - chirp_tracker.t_n;
		if (u_ret < 0.0 || u_ret >= source.chirp_duration)
		{
			return false;
		}

		phase_out =
			-2.0 * PI * source.carrier_freq * tau + source.two_pi_f0 * u_ret + source.s_pi_alpha * u_ret * u_ret;
		return true;
	}

	bool computeTriangleFmcwPhaseWithoutTracker(const core::ActiveStreamingSource& source, const RealType t_ret,
												const RealType tau, RealType& phase_out)
	{
		const RealType delta = t_ret - source.segment_start;
		const auto triangle_index = static_cast<std::size_t>(std::floor(delta / source.triangle_period));
		if (source.triangle_count.has_value() && triangle_index >= *source.triangle_count)
		{
			return false;
		}
		const RealType local_triangle_time = delta - static_cast<RealType>(triangle_index) * source.triangle_period;
		const bool down_leg = local_triangle_time >= source.chirp_duration;
		const RealType u_ret = down_leg ? local_triangle_time - source.chirp_duration : local_triangle_time;
		if (u_ret < 0.0 || u_ret >= source.chirp_duration)
		{
			return false;
		}
		const RealType phi_base = std::fmod(static_cast<RealType>(triangle_index) * source.mod_phi_tri, 2.0 * PI) +
			(down_leg ? source.mod_phi_up : 0.0);
		const RealType modular_phi_base = phi_base >= 2.0 * PI ? phi_base - 2.0 * PI : phi_base;
		const RealType linear_coeff = down_leg ? source.two_pi_f0_plus_B : source.two_pi_f0;
		const RealType quad_coeff = down_leg ? source.neg_pi_alpha : source.pi_alpha;
		phase_out = -2.0 * PI * source.carrier_freq * tau + modular_phi_base + linear_coeff * u_ret +
			quad_coeff * u_ret * u_ret;
		return true;
	}

	void advanceTriangleTracker(const core::ActiveStreamingSource& source, const RealType t_ret,
								core::FmcwChirpBoundaryTracker& chirp_tracker)
	{
		while (t_ret >= chirp_tracker.triangle_t_leg + source.chirp_duration)
		{
			chirp_tracker.triangle_t_leg += source.chirp_duration;
			chirp_tracker.triangle_leg = 1U - chirp_tracker.triangle_leg;
			if (chirp_tracker.triangle_leg == 0U)
			{
				++chirp_tracker.triangle_index;
			}
			chirp_tracker.triangle_phi_base += source.mod_phi_up;
			if (chirp_tracker.triangle_phi_base >= 2.0 * PI)
			{
				chirp_tracker.triangle_phi_base -= 2.0 * PI;
			}
		}
	}

	bool computeTriangleFmcwPhaseWithTracker(const core::ActiveStreamingSource& source, const RealType t_ret,
											 const RealType tau, core::FmcwChirpBoundaryTracker& chirp_tracker,
											 RealType& phase_out)
	{
		if (!chirp_tracker.triangle_initialized)
		{
			initializeFmcwTriangleTracker(source, t_ret, chirp_tracker);
		}

		advanceTriangleTracker(source, t_ret, chirp_tracker);
		if (source.triangle_count.has_value() && chirp_tracker.triangle_index >= *source.triangle_count)
		{
			return false;
		}

		const RealType u_ret = t_ret - chirp_tracker.triangle_t_leg;
		if (u_ret < 0.0 || u_ret >= source.chirp_duration)
		{
			return false;
		}

		const bool down_leg = chirp_tracker.triangle_leg == 1U;
		const RealType linear_coeff = down_leg ? source.two_pi_f0_plus_B : source.two_pi_f0;
		const RealType quad_coeff = down_leg ? source.neg_pi_alpha : source.pi_alpha;
		phase_out = -2.0 * PI * source.carrier_freq * tau + chirp_tracker.triangle_phi_base + linear_coeff * u_ret +
			quad_coeff * u_ret * u_ret;
		return true;
	}

	/// Computes streaming waveform phase at a receiver time.
	bool computeStreamingEvaluation(const core::ActiveStreamingSource& source, const RealType rx_time,
									const RealType tau, core::FmcwChirpBoundaryTracker* const chirp_tracker,
									StreamingWaveformEvaluation& eval)
	{
		if (source.carrier_freq <= 0.0)
		{
			return false;
		}

		const RealType t_ret = rx_time - tau;
		if (t_ret < source.segment_start || t_ret >= source.segment_end)
		{
			return false;
		}

		if (source.kind == core::StreamingWaveformKind::FmcwLinear)
		{
			if (chirp_tracker == nullptr)
			{
				return computeLinearFmcwPhaseWithoutTracker(source, t_ret, tau, eval.phase);
			}
			return computeLinearFmcwPhaseWithTracker(source, t_ret, tau, *chirp_tracker, eval.phase);
		}

		if (source.kind == core::StreamingWaveformKind::FmcwTriangle)
		{
			if (chirp_tracker == nullptr)
			{
				return computeTriangleFmcwPhaseWithoutTracker(source, t_ret, tau, eval.phase);
			}
			return computeTriangleFmcwPhaseWithTracker(source, t_ret, tau, *chirp_tracker, eval.phase);
		}

		if (source.kind == core::StreamingWaveformKind::Sfcw)
		{
			if (source.sfcw == nullptr)
			{
				return false;
			}
			const auto step = source.sfcw->activeStepAt(t_ret - source.segment_start, source.carrier_freq);
			if (!step.has_value() || step->rf_frequency <= 0.0)
			{
				return false;
			}
			eval.phase = -2.0 * PI * step->rf_frequency * tau;
			return true;
		}

		if (source.kind == core::StreamingWaveformKind::FileCw || source.kind == core::StreamingWaveformKind::FileFmcw)
		{
			if (source.file == nullptr || source.file_duration <= 0.0)
			{
				return false;
			}
			eval.phase = -2.0 * PI * source.carrier_freq * tau;
			eval.envelope = source.file->sampleAt(t_ret - source.segment_start);
			return true;
		}

		eval.phase = -2.0 * PI * source.carrier_freq * tau;
		return true;
	}

	/// Returns average radiated power for preview visualization.
	RealType previewRadiatedPower(const RadarSignal* waveform)
	{
		if (waveform == nullptr)
		{
			return 0.0;
		}
		if (const auto* fmcw = waveform->getFmcwChirpWaveform(); fmcw != nullptr)
		{
			return waveform->getPower() * (fmcw->getChirpDuration() / fmcw->getChirpPeriod());
		}
		if (waveform->isFmcwTriangle())
		{
			return waveform->getPower();
		}
		return waveform->getPower();
	}

	/// Formats received or radiated power as a dBm preview label.
	std::string formatPreviewDbmLabel(const RealType watts, const std::string_view prefix = {})
	{
		return std::format("{}{:.1f} dBm", prefix, wattsToDbm(watts));
	}

	/// Formats target illumination density as a dBW-per-square-meter preview label.
	std::string formatPreviewDbwPerSquareMeterLabel(const RealType watts)
	{
		return std::format("{:.1f} dBW/m\u00B2", wattsToDb(watts));
	}
}

namespace simulation
{
	RealType CwPhaseNoiseBuffer::sampleAt(const RealType time) const noexcept
	{
		if (samples.empty())
		{
			return 0.0;
		}
		if ((time <= start_time) || (samples.size() == 1) || (dt <= 0.0))
		{
			return samples.front();
		}

		const RealType position = (time - start_time) / dt;
		const auto last_index = static_cast<RealType>(samples.size() - 1);
		if (position >= last_index)
		{
			return samples.back();
		}

		const auto lower_index = static_cast<std::size_t>(position);
		const RealType fraction = position - static_cast<RealType>(lower_index);
		return samples[lower_index] + fraction * (samples[lower_index + 1] - samples[lower_index]);
	}

	CwPhaseNoiseLookup CwPhaseNoiseLookup::build(const std::span<const std::shared_ptr<timing::Timing>> timings,
												 const RealType start_time, const RealType end_time)
	{
		CwPhaseNoiseLookup lookup{};
		lookup.start_time = start_time;
		lookup.end_time = std::max(start_time, end_time);

		const RealType sample_rate = params::rate() * params::oversampleRatio();
		lookup.dt = sample_rate > 0.0 ? (1.0 / sample_rate) : 1.0;

		const auto sample_count =
			static_cast<std::size_t>(std::ceil((lookup.end_time - lookup.start_time) / lookup.dt) + 1.0);
		constexpr std::size_t phase_noise_warning_threshold_bytes = 500ULL * 1024ULL * 1024ULL;
		const auto bytes_per_buffer = sample_count * sizeof(RealType);

		for (const auto& timing : timings)
		{
			if (!timing || lookup.buffers.contains(timing->getId()))
			{
				continue;
			}

			CwPhaseNoiseBuffer buffer{};
			buffer.start_time = lookup.start_time;
			buffer.dt = lookup.dt;
			if (timing->isEnabled())
			{
				auto timing_clone = timing->clone();
				if (lookup.start_time > 0.0)
				{
					const auto skip_count = static_cast<std::size_t>(std::llround(lookup.start_time / lookup.dt));
					timing_clone->skipSamples(skip_count);
				}
				// TODO: Replace whole-simulation CW lookup generation with chunked/streaming generation.
				if (bytes_per_buffer > phase_noise_warning_threshold_bytes)
				{
					LOG(Level::WARNING,
						"CW phase-noise lookup for timing '{}' allocates {} bytes; large scenarios need chunked "
						"streaming.",
						timing->getName(), bytes_per_buffer);
				}
				buffer.samples.resize(sample_count);
				std::ranges::generate(buffer.samples, [&] { return timing_clone->getNextSample(); });
			}

			lookup.buffers.emplace(timing->getId(), std::move(buffer));
		}

		return lookup;
	}

	RealType CwPhaseNoiseLookup::sample(const timing::Timing* const timing, const RealType time) const noexcept
	{
		if (timing == nullptr)
		{
			return 0.0;
		}
		const auto it = buffers.find(timing->getId());
		if (it == buffers.end())
		{
			return 0.0;
		}
		return it->second.sampleAt(time);
	}

	RealType CwPhaseNoiseLookup::phaseDifference(const timing::Timing* const rx_timing, const RealType rx_time,
												 const timing::Timing* const tx_timing,
												 const RealType tx_time) const noexcept
	{
		return sample(tx_timing, tx_time) - sample(rx_timing, rx_time);
	}

	void updateCoarseStreamingCache(core::ReceiverTrackerCache& cache,
									const std::vector<propagation::PropagationPath>& next_paths,
									const RealType interval_start_time, const RealType sample_time)
	{
		for (const auto& path : next_paths)
		{
			auto& per_source = cache.coarse_path_samples[path.source_index];
			const RealType gain_mag = std::sqrt(std::norm(path.gain));
			const RealType raw_phase = std::arg(path.gain);

			const auto [it, inserted] = per_source.try_emplace(path.path_id);
			auto& entry = it->second;

			RealType hi_phase = raw_phase;
			if (inserted)
			{
				// New path found.
				// Only one sample to extrapolate from, so add a zero-gain copy.
				// This mirrors serial::Response's simpleTaperPoint fallback.
				entry.t_lo = interval_start_time;
				entry.delay_lo = path.delay;
				entry.gain_mag_lo = 0.0;
				entry.gain_phase_lo = raw_phase;
			}
			else
			{
				// Unwrap the fresh phase relative to the previous "hi" so interpolation doesn't see
				// a +/-2pi jump across samples.
				hi_phase = raw_phase + 2.0 * PI * std::round((entry.gain_phase_hi - raw_phase) / (2.0 * PI));

				entry.t_lo = entry.t_hi;
				entry.delay_lo = entry.delay_hi;
				entry.gain_mag_lo = entry.gain_mag_hi;
				entry.gain_phase_lo = entry.gain_phase_hi;
			}

			entry.t_hi = sample_time;
			entry.delay_hi = path.delay;
			entry.gain_mag_hi = gain_mag;
			entry.gain_phase_hi = hi_phase;
		}

		// Every path touched above just had its `t_hi` set to `sample_time`. Anything still holding
		// an older value wasn't in `next_paths` this round, so it's either tapering out (its first
		// missed round: extrapolate a zero-gain "hi" from its own existing lo/hi, mirroring
		// serial::Response's extrapolateTaperPoint, computed directly in rx-time since this grid has
		// no Doppler-induced tx-time/rx-time gap to reconcile) or already fully tapered (drop it).
		for (auto& per_source : cache.coarse_path_samples)
		{
			for (auto it = per_source.begin(); it != per_source.end();)
			{
				auto& entry = it->second;
				if (entry.t_hi == sample_time)
				{
					++it;
					continue;
				}

				if (entry.gain_mag_hi == 0.0)
				{
					it = per_source.erase(it);
					continue;
				}

				const RealType dt = entry.t_hi - entry.t_lo;
				RealType new_delay = entry.delay_hi;
				RealType new_phase = entry.gain_phase_hi;
				if (dt != 0.0)
				{
					const RealType extend = sample_time - entry.t_hi;
					new_delay = entry.delay_hi + (entry.delay_hi - entry.delay_lo) / dt * extend;
					new_phase = entry.gain_phase_hi + (entry.gain_phase_hi - entry.gain_phase_lo) / dt * extend;
				}

				entry.t_lo = entry.t_hi;
				entry.delay_lo = entry.delay_hi;
				entry.gain_mag_lo = entry.gain_mag_hi;
				entry.gain_phase_lo = entry.gain_phase_hi;

				entry.t_hi = sample_time;
				entry.delay_hi = new_delay;
				entry.gain_mag_hi = 0.0;
				entry.gain_phase_hi = new_phase;

				++it;
			}
		}
	}

	std::vector<propagation::PropagationPath> interpolateCoarseStreamingPaths(const core::ReceiverTrackerCache& cache,
																			  radar::Receiver& receiver,
																			  const RealType rx_time)
	{
		std::vector<propagation::PropagationPath> paths;

		for (std::size_t source_index = 0; source_index < cache.coarse_path_samples.size(); ++source_index)
		{
			for (const auto& [path_id, entry] : cache.coarse_path_samples[source_index])
			{
				RealType delay = entry.delay_hi;
				RealType gain_mag = entry.gain_mag_hi;
				RealType gain_phase = entry.gain_phase_hi;

				if (entry.t_hi != entry.t_lo)
				{
					const RealType frac = (rx_time - entry.t_lo) / (entry.t_hi - entry.t_lo);
					delay = std::lerp(entry.delay_lo, entry.delay_hi, frac);
					gain_mag = std::lerp(entry.gain_mag_lo, entry.gain_mag_hi, frac);
					gain_phase = std::lerp(entry.gain_phase_lo, entry.gain_phase_hi, frac);
				}

				paths.push_back(propagation::PropagationPath{
					.delay = delay,
					.gain = std::polar(gain_mag, gain_phase),
					.path_id = path_id,
					.source_index = source_index,
					.receiver = &receiver,
				});
			}
		}

		return paths;
	}

	bool calculateStreamingReferencePhase(const core::ActiveStreamingSource& source, const RealType timeK,
										  core::FmcwChirpBoundaryTracker* const chirp_tracker, RealType& phase_out)
	{
		StreamingWaveformEvaluation eval;
		if (!computeStreamingEvaluation(source, timeK, 0.0, chirp_tracker, eval))
		{
			return false;
		}
		phase_out = eval.phase;
		if (std::abs(eval.envelope) > 0.0)
		{
			phase_out += std::arg(eval.envelope);
		}
		return true;
	}

	bool calculateStreamingReferenceSample(const core::ActiveStreamingSource& source, const RealType timeK,
										   core::FmcwChirpBoundaryTracker* const chirp_tracker, ComplexType& sample_out)
	{
		StreamingWaveformEvaluation eval;
		if (!computeStreamingEvaluation(source, timeK, 0.0, chirp_tracker, eval))
		{
			return false;
		}
		sample_out = eval.envelope * std::polar(1.0, eval.phase);
		return true;
	}

	ComplexType calculateStreamingPathContribution(const core::ActiveStreamingSource& source, const Receiver* recv,
												   const propagation::PropagationPath& path, const RealType timeK,
												   const CwPhaseNoiseLookup* const phase_noise_lookup,
												   core::FmcwChirpBoundaryTracker* const chirp_tracker,
												   const StreamingTimingPhaseMode timing_phase_mode)
	{
		const auto* const trans = source.transmitter;
		if (trans == nullptr)
		{
			return {0.0, 0.0};
		}

		StreamingWaveformEvaluation eval;
		if (!computeStreamingEvaluation(source, timeK, path.delay, chirp_tracker, eval))
		{
			return {0.0, 0.0};
		}


		// Accumulate amplitude and phase effects
		ComplexType contribution = source.amplitude * path.gain * eval.envelope * std::polar(1.0, eval.phase);

		// Non-coherent Local Oscillator Effects
		const RealType non_coherent_phase =
			computeTimingPhase(trans, recv, timeK, timeK - path.delay, phase_noise_lookup, timing_phase_mode);
		contribution *= std::polar(1.0, non_coherent_phase);

		return contribution;
	}


	/**
	 * @brief Routes a calculated radar response to the appropriate receiver inbox or log.
	 * @param rx Pointer to the receiving radar object.
	 * @param response The calculated response to route.
	 */
	static void routeResponse(Receiver* rx, std::unique_ptr<serial::Response> response)
	{
		if (!response)
		{
			return;
		}
		if (rx->getMode() == radar::OperationMode::PULSED_MODE)
		{
			rx->addResponseToInbox(std::move(response));
		}
		else
		{
			rx->addInterferenceToLog(std::move(response));
		}
	}

	void calculateResponses(const radar::Transmitter& tx, const RealType start_tx_time,
							const propagation::PropagationModel& prop, propagation::ThreadContext* ctx)
	{
		// Skip if there's no signal associated with the transmitter or the signal has no length.
		if (tx.getSignal() == nullptr || !tx.getSignal()->isPulsed() ||
			tx.getSignal()->getPulseWaveform()->getDuration() == 0)
		{
			return;
		}

		const auto tx_duration = tx.getSignal()->getPulseWaveform()->getDuration();
		const auto end_tx_time = start_tx_time + tx_duration;
		const auto sample_period = 1.0 / params::simSamplingRate();
		const auto point_count = static_cast<uint32_t>(std::ceil(tx_duration * params::simSamplingRate()));
		std::vector<RealType> times(point_count + 1);
		for (uint32_t i = 0; i < point_count; ++i)
		{
			times[i] = start_tx_time + i * sample_period;
		}
		times[point_count] = end_tx_time;


		// This internally computes the paths for each timestep.
		// The timestepping is internal to allow for batching across timesteps.
		const auto timesteps_paths = prop.traceFromTransmitter(ctx, tx, times);

		struct OpenResponse
		{
			serial::ResponseBuilder builder;
			Receiver* receiver;
			size_t last_index;
		};
		std::unordered_map<uint64_t, OpenResponse> responses;

		for (size_t i = 0; i < timesteps_paths.size(); ++i)
		{
			// A response can only continue at the timestep immediately following its last
			// one. Anything else means its path dropped out (e.g. momentary occlusion), so
			// finalize and route it now rather than letting a later reappearance of the same
			// path_id silently bridge the gap - see Response's tx-time-interval taper, which
			// assumes each response's own points are contiguous on the sample grid.
			for (auto it = responses.begin(); it != responses.end();)
			{
				if (it->second.last_index + 1 != i)
				{
					routeResponse(it->second.receiver,
								 std::make_unique<serial::Response>(std::move(it->second.builder)));
					it = responses.erase(it);
				}
				else
				{
					++it;
				}
			}

			const auto& timesteps = timesteps_paths[i];
			const RealType left_interval = i == 0 ? 0.0 : times[i] - times[i - 1];
			const RealType right_interval = i == point_count ? 0.0 : times[i + 1] - times[i];

			for (const auto& path : timesteps.paths)
			{
				auto phase_delay = -path.delay * 2.0 * PI * tx.getSignal()->getCarrier() + std::arg(path.gain);

				const interp::InterpPoint point{.gain = std::sqrt(std::norm(path.gain)),
												.rx_time = timesteps.tx_time + path.delay,
												.delay = path.delay,
												.phase_delay = phase_delay};

				if (const auto found = responses.find(path.path_id); found != responses.end())
				{
					found->second.builder.addInterpPoint(point, right_interval);
					found->second.last_index = i;
				}
				else
				{
					responses.insert({
						path.path_id,
						{serial::ResponseBuilder(tx.getSignal(), point, left_interval, right_interval), path.receiver,
						 i},
					});
				}
			}
		}

		for (auto& [k, v] : responses)
		{
			routeResponse(v.receiver, std::make_unique<serial::Response>(std::move(v.builder)));
		}
		responses.clear(); // To be explicit: the items are invalidated by the above.
	}

	namespace
	{

		/**
		 * @brief Calculates the antenna gain for a specific direction and time.
		 * @param radar The radar object (Transmitter or Receiver).
		 * @param direction_vec The unit vector pointing AWAY from the antenna towards the target/receiver.
		 * @param time The simulation time for rotation lookup.
		 * @param lambda The signal wavelength.
		 * @return The linear gain value.
		 */
		RealType computeAntennaGain(const radar::Radar* radar, const Vec3& direction_vec, RealType time,
									RealType lambda)
		{
			return radar->getGain(SVec3(direction_vec), radar->getRotation(time), lambda);
		}

		struct PreviewTransmitterContext
		{
			const Transmitter& transmitter;
			Vec3 position;
			RealType radiated_power;
			RealType lambda;
		};

		struct PreviewReceiverContext
		{
			const Receiver& receiver;
			Vec3 position;
			bool no_loss;
		};

		PreviewTransmitterContext makePreviewTransmitterContext(const Transmitter& transmitter, const RealType time)
		{
			const auto* waveform = transmitter.getSignal();
			return PreviewTransmitterContext{.transmitter = transmitter,
											 .position = transmitter.getPosition(time),
											 .radiated_power = previewRadiatedPower(waveform),
											 .lambda =
												 (waveform != nullptr) ? (params::c() / waveform->getCarrier()) : 0.3};
		}

		PreviewReceiverContext makePreviewReceiverContext(const Receiver& receiver, const RealType time)
		{
			return PreviewReceiverContext{.receiver = receiver,
										  .position = receiver.getPosition(time),
										  .no_loss = receiver.checkFlag(Receiver::RecvFlag::FLAG_NOPROPLOSS)};
		}

		void addIlluminatorLinks(std::vector<PreviewLink>& links, const PreviewTransmitterContext& tx_ctx,
								 const core::World& world, const RealType time)
		{
			for (const auto& target : world.getTargets())
			{
				const auto target_position = target->getPosition(time);
				const Vec3 vec_tx_tgt = target_position - tx_ctx.position;
				const RealType range = vec_tx_tgt.length();
				if (range <= EPSILON)
				{
					continue;
				}

				const Vec3 u_tx_tgt = vec_tx_tgt / range;
				const RealType gain = computeAntennaGain(&tx_ctx.transmitter, u_tx_tgt, time, tx_ctx.lambda);
				const RealType power_density = (tx_ctx.radiated_power * gain) / (4.0 * PI * range * range);
				links.push_back({.type = LinkType::BistaticTxTgt,
								 .quality = LinkQuality::Strong,
								 .label = formatPreviewDbwPerSquareMeterLabel(power_density),
								 .display_value = wattsToDb(power_density),
								 .source_id = tx_ctx.transmitter.getId(),
								 .dest_id = target->getId(),
								 .origin_id = tx_ctx.transmitter.getId()});
			}
		}

		void addMonostaticLinks(std::vector<PreviewLink>& links, const PreviewTransmitterContext& tx_ctx,
								const PreviewReceiverContext& rx_ctx, const core::World& world, const RealType time)
		{
			for (const auto& target : world.getTargets())
			{
				const auto target_position = target->getPosition(time);
				const Vec3 vec_tx_tgt = target_position - tx_ctx.position;
				const RealType range = vec_tx_tgt.length();
				if (range <= EPSILON)
				{
					continue;
				}

				const Vec3 u_tx_tgt = vec_tx_tgt / range;
				const RealType gt = computeAntennaGain(&tx_ctx.transmitter, u_tx_tgt, time, tx_ctx.lambda);
				const RealType gr = computeAntennaGain(&rx_ctx.receiver, u_tx_tgt, time, tx_ctx.lambda);
				SVec3 in_angle(u_tx_tgt);
				SVec3 out_angle(-u_tx_tgt);
				const auto rcs_opt = target->getRcs(in_angle, out_angle, time);
				if (!rcs_opt)
				{
					continue;
				}

				ComplexType gain = propagation::pointscatter::computeReflectedPathGain(gt, gr, *rcs_opt, tx_ctx.lambda,
																					   range, range, rx_ctx.no_loss);
				ComplexType unit_gain = propagation::pointscatter::computeReflectedPathGain(
					gt, gr, 1.0, tx_ctx.lambda, range, range, rx_ctx.no_loss);
				if (params::pointScatterParams().polarimetric)
				{
					const auto linkage = propagation::pointscatter::computePolarimetricReflectedLinkage(
						tx_ctx.transmitter, rx_ctx.receiver, vec_tx_tgt, -vec_tx_tgt, range, range, time, time);
					gain *= linkage;
					unit_gain *= linkage;
				}
				const RealType power_ratio = std::abs(gain) * std::abs(gain);
				const RealType pr_watts = tx_ctx.radiated_power * power_ratio;
				const RealType pr_unit_watts = tx_ctx.radiated_power * std::abs(unit_gain) * std::abs(unit_gain);

				links.push_back({.type = LinkType::Monostatic,
								 .quality = isSignalStrong(pr_unit_watts, rx_ctx.receiver.getNoiseTemperature())
									 ? LinkQuality::Strong
									 : LinkQuality::Weak,
								 .label = formatPreviewDbmLabel(pr_unit_watts),
								 .display_value = wattsToDbm(pr_unit_watts),
								 .source_id = tx_ctx.transmitter.getId(),
								 .dest_id = target->getId(),
								 .origin_id = tx_ctx.transmitter.getId(),
								 .rcs = *rcs_opt,
								 .actual_power_dbm = wattsToDbm(pr_watts)});
			}
		}

		void addDirectLink(std::vector<PreviewLink>& links, const PreviewTransmitterContext& tx_ctx,
						   const PreviewReceiverContext& rx_ctx, const RealType time)
		{
			if (rx_ctx.receiver.checkFlag(Receiver::RecvFlag::FLAG_NODIRECT))
			{
				return;
			}

			const Vec3 vec_direct = rx_ctx.position - tx_ctx.position;
			const RealType range = vec_direct.length();
			if (range <= EPSILON)
			{
				return;
			}

			const Vec3 u_tx_rx = vec_direct / range;
			const RealType gt = computeAntennaGain(&tx_ctx.transmitter, u_tx_rx, time, tx_ctx.lambda);
			const RealType gr = computeAntennaGain(&rx_ctx.receiver, -u_tx_rx, time, tx_ctx.lambda);

			const RealType scalar_gain = propagation::computeDirectPathGain(gt, gr, tx_ctx.lambda, range, rx_ctx.no_loss);
			ComplexType gain{scalar_gain, 0.0};
			if (params::pointScatterParams().polarimetric)
			{
				const auto tx_pol =
					propagation::pointscatter::computeAntennaPolarisation(tx_ctx.transmitter, vec_direct, time);
				const auto rx_pol =
					propagation::pointscatter::computeAntennaPolarisation(rx_ctx.receiver, -vec_direct, time);
				const auto linkage = propagation::dot_no_conj(tx_pol, rx_pol);
				gain *= ComplexType{linkage.re, linkage.im};
			}
			const RealType power_ratio = std::abs(gain) * std::abs(gain);
			const RealType pr_watts = tx_ctx.radiated_power * power_ratio;

			links.push_back({.type = LinkType::DirectTxRx,
							 .quality = LinkQuality::Strong,
							 .label = formatPreviewDbmLabel(pr_watts, "Direct: "),
							 .display_value = wattsToDbm(pr_watts),
							 .source_id = tx_ctx.transmitter.getId(),
							 .dest_id = rx_ctx.receiver.getId(),
							 .origin_id = tx_ctx.transmitter.getId()});
		}

		void addBistaticTargetReceiverLinks(std::vector<PreviewLink>& links, const PreviewTransmitterContext& tx_ctx,
											const PreviewReceiverContext& rx_ctx, const core::World& world,
											const RealType time)
		{
			for (const auto& target : world.getTargets())
			{
				const auto target_position = target->getPosition(time);
				const Vec3 vec_tx_tgt = target_position - tx_ctx.position;
				const Vec3 vec_tgt_rx = rx_ctx.position - target_position;
				const RealType r1 = vec_tx_tgt.length();
				const RealType r2 = vec_tgt_rx.length();
				if (r1 <= EPSILON || r2 <= EPSILON)
				{
					continue;
				}

				const Vec3 u_tx_tgt = vec_tx_tgt / r1;
				const Vec3 u_tgt_rx = vec_tgt_rx / r2;

				// computePolarimetricReflectedLinkage requires tx_to_tgt and tgt_to_rx not to point in exactly
				// the same direction (forward scatter).
				if (math::dotProduct(u_tx_tgt, u_tgt_rx) > 1.0 - EPSILON)
				{
					continue;
				}
				const RealType gt = computeAntennaGain(&tx_ctx.transmitter, u_tx_tgt, time, tx_ctx.lambda);
				const RealType gr = computeAntennaGain(&rx_ctx.receiver, -u_tgt_rx, time, tx_ctx.lambda);
				SVec3 in_angle(u_tx_tgt);
				SVec3 out_angle(-u_tgt_rx);
				const auto rcs_opt = target->getRcs(in_angle, out_angle, time);
				if (!rcs_opt)
				{
					continue;
				}

				ComplexType gain = propagation::pointscatter::computeReflectedPathGain(gt, gr, *rcs_opt, tx_ctx.lambda,
																					   r1, r2, rx_ctx.no_loss);
				ComplexType unit_gain =
					propagation::pointscatter::computeReflectedPathGain(gt, gr, 1.0, tx_ctx.lambda, r1, r2,
																		 rx_ctx.no_loss);
				if (params::pointScatterParams().polarimetric)
				{
					const auto linkage = propagation::pointscatter::computePolarimetricReflectedLinkage(
						tx_ctx.transmitter, rx_ctx.receiver, vec_tx_tgt, vec_tgt_rx, r1, r2, time, time);
					gain *= linkage;
					unit_gain *= linkage;
				}
				const RealType power_ratio = std::abs(gain) * std::abs(gain);
				const RealType pr_watts = tx_ctx.radiated_power * power_ratio;
				const RealType pr_unit_watts = tx_ctx.radiated_power * std::abs(unit_gain) * std::abs(unit_gain);

				links.push_back({.type = LinkType::BistaticTgtRx,
								 .quality = isSignalStrong(pr_unit_watts, rx_ctx.receiver.getNoiseTemperature())
									 ? LinkQuality::Strong
									 : LinkQuality::Weak,
								 .label = formatPreviewDbmLabel(pr_unit_watts),
								 .display_value = wattsToDbm(pr_unit_watts),
								 .source_id = target->getId(),
								 .dest_id = rx_ctx.receiver.getId(),
								 .origin_id = tx_ctx.transmitter.getId(),
								 .rcs = *rcs_opt,
								 .actual_power_dbm = wattsToDbm(pr_watts)});
			}
		}

		void addReceiverLinks(std::vector<PreviewLink>& links, const PreviewTransmitterContext& tx_ctx,
							  const PreviewReceiverContext& rx_ctx, const core::World& world, const RealType time)
		{
			if (tx_ctx.transmitter.getAttached() == &rx_ctx.receiver)
			{
				addMonostaticLinks(links, tx_ctx, rx_ctx, world, time);
				return;
			}

			addDirectLink(links, tx_ctx, rx_ctx, time);
			addBistaticTargetReceiverLinks(links, tx_ctx, rx_ctx, world, time);
		}
	}

	std::vector<PreviewLink> calculatePreviewLinks(const core::World& world, const RealType time)
	{
		std::vector<PreviewLink> links;

		for (const auto& tx : world.getTransmitters())
		{
			if (!isComponentActive(tx->getSchedule(), time))
			{
				continue;
			}

			const auto tx_ctx = makePreviewTransmitterContext(*tx, time);
			addIlluminatorLinks(links, tx_ctx, world, time);

			for (const auto& rx : world.getReceivers())
			{
				if (!isComponentActive(rx->getSchedule(), time))
				{
					continue;
				}
				const auto rx_ctx = makePreviewReceiverContext(*rx, time);
				addReceiverLinks(links, tx_ctx, rx_ctx, world, time);
			}
		}
		return links;
	}
}
