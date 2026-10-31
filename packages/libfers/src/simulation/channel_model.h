// SPDX-License-Identifier: GPL-2.0-only
//
// Copyright (c) 2006-2008 Marc Brooker and Michael Inggs
// Copyright (c) 2008-present FERS Contributors (see AUTHORS.md).
//
// See the GNU GPLv2 LICENSE file in the FERS project root for more information.

/**
 * @file channel_model.h
 * @brief Header for radar channel propagation and interaction models.
 *
 * This file contains the declarations for functions that model the radar channel,
 * including direct and reflected signal path calculations,  and
 * power scaling according to the radar range equation. These functions form the
 * core physics engine for determining the properties of a received signal based
 * on the positions and velocities of transmitters, receivers, and targets.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

#include "core/config.h"
#include "core/sim_id.h"
#include "core/simulation_state.h"
#include "propagation/propagation_model.h"

namespace propagation
{
	class PropagationPath;
	class PropagationModel;
}
namespace core
{
	class World;
}
namespace timing
{
	class Timing;
}
namespace radar
{
	class Receiver;

	class Transmitter;

	class Target;
}

namespace serial
{
	class Response;
}

namespace fers_signal
{
	class RadarSignal;
}

namespace simulation
{
	/// Sampled phase-noise buffer for one timing source.
	struct CwPhaseNoiseBuffer
	{
		RealType start_time{}; ///< First sample time in seconds.
		RealType dt{}; ///< Time spacing between phase-noise samples in seconds.
		std::vector<RealType> samples; ///< Phase-noise samples in radians.

		/// Returns the interpolated phase-noise sample at the specified time.
		[[nodiscard]] RealType sampleAt(RealType time) const noexcept;
	};

	/// Lookup table for CW phase noise across timing sources.
	struct CwPhaseNoiseLookup
	{
		RealType start_time{}; ///< Lookup start time in seconds.
		RealType end_time{}; ///< Lookup end time in seconds.
		RealType dt{}; ///< Lookup sample spacing in seconds.
		std::unordered_map<SimId, CwPhaseNoiseBuffer> buffers; ///< Per-timing-source phase-noise buffers.

		/// Builds a phase-noise lookup for the requested timing sources and time range.
		[[nodiscard]] static CwPhaseNoiseLookup build(std::span<const std::shared_ptr<timing::Timing>> timings,
													  RealType start_time, RealType end_time);

		/// Samples phase noise for one timing source at the specified time.
		[[nodiscard]] RealType sample(const timing::Timing* timing, RealType time) const noexcept;

		/// Computes receiver-minus-transmitter phase noise at two propagation times.
		[[nodiscard]] RealType phaseDifference(const timing::Timing* rx_timing, RealType rx_time,
											   const timing::Timing* tx_timing, RealType tx_time) const noexcept;
	};

	/**
	 * @brief Ingests the next coarse sample of propagation paths for one receiver into
	 *        its per-path interpolation cache (CW coarse-rate sampling), establishing the bracket
	 *        `[interval_start_time, sample_time]` that samples in between will be interpolated
	 *        across.
	 *
	 * Mirrors `serial::Response`'s lead/trail tapering:
	 * - A path present now and previously: `hi` replaces `lo`, the new sample replaces `hi`.
	 * - A path present before but missing now (about to disappear): a synthetic zero-gain "hi" is
	 *   built at `sample_time` by linearly extrapolating delay/phase from the entry's own existing
	 *   `lo`/`hi` (the two most recent real samples), like `extrapolateTaperPoint` in response.cpp.
	 * - A path seen for the first time (just appeared): "lo" is a zero-gain copy of the next "hi"
	 *   value, placed at `interval_start_time`. The appearance-edge counterpart of the above, matching
	 *   `simpleTaperPoint`'s behavior, since there is no additional sample to estimate a rate from.
	 *
	 * @param cache The receiver's tracker cache; `coarse_path_samples` is updated in place.
	 * @param next_paths This receiver's paths from the coarse scene query just taken.
	 * @param interval_start_time The start time of the bracket a brand-new path's synthetic "lo"
	 *                            should be placed at (ignored for paths that already existed).
	 * @param sample_time The absolute simulation time `next_paths` was sampled at (the new "hi").
	 */
	void updateCoarseStreamingCache(core::ReceiverTrackerCache& cache,
									const std::vector<propagation::PropagationPath>& next_paths,
									RealType interval_start_time, RealType sample_time);

	/**
	 * @brief Produces a receiver's propagation paths at `rx_time` by linearly interpolating each
	 *        cached path's delay and gain between its bracketing `lo`/`hi` coarse samples (CW
	 *        coarse-rate sampling).
	 *
	 * Interpolating delay linearly is physically equivalent to a constant-velocity (linear
	 * Doppler) approximation of range over the coarse interval. Gain's phase here is the path's
	 * own slowly-varying antenna/polarization/scattered phase, not the fast carrier term, so linear
	 * interpolation approximates well.
	 *
	 * @param cache The receiver's tracker cache, populated by `updateCoarseStreamingCache`; the
	 *              caller is responsible for ensuring it already brackets `rx_time`.
	 * @param receiver The receiver these paths are for.
	 * @param rx_time The time to interpolate each cached path to.
	 */
	[[nodiscard]] std::vector<propagation::PropagationPath>
	interpolateCoarseStreamingPaths(const core::ReceiverTrackerCache& cache, radar::Receiver& receiver,
									RealType rx_time);

	/// Selects how timing phase noise is applied to streaming channel contributions.
	enum class StreamingTimingPhaseMode : std::uint8_t
	{
		ReceiverRelative, ///< Existing raw streaming convention: transmitter phase minus receiver LO phase.
		TransmitterOnly, ///< Incoming RF/baseband signal before receiver LO subtraction.
		None ///< Ignore timing phase noise entirely.
	};

	/**
	 * @brief Evaluates a receive-time streaming waveform phase for receiver LO/dechirp references.
	 *
	 * @param source Cached active streaming source to evaluate.
	 * @param timeK Receiver time in seconds.
	 * @param chirp_tracker Optional caller-owned FMCW boundary tracker for this source.
	 * @param phase_out Receives the waveform phase in radians when evaluation succeeds.
	 * @return True when the source is active and a phase was produced.
	 */
	[[nodiscard]] bool calculateStreamingReferencePhase(const core::ActiveStreamingSource& source, RealType timeK,
														core::FmcwChirpBoundaryTracker* chirp_tracker,
														RealType& phase_out);

	/// Evaluates the complete complex reference envelope, including file-backed amplitude modulation.
	[[nodiscard]] bool calculateStreamingReferenceSample(const core::ActiveStreamingSource& source, RealType timeK,
														 core::FmcwChirpBoundaryTracker* chirp_tracker,
														 ComplexType& sample_out);

	/**
	 * @brief Calculates a contribution from a cached streaming source.
	 *
	 * @param source Cached active streaming source to evaluate.
	 * @param recv Receiver observing the reflected signal.
	 * @param path Propagation path of the contributing radar signal.
	 * @param timeK Current receiver time in seconds.
	 * @param phase_noise_lookup Optional lookup for timing phase noise samples.
	 * @param chirp_tracker Optional caller-owned FMCW boundary tracker for this path.
	 * @param timing_phase_mode Selects how timing phase noise is applied.
	 * @return The complex I/Q sample contribution for this path.
	 */
	ComplexType calculateStreamingPathContribution(
		const core::ActiveStreamingSource& source, const radar::Receiver* recv,
		const propagation::PropagationPath& path, RealType timeK,
		const CwPhaseNoiseLookup* phase_noise_lookup = nullptr, core::FmcwChirpBoundaryTracker* chirp_tracker = nullptr,
		StreamingTimingPhaseMode timing_phase_mode = StreamingTimingPhaseMode::ReceiverRelative);

	/**
	 * @brief Creates Response objects by simulating possible radar paths/channels from
	 *        the given transmitter to all receivers.
	 *
	 * This function generates Responses: series of `InterpPoint`s describing the channel
	 * properties (gain, delay, phase delay) in a time series for each radar path
	 * across the duration of the pulse signal. This function does not sample the signal itself.
	 * These points capture the time-varying properties of the radar propagation channel.
	 * These Responses are places into the Receivers for later rendering.
	 *
	 * @param tx The transmitter of the pulse.
	 * @param prop The propagation model to determine pulse signal propagation.
	 * @param start_tx_time The absolute simulation time when the pulse transmission starts.
	 */
	void calculateResponses(const radar::Transmitter& tx, const RealType start_tx_time,
							const propagation::PropagationModel& prop, propagation::ThreadContext* ctx);

	/**
	 * @enum LinkType
	 * @brief Categorizes the visual link for rendering.
	 */
	enum class LinkType : std::uint8_t
	{
		Monostatic, ///< Combined Tx/Rx path
		BistaticTxTgt, ///< Illuminator path
		BistaticTgtRx, ///< Scattered path
		DirectTxRx ///< Interference path
	};

	/**
	 * @enum LinkQuality
	 * @brief Describes the radiometric quality of the link.
	 */
	enum class LinkQuality : std::uint8_t
	{
		Strong, ///< SNR > 0 dB
		Weak ///< SNR < 0 dB (Geometric line of sight, but below noise floor)
	};

	/**
	 * @struct PreviewLink
	 * @brief A calculated link segment for 3D visualization.
	 */
	struct PreviewLink
	{
		LinkType type; ///< Visual link category.
		LinkQuality quality; ///< Radiometric link quality.
		std::string label; ///< Human-readable label for display.
		double display_value{-999.0}; ///< Numeric value represented by label, in the label's unit.
		SimId source_id; ///< SimId at the start of this specific link segment.
		SimId dest_id; ///< SimId at the end of this specific link segment.
		SimId origin_id; ///< Original transmitted-energy source SimId.
		double rcs{-1.0}; ///< RCS in square meters, or -1.0 when not applicable.
		double actual_power_dbm{-999.0}; ///< Received power in dBm with actual RCS, or sentinel when unavailable.
	};

	/**
	 * @brief Calculates all visual links for the current world state at a specific time.
	 *
	 * This function utilizes the core radar equation helpers to determine visibility,
	 * power levels, and SNR for all Tx/Rx/Target combinations. It is lightweight
	 * and does not update simulation state.
	 *
	 * @param world The simulation world containing radar components.
	 * @param time The time at which to calculate geometry.
	 * @return A vector of renderable links.
	 */
	std::vector<PreviewLink> calculatePreviewLinks(const core::World& world, RealType time);
}
