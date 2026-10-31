// Tests for simulation::calculateStreamingPathContribution: the unified
// path -> complex sample function for CW/FMCW streaming channel contributions.
// Verifies Friis/bistatic amplitude scaling by signal power, propagation
// delay/phase, retarded-transmit-time gating, and FMCW dechirp/beat-frequency
// behavior against hand-calculated values. Path-finding physics itself
// (Friis/bistatic gain and delay) is covered by
// propagation/pointscatter/test_pointscatter.cpp; here every
// propagation::PropagationPath is built by hand.

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <memory>
#include <vector>

#include "antenna/antenna_factory.h"
#include "core/config.h"
#include "core/parameters.h"
#include "core/simulation_state.h"
#include "math/coord.h"
#include "math/geometry_ops.h"
#include "propagation/propagation_model.h"
#include "radar/platform.h"
#include "radar/receiver.h"
#include "radar/transmitter.h"
#include "signal/radar_signal.h"
#include "simulation/channel_model.h"
#include "timing/prototype_timing.h"
#include "timing/timing.h"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace
{
	struct ParamGuard
	{
		params::Parameters saved;
		ParamGuard() : saved(params::params) {}
		ParamGuard(const ParamGuard&) = delete;
		ParamGuard& operator=(const ParamGuard&) = delete;
		ParamGuard(ParamGuard&&) = delete;
		ParamGuard& operator=(ParamGuard&&) = delete;
		~ParamGuard() { params::params = saved; }
	};

	void setupPlatform(radar::Platform& plat, const math::Vec3& pos)
	{
		plat.getMotionPath()->addCoord(math::Coord{pos, 0.0});
		plat.getMotionPath()->finalize();
		plat.getRotationPath()->addCoord(math::RotationCoord{0.0, 0.0, 0.0});
		plat.getRotationPath()->finalize();
	}

	simulation::CwPhaseNoiseLookup makeLookup(const std::shared_ptr<timing::Timing>& timing)
	{
		const std::vector<std::shared_ptr<timing::Timing>> timings = {timing};
		return simulation::CwPhaseNoiseLookup::build(timings, params::startTime(), params::endTime());
	}

	RealType unwrapDelta(RealType delta)
	{
		while (delta > PI)
		{
			delta -= 2.0 * PI;
		}
		while (delta < -PI)
		{
			delta += 2.0 * PI;
		}
		return delta;
	}

	RealType rms(const std::vector<ComplexType>& samples)
	{
		RealType sum_sq = 0.0;
		for (const auto& sample : samples)
		{
			sum_sq += std::norm(sample);
		}
		return std::sqrt(sum_sq / static_cast<RealType>(samples.size()));
	}

	// Builds a direct (path_id == 0) path with a hand-computed delay/gain, the
	// counterpart of what propagation::pointscatter::PointScatterModel would have
	// found for this geometry, without needing a World/PointScatterModel here.
	propagation::PropagationPath makeDirectPath(radar::Receiver* rx, RealType dist, RealType gain)
	{
		return propagation::PropagationPath{
			.delay = dist / params::c(),
			.gain = gain,
			.path_id = 0,
			.source_index = 0,
			.receiver = rx,
		};
	}

	// Builds a reflected (path_id != 0) path with an explicit total delay/gain.
	propagation::PropagationPath makeReflectedPath(radar::Receiver* rx, RealType delay, RealType gain)
	{
		return propagation::PropagationPath{
			.delay = delay,
			.gain = gain,
			.path_id = 1,
			.source_index = 0,
			.receiver = rx,
		};
	}
}

// =============================================================================
// calculateStreamingPathContribution: direct-path CW complex sample
// =============================================================================

TEST_CASE("calculateDirectPathContribution amplitude matches Friis equation with signal power",
		  "[simulation][channel_model][streaming][direct]")
{
	ParamGuard const guard;
	params::params.reset();

	// Setup: Carrier: 1 GHz, Signal Power: 100 W, distance 1000 m
	// Expected amplitude = sqrt(Pt * Friis_factor), Friis = lambda^2 / (16*pi^2*R^2)
	const RealType c = params::c();
	const RealType carrier = 1.0e9;
	const RealType lambda = c / carrier;
	const RealType dist = 1000.0;
	const RealType power = 100.0;

	const RealType friis = (lambda * lambda) / (16.0 * PI * PI * dist * dist);
	const RealType expected_amplitude = std::sqrt(power * friis);

	radar::Platform tx_plat("tx_plat");
	setupPlatform(tx_plat, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_plat("rx_plat");
	setupPlatform(rx_plat, math::Vec3{dist, 0.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::CW_MODE);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);

	fers_signal::RadarSignal wave("sig", power, carrier, fers_signal::CwWaveform{});
	tx.setSignal(&wave);

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::CW_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);

	const auto source = core::makeActiveSource(&tx, -10.0, 10.0);
	const auto path = makeDirectPath(&rx, dist, std::sqrt(friis));

	const ComplexType result = simulation::calculateStreamingPathContribution(source, &rx, path, 0.0);

	REQUIRE_THAT(std::abs(result), WithinRel(expected_amplitude, 1e-6));
}

TEST_CASE("calculateDirectPathContribution phase matches propagation delay",
		  "[simulation][channel_model][streaming][direct]")
{
	ParamGuard const guard;
	params::params.reset();

	const RealType c = params::c();
	const RealType carrier = 1.0e9;
	const RealType dist = 1000.0;

	const RealType tau = dist / c;
	// Expected phase = -2*pi*f*tau (from carrier) + timing_phase (0 for default timing)
	const RealType expected_phase = -2.0 * PI * carrier * tau;

	radar::Platform tx_plat("tx_plat");
	setupPlatform(tx_plat, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_plat("rx_plat");
	setupPlatform(rx_plat, math::Vec3{dist, 0.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::CW_MODE);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);

	fers_signal::RadarSignal wave("sig", 1.0, carrier, fers_signal::CwWaveform{});
	tx.setSignal(&wave);

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::CW_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);

	const auto source = core::makeActiveSource(&tx, -10.0, 10.0);
	const auto path = makeDirectPath(&rx, dist, 1.0);

	const ComplexType result = simulation::calculateStreamingPathContribution(source, &rx, path, 0.0);
	const RealType result_phase = std::arg(result);

	// Compare via complex unit vectors to avoid branch-cut wrapping issues
	REQUIRE_THAT(std::cos(result_phase), WithinAbs(std::cos(expected_phase), 1e-6));
	REQUIRE_THAT(std::sin(result_phase), WithinAbs(std::sin(expected_phase), 1e-6));
}

TEST_CASE("CW streaming direct path gates schedules by retarded transmit time",
		  "[simulation][channel_model][streaming][direct]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setC(100.0);

	const RealType dist = 10.0;
	const RealType tau = dist / params::c();
	const RealType segment_start = 0.2;
	const RealType segment_end = 0.5;
	const RealType eps = 1.0e-6;

	radar::Platform tx_plat("tx_plat");
	setupPlatform(tx_plat, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_plat("rx_plat");
	setupPlatform(rx_plat, math::Vec3{dist, 0.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::CW_MODE);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);

	fers_signal::RadarSignal wave("cw", 1.0, 1.0e9, fers_signal::CwWaveform{});
	tx.setSignal(&wave);

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::CW_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);

	const auto source = core::makeActiveSource(&tx, segment_start, segment_end);
	const auto path = makeDirectPath(&rx, dist, 1.0);

	REQUIRE(std::abs(simulation::calculateStreamingPathContribution(source, &rx, path, segment_start + tau - eps)) ==
			0.0);
	REQUIRE(std::abs(simulation::calculateStreamingPathContribution(source, &rx, path, segment_start + tau + eps)) >
			0.0);
	REQUIRE(std::abs(simulation::calculateStreamingPathContribution(source, &rx, path, segment_end + 0.5 * tau)) > 0.0);
	REQUIRE(std::abs(simulation::calculateStreamingPathContribution(source, &rx, path, segment_end + tau + eps)) ==
			0.0);
}

TEST_CASE("calculateDirectPathContribution with noproploss gives distance-independent amplitude",
		  "[simulation][channel_model][streaming][direct]")
{
	ParamGuard const guard;
	params::params.reset();

	const RealType c = params::c();
	const RealType carrier = 1.0e9;
	const RealType lambda = c / carrier;
	const RealType power = 50.0;

	// With noproploss: Friis = lambda^2 / (16*pi^2) (no R^2), constant regardless of dist
	const RealType friis_noloss = (lambda * lambda) / (16.0 * PI * PI);
	const RealType expected_amplitude = std::sqrt(power * friis_noloss);

	// Test with two different distances - both should give the same amplitude, since
	// the noproploss gain (computed independently of dist here) is passed in directly.
	for (const RealType dist : {500.0, 2000.0})
	{
		radar::Platform tx_plat("tx_plat");
		setupPlatform(tx_plat, math::Vec3{0.0, 0.0, 0.0});
		radar::Platform rx_plat("rx_plat");
		setupPlatform(rx_plat, math::Vec3{dist, 0.0, 0.0});

		antenna::Isotropic iso_ant("iso");
		auto timing = std::make_shared<timing::Timing>("clk", 42);

		radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::CW_MODE);
		tx.setAntenna(&iso_ant);
		tx.setTiming(timing);

		fers_signal::RadarSignal wave("sig", power, carrier, fers_signal::CwWaveform{});
		tx.setSignal(&wave);

		radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::CW_MODE);
		rx.setAntenna(&iso_ant);
		rx.setTiming(timing);
		rx.setFlag(radar::Receiver::RecvFlag::FLAG_NOPROPLOSS);

		const auto source = core::makeActiveSource(&tx, -10.0, 10.0);
		const auto path = makeDirectPath(&rx, dist, std::sqrt(friis_noloss));

		const ComplexType result = simulation::calculateStreamingPathContribution(source, &rx, path, 0.0);

		REQUIRE_THAT(std::abs(result), WithinRel(expected_amplitude, 1e-6));
	}
}

TEST_CASE("calculateDirectPathContribution applies buffered delayed timing phase with interpolation",
		  "[simulation][channel_model][streaming][direct]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setTime(0.0, 1.0);
	params::setRate(10.0);
	params::setOversampleRatio(1);
	params::setC(10.0);

	const RealType carrier = 1.0;
	const RealType dist = 1.5; // tau = 0.15 s
	const RealType time = 0.5;
	const RealType tau = dist / params::c();
	const RealType expected_extra_phase = -2.0 * PI * tau;

	radar::Platform tx_plat("tx_plat");
	setupPlatform(tx_plat, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_plat("rx_plat");
	setupPlatform(rx_plat, math::Vec3{dist, 0.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	timing::PrototypeTiming prototype("clk");
	prototype.setFrequency(carrier);
	prototype.setFreqOffset(1.0);
	auto timing = std::make_shared<timing::Timing>("clk", 42);
	timing->initializeModel(&prototype);
	const auto lookup = makeLookup(timing);

	radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::CW_MODE);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);

	fers_signal::RadarSignal wave("sig", 1.0, carrier, fers_signal::CwWaveform{});
	tx.setSignal(&wave);

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::CW_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);

	const auto source = core::makeActiveSource(&tx, 0.0, 1.0);
	const auto path = makeDirectPath(&rx, dist, 1.0);

	const ComplexType ideal = simulation::calculateStreamingPathContribution(source, &rx, path, time);
	const ComplexType delayed = simulation::calculateStreamingPathContribution(source, &rx, path, time, &lookup);
	const RealType extra_phase = std::arg(delayed * std::conj(ideal));

	REQUIRE_THAT(std::cos(extra_phase), WithinAbs(std::cos(expected_extra_phase), 1e-6));
	REQUIRE_THAT(std::sin(extra_phase), WithinAbs(std::sin(expected_extra_phase), 1e-6));
}

// =============================================================================
// calculateStreamingPathContribution: reflected-path CW complex sample
// =============================================================================

TEST_CASE("CW streaming reflected path gates schedules by retarded transmit time",
		  "[simulation][channel_model][streaming][reflected]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setC(100.0);

	const RealType tx_target_dist = 5.0;
	const RealType target_rx_dist = 5.0;
	const RealType tau = (tx_target_dist + target_rx_dist) / params::c();
	const RealType segment_start = 0.2;
	const RealType segment_end = 0.5;
	const RealType eps = 1.0e-6;

	radar::Platform tx_plat("tx_plat");
	setupPlatform(tx_plat, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_plat("rx_plat");
	setupPlatform(rx_plat, math::Vec3{tx_target_dist + target_rx_dist, 0.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::CW_MODE);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);
	fers_signal::RadarSignal wave("cw", 1.0, 1.0e6, fers_signal::CwWaveform{});
	tx.setSignal(&wave);

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::CW_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);

	const auto source = core::makeActiveSource(&tx, segment_start, segment_end);
	const auto path = makeReflectedPath(&rx, tau, 1.0);

	REQUIRE(std::abs(simulation::calculateStreamingPathContribution(source, &rx, path, segment_start + tau - eps)) ==
			0.0);
	REQUIRE(std::abs(simulation::calculateStreamingPathContribution(source, &rx, path, segment_start + tau + eps)) >
			0.0);
	REQUIRE(std::abs(simulation::calculateStreamingPathContribution(source, &rx, path, segment_end + 0.5 * tau)) > 0.0);
	REQUIRE(std::abs(simulation::calculateStreamingPathContribution(source, &rx, path, segment_end + tau + eps)) ==
			0.0);
}

TEST_CASE("FMCW monostatic reflected path dechirps to expected stationary-target beat frequency",
		  "[simulation][channel_model][streaming][reflected][fmcw]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setRate(1.0e6);
	params::setSimSamplingRate(1.0e6);
	params::setOversampleRatio(1);
	params::setTime(0.0, 1.0e-3);

	const RealType target_range = 150.0;
	const RealType chirp_bandwidth = 1.0e6;
	const RealType chirp_duration = 1.0e-3;
	const RealType chirp_rate = chirp_bandwidth / chirp_duration;
	const RealType tau = (2.0 * target_range) / params::c();

	radar::Platform radar_platform("radar");
	setupPlatform(radar_platform, math::Vec3{0.0, 0.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&radar_platform, "tx", radar::OperationMode::FMCW_MODE, 101);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);
	fers_signal::RadarSignal wave("fmcw", 1.0, 10.0e6,
								  fers_signal::FmcwChirpWaveform(chirp_bandwidth, chirp_duration, chirp_duration), 301);
	tx.setSignal(&wave);

	radar::Receiver rx(&radar_platform, "rx", 43, radar::OperationMode::FMCW_MODE, 201);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);
	rx.setFlag(radar::Receiver::RecvFlag::FLAG_NOPROPLOSS);
	tx.setAttached(&rx);
	rx.setAttached(&tx);

	const auto source = core::makeActiveSource(&tx, 0.0, chirp_duration);
	const auto path = makeReflectedPath(&rx, tau, 1.0);
	core::FmcwChirpBoundaryTracker tracker;

	const RealType dt = 1.0 / params::simSamplingRate();
	const RealType first_time = 20.0e-6;
	const std::size_t sample_count = 500;
	std::vector<RealType> dechirped_phase;
	dechirped_phase.reserve(sample_count);

	for (std::size_t i = 0; i < sample_count; ++i)
	{
		const RealType t = first_time + static_cast<RealType>(i) * dt;
		const ComplexType sample =
			simulation::calculateStreamingPathContribution(source, &rx, path, t, nullptr, &tracker);
		const ComplexType dechirped =
			sample * std::polar(1.0, -wave.getFmcwChirpWaveform()->basebandPhaseForChirpTime(t));
		dechirped_phase.push_back(std::arg(dechirped));
	}

	RealType unwrapped_span = 0.0;
	for (std::size_t i = 1; i < dechirped_phase.size(); ++i)
	{
		unwrapped_span += unwrapDelta(dechirped_phase[i] - dechirped_phase[i - 1]);
	}

	const RealType measured_beat_hz =
		unwrapped_span / (2.0 * PI * dt * static_cast<RealType>(dechirped_phase.size() - 1));
	const RealType expected_beat_hz = -chirp_rate * tau;
	REQUIRE_THAT(measured_beat_hz, WithinRel(expected_beat_hz, 1.0e-3));
}

TEST_CASE("FMCW native dechirp convention produces positive up-chirp beat frequency",
		  "[simulation][channel_model][streaming][reflected][fmcw][dechirp]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setRate(1.0e6);
	params::setSimSamplingRate(1.0e6);
	params::setOversampleRatio(1);
	params::setTime(0.0, 1.0e-3);

	const RealType target_range = 150.0;
	const RealType chirp_bandwidth = 1.0e6;
	const RealType chirp_duration = 1.0e-3;
	const RealType chirp_rate = chirp_bandwidth / chirp_duration;
	const RealType tau = (2.0 * target_range) / params::c();

	radar::Platform radar_platform("radar");
	setupPlatform(radar_platform, math::Vec3{0.0, 0.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&radar_platform, "tx", radar::OperationMode::FMCW_MODE, 101);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);
	fers_signal::RadarSignal wave("fmcw", 1.0, 10.0e6,
								  fers_signal::FmcwChirpWaveform(chirp_bandwidth, chirp_duration, chirp_duration), 301);
	tx.setSignal(&wave);

	radar::Receiver rx(&radar_platform, "rx", 43, radar::OperationMode::FMCW_MODE, 201);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);
	rx.setFlag(radar::Receiver::RecvFlag::FLAG_NOPROPLOSS);
	tx.setAttached(&rx);
	rx.setAttached(&tx);

	const auto source = core::makeActiveSource(&tx, 0.0, chirp_duration);
	const auto path = makeReflectedPath(&rx, tau, 1.0);
	core::FmcwChirpBoundaryTracker channel_tracker;
	core::FmcwChirpBoundaryTracker reference_tracker;

	const RealType dt = 1.0 / params::simSamplingRate();
	const RealType first_time = 20.0e-6;
	const std::size_t sample_count = 500;
	std::vector<RealType> dechirped_phase;
	dechirped_phase.reserve(sample_count);

	for (std::size_t i = 0; i < sample_count; ++i)
	{
		const RealType t = first_time + static_cast<RealType>(i) * dt;
		const ComplexType sample = simulation::calculateStreamingPathContribution(
			source, &rx, path, t, nullptr, &channel_tracker, simulation::StreamingTimingPhaseMode::None);
		RealType reference_phase = 0.0;
		REQUIRE(simulation::calculateStreamingReferencePhase(source, t, &reference_tracker, reference_phase));
		const ComplexType dechirped = std::polar(1.0, reference_phase) * std::conj(sample);
		dechirped_phase.push_back(std::arg(dechirped));
	}

	RealType unwrapped_span = 0.0;
	for (std::size_t i = 1; i < dechirped_phase.size(); ++i)
	{
		unwrapped_span += unwrapDelta(dechirped_phase[i] - dechirped_phase[i - 1]);
	}

	const RealType measured_beat_hz =
		unwrapped_span / (2.0 * PI * dt * static_cast<RealType>(dechirped_phase.size() - 1));
	const RealType expected_beat_hz = chirp_rate * tau;
	REQUIRE_THAT(measured_beat_hz, WithinRel(expected_beat_hz, 1.0e-3));
}

TEST_CASE("FMCW physical dechirp preserves timing decorrelation while ideal mode removes it",
		  "[simulation][channel_model][streaming][reflected][fmcw][dechirp]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setRate(2.0e6);
	params::setSimSamplingRate(2.0e6);
	params::setOversampleRatio(1);
	params::setTime(0.0, 1.0e-3);

	const RealType target_range = 150.0;
	const RealType chirp_bandwidth = 1.0e6;
	const RealType chirp_duration = 1.0e-3;
	const RealType tau = (2.0 * target_range) / params::c();
	const RealType freq_offset = 250.0e3;
	const RealType sample_time = 40.0e-6;

	radar::Platform radar_platform("radar");
	setupPlatform(radar_platform, math::Vec3{0.0, 0.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	timing::PrototypeTiming prototype("clk");
	prototype.setFrequency(10.0e6);
	prototype.setFreqOffset(freq_offset);
	auto timing = std::make_shared<timing::Timing>("clk", 42);
	timing->initializeModel(&prototype);

	radar::Transmitter tx(&radar_platform, "tx", radar::OperationMode::FMCW_MODE, 101);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);
	fers_signal::RadarSignal wave("fmcw", 1.0, 10.0e6,
								  fers_signal::FmcwChirpWaveform(chirp_bandwidth, chirp_duration, chirp_duration), 301);
	tx.setSignal(&wave);

	radar::Receiver rx(&radar_platform, "rx", 43, radar::OperationMode::FMCW_MODE, 201);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);
	rx.setFlag(radar::Receiver::RecvFlag::FLAG_NOPROPLOSS);
	tx.setAttached(&rx);
	rx.setAttached(&tx);

	const auto source = core::makeActiveSource(&tx, 0.0, chirp_duration);
	const auto path = makeReflectedPath(&rx, tau, 1.0);
	core::FmcwChirpBoundaryTracker physical_tracker;
	core::FmcwChirpBoundaryTracker ideal_tracker;
	core::FmcwChirpBoundaryTracker reference_tracker;

	const ComplexType physical_channel =
		simulation::calculateStreamingPathContribution(source, &rx, path, sample_time, nullptr, &physical_tracker,
													   simulation::StreamingTimingPhaseMode::TransmitterOnly);
	const ComplexType ideal_channel = simulation::calculateStreamingPathContribution(
		source, &rx, path, sample_time, nullptr, &ideal_tracker, simulation::StreamingTimingPhaseMode::None);

	RealType reference_phase = 0.0;
	REQUIRE(simulation::calculateStreamingReferencePhase(source, sample_time, &reference_tracker, reference_phase));

	const RealType rx_phase = 2.0 * PI * freq_offset * sample_time;
	const ComplexType physical_dechirped = std::polar(1.0, reference_phase + rx_phase) * std::conj(physical_channel);
	const ComplexType ideal_dechirped = std::polar(1.0, reference_phase) * std::conj(ideal_channel);
	const RealType measured_delta = std::arg(physical_dechirped * std::conj(ideal_dechirped));
	const RealType expected_delta = 2.0 * PI * freq_offset * tau;

	REQUIRE_THAT(measured_delta, WithinAbs(expected_delta, 1.0e-6));
}

TEST_CASE("FMCW down-chirp monostatic reflected path reverses stationary-target beat sign",
		  "[simulation][channel_model][streaming][reflected][fmcw]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setRate(1.0e6);
	params::setSimSamplingRate(1.0e6);
	params::setOversampleRatio(1);
	params::setTime(0.0, 1.0e-3);

	const RealType target_range = 150.0;
	const RealType chirp_bandwidth = 1.0e6;
	const RealType chirp_duration = 1.0e-3;
	const RealType chirp_rate = chirp_bandwidth / chirp_duration;
	const RealType tau = (2.0 * target_range) / params::c();

	radar::Platform radar_platform("radar");
	setupPlatform(radar_platform, math::Vec3{0.0, 0.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&radar_platform, "tx", radar::OperationMode::FMCW_MODE, 101);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);
	fers_signal::RadarSignal wave("fmcw", 1.0, 10.0e6,
								  fers_signal::FmcwChirpWaveform(chirp_bandwidth, chirp_duration, chirp_duration, 0.0,
																 std::nullopt, fers_signal::FmcwChirpDirection::Down),
								  301);
	tx.setSignal(&wave);

	radar::Receiver rx(&radar_platform, "rx", 43, radar::OperationMode::FMCW_MODE, 201);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);
	rx.setFlag(radar::Receiver::RecvFlag::FLAG_NOPROPLOSS);
	tx.setAttached(&rx);
	rx.setAttached(&tx);

	const auto source = core::makeActiveSource(&tx, 0.0, chirp_duration);
	const auto path = makeReflectedPath(&rx, tau, 1.0);
	core::FmcwChirpBoundaryTracker tracker;

	const RealType dt = 1.0 / params::simSamplingRate();
	const RealType first_time = 20.0e-6;
	const std::size_t sample_count = 500;
	std::vector<RealType> dechirped_phase;
	dechirped_phase.reserve(sample_count);

	for (std::size_t i = 0; i < sample_count; ++i)
	{
		const RealType t = first_time + static_cast<RealType>(i) * dt;
		const ComplexType sample =
			simulation::calculateStreamingPathContribution(source, &rx, path, t, nullptr, &tracker);
		const ComplexType dechirped =
			sample * std::polar(1.0, -wave.getFmcwChirpWaveform()->basebandPhaseForChirpTime(t));
		dechirped_phase.push_back(std::arg(dechirped));
	}

	RealType unwrapped_span = 0.0;
	for (std::size_t i = 1; i < dechirped_phase.size(); ++i)
	{
		unwrapped_span += unwrapDelta(dechirped_phase[i] - dechirped_phase[i - 1]);
	}

	const RealType measured_beat_hz =
		unwrapped_span / (2.0 * PI * dt * static_cast<RealType>(dechirped_phase.size() - 1));
	const RealType expected_beat_hz = chirp_rate * tau;
	REQUIRE_THAT(measured_beat_hz, WithinRel(expected_beat_hz, 1.0e-3));
}

TEST_CASE("calculateReflectedPathContribution amplitude matches bistatic equation with signal power",
		  "[simulation][channel_model][streaming][reflected]")
{
	ParamGuard const guard;
	params::params.reset();

	const RealType c = params::c();
	const RealType carrier = 1.0e9;
	const RealType lambda = c / carrier;
	const RealType rcs = 10.0;
	const RealType power = 100.0;

	const math::Vec3 tx_pos{0.0, 0.0, 0.0};
	const math::Vec3 tgt_pos{1000.0, 0.0, 0.0};
	const math::Vec3 rx_pos{1000.0, 1000.0, 0.0};

	const RealType r1 = (tgt_pos - tx_pos).length();
	const RealType r2 = (rx_pos - tgt_pos).length();

	const RealType bistatic = (rcs * lambda * lambda) / (64.0 * PI * PI * PI * r1 * r1 * r2 * r2);
	const RealType expected_amplitude = std::sqrt(power * bistatic);

	radar::Platform tx_plat("tx_plat");
	setupPlatform(tx_plat, tx_pos);
	radar::Platform rx_plat("rx_plat");
	setupPlatform(rx_plat, rx_pos);

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::CW_MODE);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);

	fers_signal::RadarSignal wave("sig", power, carrier, fers_signal::CwWaveform{});
	tx.setSignal(&wave);

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::CW_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);

	const auto source = core::makeActiveSource(&tx, -10.0, 10.0);
	const auto path = makeReflectedPath(&rx, (r1 + r2) / c, std::sqrt(bistatic));

	const ComplexType result = simulation::calculateStreamingPathContribution(source, &rx, path, 0.0);

	REQUIRE_THAT(std::abs(result), WithinRel(expected_amplitude, 1e-6));
}

TEST_CASE("calculateReflectedPathContribution phase matches bistatic propagation delay",
		  "[simulation][channel_model][streaming][reflected]")
{
	ParamGuard const guard;
	params::params.reset();

	const RealType c = params::c();
	const RealType carrier = 1.0e9;

	const math::Vec3 tx_pos{0.0, 0.0, 0.0};
	const math::Vec3 tgt_pos{1000.0, 0.0, 0.0};
	const math::Vec3 rx_pos{2000.0, 0.0, 0.0};

	const RealType r1 = (tgt_pos - tx_pos).length();
	const RealType r2 = (rx_pos - tgt_pos).length();
	const RealType tau = (r1 + r2) / c;

	// Expected phase = -2*pi*f*tau + timing_phase (0 for default timing)
	const RealType expected_phase = -2.0 * PI * carrier * tau;

	radar::Platform tx_plat("tx_plat");
	setupPlatform(tx_plat, tx_pos);
	radar::Platform rx_plat("rx_plat");
	setupPlatform(rx_plat, rx_pos);

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::CW_MODE);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);

	fers_signal::RadarSignal wave("sig", 1.0, carrier, fers_signal::CwWaveform{});
	tx.setSignal(&wave);

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::CW_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);

	const auto source = core::makeActiveSource(&tx, -10.0, 10.0);
	const auto path = makeReflectedPath(&rx, tau, 1.0);

	const ComplexType result = simulation::calculateStreamingPathContribution(source, &rx, path, 0.0);
	const RealType result_phase = std::arg(result);

	// Wrap expected phase to [-pi, pi] for comparison
	RealType wrapped = std::fmod(expected_phase, 2.0 * PI);
	if (wrapped > PI)
		wrapped -= 2.0 * PI;
	if (wrapped < -PI)
		wrapped += 2.0 * PI;

	REQUIRE_THAT(result_phase, WithinAbs(wrapped, 1e-6));
}

TEST_CASE("calculateReflectedPathContribution preserves stronger phase-noise cancellation for shorter delays",
		  "[simulation][channel_model][streaming][reflected]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setTime(0.0, 1.0);
	params::setRate(10.0);
	params::setOversampleRatio(1);
	params::setC(10.0);

	const RealType carrier = 1.0;
	const RealType sample_time = 0.5;

	// near target: tx at (-0.5,0,0), rx at (0.5,0,0), target at (0,0,0) => tau = 0.1
	// far target: same tx/rx, target at (1,0,0) => tau = 0.2
	const RealType near_tau = 0.1;
	const RealType far_tau = 0.2;

	radar::Platform tx_plat("tx_plat");
	setupPlatform(tx_plat, math::Vec3{-0.5, 0.0, 0.0});
	radar::Platform rx_plat("rx_plat");
	setupPlatform(rx_plat, math::Vec3{0.5, 0.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	timing::PrototypeTiming prototype("clk");
	prototype.setFrequency(carrier);
	prototype.setFreqOffset(1.0);
	auto timing = std::make_shared<timing::Timing>("clk", 42);
	timing->initializeModel(&prototype);
	const auto lookup = makeLookup(timing);

	radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::CW_MODE);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);

	fers_signal::RadarSignal wave("sig", 1.0, carrier, fers_signal::CwWaveform{});
	tx.setSignal(&wave);

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::CW_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);

	const auto source = core::makeActiveSource(&tx, 0.0, 1.0);
	const auto near_path = makeReflectedPath(&rx, near_tau, 1.0);
	const auto far_path = makeReflectedPath(&rx, far_tau, 1.0);

	const ComplexType near_ideal = simulation::calculateStreamingPathContribution(source, &rx, near_path, sample_time);
	const ComplexType near_delayed =
		simulation::calculateStreamingPathContribution(source, &rx, near_path, sample_time, &lookup);
	const ComplexType far_ideal = simulation::calculateStreamingPathContribution(source, &rx, far_path, sample_time);
	const ComplexType far_delayed =
		simulation::calculateStreamingPathContribution(source, &rx, far_path, sample_time, &lookup);

	const RealType near_extra_phase = std::abs(std::arg(near_delayed * std::conj(near_ideal)));
	const RealType far_extra_phase = std::abs(std::arg(far_delayed * std::conj(far_ideal)));

	REQUIRE_THAT(near_extra_phase, WithinAbs(0.2 * PI, 1e-6));
	REQUIRE_THAT(far_extra_phase, WithinAbs(0.4 * PI, 1e-6));
	REQUIRE(near_extra_phase < far_extra_phase);
}

// =============================================================================
// FMCW chirp-boundary tracker tests.
// =============================================================================

TEST_CASE("FMCW streaming direct path preserves in-flight segment-end tail",
		  "[simulation][channel_model][streaming][direct][fmcw]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setRate(80.0e6);
	params::setSimSamplingRate(80.0e6);
	params::setOversampleRatio(1);
	params::setTime(0.0, 0.00502);

	const RealType dist = 1500.0;
	const RealType tau = dist / params::c();
	const RealType segment_end = 0.005;
	const RealType sample_rate = 80.0e6;
	const RealType dt = 1.0 / sample_rate;
	const RealType chirp_bandwidth = 20.0e6;
	const RealType chirp_duration = 250.0e-6;
	const RealType chirp_period = chirp_duration;
	const RealType chirp_rate = chirp_bandwidth / chirp_duration;
	const auto sample_count = static_cast<std::size_t>(std::ceil(tau / dt));

	radar::Platform tx_plat("tx_plat");
	setupPlatform(tx_plat, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_plat("rx_plat");
	setupPlatform(rx_plat, math::Vec3{dist, 0.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::FMCW_MODE);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);

	fers_signal::RadarSignal wave(
		"fmcw", 1000.0, 10.0e9, fers_signal::FmcwChirpWaveform(chirp_bandwidth, chirp_duration, chirp_period, 0.0, 20));
	tx.setSignal(&wave);
	const auto* fmcw = wave.getFmcwChirpWaveform();

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::CW_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);

	const core::ActiveStreamingSource source = core::makeActiveSource(&tx, 0.0, segment_end);
	const auto path = makeDirectPath(&rx, dist, 1.0);
	core::FmcwChirpBoundaryTracker reference_tracker;
	core::FmcwChirpBoundaryTracker tail_tracker;
	std::vector<ComplexType> reference_samples;
	std::vector<ComplexType> tail_samples;
	reference_samples.reserve(sample_count);
	tail_samples.reserve(sample_count);

	for (std::size_t i = 0; i < sample_count; ++i)
	{
		const RealType offset = static_cast<RealType>(i) * dt;
		reference_samples.push_back(simulation::calculateStreamingPathContribution(
			source, &rx, path, segment_end - tau + offset, nullptr, &reference_tracker));
		tail_samples.push_back(simulation::calculateStreamingPathContribution(source, &rx, path, segment_end + offset,
																			  nullptr, &tail_tracker));
	}

	const RealType reference_rms = rms(reference_samples);
	const RealType tail_rms = rms(tail_samples);
	REQUIRE(reference_samples.size() >= 400);
	REQUIRE(reference_rms > 0.0);
	REQUIRE(tail_rms > 0.0);
	REQUIRE_THAT(tail_rms, WithinRel(reference_rms, 1.0e-12));

	const RealType last_chirp_start = segment_end - chirp_period;
	RealType unwrapped_span = 0.0;
	RealType previous_phase = 0.0;
	for (std::size_t i = 0; i < tail_samples.size(); ++i)
	{
		const RealType t = segment_end + static_cast<RealType>(i) * dt;
		const RealType local_time = t - last_chirp_start;
		const ComplexType dechirped = tail_samples[i] * std::polar(1.0, -fmcw->basebandPhaseForChirpTime(local_time));
		const RealType phase = std::arg(dechirped);
		if (i > 0)
		{
			unwrapped_span += unwrapDelta(phase - previous_phase);
		}
		previous_phase = phase;
	}

	const RealType measured_beat_hz = unwrapped_span / (2.0 * PI * dt * static_cast<RealType>(tail_samples.size() - 1));
	const RealType expected_beat_hz = -chirp_rate * tau;
	REQUIRE_THAT(measured_beat_hz, WithinRel(expected_beat_hz, 1.0e-6));
}

TEST_CASE("FMCW streaming direct path keeps chirp cache per source",
		  "[simulation][channel_model][streaming][direct][fmcw]")
{
	ParamGuard const guard;
	params::params.reset();

	const RealType dist = 300.0;
	const RealType tau = dist / params::c();
	const RealType u_ret = 20.0e-6;
	const RealType rx_time = tau + u_ret;
	const RealType chirp_duration = 100.0e-6;
	const RealType chirp_period = 100.0e-6;

	radar::Platform tx1_plat("tx1_plat");
	setupPlatform(tx1_plat, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform tx2_plat("tx2_plat");
	setupPlatform(tx2_plat, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_plat("rx_plat");
	setupPlatform(rx_plat, math::Vec3{dist, 0.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx1(&tx1_plat, "tx1", radar::OperationMode::FMCW_MODE);
	tx1.setAntenna(&iso_ant);
	tx1.setTiming(timing);
	fers_signal::RadarSignal wave1("fmcw1", 1.0, 10.0e9,
								   fers_signal::FmcwChirpWaveform(1.0e6, chirp_duration, chirp_period));
	tx1.setSignal(&wave1);

	radar::Transmitter tx2(&tx2_plat, "tx2", radar::OperationMode::FMCW_MODE);
	tx2.setAntenna(&iso_ant);
	tx2.setTiming(timing);
	fers_signal::RadarSignal wave2("fmcw2", 1.0, 10.0e9,
								   fers_signal::FmcwChirpWaveform(4.0e6, chirp_duration, chirp_period));
	tx2.setSignal(&wave2);

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::CW_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);

	const core::ActiveStreamingSource source1 = core::makeActiveSource(&tx1, 0.0, 1.0e-3);
	const core::ActiveStreamingSource source2 = core::makeActiveSource(&tx2, 0.0, 1.0e-3);
	const auto path1 = makeDirectPath(&rx, dist, 1.0);
	const auto path2 = makeDirectPath(&rx, dist, 1.0);
	const ComplexType sample1 = simulation::calculateStreamingPathContribution(source1, &rx, path1, rx_time);
	const ComplexType sample2 = simulation::calculateStreamingPathContribution(source2, &rx, path2, rx_time);

	REQUIRE(std::abs(sample1) > 0.0);
	REQUIRE(std::abs(sample2) > 0.0);
	const RealType expected_delta =
		(source2.two_pi_f0 - source1.two_pi_f0) * u_ret + (source2.s_pi_alpha - source1.s_pi_alpha) * u_ret * u_ret;
	const RealType measured_delta = std::arg(sample2 / sample1);
	REQUIRE_THAT(unwrapDelta(measured_delta - expected_delta), WithinAbs(0.0, 1.0e-10));
}

TEST_CASE("FMCW chirp-boundary tracker matches cold-path direct contribution across chirps",
		  "[simulation][channel_model][streaming][direct][fmcw]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setRate(1.0e6);
	params::setSimSamplingRate(1.0e6);
	params::setOversampleRatio(1);
	params::setTime(0.0, 3.0e-4);

	radar::Platform tx_platform("tx_platform");
	setupPlatform(tx_platform, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_platform("rx_platform");
	setupPlatform(rx_platform, math::Vec3{300.0, 0.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 44);

	radar::Transmitter tx(&tx_platform, "tx", radar::OperationMode::FMCW_MODE, 102);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);
	fers_signal::RadarSignal wave("fmcw", 5.0, 20.0e6, fers_signal::FmcwChirpWaveform(2.0e6, 2.0e-5, 5.0e-5), 302);
	tx.setSignal(&wave);

	radar::Receiver rx(&rx_platform, "rx", 45, radar::OperationMode::CW_MODE, 202);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);

	const core::ActiveStreamingSource source = core::makeActiveSource(&tx, 0.0, 3.0e-4);
	const auto path = makeDirectPath(&rx, 300.0, 1.0);
	core::FmcwChirpBoundaryTracker tracker;
	const RealType dt = 1.0 / params::simSamplingRate();

	for (std::size_t i = 0; i < 260; ++i)
	{
		const RealType t = static_cast<RealType>(i) * dt;
		const ComplexType tracked =
			simulation::calculateStreamingPathContribution(source, &rx, path, t, nullptr, &tracker);
		const ComplexType cold = simulation::calculateStreamingPathContribution(source, &rx, path, t);
		REQUIRE_THAT(tracked.real(), WithinAbs(cold.real(), 1.0e-12));
		REQUIRE_THAT(tracked.imag(), WithinAbs(cold.imag(), 1.0e-12));
	}
	REQUIRE(tracker.initialized);
	REQUIRE(tracker.n_current > 3);
}

// =============================================================================
// File-backed CW/FMCW streaming: envelope sampling
// =============================================================================

TEST_CASE("File streaming reference follows finite HDF5-style samples for CW and FMCW",
		  "[simulation][channel_model][streaming][file][hdf5]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setOversampleRatio(1);
	constexpr RealType rate = 32.0;
	constexpr unsigned sample_count = 64;
	constexpr RealType baseband_frequency = 3.0;
	constexpr RealType carrier = 5.0;
	std::vector<ComplexType> samples(sample_count);
	for (unsigned i = 0; i < sample_count; ++i)
	{
		const RealType phase = 2.0 * PI * baseband_frequency * static_cast<RealType>(i) / rate;
		samples[i] = {std::cos(phase), std::sin(phase)};
	}

	for (const auto kind : {fers_signal::FileWaveformKind::Cw, fers_signal::FileWaveformKind::Fmcw})
	{
		fers_signal::FileWaveform wave{};
		wave.setKind(kind);
		wave.load(samples, sample_count, rate);
		fers_signal::RadarSignal const waveform("file", 4.0, carrier, std::move(wave));
		const auto source = core::makeActiveSourceFromWaveform(&waveform, 0.25, 5.0);
		const RealType local_time = 0.3125;
		ComplexType first{};
		ComplexType after_end{};
		REQUIRE(simulation::calculateStreamingReferenceSample(source, 0.25 + local_time, nullptr, first));
		REQUIRE_FALSE(
			simulation::calculateStreamingReferenceSample(source, 0.25 + local_time + 2.0, nullptr, after_end));
		const RealType expected_phase = 2.0 * PI * baseband_frequency * local_time;
		const ComplexType expected = std::polar(1.0, expected_phase);
		REQUIRE_THAT(first.real(), WithinAbs(expected.real(), 1e-6));
		REQUIRE_THAT(first.imag(), WithinAbs(expected.imag(), 1e-6));
		REQUIRE((after_end == ComplexType{0.0, 0.0}));
		REQUIRE_THAT(source.segment_end, WithinAbs(2.25, 1e-12));
	}
}

TEST_CASE("File CW direct path samples the complex envelope at retarded transmit time",
		  "[simulation][channel_model][streaming][file]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setC(100.0);
	params::setOversampleRatio(1);
	constexpr RealType rate = 32.0;
	constexpr RealType tone = 3.0;
	constexpr unsigned sample_count = 64;
	std::vector<ComplexType> samples(sample_count);
	for (unsigned i = 0; i < sample_count; ++i)
	{
		const RealType phase = 2.0 * PI * tone * static_cast<RealType>(i) / rate;
		samples[i] = std::polar(1.0, phase);
	}

	radar::Platform tx_platform("tx-platform");
	setupPlatform(tx_platform, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_platform("rx-platform");
	setupPlatform(rx_platform, math::Vec3{10.0, 0.0, 0.0});
	antenna::Isotropic antenna("isotropic");
	auto clock = std::make_shared<timing::Timing>("clock", 42);
	radar::Transmitter tx(&tx_platform, "tx", radar::OperationMode::CW_MODE);
	tx.setAntenna(&antenna);
	tx.setTiming(clock);
	fers_signal::FileWaveform wave{};
	wave.setKind(fers_signal::FileWaveformKind::Cw);
	wave.load(samples, sample_count, rate);
	fers_signal::RadarSignal waveform("file-cw", 4.0, 10.0, std::move(wave));
	tx.setSignal(&waveform);
	radar::Receiver rx(&rx_platform, "rx", 43, radar::OperationMode::CW_MODE);
	rx.setAntenna(&antenna);
	rx.setTiming(clock);

	const auto source = core::makeActiveSource(&tx, 0.0, 5.0);
	const auto path = makeDirectPath(&rx, 10.0, 1.0);
	const RealType delay = path.delay;
	const RealType first_local_time = 0.3125;
	const ComplexType first =
		simulation::calculateStreamingPathContribution(source, &rx, path, delay + first_local_time);
	const ComplexType next =
		simulation::calculateStreamingPathContribution(source, &rx, path, delay + first_local_time + 1.0 / rate);
	const ComplexType after_end =
		simulation::calculateStreamingPathContribution(source, &rx, path, delay + first_local_time + 2.0);

	REQUIRE(std::abs(first) > 0.0);
	REQUIRE_THAT(std::abs(next / first), WithinAbs(1.0, 1e-6));
	REQUIRE_THAT(std::arg(next / first), WithinAbs(2.0 * PI * tone / rate, 1e-6));
	REQUIRE(std::abs(after_end) == 0.0);
	REQUIRE_THAT(source.segment_end, WithinAbs(2.0, 1e-12));
}

// =============================================================================
// updateCoarseStreamingCache / interpolateCoarseStreamingPaths: CW coarse-rate
// sampling's bracketed-interpolation cache, exercised directly (independent of
// the full SimulationEngine/PropagationModel).
// =============================================================================

TEST_CASE("interpolateCoarseStreamingPaths fades a newly-appeared path in from zero gain",
		  "[simulation][channel_model][coarse]")
{
	radar::Platform rx_platform("rx_plat");
	setupPlatform(rx_platform, math::Vec3{0.0, 0.0, 0.0});
	radar::Receiver rx(&rx_platform, "rx", 42, radar::OperationMode::CW_MODE);

	core::ReceiverTrackerCache cache;
	cache.coarse_path_samples.resize(1);

	// A path seen for the first time: no earlier sample to taper from, so "lo" is a zero-gain copy
	// of the fresh value (serial::Response's simpleTaperPoint fallback), placed at the interval's
	// start rather than snapping straight to full gain.
	const auto path = makeDirectPath(&rx, 100.0, 0.5);
	simulation::updateCoarseStreamingCache(cache, {path}, /*interval_start_time=*/0.0, /*sample_time=*/1.0);

	const auto at_lo = simulation::interpolateCoarseStreamingPaths(cache, rx, 0.0);
	const auto at_mid = simulation::interpolateCoarseStreamingPaths(cache, rx, 0.5);
	const auto at_hi = simulation::interpolateCoarseStreamingPaths(cache, rx, 1.0);

	REQUIRE(at_lo.size() == 1);
	REQUIRE(at_mid.size() == 1);
	REQUIRE(at_hi.size() == 1);

	REQUIRE_THAT(std::abs(at_lo.front().gain), WithinAbs(0.0, 1e-12));
	REQUIRE_THAT(std::abs(at_mid.front().gain), WithinAbs(std::abs(path.gain) * 0.5, 1e-12));
	REQUIRE_THAT(std::abs(at_hi.front().gain), WithinAbs(std::abs(path.gain), 1e-12));

	// Delay is copied (not extrapolated) for the synthetic "lo", since there's no earlier sample to
	// estimate a rate from yet.
	REQUIRE_THAT(at_lo.front().delay, WithinAbs(path.delay, 1e-12));
	REQUIRE_THAT(at_hi.front().delay, WithinAbs(path.delay, 1e-12));
}

TEST_CASE("interpolateCoarseStreamingPaths linearly interpolates delay/gain between two real samples",
		  "[simulation][channel_model][coarse]")
{
	radar::Platform rx_platform("rx_plat");
	setupPlatform(rx_platform, math::Vec3{0.0, 0.0, 0.0});
	radar::Receiver rx(&rx_platform, "rx", 42, radar::OperationMode::CW_MODE);

	core::ReceiverTrackerCache cache;
	cache.coarse_path_samples.resize(1);

	const auto path_t0 = makeDirectPath(&rx, 100.0, 0.5);
	auto path_t1 = makeDirectPath(&rx, 100.0, 0.5);
	path_t1.delay = path_t0.delay + 1.0;
	path_t1.gain = path_t0.gain * 3.0;

	simulation::updateCoarseStreamingCache(cache, {path_t0}, /*interval_start_time=*/-1.0, /*sample_time=*/0.0);
	simulation::updateCoarseStreamingCache(cache, {path_t1}, /*interval_start_time=*/0.0, /*sample_time=*/1.0);

	const auto at_lo = simulation::interpolateCoarseStreamingPaths(cache, rx, 0.0);
	const auto at_mid = simulation::interpolateCoarseStreamingPaths(cache, rx, 0.5);
	const auto at_hi = simulation::interpolateCoarseStreamingPaths(cache, rx, 1.0);

	REQUIRE_THAT(at_lo.front().delay, WithinAbs(path_t0.delay, 1e-9));
	REQUIRE_THAT(std::abs(at_lo.front().gain), WithinAbs(std::abs(path_t0.gain), 1e-9));

	REQUIRE_THAT(at_mid.front().delay, WithinAbs((path_t0.delay + path_t1.delay) / 2.0, 1e-9));
	REQUIRE_THAT(std::abs(at_mid.front().gain), WithinAbs((std::abs(path_t0.gain) + std::abs(path_t1.gain)) / 2.0, 1e-9));

	REQUIRE_THAT(at_hi.front().delay, WithinAbs(path_t1.delay, 1e-9));
	REQUIRE_THAT(std::abs(at_hi.front().gain), WithinAbs(std::abs(path_t1.gain), 1e-9));
}

TEST_CASE("updateCoarseStreamingCache tapers a disappearing path to zero gain using its observed rate, "
		  "then drops it",
		  "[simulation][channel_model][coarse]")
{
	radar::Platform rx_platform("rx_plat");
	setupPlatform(rx_platform, math::Vec3{0.0, 0.0, 0.0});
	radar::Receiver rx(&rx_platform, "rx", 42, radar::OperationMode::CW_MODE);

	core::ReceiverTrackerCache cache;
	cache.coarse_path_samples.resize(1);

	// Two real samples establish a genuine delay rate of +2/s (10 -> 12 over one second).
	auto path_t0 = makeDirectPath(&rx, 100.0, 0.5);
	path_t0.delay = 10.0;
	auto path_t1 = path_t0;
	path_t1.delay = 12.0;

	simulation::updateCoarseStreamingCache(cache, {path_t0}, /*interval_start_time=*/-1.0, /*sample_time=*/0.0);
	simulation::updateCoarseStreamingCache(cache, {path_t1}, /*interval_start_time=*/0.0, /*sample_time=*/1.0);
	REQUIRE(simulation::interpolateCoarseStreamingPaths(cache, rx, 1.0).size() == 1);

	// The path is missing from this sample (dropped out) - mirror serial::Response's
	// extrapolateTaperPoint: project the delay forward at the observed rate, but force gain to
	// exactly zero, rather than snapping or bridging toward a value that no longer exists.
	simulation::updateCoarseStreamingCache(cache, {}, /*interval_start_time=*/1.0, /*sample_time=*/2.0);

	const auto at_lo = simulation::interpolateCoarseStreamingPaths(cache, rx, 1.0); // == old hi, still real
	const auto at_mid = simulation::interpolateCoarseStreamingPaths(cache, rx, 1.5);
	const auto at_hi = simulation::interpolateCoarseStreamingPaths(cache, rx, 2.0); // synthetic, zero gain

	REQUIRE(at_lo.size() == 1);
	REQUIRE(at_mid.size() == 1);
	REQUIRE(at_hi.size() == 1);

	REQUIRE_THAT(std::abs(at_lo.front().gain), WithinAbs(std::abs(path_t1.gain), 1e-9));
	REQUIRE_THAT(std::abs(at_mid.front().gain), WithinAbs(std::abs(path_t1.gain) * 0.5, 1e-9));
	REQUIRE_THAT(std::abs(at_hi.front().gain), WithinAbs(0.0, 1e-12));

	// Delay keeps extrapolating at the observed +2/s rate even as gain fades to zero.
	REQUIRE_THAT(at_hi.front().delay, WithinAbs(14.0, 1e-9));

	// Still missing next round: the path is now fully tapered out, so it's dropped rather than
	// tapered again.
	simulation::updateCoarseStreamingCache(cache, {}, /*interval_start_time=*/2.0, /*sample_time=*/3.0);
	REQUIRE(simulation::interpolateCoarseStreamingPaths(cache, rx, 2.5).empty());
}

TEST_CASE("updateCoarseStreamingCache tracks multiple paths by source index and path_id independently",
		  "[simulation][channel_model][coarse]")
{
	radar::Platform rx_platform("rx_plat");
	setupPlatform(rx_platform, math::Vec3{0.0, 0.0, 0.0});
	radar::Receiver rx(&rx_platform, "rx", 42, radar::OperationMode::CW_MODE);

	core::ReceiverTrackerCache cache;
	cache.coarse_path_samples.resize(1);

	auto path_a = makeDirectPath(&rx, 100.0, 0.5); // path_id == 0 (direct)
	auto path_b = makeDirectPath(&rx, 100.0, 0.5);
	path_b.path_id = 99;
	path_b.delay = path_a.delay + 10.0;

	simulation::updateCoarseStreamingCache(cache, {path_a, path_b}, /*interval_start_time=*/-1.0,
										   /*sample_time=*/0.0);

	const auto both = simulation::interpolateCoarseStreamingPaths(cache, rx, 0.0);
	REQUIRE(both.size() == 2);

	// Only path_a disappears; path_b must survive, with its own real values untouched.
	simulation::updateCoarseStreamingCache(cache, {path_b}, /*interval_start_time=*/0.0, /*sample_time=*/1.0);

	const auto after = simulation::interpolateCoarseStreamingPaths(cache, rx, 1.0);
	REQUIRE(after.size() == 2); // path_a is tapering (gain 0 at this instant), path_b is real.

	const auto found_b = std::ranges::find_if(after, [](const auto& p) { return p.path_id == 99; });
	REQUIRE(found_b != after.end());
	REQUIRE_THAT(std::abs(found_b->gain), WithinAbs(std::abs(path_b.gain), 1e-9));

	const auto found_a =
		std::ranges::find_if(after, [&](const auto& p) { return p.path_id == path_a.path_id; });
	REQUIRE(found_a != after.end());
	REQUIRE_THAT(std::abs(found_a->gain), WithinAbs(0.0, 1e-12)); // fully tapered at this instant

	// path_a is gone outright once it's still missing on a later round.
	simulation::updateCoarseStreamingCache(cache, {path_b}, /*interval_start_time=*/1.0, /*sample_time=*/2.0);
	const auto later = simulation::interpolateCoarseStreamingPaths(cache, rx, 2.0);
	REQUIRE(later.size() == 1);
	REQUIRE(later.front().path_id == 99);
}
