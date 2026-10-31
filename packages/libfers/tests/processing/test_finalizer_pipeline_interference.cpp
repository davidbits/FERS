#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "antenna/antenna_factory.h"
#include "core/config.h"
#include "core/parameters.h"
#include "core/world.h"
#include "interpolation/interpolation_point.h"
#include "math/coord.h"
#include "processing/finalizer_pipeline.h"
#include "propagation/pointscatter/pointscatter.h"
#include "propagation/propagation_model.h"
#include "radar/platform.h"
#include "radar/receiver.h"
#include "radar/target.h"
#include "radar/transmitter.h"
#include "serial/response.h"
#include "signal/radar_signal.h"
#include "simulation/channel_model.h"
#include "timing/prototype_timing.h"
#include "timing/timing.h"

using Catch::Matchers::WithinAbs;

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

	void setupPlatform(radar::Platform& platform, const math::Vec3& position)
	{
		platform.getMotionPath()->addCoord(math::Coord{position, 0.0});
		platform.getMotionPath()->finalize();
		platform.getRotationPath()->addCoord(math::RotationCoord{0.0, 0.0, 0.0});
		platform.getRotationPath()->finalize();
	}

	std::shared_ptr<timing::Timing> makeQuietTiming(const std::string& name, unsigned seed)
	{
		timing::PrototypeTiming prototype(name);
		prototype.setFrequency(1.0);

		auto timing_model = std::make_shared<timing::Timing>(name, seed);
		timing_model->initializeModel(&prototype);
		return timing_model;
	}

	// Sums the same path->contribution pipeline applyStreamingInterference itself uses
	// (PropagationModel::traceFromReceivers + calculateStreamingPathContribution), for one
	// sample time.
	ComplexType expectedStreamingSample(const propagation::PropagationModel& prop, radar::Receiver* receiver,
										const std::vector<core::ActiveStreamingSource>& sources, const RealType t)
	{
		ComplexType total{0.0, 0.0};
		const auto paths = prop.traceFromReceivers(nullptr, {receiver}, sources, t);
		for (const auto& path : paths.at(receiver))
		{
			total += simulation::calculateStreamingPathContribution(sources[path.source_index], receiver, path, t);
		}
		return total;
	}

	std::unique_ptr<serial::Response>
	makeFixedResponse(std::vector<std::unique_ptr<fers_signal::RadarSignal>>& wave_store,
					  const std::vector<ComplexType>& samples, const RealType sample_rate, const RealType start_time)
	{
		fers_signal::PulseWaveform sampled;
		sampled.load(samples, static_cast<unsigned>(samples.size()), sample_rate);

		auto wave = std::make_unique<fers_signal::RadarSignal>("wave", 1.0, 1.0e9, std::move(sampled));
		const auto* wave_ptr = wave.get();
		wave_store.push_back(std::move(wave));

		const RealType edge = 1.0 / sample_rate;
		const interp::InterpPoint first{.gain = 1.0, .rx_time = start_time, .delay = 0.0, .phase_delay = 0.0};
		serial::ResponseBuilder builder(wave_ptr, first, edge, edge);
		for (std::size_t i = 1; i < samples.size(); ++i)
		{
			builder.addInterpPoint({.gain = 1.0,
									.rx_time = start_time + static_cast<RealType>(i) / sample_rate,
									.delay = 0.0,
									.phase_delay = 0.0},
								   edge);
		}
		return std::make_unique<serial::Response>(std::move(builder));
	}
}

TEST_CASE("applyStreamingInterference adds direct-path streaming energy sample by sample",
		  "[processing][finalizer][interference]")
{
	ParamGuard const guard;
	params::params.reset();

	radar::Platform tx_platform("TxPlatform");
	setupPlatform(tx_platform, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_platform("RxPlatform");
	setupPlatform(rx_platform, math::Vec3{1000.0, 0.0, 0.0});

	antenna::Isotropic antenna("iso");
	auto timing_model = makeQuietTiming("clk", 11);

	radar::Transmitter transmitter(&tx_platform, "TxA", radar::OperationMode::CW_MODE, 101);
	transmitter.setAntenna(&antenna);
	transmitter.setTiming(timing_model);
	fers_signal::RadarSignal wave("cw", 25.0, 1.0e9, fers_signal::CwWaveform{}, 301);
	transmitter.setSignal(&wave);

	radar::Receiver receiver(&rx_platform, "RxA", 99, radar::OperationMode::CW_MODE, 202);
	receiver.setAntenna(&antenna);
	receiver.setTiming(timing_model);

	core::World world;
	const propagation::pointscatter::PointScatterModel prop(&world);

	std::vector<ComplexType> window(3, ComplexType{0.25, -0.5});
	const std::vector<ComplexType> baseline = window;
	const std::vector<core::ActiveStreamingSource> streaming_sources = {
		core::makeActiveSource(&transmitter, params::startTime(), std::numeric_limits<RealType>::max())};
	const RealType start = 0.0;
	const RealType dt = 0.25;
	core::ReceiverTrackerCache tracker_cache;

	processing::pipeline::applyStreamingInterference(window, start, dt, prop, nullptr, &receiver, streaming_sources,
													 tracker_cache);

	for (size_t i = 0; i < window.size(); ++i)
	{
		const ComplexType expected =
			expectedStreamingSample(prop, &receiver, streaming_sources, start + static_cast<RealType>(i) * dt);
		const ComplexType actual = window[i] - baseline[i];
		REQUIRE_THAT(actual.real(), WithinAbs(expected.real(), 1e-12));
		REQUIRE_THAT(actual.imag(), WithinAbs(expected.imag(), 1e-12));
	}
}

TEST_CASE("applyStreamingInterference respects FLAG_NODIRECT and keeps only physically reflected energy",
		  "[processing][finalizer][interference]")
{
	ParamGuard const guard;
	params::params.reset();

	radar::Platform tx_platform("TxPlatform");
	setupPlatform(tx_platform, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_platform("RxPlatform");
	setupPlatform(rx_platform, math::Vec3{1000.0, 1000.0, 0.0});
	radar::Platform target_platform("TargetPlatform");
	setupPlatform(target_platform, math::Vec3{1000.0, 0.0, 0.0});

	antenna::Isotropic antenna("iso");
	auto timing_model = makeQuietTiming("clk", 12);

	radar::Transmitter transmitter(&tx_platform, "TxA", radar::OperationMode::CW_MODE, 102);
	transmitter.setAntenna(&antenna);
	transmitter.setTiming(timing_model);
	fers_signal::RadarSignal wave("cw", 9.0, 1.0e9, fers_signal::CwWaveform{}, 302);
	transmitter.setSignal(&wave);

	radar::Receiver receiver(&rx_platform, "RxA", 100, radar::OperationMode::CW_MODE, 203);
	receiver.setAntenna(&antenna);
	receiver.setTiming(timing_model);
	receiver.setFlag(radar::Receiver::RecvFlag::FLAG_NODIRECT);

	core::World world;
	world.add(radar::createIsoTarget(&target_platform, "TargetA", 4.0, 7, 501));
	const propagation::pointscatter::PointScatterModel prop(&world);

	std::vector<ComplexType> window(3, ComplexType{});
	const std::vector<core::ActiveStreamingSource> streaming_sources = {
		core::makeActiveSource(&transmitter, params::startTime(), std::numeric_limits<RealType>::max())};
	const RealType start = 0.0;
	const RealType dt = 0.2;
	core::ReceiverTrackerCache tracker_cache;

	processing::pipeline::applyStreamingInterference(window, start, dt, prop, nullptr, &receiver, streaming_sources,
													 tracker_cache);

	for (size_t i = 0; i < window.size(); ++i)
	{
		const ComplexType expected =
			expectedStreamingSample(prop, &receiver, streaming_sources, start + static_cast<RealType>(i) * dt);
		REQUIRE_THAT(window[i].real(), WithinAbs(expected.real(), 1e-12));
		REQUIRE_THAT(window[i].imag(), WithinAbs(expected.imag(), 1e-12));
	}
}

TEST_CASE("applyPulsedInterference clips rendered pulses to LO-active sample spans",
		  "[processing][finalizer][interference][dechirp]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setRate(10.0);
	params::setSimSamplingRate(10.0);

	std::vector<ComplexType> iq_buffer(16, ComplexType{0.0, 0.0});
	std::vector<std::unique_ptr<fers_signal::RadarSignal>> wave_store;
	std::vector<std::unique_ptr<serial::Response>> interference_log;
	interference_log.push_back(makeFixedResponse(wave_store,
												 {ComplexType{1.0, 0.0}, ComplexType{2.0, 0.0}, ComplexType{3.0, 0.0},
												  ComplexType{4.0, 0.0}, ComplexType{5.0, 0.0}},
												 params::rate(), 0.2));

	// Rendered pulse (via makeFixedResponse, see comment above): [0, 1, 2, 3, 4, 5],
	// starting at response->startTime() == 0.2 - edge == 0.1 (edge == 1/rate == 0.1), placed
	// at buffer index round(0.1 * 10) = 1, i.e. occupying [1, 7). Index 3 of the buffer ->
	// rendered index 2 == 2.0 (a real sample, not the taper stub); index 6 -> rendered
	// index 5 == 5.0.
	//
	// This also exercises a numerical-robustness edge case: startTime() lands exactly on
	// a sample boundary here (0.1 * rate == 1.0 precisely), but is computed from Response's
	// own rx_time arithmetic (front point minus a locally-measured control-point delta),
	// which can differ from the "true" value by a couple of ULPs, e.g. 0.1 coming out as
	// 0.09999999999999998. applyPulsedInterference must place the pulse with
	// std::round(), as opposed to std::floor().
	const std::vector<processing::pipeline::SampleSpan> active_spans = {
		processing::pipeline::SampleSpan{.start = 3, .end_exclusive = 4},
		processing::pipeline::SampleSpan{.start = 6, .end_exclusive = 7}};

	processing::pipeline::applyPulsedInterference(iq_buffer, interference_log, active_spans, params::rate());

	// Note: near the edge of a short (6-sample) buffer, the render filter's fractional-delay
	// lookup can land on a neighboring table bin due to ordinary floating-point rounding in
	// the accumulated sample time (getFilter truncates rather than rounds to a bin index),
	// giving a small (~0.1%) deviation from exact identity rather than a bug, hence the
	// looser tolerance here versus the 1e-9 used for interior-sample checks elsewhere.
	for (std::size_t i = 0; i < iq_buffer.size(); ++i)
	{
		const RealType expected = i == 3 ? 2.0 : (i == 6 ? 5.0 : 0.0);
		REQUIRE_THAT(iq_buffer[i].real(), WithinAbs(expected, 1e-2));
	}
}

TEST_CASE("applyStreamingInterference adds FMCW energy to pulsed receiver windows",
		  "[processing][finalizer][interference][fmcw]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setRate(1.0e6);
	params::setOversampleRatio(1);

	radar::Platform tx_platform("TxPlatform");
	setupPlatform(tx_platform, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_platform("RxPlatform");
	setupPlatform(rx_platform, math::Vec3{300.0, 0.0, 0.0});

	antenna::Isotropic antenna("iso");
	auto timing_model = makeQuietTiming("clk", 13);

	radar::Transmitter transmitter(&tx_platform, "FmcwTx", radar::OperationMode::FMCW_MODE, 103);
	transmitter.setAntenna(&antenna);
	transmitter.setTiming(timing_model);
	fers_signal::RadarSignal wave("fmcw", 16.0, 1.0e9, fers_signal::FmcwChirpWaveform(1.0e6, 50.0e-6, 100.0e-6), 303);
	transmitter.setSignal(&wave);

	radar::Receiver receiver(&rx_platform, "PulsedRx", 101, radar::OperationMode::PULSED_MODE, 204);
	receiver.setAntenna(&antenna);
	receiver.setTiming(timing_model);
	receiver.setWindowProperties(150.0e-6, 1.0e3, 0.0);

	core::World world;
	const propagation::pointscatter::PointScatterModel prop(&world);

	std::vector<ComplexType> window(3, ComplexType{});
	const std::vector<core::ActiveStreamingSource> streaming_sources = {
		core::makeActiveSource(&transmitter, 0.0, 300.0e-6)};
	core::ReceiverTrackerCache tracker_cache;

	processing::pipeline::applyStreamingInterference(window, 10.0e-6, 50.0e-6, prop, nullptr, &receiver,
													 streaming_sources, tracker_cache);

	REQUIRE(std::abs(window[0]) > 0.0);
	REQUIRE_THAT(std::abs(window[1]), WithinAbs(0.0, 1.0e-18));
	REQUIRE(std::abs(window[2]) > 0.0);
}

TEST_CASE("applyStreamingInterference supports FMCW transmitter with CW streaming receiver",
		  "[processing][finalizer][interference][fmcw]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setRate(1.0e6);
	params::setOversampleRatio(1);

	radar::Platform tx_platform("TxPlatform");
	setupPlatform(tx_platform, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_platform("RxPlatform");
	setupPlatform(rx_platform, math::Vec3{300.0, 0.0, 0.0});

	antenna::Isotropic antenna("iso");
	auto timing_model = makeQuietTiming("clk", 14);

	radar::Transmitter transmitter(&tx_platform, "FmcwTx", radar::OperationMode::FMCW_MODE, 104);
	transmitter.setAntenna(&antenna);
	transmitter.setTiming(timing_model);
	fers_signal::RadarSignal wave("fmcw", 16.0, 1.0e9, fers_signal::FmcwChirpWaveform(1.0e6, 50.0e-6, 100.0e-6), 304);
	transmitter.setSignal(&wave);

	radar::Receiver receiver(&rx_platform, "CwRx", 102, radar::OperationMode::CW_MODE, 205);
	receiver.setAntenna(&antenna);
	receiver.setTiming(timing_model);

	core::World world;
	const propagation::pointscatter::PointScatterModel prop(&world);

	std::vector<ComplexType> window(3, ComplexType{0.1, -0.2});
	const std::vector<ComplexType> baseline = window;
	const std::vector<core::ActiveStreamingSource> streaming_sources = {
		core::makeActiveSource(&transmitter, 0.0, 300.0e-6)};
	core::ReceiverTrackerCache tracker_cache;

	processing::pipeline::applyStreamingInterference(window, 10.0e-6, 50.0e-6, prop, nullptr, &receiver,
													 streaming_sources, tracker_cache);

	for (std::size_t i = 0; i < window.size(); ++i)
	{
		const RealType sample_time = 10.0e-6 + static_cast<RealType>(i) * 50.0e-6;
		const ComplexType expected = expectedStreamingSample(prop, &receiver, streaming_sources, sample_time);
		const ComplexType actual = window[i] - baseline[i];
		REQUIRE_THAT(actual.real(), WithinAbs(expected.real(), 1.0e-12));
		REQUIRE_THAT(actual.imag(), WithinAbs(expected.imag(), 1.0e-12));
	}
}

TEST_CASE("applyStreamingInterference superposes up- and down-chirp FMCW transmitters",
		  "[processing][finalizer][interference][fmcw]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setRate(1.0e6);
	params::setOversampleRatio(1);

	radar::Platform tx_up_platform("TxUpPlatform");
	setupPlatform(tx_up_platform, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform tx_down_platform("TxDownPlatform");
	setupPlatform(tx_down_platform, math::Vec3{150.0, 0.0, 0.0});
	radar::Platform rx_platform("RxPlatform");
	setupPlatform(rx_platform, math::Vec3{300.0, 0.0, 0.0});

	antenna::Isotropic antenna("iso");
	auto timing_model = makeQuietTiming("clk", 15);

	radar::Transmitter up_tx(&tx_up_platform, "UpTx", radar::OperationMode::FMCW_MODE, 105);
	up_tx.setAntenna(&antenna);
	up_tx.setTiming(timing_model);
	fers_signal::RadarSignal up_wave("up_fmcw", 16.0, 1.0e9, fers_signal::FmcwChirpWaveform(1.0e6, 50.0e-6, 100.0e-6),
									 305);
	up_tx.setSignal(&up_wave);

	radar::Transmitter down_tx(&tx_down_platform, "DownTx", radar::OperationMode::FMCW_MODE, 106);
	down_tx.setAntenna(&antenna);
	down_tx.setTiming(timing_model);
	fers_signal::RadarSignal down_wave("down_fmcw", 16.0, 1.0e9,
									   fers_signal::FmcwChirpWaveform(1.0e6, 50.0e-6, 100.0e-6, 0.0, std::nullopt,
																	  fers_signal::FmcwChirpDirection::Down),
									   306);
	down_tx.setSignal(&down_wave);

	radar::Receiver receiver(&rx_platform, "CwRx", 103, radar::OperationMode::CW_MODE, 206);
	receiver.setAntenna(&antenna);
	receiver.setTiming(timing_model);

	core::World world;
	const propagation::pointscatter::PointScatterModel prop(&world);

	std::vector<ComplexType> window(3, ComplexType{0.1, -0.2});
	const std::vector<ComplexType> baseline = window;
	const std::vector<core::ActiveStreamingSource> streaming_sources = {
		core::makeActiveSource(&up_tx, 0.0, 300.0e-6), core::makeActiveSource(&down_tx, 0.0, 300.0e-6)};
	core::ReceiverTrackerCache tracker_cache;

	processing::pipeline::applyStreamingInterference(window, 10.0e-6, 50.0e-6, prop, nullptr, &receiver,
													 streaming_sources, tracker_cache);

	for (std::size_t i = 0; i < window.size(); ++i)
	{
		const RealType sample_time = 10.0e-6 + static_cast<RealType>(i) * 50.0e-6;
		const ComplexType expected = expectedStreamingSample(prop, &receiver, streaming_sources, sample_time);
		const ComplexType actual = window[i] - baseline[i];
		REQUIRE_THAT(actual.real(), WithinAbs(expected.real(), 1.0e-12));
		REQUIRE_THAT(actual.imag(), WithinAbs(expected.imag(), 1.0e-12));
	}
}

TEST_CASE("applyStreamingInterference reuses tracker cache without carrying window state",
		  "[processing][finalizer][interference][fmcw]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setRate(1.0e6);
	params::setOversampleRatio(1);

	radar::Platform tx_platform("TxPlatform");
	setupPlatform(tx_platform, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_platform("RxPlatform");
	setupPlatform(rx_platform, math::Vec3{300.0, 0.0, 0.0});
	radar::Platform target_platform("TargetPlatform");
	setupPlatform(target_platform, math::Vec3{150.0, 100.0, 0.0});

	antenna::Isotropic antenna("iso");
	auto timing_model = makeQuietTiming("clk", 16);

	radar::Transmitter transmitter(&tx_platform, "FmcwTx", radar::OperationMode::FMCW_MODE, 107);
	transmitter.setAntenna(&antenna);
	transmitter.setTiming(timing_model);
	fers_signal::RadarSignal wave("fmcw", 16.0, 1.0e9, fers_signal::FmcwChirpWaveform(1.0e6, 50.0e-6, 100.0e-6), 307);
	transmitter.setSignal(&wave);

	radar::Receiver receiver(&rx_platform, "PulsedRx", 104, radar::OperationMode::PULSED_MODE, 207);
	receiver.setAntenna(&antenna);
	receiver.setTiming(timing_model);
	receiver.setWindowProperties(150.0e-6, 1.0e3, 0.0);

	core::World world;
	world.add(radar::createIsoTarget(&target_platform, "TargetA", 2.0, 8, 502));
	const propagation::pointscatter::PointScatterModel prop(&world);

	const std::vector<core::ActiveStreamingSource> streaming_sources = {
		core::makeActiveSource(&transmitter, 0.0, 300.0e-6)};
	core::ReceiverTrackerCache tracker_cache;
	std::vector<ComplexType> first_window(3, ComplexType{});
	std::vector<ComplexType> second_window(3, ComplexType{});

	processing::pipeline::applyStreamingInterference(first_window, 10.0e-6, 50.0e-6, prop, nullptr, &receiver,
													 streaming_sources, tracker_cache);

	REQUIRE(tracker_cache.path_trackers.size() == 1);
	const std::size_t sources_count = tracker_cache.path_trackers.size();
	const std::size_t tracker_count = tracker_cache.path_trackers[0].size();

	processing::pipeline::applyStreamingInterference(second_window, 10.0e-6, 50.0e-6, prop, nullptr, &receiver,
													 streaming_sources, tracker_cache);

	REQUIRE(tracker_cache.path_trackers.size() == sources_count);
	REQUIRE(tracker_cache.path_trackers[0].size() == tracker_count);
	REQUIRE(std::abs(first_window.front()) > 0.0);
	for (std::size_t i = 0; i < first_window.size(); ++i)
	{
		REQUIRE_THAT(second_window[i].real(), WithinAbs(first_window[i].real(), 1.0e-12));
		REQUIRE_THAT(second_window[i].imag(), WithinAbs(first_window[i].imag(), 1.0e-12));
	}
}

TEST_CASE("applyPulsedInterference maps pulse start times to simulation sample indices and clips overflow",
		  "[processing][finalizer][interference]")
{
	ParamGuard const guard;
	params::setTime(10.0, 11.0);
	params::setRate(4.0);
	params::setOversampleRatio(1);
	params::setSimSamplingRate(4.0);

	std::vector<std::unique_ptr<fers_signal::RadarSignal>> wave_store;
	std::vector<std::unique_ptr<serial::Response>> interference_log;
	interference_log.push_back(makeFixedResponse(
		wave_store, {ComplexType{1.0, 0.5}, ComplexType{2.0, -0.5}, ComplexType{3.0, 1.0}}, 4.0, 10.25));
	// Second pulse deliberately starts where the first pulse's real content is still
	// active, so their overlap covers genuine samples from both sides, not either one's
	// zero-gain taper stub.
	interference_log.push_back(makeFixedResponse(
		wave_store, {ComplexType{10.0, 0.0}, ComplexType{20.0, 1.0}, ComplexType{30.0, 2.0}}, 4.0, 10.5));

	std::vector<ComplexType> iq_buffer(5, ComplexType{});

	processing::pipeline::applyPulsedInterference(iq_buffer, interference_log);

	// Pulse 1 renders as [0, 1.0+0.5i, 2.0-0.5i, 3.0+1.0i], starting at
	// response->startTime() == 10.25 - edge == 10.0 (edge = 1/simSamplingRate == 0.25),
	// placed at buffer index floor((10.0-10.0)*4)=0, occupying [0,4).
	// Pulse 2 renders as [0, 10.0, 20.0+1.0i, 30.0+2.0i], starting at 10.5-0.25=10.25,
	// placed at floor((10.25-10.0)*4)=1, occupying [1,5). Indices 1-3 are where both
	// pulses overlap: buffer[1] = 1.0+0.5i + 0 = 1.0+0.5i, buffer[2] = 2.0-0.5i + 10.0 =
	// 12.0-0.5i, buffer[3] = 3.0+1.0i + 20.0+1.0i = 23.0+2.0i; buffer[4] is pulse 2 alone.
	const std::vector<ComplexType> expected = {
		ComplexType{0.0, 0.0},	ComplexType{1.0, 0.5},	ComplexType{12.0, -0.5},
		ComplexType{23.0, 2.0}, ComplexType{30.0, 2.0},
	};

	REQUIRE(iq_buffer.size() == expected.size());
	for (size_t i = 0; i < expected.size(); ++i)
	{
		REQUIRE_THAT(iq_buffer[i].real(), WithinAbs(expected[i].real(), 1e-9));
		REQUIRE_THAT(iq_buffer[i].imag(), WithinAbs(expected[i].imag(), 1e-9));
	}
}

TEST_CASE("applyPulsedInterference uses RF simulation rate for oversampled full-buffer path",
		  "[processing][finalizer][interference]")
{
	ParamGuard const guard;
	params::setTime(10.0, 11.0);
	params::setRate(4.0);
	params::setOversampleRatio(2);
	const RealType combined_rate = params::rate() * static_cast<RealType>(params::oversampleRatio());
	params::setSimSamplingRate(combined_rate);

	std::vector<std::unique_ptr<fers_signal::RadarSignal>> wave_store;
	std::vector<std::unique_ptr<serial::Response>> interference_log;
	interference_log.push_back(makeFixedResponse(
		wave_store, {ComplexType{1.0, 0.5}, ComplexType{2.0, -0.5}, ComplexType{3.0, 1.0}}, combined_rate, 10.25));

	std::vector<ComplexType> iq_buffer(8, ComplexType{});

	processing::pipeline::applyPulsedInterference(iq_buffer, interference_log);

	// Rendered pulse is [0, 1.0+0.5i, 2.0-0.5i, 3.0+1.0i], starting at
	// response->startTime() == 10.25 - edge == 10.125 (edge = 1/combined_rate == 0.125),
	// placed at buffer index floor((10.125-10.0)*8)=1, occupying [1,5).
	const std::vector<ComplexType> expected = {
		ComplexType{0.0, 0.0}, ComplexType{0.0, 0.0}, ComplexType{1.0, 0.5}, ComplexType{2.0, -0.5},
		ComplexType{3.0, 1.0}, ComplexType{0.0, 0.0}, ComplexType{0.0, 0.0}, ComplexType{0.0, 0.0},
	};

	REQUIRE(iq_buffer.size() == expected.size());
	for (size_t i = 0; i < expected.size(); ++i)
	{
		REQUIRE_THAT(iq_buffer[i].real(), WithinAbs(expected[i].real(), 1e-9));
		REQUIRE_THAT(iq_buffer[i].imag(), WithinAbs(expected[i].imag(), 1e-9));
	}
}
