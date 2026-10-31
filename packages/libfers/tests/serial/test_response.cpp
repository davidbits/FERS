#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

#include "core/config.h"
#include "core/parameters.h"
#include "interpolation/interpolation_point.h"
#include "serial/response.h"
#include "signal/radar_signal.h"

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

	fers_signal::RadarSignal makeSampledRadarSignal(const std::string& name, const RealType power,
													std::vector<ComplexType> samples, const RealType rate)
	{
		fers_signal::PulseWaveform sampled;
		sampled.load(samples, static_cast<unsigned>(samples.size()), rate);
		return {name, power, 1.0e9, std::move(sampled)};
	}
}

TEST_CASE("Response start/end time always pad a single real point by the caller-supplied interval on both sides",
		  "[serial][response]")
{
	ParamGuard const guard;
	params::setOversampleRatio(1);
	params::setSimSamplingRate(1000.0);

	const auto wave = makeSampledRadarSignal("wave", 1.0, {ComplexType{0.0, 0.0}}, 1000.0);
	const interp::InterpPoint first{.gain = 1.0, .rx_time = 0.5, .delay = 0.0, .phase_delay = 0.0};
	const RealType edge = 1.0 / params::simSamplingRate();
	serial::ResponseBuilder builder(&wave, first, edge, edge);
	const serial::Response response(std::move(builder));

	// A single-sample response (the common case for ephemeral ray-traced paths rather than
	// a whole transmitted pulse) still gets a lead/tail taper padding it on each side.
	REQUIRE_THAT(response.startTime(), WithinAbs(0.5 - 1.0e-3, 1e-12));
	REQUIRE_THAT(response.endTime(), WithinAbs(0.5 + 1.0e-3, 1e-12));
	REQUIRE_THAT(response.getRxDuration(), WithinAbs(2.0e-3, 1e-12));
}

TEST_CASE("Response taper width and delay/phase extrapolate using the caller-supplied interval, "
		  "Doppler-scaled by the observed rx/tx rate",
		  "[serial][response]")
{
	ParamGuard const guard;
	params::setOversampleRatio(1);
	params::setSimSamplingRate(1000.0);

	const auto wave = makeSampledRadarSignal("wave", 1.0, {ComplexType{0.0, 0.0}}, 1000.0);

	// p1: tx = rx_time - delay = 8.0. p2: tx = 8.5. tx_gap = 0.5, rx_gap = 1.0 - an
	// exaggerated 2x local Doppler compression, chosen so "scaled by the supplied interval"
	// is unambiguously distinct from "reuses the pair's own 0.5s tx_gap directly".
	const interp::InterpPoint p1{.gain = 1.0, .rx_time = 10.0, .delay = 2.0, .phase_delay = 1.0};
	const interp::InterpPoint p2{.gain = 1.0, .rx_time = 11.0, .delay = 2.5, .phase_delay = 1.6};

	// The supplied interval (0.2 tx-seconds) comes from the true sample grid (see
	// calculateResponses), not from this pair's own spacing - that's the whole point of the
	// fix: two adjacent path_id channels sharing a short grid gap must each taper by that
	// same short interval, not by whatever their own last two points happen to be spaced by,
	// or their tapers overlap (double-count) or gap right at the seam between them.
	serial::ResponseBuilder builder(&wave, p1, /*left_interval=*/0.2, /*right_interval=*/0.2);
	builder.addInterpPoint(p2, /*right_interval=*/0.2);
	const serial::Response response(std::move(builder));

	const auto points = response.points();
	REQUIRE(points.size() == 4); // lead, p1, p2, tail
	const auto lead = points[0];
	const auto tail = points[3];

	// Local rate = (delay2-delay1)/tx_gap = 0.5/0.5 = 1.0 delay-second per tx-second.
	// Extrapolating 0.2 more tx-seconds carries 0.2 more delay with it either direction.
	REQUIRE_THAT(lead.gain, WithinAbs(0.0, 1e-12));
	REQUIRE_THAT(lead.delay, WithinAbs(2.0 - 0.2, 1e-9));
	REQUIRE_THAT(lead.phase_delay, WithinAbs(1.0 - 0.24, 1e-9));
	REQUIRE_THAT(lead.rx_time, WithinAbs(10.0 - 0.4, 1e-9)); // rx_width = interval + delay_delta = 0.4

	REQUIRE_THAT(tail.gain, WithinAbs(0.0, 1e-12));
	REQUIRE_THAT(tail.delay, WithinAbs(2.5 + 0.2, 1e-9));
	REQUIRE_THAT(tail.phase_delay, WithinAbs(1.6 + 0.24, 1e-9));
	REQUIRE_THAT(tail.rx_time, WithinAbs(11.0 + 0.4, 1e-9));

	REQUIRE_THAT(response.startTime(), WithinAbs(10.0 - 0.4, 1e-9));
	REQUIRE_THAT(response.endTime(), WithinAbs(11.0 + 0.4, 1e-9));
}

TEST_CASE("Response taper degenerates to the real point when its interval is 0", "[serial][response]")
{
	ParamGuard const guard;
	params::setOversampleRatio(1);
	params::setSimSamplingRate(1000.0);

	const auto wave = makeSampledRadarSignal("wave", 1.0, {ComplexType{0.0, 0.0}}, 1000.0);

	// A 0 interval means this side sits at the transmission's actual start/end - nothing
	// ever renders out there, so there's nothing meaningful to taper into.
	serial::ResponseBuilder builder(&wave, {.gain = 1.0, .rx_time = 5.0, .delay = 0.0, .phase_delay = 0.0},
									/*left_interval=*/0.0, /*right_interval=*/0.0);
	builder.addInterpPoint({.gain = 1.0, .rx_time = 5.001, .delay = 0.0, .phase_delay = 0.0},
						   /*right_interval=*/0.0);
	const serial::Response response(std::move(builder));

	// No taper point is materialized on either side - just the 2 real points.
	REQUIRE(response.points().size() == 2);
	REQUIRE_THAT(response.startTime(), WithinAbs(5.0, 1e-12));
	REQUIRE_THAT(response.endTime(), WithinAbs(5.001, 1e-12));
}

TEST_CASE("Response renderSlice reproduces the source waveform at zero delay", "[serial][response]")
{
	ParamGuard const guard;
	params::setOversampleRatio(1);
	params::setSimSamplingRate(1000.0);

	const std::vector<ComplexType> samples = {
		ComplexType{1.0, -1.0},
		ComplexType{0.5, 0.25},
		ComplexType{-0.25, 0.75},
		ComplexType{0.0, -0.5},
	};
	constexpr RealType rate = 1000.0;
	const auto wave = makeSampledRadarSignal("wave", 1.0, samples, rate);

	const interp::InterpPoint first{.gain = 1.0, .rx_time = 0.0, .delay = 0.0, .phase_delay = 0.0};
	// The interval must be nonzero: a 0-width lead/tail segment collapses onto the
	// adjacent real point's own rx_time, leaving the render loop's boundary sample with a
	// degenerate zero-length interpolation span instead of a well-defined one.
	const RealType edge = 1.0 / rate;
	serial::ResponseBuilder builder(&wave, first, edge, edge);
	// One real control point per native sample (matching how the point-scatter model
	// covers a whole pulse) keeps gain == 1 across the entire buffer under test; a
	// single real point would only hold full gain exactly at that one point; the very
	// next native sample already falls inside the response's own trailing taper.
	for (std::size_t i = 1; i < samples.size(); ++i)
	{
		builder.addInterpPoint(
			{.gain = 1.0, .rx_time = static_cast<RealType>(i) / rate, .delay = 0.0, .phase_delay = 0.0}, edge);
	}
	const serial::Response response(std::move(builder));

	// At exactly zero fractional delay the render filter is an exact identity (its
	// center tap is a bit-exact 1.0; every other tap is sinc(integer) ~ 0), so with
	// delay == 0.0 throughout, rendered samples must reproduce the loaded ones
	// verbatim. Native sample 0 lines up with `first` itself (the true first real
	// point), not with the synthetic taper. The taper is a padded, separate concern
	// (see the startTime()/endTime() tests) that doesn't shift the native buffer.
	const auto data = response.renderSlice(rate, first.rx_time, samples.size(), 0.0);

	REQUIRE(data.size() == samples.size());
	for (std::size_t i = 0; i < samples.size(); ++i)
	{
		REQUIRE_THAT(data[i].real(), WithinAbs(samples[i].real(), 1e-9));
		REQUIRE_THAT(data[i].imag(), WithinAbs(samples[i].imag(), 1e-9));
	}
}

TEST_CASE("Response render keeps full gain at an untapered (zero-interval) edge sample", "[serial][response]")
{
	ParamGuard const guard;
	params::setOversampleRatio(1);
	params::setSimSamplingRate(1000.0);

	// A pulsed response with no mid-transmission occlusion: both edges sit at the
	// transmission's actual start/end, so left_interval/right_interval are 0 and there is
	// nothing to taper. Every real point shares the same gain, so every rendered native
	// sample - including the first and last - must come out at that same gain.
	const std::vector<ComplexType> samples = {ComplexType{1.0, 0.0}, ComplexType{1.0, 0.0}, ComplexType{1.0, 0.0}};
	const auto wave = makeSampledRadarSignal("wave", 1.0, samples, 1000.0);

	serial::ResponseBuilder builder(&wave, {.gain = 1.0, .rx_time = 0.0, .delay = 0.0, .phase_delay = 0.0}, 0.0, 0.0);
	builder.addInterpPoint({.gain = 1.0, .rx_time = 0.001, .delay = 0.0, .phase_delay = 0.0}, 0.0);
	builder.addInterpPoint({.gain = 1.0, .rx_time = 0.002, .delay = 0.0, .phase_delay = 0.0}, 0.0);
	const serial::Response response(std::move(builder));

	const auto data = response.render(0.0);
	REQUIRE(data.size() == 3);
	for (auto i : data)
	{
		REQUIRE_THAT(i.real(), WithinAbs(1.0, 1e-9));
		REQUIRE_THAT(i.imag(), WithinAbs(0.0, 1e-9));
	}
}
