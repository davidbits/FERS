#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "core/simulation_state.h"
#include "propagation/common.h"
#include "propagation/utils.h"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
namespace prop = propagation;

// --- sampleCarrierFrequency: Constant ------------------------------------------------------ //

TEST_CASE("sampleCarrierFrequency's Constant kind is invariant to time", "[propagation][carrier]")
{
	prop::CarrierModel<float> carrier{};
	carrier.kind = prop::CarrierModelKind::Constant;
	carrier.params.constant.frequency = 1234.5f;

	REQUIRE_THAT(prop::sampleCarrierFrequency(carrier, -1e6), WithinAbs(1234.5, 1e-3));
	REQUIRE_THAT(prop::sampleCarrierFrequency(carrier, 0.0), WithinAbs(1234.5, 1e-3));
	REQUIRE_THAT(prop::sampleCarrierFrequency(carrier, 1e6), WithinAbs(1234.5, 1e-3));
}

// --- sampleCarrierFrequency: Stairs (SFCW) -------------------------------------------------- //

namespace
{
	prop::CarrierModel<float> makeStairs(double start_time, float step_duration, float step_count,
										 float start_frequency, float step_frequency)
	{
		prop::CarrierModel<float> carrier{};
		carrier.kind = prop::CarrierModelKind::Stairs;
		carrier.params.stairs.start_time = start_time;
		carrier.params.stairs.step_duration = step_duration;
		carrier.params.stairs.step_count = step_count;
		carrier.params.stairs.start_frequency = start_frequency;
		carrier.params.stairs.step_frequency = step_frequency;
		return carrier;
	}
}

TEST_CASE("sampleCarrierFrequency's Stairs kind steps up once per step_duration and wraps after "
		  "step_count steps",
		  "[propagation][carrier]")
{
	const auto carrier =
		makeStairs(/*start_time=*/2.0, /*step_duration=*/1.0f, /*step_count=*/4.0f, /*start_frequency=*/1000.0f,
				   /*step_frequency=*/50.0f);

	REQUIRE_THAT(prop::sampleCarrierFrequency(carrier, 2.0), WithinAbs(1000.0, 1e-3)); // step 0
	REQUIRE_THAT(prop::sampleCarrierFrequency(carrier, 3.0), WithinAbs(1050.0, 1e-3)); // step 1
	REQUIRE_THAT(prop::sampleCarrierFrequency(carrier, 5.0), WithinAbs(1150.0, 1e-3)); // step 3 (last of sweep)
	REQUIRE_THAT(prop::sampleCarrierFrequency(carrier, 6.0), WithinAbs(1000.0, 1e-3)); // step 4 -> wraps to step 0
}

TEST_CASE("sampleCarrierFrequency's Stairs kind extends periodically before start_time", "[propagation][carrier]")
{
	// Regression target: step_index_within_stair uses a floor-based modulo, not `%`, so a
	// negative step_index must still land on the *last* step of the previous sweep (step 3),
	// not step -1 or an unrelated value.
	const auto carrier = makeStairs(2.0, 1.0f, 4.0f, 1000.0f, 50.0f);

	REQUIRE_THAT(prop::sampleCarrierFrequency(carrier, 1.0), WithinAbs(1150.0, 1e-3)); // dt=-1 -> step 3
}

TEST_CASE("sampleCarrierFrequency works for CarrierModel<double> too", "[propagation][carrier]")
{
	prop::CarrierModel<double> carrier{};
	carrier.kind = prop::CarrierModelKind::Stairs;
	carrier.params.stairs.start_time = 1.0;
	carrier.params.stairs.step_duration = 1.0;
	carrier.params.stairs.start_frequency = 1000.0;
	carrier.params.stairs.step_frequency = 100.0;
	carrier.params.stairs.step_count = 100;

	// second step
	REQUIRE_THAT(prop::sampleCarrierFrequency(carrier, 2.5), WithinAbs(1100.0, 1e-9));
}

// --- buildPropagationCarrier ---------------------------------------------------------------- //

namespace
{
	core::ActiveStreamingSource makeBaseSource(core::StreamingWaveformKind kind)
	{
		core::ActiveStreamingSource source{};
		source.kind = kind;
		source.segment_start = 10.0;
		source.carrier_freq = 1e9;
		source.start_freq_off = 5e6;
		source.chirp_period = 0.001;
		source.chirp_rate = 1e11;
		source.triangle_period = 0.002;
		source.sfcw_step_period = 0.0005;
		source.sfcw_step_count = 8;
		source.sfcw_step_size = 2e6;
		return source;
	}
}

TEST_CASE("buildPropagationCarrier maps CW-like kinds to a Constant carrier", "[propagation][carrier]")
{
	for (auto kind :
		 {core::StreamingWaveformKind::Cw, core::StreamingWaveformKind::FileCw, core::StreamingWaveformKind::FileFmcw})
	{
		const auto source = makeBaseSource(kind);
		const auto carrier = prop::buildPropagationCarrier<float>(source);

		CHECK(carrier.kind == prop::CarrierModelKind::Constant);
		CHECK_THAT(double(carrier.params.constant.frequency), WithinRel(source.carrier_freq, 1e-6));
	}
}

TEST_CASE("buildPropagationCarrier maps Sfcw to a Stairs carrier", "[propagation][carrier]")
{
	const auto source = makeBaseSource(core::StreamingWaveformKind::Sfcw);
	const auto carrier = prop::buildPropagationCarrier<float>(source);

	REQUIRE(carrier.kind == prop::CarrierModelKind::Stairs);
	CHECK_THAT(carrier.params.stairs.start_time, WithinRel(source.segment_start, 1e-9));
	CHECK_THAT(double(carrier.params.stairs.step_duration), WithinRel(source.sfcw_step_period, 1e-6));
	CHECK_THAT(double(carrier.params.stairs.step_count), WithinRel(double(source.sfcw_step_count), 1e-6));
	CHECK_THAT(double(carrier.params.stairs.start_frequency),
			   WithinRel(source.carrier_freq + source.start_freq_off, 1e-6));
	CHECK_THAT(double(carrier.params.stairs.step_frequency), WithinRel(source.sfcw_step_size, 1e-6));
}
