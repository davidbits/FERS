#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <string>

#include "core/parameters.h"
#include "core/portable_utils.h"

using Catch::Matchers::ContainsSubstring;
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
}

TEST_CASE("Parameters default values are consistent", "[core][parameters]")
{
	ParamGuard const guard;
	params::params.reset();

	REQUIRE_THAT(params::c(), WithinAbs(params::Parameters::DEFAULT_C, 0.0));
	REQUIRE_THAT(params::boltzmannK(), WithinAbs(params::Parameters::DEFAULT_BOLTZMANN_K, 0.0));
	REQUIRE_THAT(params::startTime(), WithinAbs(0.0, 0.0));
	REQUIRE_THAT(params::endTime(), WithinAbs(0.0, 0.0));
	REQUIRE_THAT(params::simSamplingRate(), WithinAbs(1000.0, 0.0));
	REQUIRE_THAT(params::rate(), WithinAbs(0.0, 0.0));
	REQUIRE(params::randomSeed() == 0u);
	REQUIRE(params::adcBits() == 0u);
	REQUIRE(params::renderFilterLength() == 33u);
	REQUIRE(params::workerThreads() == core::countProcessors());
	REQUIRE(params::oversampleRatio() == 1u);
	REQUIRE(params::coordinateFrame() == params::CoordinateFrame::ENU);
	REQUIRE(params::rotationAngleUnit() == params::RotationAngleUnit::Degrees);
	REQUIRE(params::utmZone() == 0);
	REQUIRE(params::utmNorthHemisphere());
}

TEST_CASE("RayTracingParameters derives ray counts and solid angles from boresight width and tube weights",
		  "[core][parameters]")
{
	params::RayTracingParameters rt;
	rt.boresight_width = PI / 2.0; // 90 degree full-width cone -> 45 degree half-angle.
	rt.boresight_tube_solid_angle = 1e-3;
	rt.off_boresight_tube_solid_angle = 1e-2;

	const double expected_cone_sr = 2.0 * PI * (1.0 - std::cos(PI / 4.0));
	REQUIRE_THAT(rt.boresightSolidAngle(), WithinAbs(expected_cone_sr, 1e-9));

	const auto expected_boresight_rays = static_cast<unsigned>(std::llround(expected_cone_sr / 1e-3));
	REQUIRE(rt.boresightDirections() == expected_boresight_rays);

	const double expected_off_sr = 4.0 * PI - expected_cone_sr;
	const auto expected_off_rays = static_cast<unsigned>(std::llround(expected_off_sr / 1e-2));
	REQUIRE(rt.offBoresightDirections() == expected_off_rays);

	REQUIRE(rt.directionsPerSource() == expected_boresight_rays + expected_off_rays);
}

TEST_CASE("RayTracingParameters treats a full-turn boresight width as covering the whole sphere", "[core][parameters]")
{
	params::RayTracingParameters rt;
	rt.boresight_width = 2.0 * PI;
	rt.boresight_tube_solid_angle = 1e-3;

	REQUIRE_THAT(rt.boresightSolidAngle(), WithinAbs(4.0 * PI, 1e-9));
	REQUIRE(rt.offBoresightDirections() == 0u);
}

TEST_CASE("Parameters setters update getters", "[core][parameters]")
{
	ParamGuard const guard;
	params::params.reset();

	params::setC(300000000.0);
	REQUIRE_THAT(params::c(), WithinAbs(300000000.0, 1e-9));

	params::setTime(1.25, 9.5);
	REQUIRE_THAT(params::startTime(), WithinAbs(1.25, 1e-12));
	REQUIRE_THAT(params::endTime(), WithinAbs(9.5, 1e-12));

	params::setSimSamplingRate(2048.0);
	REQUIRE_THAT(params::simSamplingRate(), WithinAbs(2048.0, 1e-12));

	params::setRate(512.0);
	REQUIRE_THAT(params::rate(), WithinAbs(512.0, 1e-12));

	params::setRandomSeed(42);
	REQUIRE(params::randomSeed() == 42u);

	params::setAdcBits(12);
	REQUIRE(params::adcBits() == 12u);

	params::setOversampleRatio(4);
	REQUIRE(params::oversampleRatio() == 4u);

	params::setRotationAngleUnit(params::RotationAngleUnit::Radians);
	REQUIRE(params::rotationAngleUnit() == params::RotationAngleUnit::Radians);
}

TEST_CASE("Parameters setters validate inputs", "[core][parameters]")
{
	ParamGuard const guard;
	params::params.reset();

	REQUIRE_THROWS_AS(params::setRate(0.0), std::runtime_error);
	REQUIRE_THROWS_AS(params::setRate(-1.0), std::runtime_error);

	REQUIRE_THROWS_AS(params::setOversampleRatio(0), std::runtime_error);
	REQUIRE_THROWS_WITH(params::setOversampleRatio(9), ContainsSubstring("Oversampling ratios > 8 are not supported"));
}

TEST_CASE("Parameters origin and coordinate settings", "[core][parameters]")
{
	ParamGuard const guard;
	params::params.reset();

	params::setOrigin(1.5, 2.5, 3.5);
	REQUIRE_THAT(params::originLatitude(), WithinAbs(1.5, 1e-12));
	REQUIRE_THAT(params::originLongitude(), WithinAbs(2.5, 1e-12));
	REQUIRE_THAT(params::originAltitude(), WithinAbs(3.5, 1e-12));

	params::setCoordinateSystem(params::CoordinateFrame::UTM, 33, false);
	REQUIRE(params::coordinateFrame() == params::CoordinateFrame::UTM);
	REQUIRE(params::utmZone() == 33);
	REQUIRE_FALSE(params::utmNorthHemisphere());
}

TEST_CASE("Parameters setThreads returns expected", "[core][parameters]")
{
	ParamGuard const guard;
	params::params.reset();

	const auto error = params::setThreads(0);
	REQUIRE_FALSE(error.has_value());

	const auto ok = params::setThreads(4);
	REQUIRE(ok.has_value());
	REQUIRE(params::workerThreads() == 4u);
}

TEST_CASE("Parameters reset restores defaults", "[core][parameters]")
{
	ParamGuard const guard;
	params::setC(1.0);
	params::setTime(5.0, 10.0);
	params::setSimSamplingRate(500.0);
	params::setRate(100.0);
	params::setRandomSeed(7);
	params::setAdcBits(8);
	params::setOversampleRatio(2);
	params::setOrigin(9.0, 8.0, 7.0);
	params::setCoordinateSystem(params::CoordinateFrame::ECEF, 12, false);

	params::params.reset();

	REQUIRE_THAT(params::c(), WithinAbs(params::Parameters::DEFAULT_C, 0.0));
	REQUIRE_THAT(params::startTime(), WithinAbs(0.0, 0.0));
	REQUIRE_THAT(params::endTime(), WithinAbs(0.0, 0.0));
	REQUIRE_THAT(params::simSamplingRate(), WithinAbs(1000.0, 0.0));
	REQUIRE_THAT(params::rate(), WithinAbs(0.0, 0.0));
	REQUIRE(params::randomSeed() == 0u);
	REQUIRE(params::adcBits() == 0u);
	REQUIRE(params::oversampleRatio() == 1u);
	REQUIRE(params::coordinateFrame() == params::CoordinateFrame::ENU);
	REQUIRE(params::rotationAngleUnit() == params::RotationAngleUnit::Degrees);
	REQUIRE(params::utmZone() == 0);
	REQUIRE(params::utmNorthHemisphere());
}
