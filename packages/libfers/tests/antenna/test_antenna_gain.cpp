#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "antenna/antenna_gain.h"
#include "core/portable_utils.h"

using Catch::Matchers::WithinRel;
using namespace antenna::gain;

TEST_CASE("besselJ1Approx matches core::besselJ1 (libm j1) across a range of x", "[antenna][gain]")
{
	for (const double x : {0.001, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.9, 8.0, 8.1, 10.0, 20.0, 50.0, 100.0})
	{
		const double expected = core::besselJ1(x);
		const double actual = besselJ1Approx(x);
		REQUIRE_THAT(actual, WithinRel(expected, 1e-6));
	}
}

TEST_CASE("besselJ1Approx is odd (J1(-x) = -J1(x))", "[antenna][gain]")
{
	for (const double x : {0.5, 2.0, 9.0})
	{
		REQUIRE_THAT(besselJ1Approx(-x), WithinRel(-besselJ1Approx(x), 1e-9));
	}
}

TEST_CASE("besselJ1OverX approaches 1/2 as x approaches 0", "[antenna][gain]")
{
	REQUIRE_THAT(besselJ1OverX(0.0), Catch::Matchers::WithinAbs(0.5, 1e-9));
	REQUIRE_THAT(besselJ1OverX(1e-4), Catch::Matchers::WithinAbs(0.5, 1e-6));
}
