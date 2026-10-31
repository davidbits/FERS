#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>

#include "core/logging.h"
#include "core/portable_utils.h"

using Catch::Matchers::WithinRel;

namespace
{
	double expectedBesselJ1(const double x) noexcept
	{
#ifdef _MSC_VER
		return _j1(x);
#else
		return j1(x);
#endif
	}
}

TEST_CASE("besselJ1 matches standard library j1", "[core][portable]")
{
	const double x1 = 0.0;
	const double x2 = 1.0;
	const double x3 = -2.5;

	REQUIRE_THAT(core::besselJ1(x1), WithinRel(expectedBesselJ1(x1), 1e-12));
	REQUIRE_THAT(core::besselJ1(x2), WithinRel(expectedBesselJ1(x2), 1e-12));
	REQUIRE_THAT(core::besselJ1(x3), WithinRel(expectedBesselJ1(x3), 1e-12));
}

TEST_CASE("countProcessors returns a valid count", "[core][portable]")
{
	const unsigned count = core::countProcessors();
	REQUIRE(count >= 1u);
}

TEST_CASE("countProcessors fallback logging not directly testable", "[core][portable]")
{
	// TODO: Cannot reliably force std::thread::hardware_concurrency() to return 0 to hit the error logging branch.
	REQUIRE(true);
}
