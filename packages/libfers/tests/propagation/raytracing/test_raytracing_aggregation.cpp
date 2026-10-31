#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>
#include <optional>
#include <vector>

#include "math/coord.h"
#include "propagation/propagation_model.h"
#include "propagation/raytracing/raytracing_model.h"
#include "radar/platform.h"
#include "radar/receiver.h"

using Catch::Matchers::WithinAbs;
namespace rt = propagation::raytracing;
namespace prop = propagation;

namespace
{
	rt::Contribution makeContribution(prop::CFloat voltage, double delay, uint64_t path_id = 0,
									  uint32_t source_times_index = 0, uint32_t dest_index = 0)
	{
		rt::Contribution c{};
		c.path_gain = voltage;
		c.delay = delay;
		c.path_id = path_id;
		c.source_times_index = source_times_index;
		c.dest_index = dest_index;
		return c;
	}

	// pruneWeakPaths only looks at `.gain`; the receiver reference just needs to be valid.
	void setupPlatform(radar::Platform& plat)
	{
		plat.getMotionPath()->addCoord(math::Coord{math::Vec3{0.0, 0.0, 0.0}, 0.0});
		plat.getMotionPath()->finalize();
		plat.getRotationPath()->addCoord(math::RotationCoord{0.0, 0.0, 0.0});
		plat.getRotationPath()->finalize();
	}

	prop::PropagationPath makePath(radar::Receiver& rx, const RealType gain_mag, const uint64_t path_id = 0)
	{
		return prop::PropagationPath{
			.delay = 0.0,
			.gain = ComplexType{gain_mag, 0.0},
			.path_id = path_id,
			.source_index = 0,
			.receiver = &rx,
		};
	}
}

// --- pruneWeakPaths ---------------------------------------------------------------------------

TEST_CASE("pruneWeakPaths is a no-op when below_max_db is unset", "[raytracing][aggregation][prune]")
{
	radar::Platform plat("plat");
	setupPlatform(plat);
	radar::Receiver rx(&plat, "rx", 0, radar::OperationMode::CW_MODE);

	std::vector<prop::PropagationPath> paths{makePath(rx, 1.0, 1), makePath(rx, 1e-9, 2)};
	rt::pruneWeakPaths(paths, std::nullopt);

	REQUIRE(paths.size() == 2);
}

TEST_CASE("pruneWeakPaths is a no-op for zero or one paths", "[raytracing][aggregation][prune]")
{
	radar::Platform plat("plat");
	setupPlatform(plat);
	radar::Receiver rx(&plat, "rx", 0, radar::OperationMode::CW_MODE);

	std::vector<prop::PropagationPath> paths;
	rt::pruneWeakPaths(paths, 60.0);
	REQUIRE(paths.empty());
	paths.push_back(makePath(rx, 1.0, 1));
	rt::pruneWeakPaths(paths, 60.0);
	REQUIRE(paths.size() == 1);
}

TEST_CASE("pruneWeakPaths drops paths more than N dB below the strongest path", "[raytracing][aggregation][prune]")
{
	radar::Platform plat("plat");
	setupPlatform(plat);
	radar::Receiver rx(&plat, "rx", 0, radar::OperationMode::CW_MODE);

	// Relative to the strongest path (mag 1.0): -40dB (kept, above the -60dB floor) and
	// -80dB (dropped, below it).
	std::vector<prop::PropagationPath> paths{
		makePath(rx, 1.0, /*path_id=*/1),
		makePath(rx, 1e-2, /*path_id=*/2),
		makePath(rx, 1e-4, /*path_id=*/3),
	};

	rt::pruneWeakPaths(paths, 60.0);

	REQUIRE(paths.size() == 2);
	CHECK(paths[0].path_id == 1);
	CHECK(paths[1].path_id == 2);
}

TEST_CASE("pruneWeakPaths keeps a path exactly at the threshold", "[raytracing][aggregation][prune]")
{
	radar::Platform plat("plat");
	setupPlatform(plat);
	radar::Receiver rx(&plat, "rx", 0, radar::OperationMode::CW_MODE);

	// 1e-3 is exactly -60dB relative to 1.0.
	std::vector<prop::PropagationPath> paths{makePath(rx, 1.0, 1), makePath(rx, 1e-3, 2)};
	rt::pruneWeakPaths(paths, 60.0);

	REQUIRE(paths.size() == 2);
}

// --- aggregateContributionGroup -------------------------------------------------------------

TEST_CASE("aggregateContributionGroup passes a single contribution through unchanged", "[raytracing][aggregation]")
{
	const auto c = makeContribution(prop::CFloat(2.0f, -1.0f), 7.5, 42);
	const std::vector<const rt::Contribution*> group{&c};

	const auto gain = rt::aggregateContributionGroup(group, /*group_delay=*/7.5, /*carrier_frequency=*/1e9);

	REQUIRE_THAT(gain.real(), WithinAbs(2.0, 1e-9));
	REQUIRE_THAT(gain.imag(), WithinAbs(-1.0, 1e-9));
}

TEST_CASE("contributionGroupDelay returns the group's minimum - the most direct path", "[raytracing][aggregation]")
{
	// This test is guarding the convention `contributionGroupDelay` uses against unintended regression.
	// It uses the minimum-delay i.e. most direct path's delay (which is likely closest to the highest-return GO path).
	const auto weak = makeContribution(prop::CFloat(1.0f, 0.0f), 10.0);
	const auto strong = makeContribution(prop::CFloat(3.0f, 0.0f), 20.0);

	const std::vector<const rt::Contribution*> group{&weak, &strong};
	const auto delay = rt::contributionGroupDelay(group);

	REQUIRE_THAT(delay, WithinAbs(10.0, 1e-9));
}

TEST_CASE("aggregateContributionGroup sums same-delay contributions coherently", "[raytracing][aggregation]")
{
	const auto c1 = makeContribution(prop::CFloat(1.0f, 0.0f), 1.0);
	const auto c2 = makeContribution(prop::CFloat(1.0f, 0.0f), 1.0);
	const std::vector<const rt::Contribution*> group{&c1, &c2};

	const auto gain = rt::aggregateContributionGroup(group, rt::contributionGroupDelay(group),
													 /*carrier_frequency=*/1e9);

	REQUIRE_THAT(gain.real(), WithinAbs(2.0, 1e-9));
	REQUIRE_THAT(gain.imag(), WithinAbs(0.0, 1e-9));
}

TEST_CASE("aggregateContributionGroup coherently sums a half-period delay offset destructively",
		  "[raytracing][aggregation]")
{
	constexpr double carrier_freq_hz = 10e9; // 10 GHz
	constexpr double period = 1.0 / carrier_freq_hz; // 100 ps
	constexpr double base_delay = 1e-6; // 1 us

	const auto c1 = makeContribution(prop::CFloat(1.0f, 0.0f), base_delay);
	const auto c2 = makeContribution(prop::CFloat(1.0f, 0.0f), base_delay + period / 2.0);
	const std::vector<const rt::Contribution*> group{&c1, &c2};

	const auto delay = rt::contributionGroupDelay(group);
	const auto gain = rt::aggregateContributionGroup(group, delay, carrier_freq_hz);

	REQUIRE_THAT(delay, WithinAbs(base_delay, 1e-15));
	REQUIRE_THAT(std::abs(gain), WithinAbs(0.0, 1e-6));
}

TEST_CASE("aggregateContributionGroup coherently sums a full-period delay offset constructively",
		  "[raytracing][aggregation]")
{
	constexpr double carrier_freq_hz = 10e9;
	constexpr double period = 1.0 / carrier_freq_hz;
	constexpr double base_delay = 1e-9;

	const auto c1 = makeContribution(prop::CFloat(1.0f, 0.0f), base_delay);
	const auto c2 = makeContribution(prop::CFloat(1.0f, 0.0f), base_delay + period); // one full cycle later
	const std::vector<const rt::Contribution*> group{&c1, &c2};

	const auto gain = rt::aggregateContributionGroup(group, rt::contributionGroupDelay(group), carrier_freq_hz);

	REQUIRE_THAT(std::abs(gain), WithinAbs(2.0, 1e-6));
}

// --- groupContributions -----------------------------------------------------------------------

TEST_CASE("groupContributions merges contributions by path_id", "[raytracing][aggregation]")
{
	// Two ray-tube samples on the same TX->facet-sequence->RX path (same path_id) should merge...
	const auto same_path_a = makeContribution(prop::CFloat(1.0f, 0.0f), 10.0, /*path_id=*/1);
	const auto same_path_b = makeContribution(prop::CFloat(1.0f, 0.0f), 10.1, /*path_id=*/1);
	// ...but a geometrically distinct path (different path_id) to the same destination at the same
	// timestep must not be merged with it.
	const auto distinct_path = makeContribution(prop::CFloat(1.0f, 0.0f), 500.0, /*path_id=*/2);

	const std::vector<rt::Contribution> contributions{same_path_a, same_path_b, distinct_path};
	const auto groups = rt::groupContributions(contributions);

	REQUIRE(groups.size() == 2);

	size_t merged_groups = 0;
	for (const auto& [key, group] : groups)
	{
		if (group.size() == 2)
		{
			merged_groups++;
			CHECK(group[0]->path_id == 1);
			CHECK(group[1]->path_id == 1);
		}
		else
		{
			REQUIRE(group.size() == 1);
			CHECK(group.front()->path_id == 2);
		}
	}
	REQUIRE(merged_groups == 1);
}

TEST_CASE("groupContributions keeps different timesteps/destinations separate even with the same path_id",
		  "[raytracing][aggregation]")
{
	// path_id alone doesn't encode time or destination (see resolveCarrierFreq's docs) - the full key
	// must include source_times_index and dest_index too.
	const auto t0 = makeContribution(prop::CFloat(1.0f, 0.0f), 10.0, /*path_id=*/7, /*source_times_index=*/0);
	const auto t1 = makeContribution(prop::CFloat(1.0f, 0.0f), 10.0, /*path_id=*/7, /*source_times_index=*/1);
	const auto d1 = makeContribution(prop::CFloat(1.0f, 0.0f), 10.0, /*path_id=*/7, /*source_times_index=*/0,
									 /*dest_index=*/1);

	const std::vector<rt::Contribution> contributions{t0, t1, d1};
	const auto groups = rt::groupContributions(contributions);

	REQUIRE(groups.size() == 3);
	for (const auto& [key, group] : groups)
		REQUIRE(group.size() == 1);
}
