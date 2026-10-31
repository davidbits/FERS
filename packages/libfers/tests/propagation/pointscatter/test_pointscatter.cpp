// Tests for propagation::pointscatter::PointScatterModel's path-finding
// physics: the Friis transmission equation (direct paths) and the bistatic
// radar range equation (reflected paths).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <memory>
#include <vector>

#include "antenna/antenna_factory.h"
#include "core/config.h"
#include "core/parameters.h"
#include "core/simulation_state.h"
#include "core/world.h"
#include "math/coord.h"
#include "math/geometry_ops.h"
#include "propagation/common.h"
#include "propagation/pointscatter/pointscatter.h"
#include "propagation/propagation_model.h"
#include "radar/platform.h"
#include "radar/receiver.h"
#include "radar/target.h"
#include "radar/transmitter.h"
#include "signal/radar_signal.h"
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

	// Orients the platform's boresight towards look_at. Antenna polarisation is defined relative
	// to boresight, so two antennas pointed at each other couple at full polarisation regardless
	// of their separation's direction.
	void setupPlatformFacing(radar::Platform& plat, const math::Vec3& pos, const math::Vec3& look_at)
	{
		plat.getMotionPath()->addCoord(math::Coord{pos, 0.0});
		plat.getMotionPath()->finalize();
		const math::SVec3 direction(look_at - pos);
		plat.getRotationPath()->addCoord(math::RotationCoord{direction.azimuth, direction.elevation, 0.0});
		plat.getRotationPath()->finalize();
	}

	// path_id is a hash_mix of tx/rx (direct) or tx/target/rx (bistatic).
	const propagation::PropagationPath* findDirectPath(const std::vector<propagation::PropagationPath>& paths,
													   const radar::Transmitter& tx, const radar::Receiver& rx)
	{
		const auto expected_id =
			propagation::hash_mix(propagation::hash_mix(propagation::HASH_SEED, tx.getId()), rx.getId());
		for (const auto& path : paths)
		{
			if (path.path_id == expected_id)
			{
				return &path;
			}
		}
		return nullptr;
	}

	const propagation::PropagationPath* findBistaticPath(const std::vector<propagation::PropagationPath>& paths,
														 const radar::Transmitter& tx, const radar::Target& target,
														 const radar::Receiver& rx)
	{
		const auto expected_id = propagation::hash_mix(
			propagation::hash_mix(propagation::hash_mix(propagation::HASH_SEED, tx.getId()), target.getId()),
			rx.getId());
		for (const auto& path : paths)
		{
			if (path.path_id == expected_id)
			{
				return &path;
			}
		}
		return nullptr;
	}
}

// =============================================================================
// Direct path: Friis transmission equation verification
// =============================================================================

TEST_CASE("direct path computes correct Friis power for isotropic antennas", "[propagation][pointscatter][direct]")
{
	ParamGuard const guard;
	params::params.reset();

	// Setup: Tx at origin, Rx at (1000, 0, 0)
	// Carrier frequency: 1 GHz => lambda = c / 1e9
	// Distance: 1000 m
	// Isotropic antennas: Gt = Gr = 1
	// Expected Friis: Pr/Pt = Gt * Gr * lambda^2 / (16 * pi^2 * R^2)

	const RealType c = params::c();
	const RealType carrier = 1.0e9;
	const RealType lambda = c / carrier;
	const RealType dist = 1000.0;

	const RealType expected_gain = (lambda * lambda) / (16.0 * PI * PI * dist * dist);
	const RealType expected_delay = dist / c;
	const RealType expected_phase = -expected_delay * 2.0 * PI * carrier;

	radar::Platform tx_plat("tx_plat");
	setupPlatform(tx_plat, math::Vec3{0.0, 0.0, 0.0});

	radar::Platform rx_plat("rx_plat");
	setupPlatform(rx_plat, math::Vec3{dist, 0.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::PULSED_MODE);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);

	fers_signal::RadarSignal wave("sig", 1.0, carrier, fers_signal::CwWaveform{});
	tx.setSignal(&wave);

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);

	core::World world;
	const propagation::pointscatter::PointScatterModel prop(&world);

	const auto source = core::makeActiveSource(&tx, 0.0, 1.0);
	const auto paths = prop.traceFromReceivers(nullptr, {&rx}, {source}, 0.0).at(&rx);
	const auto* path = findDirectPath(paths, tx, rx);
	REQUIRE(path != nullptr);

	REQUIRE_THAT(std::norm(path->gain), WithinRel(expected_gain, 1e-9));
	REQUIRE_THAT(path->delay, WithinRel(expected_delay, 1e-9));
	REQUIRE_THAT(-path->delay * 2.0 * PI * carrier, WithinRel(expected_phase, 1e-9));
}

TEST_CASE("direct path Friis equation with different distances", "[propagation][pointscatter][direct]")
{
	ParamGuard const guard;
	params::params.reset();

	// Verify inverse-square law: doubling distance should quarter the power
	const RealType carrier = 3.0e9; // 3 GHz

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	fers_signal::RadarSignal wave1("sig1", 1.0, carrier, fers_signal::CwWaveform{});

	fers_signal::RadarSignal wave2("sig2", 1.0, carrier, fers_signal::CwWaveform{});

	// Distance 500 m
	radar::Platform tx_plat1("tx1");
	setupPlatform(tx_plat1, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_plat1("rx1");
	setupPlatform(rx_plat1, math::Vec3{500.0, 0.0, 0.0});

	radar::Transmitter tx1(&tx_plat1, "tx1", radar::OperationMode::PULSED_MODE);
	tx1.setAntenna(&iso_ant);
	tx1.setTiming(timing);
	tx1.setSignal(&wave1);

	radar::Receiver rx1(&rx_plat1, "rx1", 42, radar::OperationMode::PULSED_MODE);
	rx1.setAntenna(&iso_ant);
	rx1.setTiming(timing);

	core::World world1;
	const propagation::pointscatter::PointScatterModel prop1(&world1);
	const auto source1 = core::makeActiveSource(&tx1, 0.0, 1.0);
	const auto paths1 = prop1.traceFromReceivers(nullptr, {&rx1}, {source1}, 0.0).at(&rx1);
	const auto* path1 = findDirectPath(paths1, tx1, rx1);
	REQUIRE(path1 != nullptr);

	// Distance 1000 m (doubled)
	radar::Platform tx_plat2("tx2");
	setupPlatform(tx_plat2, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_plat2("rx2");
	setupPlatform(rx_plat2, math::Vec3{1000.0, 0.0, 0.0});

	radar::Transmitter tx2(&tx_plat2, "tx2", radar::OperationMode::PULSED_MODE);
	tx2.setAntenna(&iso_ant);
	tx2.setTiming(timing);
	tx2.setSignal(&wave2);

	radar::Receiver rx2(&rx_plat2, "rx2", 42, radar::OperationMode::PULSED_MODE);
	rx2.setAntenna(&iso_ant);
	rx2.setTiming(timing);

	core::World world2;
	const propagation::pointscatter::PointScatterModel prop2(&world2);
	const auto source2 = core::makeActiveSource(&tx2, 0.0, 1.0);
	const auto paths2 = prop2.traceFromReceivers(nullptr, {&rx2}, {source2}, 0.0).at(&rx2);
	const auto* path2 = findDirectPath(paths2, tx2, rx2);
	REQUIRE(path2 != nullptr);

	// Power ratio should be 4:1 (inverse square)
	REQUIRE_THAT(std::norm(path1->gain) / std::norm(path2->gain), WithinRel(4.0, 1e-9));

	// Delay ratio should be 1:2
	REQUIRE_THAT(path2->delay / path1->delay, WithinRel(2.0, 1e-9));
}

TEST_CASE("direct path with noproploss flag ignores distance in power", "[propagation][pointscatter][direct]")
{
	ParamGuard const guard;
	params::params.reset();

	const RealType c = params::c();
	const RealType carrier = 1.0e9;
	const RealType lambda = c / carrier;
	const RealType dist = 5000.0;

	// With no_prop_loss: Pr/Pt = Gt * Gr * lambda^2 / (16 * pi^2) (no R^2 in denominator)
	const RealType expected_gain = (lambda * lambda) / (16.0 * PI * PI);

	radar::Platform tx_plat("tx_plat");
	setupPlatform(tx_plat, math::Vec3{0.0, 0.0, 0.0});

	radar::Platform rx_plat("rx_plat");
	setupPlatform(rx_plat, math::Vec3{dist, 0.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::PULSED_MODE);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);

	fers_signal::RadarSignal wave("sig", 1.0, carrier, fers_signal::CwWaveform{});
	tx.setSignal(&wave);

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);
	rx.setFlag(radar::Receiver::RecvFlag::FLAG_NOPROPLOSS);

	core::World world;
	const propagation::pointscatter::PointScatterModel prop(&world);
	const auto source = core::makeActiveSource(&tx, 0.0, 1.0);
	const auto paths = prop.traceFromReceivers(nullptr, {&rx}, {source}, 0.0).at(&rx);
	const auto* path = findDirectPath(paths, tx, rx);
	REQUIRE(path != nullptr);

	REQUIRE_THAT(std::norm(path->gain), WithinRel(expected_gain, 1e-9));

	// Delay is still computed based on actual distance
	const RealType expected_delay = dist / c;
	REQUIRE_THAT(path->delay, WithinRel(expected_delay, 1e-9));
}

TEST_CASE("direct path phase is proportional to carrier frequency", "[propagation][pointscatter][direct]")
{
	ParamGuard const guard;
	params::params.reset();

	// Phase = -delay * 2 * pi * carrier
	// For same distance, doubling carrier should double the phase magnitude
	const RealType dist = 300.0;

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	// Carrier at 1 GHz
	fers_signal::RadarSignal wave1("sig1", 1.0, 1.0e9, fers_signal::CwWaveform{});

	// Carrier at 2 GHz
	fers_signal::RadarSignal wave2("sig2", 1.0, 2.0e9, fers_signal::CwWaveform{});

	radar::Platform tx_plat1("tx1");
	setupPlatform(tx_plat1, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_plat1("rx1");
	setupPlatform(rx_plat1, math::Vec3{dist, 0.0, 0.0});

	radar::Transmitter tx1(&tx_plat1, "tx1", radar::OperationMode::PULSED_MODE);
	tx1.setAntenna(&iso_ant);
	tx1.setTiming(timing);
	tx1.setSignal(&wave1);

	radar::Receiver rx1(&rx_plat1, "rx1", 42, radar::OperationMode::PULSED_MODE);
	rx1.setAntenna(&iso_ant);
	rx1.setTiming(timing);

	core::World world1;
	const propagation::pointscatter::PointScatterModel prop1(&world1);
	const auto source1 = core::makeActiveSource(&tx1, 0.0, 1.0);
	const auto paths1 = prop1.traceFromReceivers(nullptr, {&rx1}, {source1}, 0.0).at(&rx1);
	const auto* path1 = findDirectPath(paths1, tx1, rx1);
	REQUIRE(path1 != nullptr);

	radar::Platform tx_plat2("tx2");
	setupPlatform(tx_plat2, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_plat2("rx2");
	setupPlatform(rx_plat2, math::Vec3{dist, 0.0, 0.0});

	radar::Transmitter tx2(&tx_plat2, "tx2", radar::OperationMode::PULSED_MODE);
	tx2.setAntenna(&iso_ant);
	tx2.setTiming(timing);
	tx2.setSignal(&wave2);

	radar::Receiver rx2(&rx_plat2, "rx2", 42, radar::OperationMode::PULSED_MODE);
	rx2.setAntenna(&iso_ant);
	rx2.setTiming(timing);

	core::World world2;
	const propagation::pointscatter::PointScatterModel prop2(&world2);
	const auto source2 = core::makeActiveSource(&tx2, 0.0, 1.0);
	const auto paths2 = prop2.traceFromReceivers(nullptr, {&rx2}, {source2}, 0.0).at(&rx2);
	const auto* path2 = findDirectPath(paths2, tx2, rx2);
	REQUIRE(path2 != nullptr);

	// Same delay (same distance)
	REQUIRE_THAT(path1->delay, WithinRel(path2->delay, 1e-12));

	// Phase ratio should be 1:2 (derived from delay, since PropagationPath no longer carries phase)
	const RealType phase1 = -path1->delay * 2.0 * PI * 1.0e9;
	const RealType phase2 = -path2->delay * 2.0 * PI * 2.0e9;
	REQUIRE_THAT(phase2 / phase1, WithinRel(2.0, 1e-9));
}

TEST_CASE("direct path at 3D separation produces correct distance and delay", "[propagation][pointscatter][direct]")
{
	ParamGuard const guard;
	params::params.reset();

	// Tx at (100, 200, 300), Rx at (400, 600, 800)
	// Distance = sqrt(300^2 + 400^2 + 500^2) = sqrt(500000)
	const RealType c = params::c();
	const RealType carrier = 2.0e9;
	const RealType lambda = c / carrier;

	const math::Vec3 tx_pos{100.0, 200.0, 300.0};
	const math::Vec3 rx_pos{400.0, 600.0, 800.0};
	const RealType dist = std::sqrt(300.0 * 300.0 + 400.0 * 400.0 + 500.0 * 500.0);

	const RealType expected_delay = dist / c;
	const RealType expected_gain = (lambda * lambda) / (16.0 * PI * PI * dist * dist);
	const RealType expected_phase = -expected_delay * 2.0 * PI * carrier;

	radar::Platform tx_plat("tx_plat");
	setupPlatformFacing(tx_plat, tx_pos, rx_pos);

	radar::Platform rx_plat("rx_plat");
	setupPlatformFacing(rx_plat, rx_pos, tx_pos);

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::PULSED_MODE);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);

	fers_signal::RadarSignal wave("sig", 1.0, carrier, fers_signal::CwWaveform{});
	tx.setSignal(&wave);

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);

	core::World world;
	const propagation::pointscatter::PointScatterModel prop(&world);
	const auto source = core::makeActiveSource(&tx, 0.0, 1.0);
	const auto paths = prop.traceFromReceivers(nullptr, {&rx}, {source}, 0.0).at(&rx);
	const auto* path = findDirectPath(paths, tx, rx);
	REQUIRE(path != nullptr);

	REQUIRE_THAT(path->delay, WithinRel(expected_delay, 1e-9));
	REQUIRE_THAT(std::norm(path->gain), WithinRel(expected_gain, 1e-9));
	REQUIRE_THAT(-path->delay * 2.0 * PI * carrier, WithinRel(expected_phase, 1e-9));
}

TEST_CASE("direct path power scales with lambda squared", "[propagation][pointscatter][direct]")
{
	ParamGuard const guard;
	params::params.reset();

	// Friis: Pr/Pt ∝ lambda^2. At f2 = 2*f1, lambda2 = lambda1/2, so power ratio = 4.
	const RealType dist = 1000.0;

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	// Test at 1 GHz
	fers_signal::RadarSignal wave1("sig1", 1.0, 1.0e9, fers_signal::CwWaveform{});

	radar::Platform tx_plat1("tx1");
	setupPlatform(tx_plat1, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_plat1("rx1");
	setupPlatform(rx_plat1, math::Vec3{dist, 0.0, 0.0});

	radar::Transmitter tx1(&tx_plat1, "tx1", radar::OperationMode::PULSED_MODE);
	tx1.setAntenna(&iso_ant);
	tx1.setTiming(timing);
	tx1.setSignal(&wave1);

	radar::Receiver rx1(&rx_plat1, "rx1", 42, radar::OperationMode::PULSED_MODE);
	rx1.setAntenna(&iso_ant);
	rx1.setTiming(timing);

	core::World world1;
	const propagation::pointscatter::PointScatterModel prop1(&world1);
	const auto source1 = core::makeActiveSource(&tx1, 0.0, 1.0);
	const auto paths1 = prop1.traceFromReceivers(nullptr, {&rx1}, {source1}, 0.0).at(&rx1);
	const auto* path1 = findDirectPath(paths1, tx1, rx1);
	REQUIRE(path1 != nullptr);

	// Test at 2 GHz
	fers_signal::RadarSignal wave2("sig2", 1.0, 2.0e9, fers_signal::CwWaveform{});

	radar::Platform tx_plat2("tx2");
	setupPlatform(tx_plat2, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_plat2("rx2");
	setupPlatform(rx_plat2, math::Vec3{dist, 0.0, 0.0});

	radar::Transmitter tx2(&tx_plat2, "tx2", radar::OperationMode::PULSED_MODE);
	tx2.setAntenna(&iso_ant);
	tx2.setTiming(timing);
	tx2.setSignal(&wave2);

	radar::Receiver rx2(&rx_plat2, "rx2", 42, radar::OperationMode::PULSED_MODE);
	rx2.setAntenna(&iso_ant);
	rx2.setTiming(timing);

	core::World world2;
	const propagation::pointscatter::PointScatterModel prop2(&world2);
	const auto source2 = core::makeActiveSource(&tx2, 0.0, 1.0);
	const auto paths2 = prop2.traceFromReceivers(nullptr, {&rx2}, {source2}, 0.0).at(&rx2);
	const auto* path2 = findDirectPath(paths2, tx2, rx2);
	REQUIRE(path2 != nullptr);

	// Power at 1 GHz should be 4x power at 2 GHz
	REQUIRE_THAT(std::norm(path1->gain) / std::norm(path2->gain), WithinRel(4.0, 1e-9));
}

// =============================================================================
// Reflected path: bistatic radar range equation verification
// =============================================================================

TEST_CASE("bistatic path handler computes correct bistatic power for isotropic antennas",
		  "[propagation][pointscatter][reflected]")
{
	ParamGuard const guard;
	params::params.reset();

	// Setup: Tx at (0,0,0), Target at (500,0,0), Rx at (500,500,0)
	// Carrier: 1 GHz, RCS: 10 m^2
	// Bistatic: Pr/Pt = Gt * Gr * sigma * lambda^2 / (64 * pi^3 * R1^2 * R2^2)

	const RealType c = params::c();
	const RealType carrier = 1.0e9;
	const RealType lambda = c / carrier;
	const RealType rcs = 10.0;

	const math::Vec3 tx_pos{0.0, 0.0, 0.0};
	const math::Vec3 tgt_pos{500.0, 0.0, 0.0};
	const math::Vec3 rx_pos{500.0, 500.0, 0.0};

	const RealType r1 = (tgt_pos - tx_pos).length(); // 500 m
	const RealType r2 = (rx_pos - tgt_pos).length(); // 500 m

	const RealType expected_gain = (rcs * lambda * lambda) / (64.0 * PI * PI * PI * r1 * r1 * r2 * r2);
	const RealType expected_delay = (r1 + r2) / c;
	const RealType expected_phase = -expected_delay * 2.0 * PI * carrier;

	radar::Platform tx_plat("tx_plat");
	setupPlatform(tx_plat, tx_pos);

	radar::Platform tgt_plat("tgt_plat");
	setupPlatform(tgt_plat, tgt_pos);

	radar::Platform rx_plat("rx_plat");
	setupPlatform(rx_plat, rx_pos);

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::PULSED_MODE);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);

	fers_signal::RadarSignal wave("sig", 1.0, carrier, fers_signal::CwWaveform{});
	tx.setSignal(&wave);

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);

	core::World world;
	world.add(radar::createIsoTarget(&tgt_plat, "tgt", rcs, 42));
	const propagation::pointscatter::PointScatterModel prop(&world);
	const auto source = core::makeActiveSource(&tx, 0.0, 1.0);
	const auto paths = prop.traceFromReceivers(nullptr, {&rx}, {source}, 0.0).at(&rx);
	const auto* path = findBistaticPath(paths, tx, *world.getTargets().front(), rx);
	REQUIRE(path != nullptr);

	REQUIRE_THAT(std::norm(path->gain), WithinRel(expected_gain, 1e-6));
	REQUIRE_THAT(path->delay, WithinRel(expected_delay, 1e-9));
	REQUIRE_THAT(-path->delay * 2.0 * PI * carrier, WithinRel(expected_phase, 1e-9));
}

TEST_CASE("targets with no RCS contribute no bistatic path", "[propagation][pointscatter][reflected]")
{
	ParamGuard const guard;
	params::params.reset();

	const RealType carrier = 1.0e9;

	const math::Vec3 tx_pos{0.0, 0.0, 0.0};
	const math::Vec3 tgt_pos{500.0, 0.0, 0.0};
	const math::Vec3 rx_pos{500.0, 500.0, 0.0};

	radar::Platform tx_plat("tx_plat");
	setupPlatform(tx_plat, tx_pos);

	radar::Platform tgt_plat("tgt_plat");
	setupPlatform(tgt_plat, tgt_pos);

	radar::Platform rx_plat("rx_plat");
	setupPlatform(rx_plat, rx_pos);

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::PULSED_MODE);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);

	fers_signal::RadarSignal wave("sig", 1.0, carrier, fers_signal::CwWaveform{});
	tx.setSignal(&wave);

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);
	rx.setFlag(radar::Receiver::RecvFlag::FLAG_NODIRECT); // isolate the bistatic (reflected) contribution

	core::World world;
	world.add(std::make_unique<radar::Target>(&tgt_plat, "tgt")); // geometry-only, no RCS attached
	const propagation::pointscatter::PointScatterModel prop(&world);
	const auto source = core::makeActiveSource(&tx, 0.0, 1.0);
	const auto paths = prop.traceFromReceivers(nullptr, {&rx}, {source}, 0.0).at(&rx);

	REQUIRE(findBistaticPath(paths, tx, *world.getTargets().front(), rx) == nullptr);
	REQUIRE(paths.empty());
}

TEST_CASE("bistatic path power scales linearly with RCS", "[propagation][pointscatter][reflected]")
{
	ParamGuard const guard;
	params::params.reset();

	// RCS is a linear multiplier in the bistatic equation; doubling RCS should double the power
	const RealType carrier = 1.0e9;

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	// RCS = 5 m^2. Off-axis (non-collinear, non-coplanar-null) geometry: forward scatter and
	// several coordinate-plane-aligned bistatic angles are degenerate/null for this point-target
	// model's polarimetric scattering (see pointscatter.cpp's computePolarimetricReflectedLinkage).
	radar::Platform tx_plat1("tx1");
	setupPlatform(tx_plat1, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform tgt_plat1("tgt1");
	setupPlatform(tgt_plat1, math::Vec3{1000.0, 0.0, 0.0});
	radar::Platform rx_plat1("rx1");
	setupPlatform(rx_plat1, math::Vec3{1300.0, 400.0, 600.0});

	radar::Transmitter tx1(&tx_plat1, "tx1", radar::OperationMode::PULSED_MODE);
	tx1.setAntenna(&iso_ant);
	tx1.setTiming(timing);
	fers_signal::RadarSignal wave1("sig1", 1.0, carrier, fers_signal::CwWaveform{});
	tx1.setSignal(&wave1);

	radar::Receiver rx1(&rx_plat1, "rx1", 42, radar::OperationMode::PULSED_MODE);
	rx1.setAntenna(&iso_ant);
	rx1.setTiming(timing);

	core::World world1;
	world1.add(radar::createIsoTarget(&tgt_plat1, "tgt1", 5.0, 42));
	const propagation::pointscatter::PointScatterModel prop1(&world1);
	const auto source1 = core::makeActiveSource(&tx1, 0.0, 1.0);
	const auto paths1 = prop1.traceFromReceivers(nullptr, {&rx1}, {source1}, 0.0).at(&rx1);
	const auto* path1 = findBistaticPath(paths1, tx1, *world1.getTargets().front(), rx1);
	REQUIRE(path1 != nullptr);

	// RCS = 10 m^2 (doubled). Same geometry as above.
	radar::Platform tx_plat2("tx2");
	setupPlatform(tx_plat2, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform tgt_plat2("tgt2");
	setupPlatform(tgt_plat2, math::Vec3{1000.0, 0.0, 0.0});
	radar::Platform rx_plat2("rx2");
	setupPlatform(rx_plat2, math::Vec3{1300.0, 400.0, 600.0});

	radar::Transmitter tx2(&tx_plat2, "tx2", radar::OperationMode::PULSED_MODE);
	tx2.setAntenna(&iso_ant);
	tx2.setTiming(timing);
	fers_signal::RadarSignal wave2("sig2", 1.0, carrier, fers_signal::CwWaveform{});
	tx2.setSignal(&wave2);

	radar::Receiver rx2(&rx_plat2, "rx2", 42, radar::OperationMode::PULSED_MODE);
	rx2.setAntenna(&iso_ant);
	rx2.setTiming(timing);

	core::World world2;
	world2.add(radar::createIsoTarget(&tgt_plat2, "tgt2", 10.0, 42));
	const propagation::pointscatter::PointScatterModel prop2(&world2);
	const auto source2 = core::makeActiveSource(&tx2, 0.0, 1.0);
	const auto paths2 = prop2.traceFromReceivers(nullptr, {&rx2}, {source2}, 0.0).at(&rx2);
	const auto* path2 = findBistaticPath(paths2, tx2, *world2.getTargets().front(), rx2);
	REQUIRE(path2 != nullptr);

	REQUIRE_THAT(std::norm(path2->gain) / std::norm(path1->gain), WithinRel(2.0, 1e-9));
}

TEST_CASE("bistatic path delay is sum of Tx-Tgt and Tgt-Rx distances divided by c",
		  "[propagation][pointscatter][reflected]")
{
	ParamGuard const guard;
	params::params.reset();

	const RealType c = params::c();
	const RealType carrier = 1.0e9;

	// Tx at origin, Tgt at 1000 m out on +X, Rx 2000 m out on +Y from the target (not collinear
	// with Tx and Tgt: exact forward scatter isn't covered by this point-target RCS model).
	const RealType r1 = 1000.0;
	const RealType r2 = 2000.0;
	const RealType expected_delay = (r1 + r2) / c;

	radar::Platform tx_plat("tx_plat");
	setupPlatform(tx_plat, math::Vec3{0.0, 0.0, 0.0});

	radar::Platform tgt_plat("tgt_plat");
	setupPlatform(tgt_plat, math::Vec3{r1, 0.0, 0.0});

	radar::Platform rx_plat("rx_plat");
	setupPlatform(rx_plat, math::Vec3{r1, r2, 0.0});

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::PULSED_MODE);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);

	fers_signal::RadarSignal wave("sig", 1.0, carrier, fers_signal::CwWaveform{});
	tx.setSignal(&wave);

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);

	core::World world;
	world.add(radar::createIsoTarget(&tgt_plat, "tgt", 1.0, 42));
	const propagation::pointscatter::PointScatterModel prop(&world);
	const auto source = core::makeActiveSource(&tx, 0.0, 1.0);
	const auto paths = prop.traceFromReceivers(nullptr, {&rx}, {source}, 0.0).at(&rx);
	const auto* path = findBistaticPath(paths, tx, *world.getTargets().front(), rx);
	REQUIRE(path != nullptr);

	REQUIRE_THAT(path->delay, WithinRel(expected_delay, 1e-9));
}

TEST_CASE("bistatic path with noproploss flag ignores distance in power", "[propagation][pointscatter][reflected]")
{
	ParamGuard const guard;
	params::params.reset();

	const RealType c = params::c();
	const RealType carrier = 1.0e9;
	const RealType lambda = c / carrier;
	const RealType rcs = 10.0;

	// With no_prop_loss: Pr/Pt = Gt * Gr * sigma * lambda^2 / (64 * pi^3) (no R terms)
	const RealType expected_gain = (rcs * lambda * lambda) / (64.0 * PI * PI * PI);

	radar::Platform tx_plat("tx_plat");
	setupPlatform(tx_plat, math::Vec3{0.0, 0.0, 0.0});

	radar::Platform tgt_plat("tgt_plat");
	setupPlatform(tgt_plat, math::Vec3{2000.0, 0.0, 0.0});

	radar::Platform rx_plat("rx_plat");
	setupPlatform(rx_plat, math::Vec3{2000.0, 3000.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::PULSED_MODE);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);

	fers_signal::RadarSignal wave("sig", 1.0, carrier, fers_signal::CwWaveform{});
	tx.setSignal(&wave);

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);
	rx.setFlag(radar::Receiver::RecvFlag::FLAG_NOPROPLOSS);

	core::World world;
	world.add(radar::createIsoTarget(&tgt_plat, "tgt", rcs, 42));
	const propagation::pointscatter::PointScatterModel prop(&world);
	const auto source = core::makeActiveSource(&tx, 0.0, 1.0);
	const auto paths = prop.traceFromReceivers(nullptr, {&rx}, {source}, 0.0).at(&rx);
	const auto* path = findBistaticPath(paths, tx, *world.getTargets().front(), rx);
	REQUIRE(path != nullptr);

	REQUIRE_THAT(std::norm(path->gain), WithinRel(expected_gain, 1e-6));
}

TEST_CASE("bistatic path power follows R^-4 for monostatic geometry", "[propagation][pointscatter][reflected]")
{
	ParamGuard const guard;
	params::params.reset();

	// For monostatic-like geometry (Tx and Rx co-located, target along x-axis):
	// Pr/Pt = sigma * lambda^2 / (64 * pi^3 * R^4) for r1 = r2 = R
	// Doubling R should reduce power by factor of 16
	const RealType carrier = 1.0e9;

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	// R = 500 m
	radar::Platform tx_plat1("tx1");
	setupPlatform(tx_plat1, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_plat1("rx1");
	setupPlatform(rx_plat1, math::Vec3{0.0, 0.1, 0.0}); // Slightly offset to avoid co-location skip
	radar::Platform tgt_plat1("tgt1");
	setupPlatform(tgt_plat1, math::Vec3{500.0, 0.0, 0.0});

	radar::Transmitter tx1(&tx_plat1, "tx1", radar::OperationMode::PULSED_MODE);
	tx1.setAntenna(&iso_ant);
	tx1.setTiming(timing);
	fers_signal::RadarSignal wave1("sig1", 1.0, carrier, fers_signal::CwWaveform{});
	tx1.setSignal(&wave1);

	radar::Receiver rx1(&rx_plat1, "rx1", 42, radar::OperationMode::PULSED_MODE);
	rx1.setAntenna(&iso_ant);
	rx1.setTiming(timing);

	core::World world1;
	world1.add(radar::createIsoTarget(&tgt_plat1, "tgt1", 1.0, 42));
	const propagation::pointscatter::PointScatterModel prop1(&world1);
	const auto source1 = core::makeActiveSource(&tx1, 0.0, 1.0);
	const auto paths1 = prop1.traceFromReceivers(nullptr, {&rx1}, {source1}, 0.0).at(&rx1);
	const auto* path1 = findBistaticPath(paths1, tx1, *world1.getTargets().front(), rx1);
	REQUIRE(path1 != nullptr);

	// R = 1000 m (doubled)
	radar::Platform tx_plat2("tx2");
	setupPlatform(tx_plat2, math::Vec3{0.0, 0.0, 0.0});
	radar::Platform rx_plat2("rx2");
	setupPlatform(rx_plat2, math::Vec3{0.0, 0.1, 0.0});
	radar::Platform tgt_plat2("tgt2");
	setupPlatform(tgt_plat2, math::Vec3{1000.0, 0.0, 0.0});

	radar::Transmitter tx2(&tx_plat2, "tx2", radar::OperationMode::PULSED_MODE);
	tx2.setAntenna(&iso_ant);
	tx2.setTiming(timing);
	fers_signal::RadarSignal wave2("sig2", 1.0, carrier, fers_signal::CwWaveform{});
	tx2.setSignal(&wave2);

	radar::Receiver rx2(&rx_plat2, "rx2", 42, radar::OperationMode::PULSED_MODE);
	rx2.setAntenna(&iso_ant);
	rx2.setTiming(timing);

	core::World world2;
	world2.add(radar::createIsoTarget(&tgt_plat2, "tgt2", 1.0, 42));
	const propagation::pointscatter::PointScatterModel prop2(&world2);
	const auto source2 = core::makeActiveSource(&tx2, 0.0, 1.0);
	const auto paths2 = prop2.traceFromReceivers(nullptr, {&rx2}, {source2}, 0.0).at(&rx2);
	const auto* path2 = findBistaticPath(paths2, tx2, *world2.getTargets().front(), rx2);
	REQUIRE(path2 != nullptr);

	// Note: The Rx is offset by 0.1m, so this is approximately R^-4 but not exact;
	// we use a looser tolerance to account for the 0.1m offset.
	REQUIRE_THAT(std::norm(path1->gain) / std::norm(path2->gain), WithinRel(16.0, 0.01));
}

TEST_CASE("bistatic path with 3D geometry computes correct bistatic range (polarimetric)",
		 "[propagation][pointscatter][reflected]")
{
	ParamGuard const guard;
	params::params.reset();
	params::params.point_scatterer_params.polarimetric = true;

	const RealType c = params::c();
	const RealType carrier = 2.0e9;
	const RealType lambda = c / carrier;
	const RealType rcs = 5.0;

	// 3D positions
	const math::Vec3 tx_pos{0.0, 0.0, 0.0};
	const math::Vec3 tgt_pos{300.0, 400.0, 0.0}; // r1 = 500
	const math::Vec3 rx_pos{300.0, 400.0, 600.0}; // r2 = 600

	const RealType r1 = (tgt_pos - tx_pos).length();
	const RealType r2 = (rx_pos - tgt_pos).length();

	REQUIRE_THAT(r1, WithinAbs(500.0, 1e-9));
	REQUIRE_THAT(r2, WithinAbs(600.0, 1e-9));

	// Both antennas keep the default horizontal (Ludwig-3) polarisation and neither is
	// boresight-pointed along the path, and the Tx-Tgt-Rx plane isn't aligned with world Z
	// (elevation's reference axis), so the reflection's co-pol coupling isn't unity here. For this
	// geometry: k_in=(0.6,0.8,0), so Tx's radiated field at Tgt is purely TE relative to the
	// scattering plane (e_te=-1, e_tm=0). The PEC Sinclair matrix keeps a TE component TE on
	// reflection, along (0.8,-0.6,0). Rx's own horizontal reference for its viewing direction
	// (looking straight down, -k_scat=(0,0,-1)) is world +Y, giving a coupling of
	// dot((0.8,-0.6,0), (0,1,0)) = -0.6 in voltage, 0.36 in power.
	constexpr RealType polarisation_coupling = 0.36;
	const RealType expected_gain =
		(rcs * lambda * lambda) / (64.0 * PI * PI * PI * r1 * r1 * r2 * r2) * polarisation_coupling;
	const RealType expected_delay = (r1 + r2) / c;

	radar::Platform tx_plat("tx_plat");
	setupPlatform(tx_plat, tx_pos);

	radar::Platform tgt_plat("tgt_plat");
	setupPlatform(tgt_plat, tgt_pos);

	radar::Platform rx_plat("rx_plat");
	setupPlatform(rx_plat, rx_pos);

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::PULSED_MODE);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);

	fers_signal::RadarSignal wave("sig", 1.0, carrier, fers_signal::CwWaveform{});
	tx.setSignal(&wave);

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);

	core::World world;
	world.add(radar::createIsoTarget(&tgt_plat, "tgt", rcs, 42));
	const propagation::pointscatter::PointScatterModel prop(&world);
	const auto source = core::makeActiveSource(&tx, 0.0, 1.0);
	const auto paths = prop.traceFromReceivers(nullptr, {&rx}, {source}, 0.0).at(&rx);
	const auto* path = findBistaticPath(paths, tx, *world.getTargets().front(), rx);
	REQUIRE(path != nullptr);

	REQUIRE_THAT(std::norm(path->gain), WithinRel(expected_gain, 1e-6));
	REQUIRE_THAT(path->delay, WithinRel(expected_delay, 1e-9));
}

TEST_CASE("bistatic path with 3D geometry computes correct bistatic range (non-polarimetric)",
		 "[propagation][pointscatter][reflected]")
{
	ParamGuard const guard;
	params::params.reset();

	const RealType c = params::c();
	const RealType carrier = 2.0e9;
	const RealType lambda = c / carrier;
	const RealType rcs = 5.0;

	// 3D positions
	const math::Vec3 tx_pos{0.0, 0.0, 0.0};
	const math::Vec3 tgt_pos{300.0, 400.0, 0.0}; // r1 = 500
	const math::Vec3 rx_pos{300.0, 400.0, 600.0}; // r2 = 600

	const RealType r1 = (tgt_pos - tx_pos).length();
	const RealType r2 = (rx_pos - tgt_pos).length();

	REQUIRE_THAT(r1, WithinAbs(500.0, 1e-9));
	REQUIRE_THAT(r2, WithinAbs(600.0, 1e-9));

	// The non-polarimetric model has no antenna-to-antenna coupling term, so unlike the
	// polarimetric case this geometry's 3D offset has no effect on the expected gain.
	const RealType expected_gain = (rcs * lambda * lambda) / (64.0 * PI * PI * PI * r1 * r1 * r2 * r2);
	const RealType expected_delay = (r1 + r2) / c;

	radar::Platform tx_plat("tx_plat");
	setupPlatform(tx_plat, tx_pos);

	radar::Platform tgt_plat("tgt_plat");
	setupPlatform(tgt_plat, tgt_pos);

	radar::Platform rx_plat("rx_plat");
	setupPlatform(rx_plat, rx_pos);

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	radar::Transmitter tx(&tx_plat, "tx", radar::OperationMode::PULSED_MODE);
	tx.setAntenna(&iso_ant);
	tx.setTiming(timing);

	fers_signal::RadarSignal wave("sig", 1.0, carrier, fers_signal::CwWaveform{});
	tx.setSignal(&wave);

	radar::Receiver rx(&rx_plat, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx.setAntenna(&iso_ant);
	rx.setTiming(timing);

	core::World world;
	world.add(radar::createIsoTarget(&tgt_plat, "tgt", rcs, 42));
	const propagation::pointscatter::PointScatterModel prop(&world);
	const auto source = core::makeActiveSource(&tx, 0.0, 1.0);
	const auto paths = prop.traceFromReceivers(nullptr, {&rx}, {source}, 0.0).at(&rx);
	const auto* path = findBistaticPath(paths, tx, *world.getTargets().front(), rx);
	REQUIRE(path != nullptr);

	REQUIRE_THAT(std::norm(path->gain), WithinRel(expected_gain, 1e-6));
	REQUIRE_THAT(path->delay, WithinRel(expected_delay, 1e-9));
}
