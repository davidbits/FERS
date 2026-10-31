// Tests for simulation::calculateResponses. Verifies co-location skipping,
// and that direct/reflected-path responses are generated and routed to the
// correct receiver queue (pulsed inbox vs. streaming-mode interference log).

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <memory>
#include <vector>

#include "antenna/antenna_factory.h"
#include "core/config.h"
#include "core/parameters.h"
#include "core/world.h"
#include "math/coord.h"
#include "math/geometry_ops.h"
#include "propagation/pointscatter/pointscatter.h"
#include "propagation/propagation_model.h"
#include "radar/platform.h"
#include "radar/receiver.h"
#include "radar/target.h"
#include "radar/transmitter.h"
#include "serial/response.h"
#include "signal/radar_signal.h"
#include "simulation/channel_model.h"
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

	// A propagation model that hands back a scripted, per-timestep-index path presence for a
	// single path_id, ignoring actual geometry - used to test calculateResponses's own
	// bookkeeping (splitting a path that drops out mid-pulse) in isolation from any real
	// propagation model's behavior.
	class ScriptedPropagationModel final : public propagation::PropagationModel
	{
	public:
		ScriptedPropagationModel(std::vector<bool> present_at_index, const RealType delay, radar::Receiver* receiver) :
			_present_at_index(std::move(present_at_index)), _delay(delay), _receiver(receiver)
		{
		}

		[[nodiscard]] std::vector<propagation::PathsAtTime>
		traceFromTransmitter(propagation::ThreadContext*, const radar::Transmitter&,
						const std::vector<RealType>& times) const override
		{
			std::vector<propagation::PathsAtTime> result(times.size());
			for (size_t i = 0; i < times.size(); ++i)
			{
				result[i].tx_time = times[i];
				if (i < _present_at_index.size() && _present_at_index[i])
				{
					result[i].paths.push_back(propagation::PropagationPath{.delay = _delay,
																		   .gain = ComplexType{1.0, 0.0},
																		   .path_id = 1,
																		   .source_index = 0,
																		   .receiver = _receiver});
				}
			}
			return result;
		}

		[[nodiscard]] std::unordered_map<radar::Receiver*, std::vector<propagation::PropagationPath>>
		traceFromReceivers(propagation::ThreadContext*, const std::vector<radar::Receiver*>&,
						   const std::vector<core::ActiveStreamingSource>&, RealType) const override
		{
			return {};
		}

		[[nodiscard]] unsigned int maxPropagationLegs() const noexcept override { return 1; }

	private:
		std::vector<bool> _present_at_index;
		RealType _delay;
		radar::Receiver* _receiver;
	};

	void setupPlatform(radar::Platform& plat, const math::Vec3& pos)
	{
		plat.getMotionPath()->addCoord(math::Coord{pos, 0.0});
		plat.getMotionPath()->finalize();
		plat.getRotationPath()->addCoord(math::RotationCoord{0.0, 0.0, 0.0});
		plat.getRotationPath()->finalize();
	}

	// calculateResponses requires a Response's underlying RadarSignal to be
	// isPulsed() (it throws otherwise), so every transmitter here carries a
	// PulseWaveform rather than a CwWaveform or a mock.
	fers_signal::RadarSignal makeSampledWave(const std::string& name, const RealType power, const RealType carrier,
											 std::vector<ComplexType> samples, const RealType rate)
	{
		fers_signal::PulseWaveform sampled;
		sampled.load(samples, static_cast<unsigned>(samples.size()), rate);
		return {name, power, carrier, std::move(sampled)};
	}
}

// =============================================================================
// calculateResponses: co-location skipping produces no responses
// =============================================================================

TEST_CASE("calculateResponses produces no responses when Tx and Rx share the same platform (direct path)",
		  "[simulation][channel_model][response]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setSimSamplingRate(1000.0);

	core::World world;

	auto shared_plat = std::make_unique<radar::Platform>("shared");
	setupPlatform(*shared_plat, math::Vec3{0.0, 0.0, 0.0});
	auto* shared_plat_ptr = shared_plat.get();

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	auto tx = std::make_unique<radar::Transmitter>(shared_plat_ptr, "tx", radar::OperationMode::PULSED_MODE);
	tx->setAntenna(&iso_ant);
	tx->setTiming(timing);

	auto wave = makeSampledWave("sig", 1.0, 1e9, std::vector<ComplexType>{ComplexType{1.0, 0.0}, ComplexType{1.0, 0.0}},
								1000.0);
	tx->setSignal(&wave);

	auto rx = std::make_unique<radar::Receiver>(shared_plat_ptr, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx->setAntenna(&iso_ant);
	rx->setTiming(timing);
	auto* rx_ptr = rx.get();
	auto* tx_ptr = tx.get();

	world.add(std::move(shared_plat));
	world.add(std::move(tx));
	world.add(std::move(rx));

	const propagation::pointscatter::PointScatterModel prop(&world);
	simulation::calculateResponses(*tx_ptr, 0.0, prop, nullptr);

	REQUIRE(rx_ptr->drainInbox().empty());
}

TEST_CASE("calculateResponses produces no direct-path response for a co-located monostatic pair",
		  "[simulation][channel_model][response]")
{
	// Real monostatic Tx/Rx pairs (see parseMonostatic) are always parsed onto the
	// same platform, so co-location alone (not attachment) is what pointscatter's
	// directPath() uses to suppress self-interference. Exercise that realistic case.
	ParamGuard const guard;
	params::params.reset();
	params::setSimSamplingRate(1000.0);

	core::World world;

	auto plat = std::make_unique<radar::Platform>("mono_plat");
	setupPlatform(*plat, math::Vec3{0.0, 0.0, 0.0});
	auto* plat_ptr = plat.get();

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	auto tx = std::make_unique<radar::Transmitter>(plat_ptr, "tx", radar::OperationMode::PULSED_MODE);
	tx->setAntenna(&iso_ant);
	tx->setTiming(timing);

	auto wave = makeSampledWave("sig", 1.0, 1e9, std::vector<ComplexType>{ComplexType{1.0, 0.0}, ComplexType{1.0, 0.0}},
								1000.0);
	tx->setSignal(&wave);

	auto rx = std::make_unique<radar::Receiver>(plat_ptr, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx->setAntenna(&iso_ant);
	rx->setTiming(timing);
	auto* rx_ptr = rx.get();
	auto* tx_ptr = tx.get();

	tx_ptr->setAttached(rx_ptr);
	rx_ptr->setAttached(tx_ptr);

	world.add(std::move(plat));
	world.add(std::move(tx));
	world.add(std::move(rx));

	const propagation::pointscatter::PointScatterModel prop(&world);
	simulation::calculateResponses(*tx_ptr, 0.0, prop, nullptr);

	REQUIRE(rx_ptr->drainInbox().empty());
}

TEST_CASE("calculateResponses produces no reflected-path response when target co-located with Tx",
		  "[simulation][channel_model][response]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setSimSamplingRate(1000.0);

	core::World world;

	auto tx_plat = std::make_unique<radar::Platform>("tx_plat");
	setupPlatform(*tx_plat, math::Vec3{0.0, 0.0, 0.0});
	auto* tx_plat_ptr = tx_plat.get();

	auto rx_plat = std::make_unique<radar::Platform>("rx_plat");
	setupPlatform(*rx_plat, math::Vec3{1000.0, 0.0, 0.0});
	auto* rx_plat_ptr = rx_plat.get();

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	auto tx = std::make_unique<radar::Transmitter>(tx_plat_ptr, "tx", radar::OperationMode::PULSED_MODE);
	tx->setAntenna(&iso_ant);
	tx->setTiming(timing);

	auto wave = makeSampledWave("sig", 1.0, 1e9, std::vector<ComplexType>{ComplexType{1.0, 0.0}, ComplexType{1.0, 0.0}},
								1000.0);
	tx->setSignal(&wave);

	auto rx = std::make_unique<radar::Receiver>(rx_plat_ptr, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx->setAntenna(&iso_ant);
	rx->setTiming(timing);
	// Suppress the (co-location-independent) direct path so only the co-located
	// target's reflected-path skip is under test.
	rx->setFlag(radar::Receiver::RecvFlag::FLAG_NODIRECT);
	auto* rx_ptr = rx.get();
	auto* tx_ptr = tx.get();

	// Target on same platform as Tx
	auto tgt = radar::createIsoTarget(tx_plat_ptr, "tgt", 10.0, 42);

	world.add(std::move(tx_plat));
	world.add(std::move(rx_plat));
	world.add(std::move(tx));
	world.add(std::move(rx));
	world.add(std::move(tgt));

	const propagation::pointscatter::PointScatterModel prop(&world);
	simulation::calculateResponses(*tx_ptr, 0.0, prop, nullptr);

	REQUIRE(rx_ptr->drainInbox().empty());
}

TEST_CASE("calculateResponses produces no reflected-path response when target co-located with Rx",
		  "[simulation][channel_model][response]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setSimSamplingRate(1000.0);

	core::World world;

	auto tx_plat = std::make_unique<radar::Platform>("tx_plat");
	setupPlatform(*tx_plat, math::Vec3{0.0, 0.0, 0.0});
	auto* tx_plat_ptr = tx_plat.get();

	auto rx_plat = std::make_unique<radar::Platform>("rx_plat");
	setupPlatform(*rx_plat, math::Vec3{1000.0, 0.0, 0.0});
	auto* rx_plat_ptr = rx_plat.get();

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	auto tx = std::make_unique<radar::Transmitter>(tx_plat_ptr, "tx", radar::OperationMode::PULSED_MODE);
	tx->setAntenna(&iso_ant);
	tx->setTiming(timing);

	auto wave = makeSampledWave("sig", 1.0, 1e9, std::vector<ComplexType>{ComplexType{1.0, 0.0}, ComplexType{1.0, 0.0}},
								1000.0);
	tx->setSignal(&wave);

	auto rx = std::make_unique<radar::Receiver>(rx_plat_ptr, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx->setAntenna(&iso_ant);
	rx->setTiming(timing);
	rx->setFlag(radar::Receiver::RecvFlag::FLAG_NODIRECT);
	auto* rx_ptr = rx.get();
	auto* tx_ptr = tx.get();

	// Target on same platform as Rx
	auto tgt = radar::createIsoTarget(rx_plat_ptr, "tgt", 10.0, 42);

	world.add(std::move(tx_plat));
	world.add(std::move(rx_plat));
	world.add(std::move(tx));
	world.add(std::move(rx));
	world.add(std::move(tgt));

	const propagation::pointscatter::PointScatterModel prop(&world);
	simulation::calculateResponses(*tx_ptr, 0.0, prop, nullptr);

	REQUIRE(rx_ptr->drainInbox().empty());
}

// =============================================================================
// calculateResponses: valid paths produce routed responses with correct timing
// =============================================================================

TEST_CASE("calculateResponses direct path produces a routed response with interp points",
		  "[simulation][channel_model][response]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setSimSamplingRate(1000.0); // 1000 samples/sec

	core::World world;

	auto tx_plat = std::make_unique<radar::Platform>("tx_plat");
	setupPlatform(*tx_plat, math::Vec3{0.0, 0.0, 0.0});
	auto* tx_plat_ptr = tx_plat.get();

	auto rx_plat = std::make_unique<radar::Platform>("rx_plat");
	setupPlatform(*rx_plat, math::Vec3{1000.0, 0.0, 0.0});
	auto* rx_plat_ptr = rx_plat.get();

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	auto tx = std::make_unique<radar::Transmitter>(tx_plat_ptr, "tx", radar::OperationMode::PULSED_MODE);
	tx->setAntenna(&iso_ant);
	tx->setTiming(timing);

	auto wave = makeSampledWave("sig", 1.0, 1e9, std::vector<ComplexType>{ComplexType{1.0, 0.0}, ComplexType{1.0, 0.0}},
								1000.0);
	tx->setSignal(&wave);

	auto rx = std::make_unique<radar::Receiver>(rx_plat_ptr, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx->setAntenna(&iso_ant);
	rx->setTiming(timing);
	auto* rx_ptr = rx.get();
	auto* tx_ptr = tx.get();

	world.add(std::move(tx_plat));
	world.add(std::move(rx_plat));
	world.add(std::move(tx));
	world.add(std::move(rx));

	const propagation::pointscatter::PointScatterModel prop(&world);
	simulation::calculateResponses(*tx_ptr, 0.0, prop, nullptr);

	auto inbox = rx_ptr->drainInbox();
	REQUIRE(inbox.size() == 1);

	// This path is present at every sample from the transmission's first to its last, so its
	// Response has no lead taper at all (left_interval is 0 at the transmission's true
	// start) and startTime() equals the propagation delay itself. Still compare with a
	// generous tolerance rather than an exact match, in case that ever changes.
	const RealType expected_delay = 1000.0 / params::c();
	const RealType edge = 1.0 / params::simSamplingRate();
	REQUIRE_THAT(inbox[0]->startTime(), WithinAbs(expected_delay, 3.0 * edge));
}

TEST_CASE("calculateResponses reflected path produces a routed response", "[simulation][channel_model][response]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setSimSamplingRate(1000.0);

	core::World world;

	auto tx_plat = std::make_unique<radar::Platform>("tx_plat");
	setupPlatform(*tx_plat, math::Vec3{0.0, 0.0, 0.0});

	auto tgt_plat = std::make_unique<radar::Platform>("tgt_plat");
	setupPlatform(*tgt_plat, math::Vec3{500.0, 0.0, 0.0});
	auto* tgt_plat_ptr = tgt_plat.get();

	auto rx_plat = std::make_unique<radar::Platform>("rx_plat");
	setupPlatform(*rx_plat, math::Vec3{500.0, 500.0, 0.0});
	auto* rx_plat_ptr = rx_plat.get();

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	auto tx = std::make_unique<radar::Transmitter>(tx_plat.get(), "tx", radar::OperationMode::PULSED_MODE);
	tx->setAntenna(&iso_ant);
	tx->setTiming(timing);

	auto wave = makeSampledWave("sig", 1.0, 1e9, std::vector<ComplexType>{ComplexType{1.0, 0.0}, ComplexType{1.0, 0.0}},
								1000.0);
	tx->setSignal(&wave);

	auto rx = std::make_unique<radar::Receiver>(rx_plat_ptr, "rx", 42, radar::OperationMode::PULSED_MODE);
	rx->setAntenna(&iso_ant);
	rx->setTiming(timing);
	// Isolate the reflected path from this geometry's (perfectly valid) direct path.
	rx->setFlag(radar::Receiver::RecvFlag::FLAG_NODIRECT);
	auto* rx_ptr = rx.get();
	auto* tx_ptr = tx.get();

	auto tgt = radar::createIsoTarget(tgt_plat_ptr, "tgt", 10.0, 42);

	world.add(std::move(tx_plat));
	world.add(std::move(rx_plat));
	world.add(std::move(tgt_plat));
	world.add(std::move(tx));
	world.add(std::move(rx));
	world.add(std::move(tgt));

	const propagation::pointscatter::PointScatterModel prop(&world);
	simulation::calculateResponses(*tx_ptr, 0.0, prop, nullptr);

	auto inbox = rx_ptr->drainInbox();
	REQUIRE(inbox.size() == 1);

	const RealType r1 = 500.0;
	const RealType r2 = 500.0;
	const RealType expected_delay = (r1 + r2) / params::c();
	const RealType edge = 1.0 / params::simSamplingRate();
	REQUIRE_THAT(inbox[0]->startTime(), WithinAbs(expected_delay, 3.0 * edge));
}

TEST_CASE("calculateResponses direct path response spans approximately the signal duration",
		  "[simulation][channel_model][response]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setSimSamplingRate(10000.0); // High sample rate for many points

	const RealType dist = 2000.0;
	const RealType carrier = 1.0e9;

	core::World world;

	auto tx_plat = std::make_unique<radar::Platform>("tx_plat");
	setupPlatform(*tx_plat, math::Vec3{0.0, 0.0, 0.0});

	auto rx_plat = std::make_unique<radar::Platform>("rx_plat");
	setupPlatform(*rx_plat, math::Vec3{dist, 0.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	auto tx = std::make_unique<radar::Transmitter>(tx_plat.get(), "tx", radar::OperationMode::PULSED_MODE);
	tx->setAntenna(&iso_ant);
	tx->setTiming(timing);

	// 5e-3 s worth of samples at 10000 Hz.
	const std::vector<ComplexType> samples(50, ComplexType{1.0, 0.0});
	auto wave = makeSampledWave("sig", 1.0, carrier, samples, 10000.0);
	tx->setSignal(&wave);

	auto rx = std::make_unique<radar::Receiver>(rx_plat.get(), "rx", 42, radar::OperationMode::PULSED_MODE);
	rx->setAntenna(&iso_ant);
	rx->setTiming(timing);
	auto* rx_ptr = rx.get();
	auto* tx_ptr = tx.get();

	world.add(std::move(tx_plat));
	world.add(std::move(rx_plat));
	world.add(std::move(tx));
	world.add(std::move(rx));

	const propagation::pointscatter::PointScatterModel prop(&world);
	simulation::calculateResponses(*tx_ptr, 0.0, prop, nullptr);

	auto inbox = rx_ptr->drainInbox();
	REQUIRE(inbox.size() == 1);

	// This path spans the whole transmission, so its Response has no taper padding at either
	// end (both left_interval and right_interval are 0 at the transmission's true
	// start/end) - length should track the signal length closely.
	const RealType edge = 1.0 / params::simSamplingRate();
	REQUIRE_THAT(inbox[0]->getRxDuration(), WithinAbs(5e-3, 4.0 * edge));
}

// =============================================================================
// calculateResponses: a path_id dropping out mid-pulse splits into separate Responses
// =============================================================================

TEST_CASE("calculateResponses splits a path into separate Responses when it drops out mid-pulse",
		  "[simulation][channel_model][response]")
{
	ParamGuard const guard;
	params::params.reset();
	params::setSimSamplingRate(1000.0); // sample_period = 1ms

	core::World world;

	auto tx_plat = std::make_unique<radar::Platform>("tx_plat");
	setupPlatform(*tx_plat, math::Vec3{0.0, 0.0, 0.0});

	auto rx_plat = std::make_unique<radar::Platform>("rx_plat");
	setupPlatform(*rx_plat, math::Vec3{1000.0, 0.0, 0.0});

	antenna::Isotropic iso_ant("iso");
	auto timing = std::make_shared<timing::Timing>("clk", 42);

	auto tx = std::make_unique<radar::Transmitter>(tx_plat.get(), "tx", radar::OperationMode::PULSED_MODE);
	tx->setAntenna(&iso_ant);
	tx->setTiming(timing);

	// 6ms of samples at 1000Hz -> point_count = 6, times = {0, 1, 2, 3, 4, 5, 6} ms.
	const std::vector<ComplexType> samples(6, ComplexType{1.0, 0.0});
	auto wave = makeSampledWave("sig", 1.0, 1e9, samples, 1000.0);
	tx->setSignal(&wave);

	auto rx = std::make_unique<radar::Receiver>(rx_plat.get(), "rx", 42, radar::OperationMode::PULSED_MODE);
	rx->setAntenna(&iso_ant);
	rx->setTiming(timing);
	auto* rx_ptr = rx.get();
	auto* tx_ptr = tx.get();

	world.add(std::move(tx_plat));
	world.add(std::move(rx_plat));
	world.add(std::move(tx));
	world.add(std::move(rx));

	// Same path_id present at indices 0-2 and 4-6, missing at index 3 (momentary occlusion) -
	// a fixed, geometry-independent delay keeps the expected timings exact (no Doppler).
	constexpr RealType delay = 0.5e-3;
	const ScriptedPropagationModel prop({true, true, true, false, true, true, true}, delay, rx_ptr);
	simulation::calculateResponses(*tx_ptr, 0.0, prop, nullptr);

	auto inbox = rx_ptr->drainInbox();
	REQUIRE(inbox.size() == 2);

	std::ranges::sort(inbox, {}, [](const auto& r) { return r->startTime(); });

	constexpr RealType sample_period = 1.0e-3;

	// First response: real points at t=0,1,2ms (+delay). left_interval=0 (index 0 is the
	// transmission's true start), so startTime() lands exactly on the first real point.
	REQUIRE_THAT(inbox[0]->startTime(), WithinAbs(delay, 1e-9));
	// It's evicted when index 3 doesn't match, so its tail taper used the normal grid gap
	// to index 3 (unrelated to the transmission's own end) - meeting the second response's
	// lead taper exactly at index 3's own time, with no overlap or gap between them.
	REQUIRE_THAT(inbox[0]->endTime(), WithinAbs(3 * sample_period + delay, 1e-9));

	// Second response: real points at t=4,5,6ms (+delay).
	REQUIRE_THAT(inbox[1]->startTime(), WithinAbs(3 * sample_period + delay, 1e-9));
	// right_interval=0 at index 6 (the transmission's true end), so endTime() lands exactly
	// on the last real point.
	REQUIRE_THAT(inbox[1]->endTime(), WithinAbs(6 * sample_period + delay, 1e-9));
}
