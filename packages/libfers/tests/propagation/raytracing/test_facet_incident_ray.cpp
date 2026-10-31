#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

#include "propagation/raytracing/sbr_impl.h"
#include "propagation/raytracing/sbr_shared.h"

using Catch::Matchers::WithinAbs;
namespace rt = propagation::raytracing;
namespace prop = propagation;

namespace
{
	// One source/dest antenna's carrier: 30mm wavelength.
	std::vector<prop::CarrierModel<float>> makeCarriers()
	{
		prop::CarrierModel<float> carrier{};
		carrier.kind = prop::CarrierModelKind::Constant;
		carrier.params.constant.frequency = prop::C<float> / 0.03f;
		return {carrier};
	}

	std::vector<double> times{0.0};

	// A single right triangle in the z=0 plane: v0=(0,0,0), v1=(1,0,0), v2=(0,1,0), area 0.5,
	// outward normal +Z (CCW winding as seen from +Z). A lossless dielectric (relative_permittivity
	// != 1) so the facet has a nonzero reflection coefficient.
	struct TestScene
	{
		std::vector<prop::Float3> vertices{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
		std::vector<prop::Uint3> indices{{0, 1, 2}};
		std::vector<rt::Material> materials{{4.0f, 0.0f}};
		std::vector<rt::AntennaModel> antenna_models{{.kind = rt::AntennaKind::Isotropic,
													   .efficiency = 1.0,
													   .params = {},
													   .horizontal_pol = prop::CFloat{1.0f, 0.0f},
													   .vertical_pol = prop::CFloat{0.0f, 0.0f}}};

		// A source antenna sitting above the triangle, boresight (local +X) pointing straight down
		// (world -Z), matching the ray's launch direction below.
		std::vector<rt::ActiveAntenna> source_antennas{{.position = prop::Double3{1.0 / 3.0, 1.0 / 3.0, 5.0},
														.direction = prop::FloatAzEl{0.0f, -prop::PI_V<float> / 2.0f},
														.antenna_model_index = 0}};

		// Identity object-to-world / world-to-object transforms.
		prop::Float3x4 identity{prop::Float3{1, 0, 0}, prop::Float3{0, 1, 0}, prop::Float3{0, 0, 1}, prop::Float3{0, 0, 0}};

		[[nodiscard]] rt::HitInfo hitAtCentroid() const
		{
			rt::HitInfo hit{};
			hit.triangle_index = 0;
			hit.material_index = 0;
			hit.bary_u = 1.0f / 3.0f;
			hit.bary_v = 1.0f / 3.0f;
			hit.world_to_obj = identity;
			hit.obj_to_world = identity;
			return hit;
		}
	};

	// Ray originating 5 units above the triangle's centroid, travelling straight down (normal
	// incidence), with no prior accumulated path length. Matches TestScene::source_antennas[0]'s
	// position/boresight, so the source antenna's gain/polarisation sampling is well-defined.
	rt::PathState makeIncidentRay(float weight)
	{
		rt::PathState ray{};
		ray.path_vertices[0] = prop::Double3{1.0 / 3.0, 1.0 / 3.0, 5.0};
		ray.trace_directions[0] = prop::Float3{0.0f, 0.0f, -1.0f};
		ray.local_emmission_direction = prop::Float3{1.0f, 0.0f, 0.0f}; // local boresight, matching source direction
		ray.length = 0.0;
		ray.ray_count = 0;
		ray.hash = 0;
		ray.weight = weight;
		ray.source_index = 0;
		ray.time_index = 0;
		return ray;
	}

	std::vector<rt::ActiveAntenna> makeDestAntennas()
	{
		return {{.position = prop::Double3{1.0 / 3.0, 1.0 / 3.0, 100.0},
				.direction = prop::FloatAzEl{0.0f, 0.0f},
				.antenna_model_index = 0}};
	}

	rt::SbrParams makeParams(const TestScene& scene, const std::vector<prop::CarrierModel<float>>& carriers,
							 const std::vector<rt::ActiveAntenna>& dst_antennas, std::vector<rt::RxFlags>& rx_flags)
	{
		rt::SbrParams params{};
		params.scatter_limit = 4;
		params.directions_per_source = 1000;
		params.vertices = scene.vertices.data();
		params.indices = scene.indices.data();
		params.materials = scene.materials.data();
		params.times = times.data();
		params.carrier_models = carriers.data();
		params.antenna_models = scene.antenna_models.data();
		params.source_antennas = scene.source_antennas.data();
		params.source_antenna_count = static_cast<uint32_t>(scene.source_antennas.size());
		params.dest_antennas = dst_antennas.data();
		params.dest_antenna_count = static_cast<uint32_t>(dst_antennas.size());
		params.rx_flags = rx_flags.data();
		params.rx_to_tx = false;
		return params;
	}

	// Always-unoccluded shadow test.
	auto unoccluded = [](prop::Float3, prop::Float3) { return INFINITY; };
}

TEST_CASE("facetIncidentRay computes delay as the full source-to-hit-to-destination distance",
		  "[raytracing][facet_hit]")
{
	const TestScene scene;
	const auto hit = scene.hitAtCentroid();
	const auto carriers = makeCarriers();
	auto ray = makeIncidentRay(0.001f);

	const auto dst_antennas = makeDestAntennas();
	std::vector<rt::RxFlags> rx_flags{static_cast<rt::RxFlags>(0)};
	const auto params = makeParams(scene, carriers, dst_antennas, rx_flags);

	rt::Contribution out{};
	auto get_ptr = [&out] { return &out; };

	(void)rt::facetHit(params, &ray, hit, unoccluded, get_ptr);

	// go_dist (source to hit) = 5, po_dist (hit to destination) = 100, converted to a time delay.
	REQUIRE_THAT(out.delay, WithinAbs(105.0 / prop::C<double>, 1e-15));
}

TEST_CASE("facetIncidentRay's contribution scales linearly with ray weight in the sub-triangle regime",
		  "[raytracing][facet_hit]")
{
	const TestScene scene;
	const auto hit = scene.hitAtCentroid();
	const auto carriers = makeCarriers();
	const auto dst_antennas = makeDestAntennas();

	// go_dist=5, so a_tube = 25*weight; both weights below keep a_tube well under the 0.5 triangle
	// area (0.025 and 0.05), i.e. in the linear (uncapped) regime.
	auto run = [&](float weight) -> float
	{
		std::vector<rt::RxFlags> rx_flags{static_cast<rt::RxFlags>(0)};
		const auto params = makeParams(scene, carriers, dst_antennas, rx_flags);

		rt::Contribution out{};
		auto get_ptr = [&out] { return &out; };

		auto ray = makeIncidentRay(weight);
		(void)rt::facetHit(params, &ray, hit, unoccluded, get_ptr);
		return prop::cabs(out.path_gain);
	};

	const float mag1 = run(0.001f);
	const float mag2 = run(0.002f);

	REQUIRE(mag1 > 0.0f);
	REQUIRE_THAT(mag2 / mag1, WithinAbs(2.0, 1e-6));
}

TEST_CASE("facetIncidentRay's contribution saturates once the tube footprint exceeds the triangle",
		  "[raytracing][facet_hit]")
{
	const TestScene scene;
	const auto hit = scene.hitAtCentroid();
	const auto carriers = makeCarriers();
	const auto dst_antennas = makeDestAntennas();

	// go_dist=5, so a_tube = 25*weight; both weights well exceed the 0.5 triangle area (25 and 50),
	// i.e. both should clamp to the same full-triangle patch and produce identical contributions.
	auto run = [&](float weight) -> float
	{
		std::vector<rt::RxFlags> rx_flags{static_cast<rt::RxFlags>(0)};
		const auto params = makeParams(scene, carriers, dst_antennas, rx_flags);

		rt::Contribution out{};
		auto get_ptr = [&out] { return &out; };

		auto ray = makeIncidentRay(weight);
		(void)rt::facetHit(params, &ray, hit, unoccluded, get_ptr);
		return prop::cabs(out.path_gain);
	};

	const float mag1 = run(1.0f);
	const float mag2 = run(2.0f);

	REQUIRE_THAT(mag1, WithinAbs(double(mag2), 1e-6));
}

namespace
{
	// A two-facet scene exercising the on-demand radiation cache's extension path across
	// sequential facetIncidentRay calls on the same PathState:
	//
	// Facet A: the z=0 triangle above, hit by a ray from (1/3,1/3,5) travelling straight down;
	//   reflects it straight back up.
	// Facet B: a downward-facing triangle at z=10, directly above facet A's hit point (so it's
	//   hit at the same barycentric weights by construction); reflects the ray straight back down.
	// A destination antenna sits below facet A, so only facet B's hit faces it (facet A's +Z
	// normal doesn't), producing exactly one PO contribution built from both bounces' cached
	// surface current.
	struct TwoBounceScene
	{
		// Facet A (index 0): v0=(0,0,0), v1=(1,0,0), v2=(0,1,0), outward normal +Z.
		// Facet B (index 1): v0=(0,0,10), v1=(0,1,10), v2=(1,0,10), outward normal -Z.
		std::vector<prop::Float3> vertices{{0.0f, 0.0f, 0.0f},	{1.0f, 0.0f, 0.0f},	 {0.0f, 1.0f, 0.0f},
										   {0.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 10.0f}, {1.0f, 0.0f, 10.0f}};
		std::vector<prop::Uint3> indices{{0, 1, 2}, {3, 4, 5}};
		std::vector<rt::Material> materials{{4.0f, 0.0f}};
		std::vector<rt::AntennaModel> antenna_models{{.kind = rt::AntennaKind::Isotropic,
													  .efficiency = 1.0,
													  .params = {},
													  .horizontal_pol = prop::CFloat{1.0f, 0.0f},
													  .vertical_pol = prop::CFloat{0.0f, 0.0f}}};
		std::vector<rt::ActiveAntenna> source_antennas{{.position = prop::Double3{1.0 / 3.0, 1.0 / 3.0, 5.0},
														.direction = prop::FloatAzEl{0.0f, -prop::PI_V<float> / 2.0f},
														.antenna_model_index = 0}};

		prop::Float3x4 identity{prop::Float3{1, 0, 0}, prop::Float3{0, 1, 0}, prop::Float3{0, 0, 1},
								prop::Float3{0, 0, 0}};

		[[nodiscard]] rt::HitInfo hitAt(uint32_t triangle_index) const
		{
			rt::HitInfo hit{};
			hit.triangle_index = triangle_index;
			hit.material_index = 0;
			hit.bary_u = 1.0f / 3.0f;
			hit.bary_v = 1.0f / 3.0f;
			hit.world_to_obj = identity;
			hit.obj_to_world = identity;
			return hit;
		}
	};
}

TEST_CASE("facetIncidentRay extends the radiation cache across a second bounce without throwing",
		  "[raytracing][facet_hit]")
{
	const TwoBounceScene scene;
	const auto carriers = makeCarriers();
	const std::vector<rt::ActiveAntenna> dst_antennas{{.position = prop::Double3{1.0 / 3.0, 1.0 / 3.0, -50.0},
													   .direction = prop::FloatAzEl{0.0f, 0.0f},
													   .antenna_model_index = 0}};
	std::vector<rt::RxFlags> rx_flags{static_cast<rt::RxFlags>(0)};
	std::vector<rt::Contribution> contribs(2);

	rt::SbrParams params{};
	params.scatter_limit = 4;
	params.directions_per_source = 1000;
	params.vertices = scene.vertices.data();
	params.indices = scene.indices.data();
	params.materials = scene.materials.data();
	params.times = times.data();
	params.carrier_models = carriers.data();
	params.antenna_models = scene.antenna_models.data();
	params.source_antennas = scene.source_antennas.data();
	params.source_antenna_count = static_cast<uint32_t>(scene.source_antennas.size());
	params.dest_antennas = dst_antennas.data();
	params.dest_antenna_count = static_cast<uint32_t>(dst_antennas.size());
	params.rx_flags = rx_flags.data();
	params.rx_to_tx = false;

	auto ray = makeIncidentRay(0.001f);

	uint32_t next_idx = 0;
	auto get_idx = [&next_idx] { return next_idx++; };

	bool should_continue = false;
	REQUIRE_NOTHROW(should_continue = rt::facetHit(params, &ray, scene.hitAt(0), unoccluded, get_idx));
	REQUIRE(should_continue);
	REQUIRE(next_idx == 0); // facet A doesn't face the destination - no contribution yet.

	REQUIRE_NOTHROW(rt::facetHit(params, &ray, scene.hitAt(1), unoccluded, get_idx));
	REQUIRE(next_idx == 1);

	// source->A (5) + A->B (10) + B->dest (60) = 75 units.
	REQUIRE_THAT(contribs[0].delay, WithinAbs(75.0 / prop::C<double>, 1e-12));
}
