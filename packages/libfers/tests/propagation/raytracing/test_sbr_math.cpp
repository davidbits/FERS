#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <vector>

#include "math/geometry_ops.h"
#include "propagation/common.h"
#include "propagation/math.h"
#include "propagation/raytracing/sbr_impl.h"

using Catch::Matchers::WithinAbs;
namespace rt = propagation::raytracing;
namespace prop = propagation;

TEST_CASE("rotateLocalToWorld's boresight matches Vec3(SVec3(...))", "[raytracing][math]")
{
	const prop::AzEl rot{0.3, -0.2};
	const prop::Double3 local_boresight{1.0, 0.0, 0.0};
	const auto world = prop::rotateLocalToWorld(rot, local_boresight);

	const math::Vec3 expected(math::SVec3(1.0, rot.azimuth, rot.elevation));
	REQUIRE_THAT(world.x, WithinAbs(expected.x, 1e-12));
	REQUIRE_THAT(world.y, WithinAbs(expected.y, 1e-12));
	REQUIRE_THAT(world.z, WithinAbs(expected.z, 1e-12));
}

TEST_CASE("rotateLocalToWorld is the identity at zero rotation", "[raytracing][math]")
{
	const prop::AzEl rot{0.0, 0.0};
	const prop::Double3 local{std::cos(0.6) * std::cos(-0.4), std::sin(0.6) * std::cos(-0.4), std::sin(-0.4)};

	const auto world = prop::rotateLocalToWorld(rot, local);
	REQUIRE_THAT(world.x, WithinAbs(local.x, 1e-12));
	REQUIRE_THAT(world.y, WithinAbs(local.y, 1e-12));
	REQUIRE_THAT(world.z, WithinAbs(local.z, 1e-12));
}

TEST_CASE("rotateLocalToWorld/rotateWorldToLocal round-trip", "[raytracing][math]")
{
	const prop::AzEl rot{0.9, 0.4};
	constexpr double az_l = -0.6, el_l = 0.25;
	const prop::Double3 local{std::cos(az_l) * std::cos(el_l), std::sin(az_l) * std::cos(el_l), std::sin(el_l)};

	const auto world = prop::rotateLocalToWorld(rot, local);
	const auto back = prop::rotateWorldToLocal(rot, world);

	REQUIRE_THAT(back.x, WithinAbs(local.x, 1e-12));
	REQUIRE_THAT(back.y, WithinAbs(local.y, 1e-12));
	REQUIRE_THAT(back.z, WithinAbs(local.z, 1e-12));
}

TEST_CASE("sampleAntennaPattern reproduces exact grid values at grid points", "[raytracing][math]")
{
	// 3x3 grid: az in {-pi, 0, pi}, el in {-pi/2, 0, pi/2}, row-major by elevation.
	std::vector<float> gains{
		1.0f, 2.0f, 3.0f, // el = -pi/2
		4.0f, 5.0f, 6.0f, // el = 0
		7.0f, 8.0f, 9.0f, // el = pi/2
	};
	const rt::AntennaPatternView view{0, 3, 3};
	rt::SbrParams params{};
	params.antenna_gains = gains.data();

	REQUIRE_THAT(rt::sampleAntennaPattern(params, view, -prop::PI_V<float>, -prop::PI_V<float> / 2.0f),
				 WithinAbs(1.0, 1e-6));
	REQUIRE_THAT(rt::sampleAntennaPattern(params, view, 0.0f, -prop::PI_V<float> / 2.0f), WithinAbs(2.0, 1e-6));
	REQUIRE_THAT(rt::sampleAntennaPattern(params, view, 0.0f, 0.0f), WithinAbs(5.0, 1e-6));
	REQUIRE_THAT(rt::sampleAntennaPattern(params, view, prop::PI_V<float>, prop::PI_V<float> / 2.0f),
				 WithinAbs(9.0, 1e-6));
}

TEST_CASE("sampleAntennaPattern bilinearly interpolates between grid points", "[raytracing][math]")
{
	// 2x2 grid: (az,el) corners -> gain: (-pi,-pi/2)=0, (pi,-pi/2)=10, (-pi,pi/2)=20, (pi,pi/2)=30.
	std::vector<float> gains{0.0f, 10.0f, 20.0f, 30.0f};
	const rt::AntennaPatternView view{0, 2, 2};
	rt::SbrParams params{};
	params.antenna_gains = gains.data();

	REQUIRE_THAT(rt::sampleAntennaPattern(params, view, 0.0f, -prop::PI_V<float> / 2.0f),
				 WithinAbs(5.0, 1e-6)); // az midpoint, el0 row
	REQUIRE_THAT(rt::sampleAntennaPattern(params, view, -prop::PI_V<float>, 0.0f),
				 WithinAbs(10.0, 1e-6)); // az0, el midpoint
	REQUIRE_THAT(rt::sampleAntennaPattern(params, view, 0.0f, 0.0f), WithinAbs(15.0, 1e-6)); // center of all 4 corners
}

TEST_CASE("sampleAntennaPattern clamps queries outside the grid's range", "[raytracing][math]")
{
	std::vector<float> gains{0.0f, 10.0f, 20.0f, 30.0f};
	const rt::AntennaPatternView view{0, 2, 2};
	rt::SbrParams params{};
	params.antenna_gains = gains.data();

	REQUIRE_THAT(rt::sampleAntennaPattern(params, view, -100.0f, -100.0f), WithinAbs(0.0, 1e-6));
	REQUIRE_THAT(rt::sampleAntennaPattern(params, view, 100.0f, 100.0f), WithinAbs(30.0, 1e-6));
}

TEST_CASE("fibonacciDirFromX produces unit vectors spanning the sphere", "[raytracing][math]")
{
	constexpr uint32_t num_rays = 2000;
	prop::Double3 sum{0.0, 0.0, 0.0};
	double min_x = 0, max_x = -1;
	double min_y = 0, max_y = -1;
	double min_z = 0, max_z = -1;

	for (uint32_t i = 0; i < num_rays; ++i)
	{
		const float one_plus_x = 2.0f * ((float(i) + 0.5f) / float(num_rays));
		const auto dir = prop::Double3{rt::fibonacciDirFromX(2 - one_plus_x, one_plus_x, i)};
		const double len = std::sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
		// dir is float-precision, so 1e-9 is unachievable; float32 gives ~1e-7 relative.
		REQUIRE_THAT(len, WithinAbs(1.0, 1e-6));

		sum.x += dir.x;
		sum.y += dir.y;
		sum.z += dir.z;
		min_x = std::min(min_x, double(dir.x));
		max_x = std::max(max_x, double(dir.x));
		min_y = std::min(min_y, double(dir.y));
		max_y = std::max(max_y, double(dir.y));
		min_z = std::min(min_z, double(dir.z));
		max_z = std::max(max_z, double(dir.z));
	}

	// Points should be roughly symmetric about the origin for a large, evenly-spread sample.
	REQUIRE_THAT(sum.x / num_rays, WithinAbs(0.0, 1e-2));
	REQUIRE_THAT(sum.y / num_rays, WithinAbs(0.0, 1e-2));
	REQUIRE_THAT(sum.z / num_rays, WithinAbs(0.0, 1e-2));

	// And should span across the sphere in all dimensions.
	REQUIRE(min_x < -0.99);
	REQUIRE(max_x > 0.99);
	REQUIRE(min_y < -0.99);
	REQUIRE(max_y > 0.99);
	REQUIRE(min_z < -0.99);
	REQUIRE(max_z > 0.99);
}

TEST_CASE("fibonacciDirFromX spaced evenly at dense boresight x", "[raytracing][math]")
{
	// This test guards against floating-point handling regressions.

	// These are the (1-x) values we'll be using.
	// These are tiny. x = 1 - (1-x) would cancel to 1 here as the ULP of floats around 1 is much larger than 10e-9.
	const auto x1 = 1e-9f;
	const auto x2 = 2e-9f;
	const auto x3 = 3e-9f;

	// These are huge. The ULP of floats here is larger than 2pi.
	const auto i1 = 100000000;
	const auto i2 = 100000001;
	const auto i3 = 100000002;

	const auto d1 = prop::Double3{rt::fibonacciDirFromX(x1, 2 - x1, i1)};
	const auto d2 = prop::Double3{rt::fibonacciDirFromX(x2, 2 - x2, i2)};
	const auto d3 = prop::Double3{rt::fibonacciDirFromX(x3, 2 - x3, i3)};

	// check that the theta directions are still spiralling accurately at the golden angle
	const auto theta1 = std::atan2(d1.y, d1.z);
	const auto theta2 = std::atan2(d2.y, d2.z);
	const auto theta3 = std::atan2(d3.y, d3.z);
	const auto golden_angle = PI * (3 - std::sqrt(5));
	const auto theta12 = theta2 - theta1 + (theta2 > theta1 ? 0.0 : 2.0 * PI);
	const auto theta23 = theta3 - theta2 + (theta3 > theta2 ? 0.0 : 2.0 * PI);
	REQUIRE_THAT(theta12, WithinAbs(golden_angle, 1e-6));
	REQUIRE_THAT(theta23, WithinAbs(golden_angle, 1e-6));

	// check that the radius (distance from x axis) was computed accurately
	const auto r1 = std::sqrt(d1.y * d1.y + d1.z * d1.z);
	const auto r2 = std::sqrt(d2.y * d2.y + d2.z * d2.z);
	const auto r3 = std::sqrt(d3.y * d3.y + d3.z * d3.z);
	REQUIRE_THAT(r1, WithinAbs(std::sqrt(double(x1) * (2 - double(x1))), 1e-9));
	REQUIRE_THAT(r2, WithinAbs(std::sqrt(double(x2) * (2 - double(x2))), 1e-9));
	REQUIRE_THAT(r3, WithinAbs(std::sqrt(double(x3) * (2 - double(x3))), 1e-9));
	REQUIRE(r1 != r2);
	REQUIRE(r2 != r3);
}

TEST_CASE("boresightWeightedSample assigns per-regime weights that sum to 4*pi", "[raytracing][math]")
{
	rt::SbrParams params{};
	params.directions_per_source = 1000;
	params.boresight_rays = 200;
	params.boresight_fraction = 0.1f;

	double total_weight = 0.0;
	for (uint32_t i = 0; i < params.directions_per_source; ++i)
	{
		float weight = 0.0f;
		const auto dir = rt::boresightWeightedSample(params, i, weight);
		const double len =
			std::sqrt(double(dir.x) * double(dir.x) + double(dir.y) * double(dir.y) + double(dir.z) * double(dir.z));
		REQUIRE_THAT(len, WithinAbs(1.0, 1e-6));
		REQUIRE(weight > 0.0f);
		total_weight += double(weight);
	}

	REQUIRE_THAT(total_weight, WithinAbs(4.0 * double(prop::PI_V<float>), 1e-3));
}

TEST_CASE("boresightWeightedSample gives the denser (boresight) regime a smaller per-ray weight",
		  "[raytracing][math]")
{
	rt::SbrParams params{};
	params.directions_per_source = 1000;
	params.boresight_rays = 200;
	params.boresight_fraction = 0.1f; // 10% of the sphere's solid angle holds 20% of the rays.

	float boresight_weight = 0.0f;
	float off_boresight_weight = 0.0f;
	(void)rt::boresightWeightedSample(params, 0, boresight_weight);
	(void)rt::boresightWeightedSample(params, params.boresight_rays, off_boresight_weight);

	const float expected_boresight_weight =
		4.0f * prop::PI_V<float> * params.boresight_fraction / float(params.boresight_rays);
	const float expected_off_boresight_weight = 4.0f * prop::PI_V<float> * (1.0f - params.boresight_fraction) /
		float(params.directions_per_source - params.boresight_rays);

	REQUIRE_THAT(boresight_weight, WithinAbs(double(expected_boresight_weight), 1e-9));
	REQUIRE_THAT(off_boresight_weight, WithinAbs(double(expected_off_boresight_weight), 1e-9));
	REQUIRE(boresight_weight < off_boresight_weight);
}

namespace
{
	// v0=(0,0,0), v1=(1,0,0), v2=(0,1,0): a right triangle in the z=0 plane, area 0.5, outward
	// normal +Z (CCW as seen from +Z, matching TriangleMeshView's winding convention).
	const std::array<const prop::Double3, 3> test_tri{prop::Double3{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
	constexpr float test_tri_area = 0.5f;
	const prop::Float3 test_norm{0.0f, 0.0f, 1.0f};
	const prop::Double3 test_hit_pos{1.0 / 3.0, 1.0 / 3.0, 0.0}; // centroid
	const prop::Float3 test_incoming_dir{0.0f, 0.0f, -1.0f}; // normal incidence, travelling in -Z
}

TEST_CASE("computeTubePatch shrinks to the tube footprint when it's smaller than the triangle",
		  "[raytracing][math]")
{
	constexpr float dist = 1.0f;
	constexpr float solid_angle = 0.01f; // a_tube = dist^2 * solid_angle / cos(0) = 0.01 < 0.5
	const auto patch =
		rt::po::computeTubePatch(test_tri, test_norm, test_incoming_dir, test_tri_area, dist, solid_angle);

	REQUIRE_THAT(patch.area, WithinAbs(0.01, 1e-6));
	// Scaled toward, and centroid-centered on, the triangle's own centroid
	// (which equals test_hit_pos for this right triangle).
	const double scale = std::sqrt(0.01 / 0.5);
	for (size_t i = 0; i < 3; ++i)
	{
		const auto expected = scale * (test_tri.at(i) - test_hit_pos);
		REQUIRE_THAT(patch.verts.at(i).x, WithinAbs(expected.x, 1e-6));
		REQUIRE_THAT(patch.verts.at(i).y, WithinAbs(expected.y, 1e-6));
		REQUIRE_THAT(patch.verts.at(i).z, WithinAbs(expected.z, 1e-6));
	}
}

TEST_CASE("computeTubePatch clamps to the full triangle when the tube footprint is larger",
		  "[raytracing][math]")
{
	constexpr float dist = 100.0f;
	constexpr float solid_angle = 1.0f; // a_tube = 10000, way over the 0.5 triangle area
	const auto patch =
		rt::po::computeTubePatch(test_tri, test_norm, test_incoming_dir, test_tri_area, dist, solid_angle);

	REQUIRE_THAT(patch.area, WithinAbs(double(test_tri_area), 1e-9));
	// scale=1 here (no shrinking), but verts are still centroid-centered (which equals test_hit_pos
	// for this right triangle - see its definition above).
	for (size_t i = 0; i < 3; ++i)
	{
		const auto expected = test_tri.at(i) - test_hit_pos;
		REQUIRE_THAT(patch.verts.at(i).x, WithinAbs(expected.x, 1e-6));
		REQUIRE_THAT(patch.verts.at(i).y, WithinAbs(expected.y, 1e-6));
		REQUIRE_THAT(patch.verts.at(i).z, WithinAbs(expected.z, 1e-6));
	}
}

TEST_CASE("computeTubePatch stays finite at grazing incidence", "[raytracing][math]")
{
	// Travelling almost parallel to the triangle plane - cos(theta_i) would be near 0 unclamped.
	const prop::Float3 grazing_dir = normalize(prop::Float3{1.0, 0.0, -1e-5f});
	const auto patch = rt::po::computeTubePatch(test_tri, test_norm, grazing_dir, test_tri_area, 1.0f, 0.001f);

	REQUIRE(std::isfinite(patch.area));
	REQUIRE(patch.area >= 0.0f);
	REQUIRE(patch.area <= test_tri_area + 1e-6f);
}

TEST_CASE("arbitraryPerpendicular returns a unit vector perpendicular to its input", "[raytracing][math]")
{
	const std::vector<prop::Double3> dirs{
		{1.0, 0.0, 0.0},  {0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0}, // near/at the "up" pole - exercises the fallback branch
		{0.0, 0.0, -1.0}, {0.5773502691896258, 0.5773502691896258, 0.5773502691896258}, // arbitrary diagonal
	};

	for (const auto& dir : dirs)
	{
		const auto perp = prop::arbitraryPerpendicular(dir);
		const double len = std::sqrt(perp.x * perp.x + perp.y * perp.y + perp.z * perp.z);
		const double dot = perp.x * dir.x + perp.y * dir.y + perp.z * dir.z;

		REQUIRE_THAT(len, WithinAbs(1.0, 1e-9));
		REQUIRE_THAT(dot, WithinAbs(0.0, 1e-9));
	}
}

// --- csqrt: principal branch and branch-cut behaviour --------------------------------------- //

TEST_CASE("csqrt returns the principal branch and matches std::sqrt(std::complex<double>)",
		  "[raytracing][math][fresnel]")
{
	const std::vector<std::pair<float, float>> points{
		{4.0f, 0.0f}, {0.0f, 8.0f}, {0.0f, -8.0f}, {-4.0f, 0.0f}, {3.0f, 4.0f},
		{-3.0f, 4.0f}, {-3.0f, -4.0f}, {3.0f, -4.0f}, {1e-3f, 1e-3f}, {100.0f, -0.01f},
	};

	for (const auto& [re, im] : points)
	{
		const prop::CFloat got = prop::csqrt(prop::CFloat{re, im});
		const std::complex<double> expected = std::sqrt(std::complex<double>(double(re), double(im)));

		// Principal branch: Re(sqrt(z)) >= 0 everywhere in the complex plane.
		REQUIRE(got.re >= 0.0f);
		REQUIRE_THAT(double(got.re), WithinAbs(expected.real(), 1e-4));
		REQUIRE_THAT(double(got.im), WithinAbs(expected.imag(), 1e-4));

		// Squaring the result must recover the input.
		const prop::CFloat squared = got * got;
		REQUIRE_THAT(double(squared.re), WithinAbs(double(re), 1e-2));
		REQUIRE_THAT(double(squared.im), WithinAbs(double(im), 1e-2));
	}
}

TEST_CASE("csqrt's imaginary part flips sign crossing the negative-real-axis branch cut",
		  "[raytracing][math][fresnel]")
{
	// Just above and just below the branch cut (the negative real axis), the principal
	// square root's imaginary part must have opposite sign, matching the side it was
	// approached from, while staying continuous in magnitude.
	const auto above = prop::csqrt(prop::CFloat{-4.0f, 1e-4f});
	const auto below = prop::csqrt(prop::CFloat{-4.0f, -1e-4f});
	const auto on_cut = prop::csqrt(prop::CFloat{-4.0f, 0.0f});

	REQUIRE(above.im > 0.0f);
	REQUIRE(below.im < 0.0f);
	REQUIRE(on_cut.im > 0.0f); // the +0 side, matching the `a.im >= 0` tie-break in csqrt.
	// Near the cut, sqrt(-4+j*eps) ~ (eps/4, 2): the real part stays tiny and the imaginary part
	// stays near the branch value of sqrt(4)=2, on whichever side the sign came from.
	REQUIRE_THAT(double(above.re), WithinAbs(0.0, 1e-3));
	REQUIRE_THAT(double(below.re), WithinAbs(0.0, 1e-3));
	REQUIRE_THAT(double(above.im), WithinAbs(2.0, 1e-2));
	REQUIRE_THAT(double(below.im), WithinAbs(-2.0, 1e-2));
}

// --- indexOfRefraction -------------------------------------------------------------------------- //

TEST_CASE("indexOfRefraction is purely real for a lossless dielectric", "[raytracing][math][fresnel]")
{
	const auto n = rt::go::indexOfRefraction(4.0f, 0.0f, 1e9f);
	REQUIRE_THAT(double(n.re), WithinAbs(2.0, 1e-6));
	REQUIRE_THAT(double(n.im), WithinAbs(0.0, 1e-6));
}

TEST_CASE("indexOfRefraction matches sqrt(eps_r - j*sigma/(eps0*omega)), with a negative imaginary part "
		  "for a passive, lossy medium",
		  "[raytracing][math][fresnel]")
{
	constexpr float eps_r = 4.0f, sigma = 0.02f, freq = 1e9f;
	const auto n = rt::go::indexOfRefraction(eps_r, sigma, freq);

	const double omega = 2.0 * prop::PI_V<double> * double(freq);
	const double loss = double(sigma) / (double(prop::E0) * omega);
	const std::complex<double> expected = std::sqrt(std::complex<double>(double(eps_r), -loss));

	REQUIRE_THAT(double(n.re), WithinAbs(expected.real(), 1e-4));
	REQUIRE_THAT(double(n.im), WithinAbs(expected.imag(), 1e-4));

	// With this codebase's e^{+jx} = cexp_i(x) convention and far-field phase e^{-jkR}, a
	// forward-propagating wave through a passive (lossy) medium must decay, not grow: that
	// requires Im(n) < 0, since k = n*k0 and R grows along the propagation direction.
	REQUIRE(n.im < 0.0f);
	REQUIRE(n.re > 0.0f);
}

TEST_CASE("indexOfRefraction approaches the 45-degree good-conductor asymptote at high conductivity, "
		  "and matches it exactly at conductivity == INFINITY",
		  "[raytracing][math][fresnel]")
{
	constexpr float eps_r = 4.0f, freq = 1e9f;

	// Good-conductor approximation: n ~ (1-j) * sqrt(sigma / (eps0 * omega)), i.e. Re(n) == -Im(n).
	const auto n = rt::go::indexOfRefraction(eps_r, 1e7f, freq); // copper-like conductivity
	REQUIRE_THAT(double(n.re), WithinAbs(double(-n.im), double(n.re) * 1e-3));
	REQUIRE(n.re > 100.0f); // sanity: dominated by conductivity, not relative_permittivity.

	const auto pec = rt::go::indexOfRefraction(eps_r, INFINITY, freq);
	REQUIRE(std::isinf(pec.re));
	REQUIRE(std::isinf(pec.im));
	REQUIRE(pec.re > 0.0f);
	REQUIRE(pec.im < 0.0f);
}

// --- complexSnellsLaw: Fresnel reflection coefficients ------------------------------------------ //

TEST_CASE("complexSnellsLaw gives TE and TM reflection coefficients that are negatives of each other at "
		  "normal incidence",
		  "[raytracing][math][fresnel]")
{
	// At normal incidence the plane of incidence (and hence the TE/TM split) is degenerate, and
	// `basis.e_te` falls back to an arbitrary tangent. The reflected result stays independent of
	// that arbitrary choice only because e_tm_out == -e_tm_in exactly at normal incidence - which
	// requires gamma_te == -gamma_tm here (not gamma_te == gamma_tm).
	const prop::Float3 norm{0.0f, 0.0f, 1.0f};
	const prop::Float3 k_in{0.0f, 0.0f, -1.0f};
	const prop::CFloat n2{3.0f, -0.5f};

	prop::CFloat gamma_tm, gamma_te;
	rt::go::complexSnellsLaw(norm, k_in, n2, gamma_tm, gamma_te);

	const prop::CFloat expected_te = (prop::CFloat{1.0f, 0.0f} - n2) / (prop::CFloat{1.0f, 0.0f} + n2);
	REQUIRE_THAT(double(gamma_te.re), WithinAbs(double(expected_te.re), 1e-5));
	REQUIRE_THAT(double(gamma_te.im), WithinAbs(double(expected_te.im), 1e-5));
	REQUIRE_THAT(double(gamma_tm.re), WithinAbs(-double(expected_te.re), 1e-5));
	REQUIRE_THAT(double(gamma_tm.im), WithinAbs(-double(expected_te.im), 1e-5));
}

TEST_CASE("complexSnellsLaw matches the Fresnel equations at oblique incidence for a lossless dielectric",
		  "[raytracing][math][fresnel]")
{
	// This is a regression test for the refraction angle used internally: Snell's law requires
	// sin(theta_t) = (n1/n2) sin(theta_i), i.e. the *squared* ratio inside cos(theta_t), not the
	// bare ratio - using the bare ratio silently refracts every oblique ray to the wrong angle.
	const double theta_i = 60.0 * prop::PI_V<double> / 180.0;
	const prop::Float3 norm{0.0f, 0.0f, 1.0f};
	const prop::Float3 k_in{float(std::sin(theta_i)), 0.0f, float(-std::cos(theta_i))};
	const prop::CFloat n2{2.0f, 0.0f}; // real, lossless, index 2.

	prop::CFloat gamma_tm, gamma_te;
	rt::go::complexSnellsLaw(norm, k_in, n2, gamma_tm, gamma_te);

	const double cos_i = std::cos(theta_i);
	const double sin_t = std::sin(theta_i) / 2.0; // n1 = 1, n2 = 2.
	const double cos_t = std::sqrt(1.0 - sin_t * sin_t);

	const double r_s = (cos_i - 2.0 * cos_t) / (cos_i + 2.0 * cos_t);
	const double r_p = (2.0 * cos_i - cos_t) / (2.0 * cos_i + cos_t);

	REQUIRE_THAT(double(gamma_te.re), WithinAbs(r_s, 1e-4));
	REQUIRE_THAT(double(gamma_te.im), WithinAbs(0.0, 1e-5));
	REQUIRE_THAT(double(gamma_tm.re), WithinAbs(r_p, 1e-4));
	REQUIRE_THAT(double(gamma_tm.im), WithinAbs(0.0, 1e-5));
}

TEST_CASE("complexSnellsLaw: TM (parallel) reflection vanishes at Brewster's angle, TE does not",
		  "[raytracing][math][fresnel]")
{
	constexpr float n2_val = 1.5f;
	const double theta_b = std::atan(double(n2_val)); // Brewster's angle: theta_i + theta_t = 90 degrees.
	const prop::Float3 norm{0.0f, 0.0f, 1.0f};
	const prop::Float3 k_in{float(std::sin(theta_b)), 0.0f, float(-std::cos(theta_b))};
	const prop::CFloat n2{n2_val, 0.0f};

	prop::CFloat gamma_tm, gamma_te;
	rt::go::complexSnellsLaw(norm, k_in, n2, gamma_tm, gamma_te);

	REQUIRE_THAT(double(gamma_tm.re), WithinAbs(0.0, 1e-4));
	REQUIRE_THAT(double(gamma_tm.im), WithinAbs(0.0, 1e-5));
	REQUIRE(std::abs(gamma_te.re) > 0.1f); // TE reflection is clearly nonzero at the same angle.
}

TEST_CASE("complexSnellsLaw reflection coefficients stay within unit magnitude for a lossless dielectric",
		  "[raytracing][math][fresnel]")
{
	// A passive, lossless interface cannot reflect more power than it receives.
	const prop::Float3 norm{0.0f, 0.0f, 1.0f};
	const prop::CFloat n2{1.8f, 0.0f};

	for (int deg = 0; deg < 90; deg += 3)
	{
		const double theta_i = double(deg) * prop::PI_V<double> / 180.0;
		const prop::Float3 k_in{float(std::sin(theta_i)), 0.0f, float(-std::cos(theta_i))};

		prop::CFloat gamma_tm, gamma_te;
		rt::go::complexSnellsLaw(norm, k_in, n2, gamma_tm, gamma_te);

		REQUIRE(prop::cabs(gamma_te) <= 1.0f + 1e-4f);
		REQUIRE(prop::cabs(gamma_tm) <= 1.0f + 1e-4f);
	}
}

TEST_CASE("complexSnellsLaw approaches the hardcoded PEC reflection coefficients at very high conductivity",
		  "[raytracing][math][fresnel]")
{
	// fresnelGoReflection special-cases true PECs (conductivity == INFINITY) with hardcoded
	// gamma_tm = 1, gamma_te = -1. The general formula, fed a very conductive (but finite) index
	// of refraction, must agree with that in the limit.
	const double theta_i = 40.0 * prop::PI_V<double> / 180.0;
	const prop::Float3 norm{0.0f, 0.0f, 1.0f};
	const prop::Float3 k_in{float(std::sin(theta_i)), 0.0f, float(-std::cos(theta_i))};

	const auto n2 = rt::go::indexOfRefraction(1.0f, 1e9f, 1e9f); // very conductive, not literally infinite.

	prop::CFloat gamma_tm, gamma_te;
	rt::go::complexSnellsLaw(norm, k_in, n2, gamma_tm, gamma_te);

	REQUIRE_THAT(double(gamma_te.re), WithinAbs(-1.0, 1e-3));
	REQUIRE_THAT(double(gamma_te.im), WithinAbs(0.0, 1e-3));
	REQUIRE_THAT(double(gamma_tm.re), WithinAbs(1.0, 1e-3));
	REQUIRE_THAT(double(gamma_tm.im), WithinAbs(0.0, 1e-3));
}

// --- fresnelGoReflection: full basis + Fresnel pipeline ----------------------------------------- //

namespace
{
	const rt::Material pec_material{1.0f, INFINITY};
}

TEST_CASE("fresnelGoReflection cancels the tangential field at a PEC surface at normal incidence",
		  "[raytracing][math][fresnel]")
{
	const prop::Float3 norm{0.0f, 0.0f, 1.0f};
	const prop::Float3 u_inc{0.0f, 0.0f, -1.0f};
	const prop::Float3 u_refl = rt::go::reflect(u_inc, norm);
	const prop::CFloat3 e_inc{{0.6f, 0.1f}, {-0.3f, 0.2f}, {0.0f, 0.0f}}; // arbitrary, tangential to norm.

	const auto e_refl = rt::go::fresnelGoReflection(e_inc, norm, u_inc, u_refl, &pec_material, 1e9f);

	const auto total = e_inc + e_refl;
	REQUIRE_THAT(double(total.x.re), WithinAbs(0.0, 1e-5));
	REQUIRE_THAT(double(total.x.im), WithinAbs(0.0, 1e-5));
	REQUIRE_THAT(double(total.y.re), WithinAbs(0.0, 1e-5));
	REQUIRE_THAT(double(total.y.im), WithinAbs(0.0, 1e-5));
}

TEST_CASE("fresnelGoReflection cancels the tangential field at a PEC surface at oblique incidence",
		  "[raytracing][math][fresnel]")
{
	const double theta_i = 35.0 * prop::PI_V<double> / 180.0;
	const prop::Float3 norm{0.0f, 0.0f, 1.0f};
	const prop::Float3 u_inc{float(std::sin(theta_i)), 0.0f, float(-std::cos(theta_i))};
	const prop::Float3 u_refl = rt::go::reflect(u_inc, norm);

	// An arbitrary complex incident field, built from the incidence-plane basis so it's
	// transverse to u_inc (a physical plane-wave field).
	const auto basis = prop::incidencePlaneBasis(norm, u_inc);
	const prop::CFloat3 e_inc = basis.e_te * prop::CFloat{0.7f, -0.3f} + basis.e_tm_in * prop::CFloat{-0.4f, 0.5f};

	const auto e_refl = rt::go::fresnelGoReflection(e_inc, norm, u_inc, u_refl, &pec_material, 1e9f);

	// Tangential (x, y) components of the total field must vanish at a PEC surface; the normal
	// (z) component is unconstrained.
	const auto total = e_inc + e_refl;
	REQUIRE_THAT(double(total.x.re), WithinAbs(0.0, 1e-4));
	REQUIRE_THAT(double(total.x.im), WithinAbs(0.0, 1e-4));
	REQUIRE_THAT(double(total.y.re), WithinAbs(0.0, 1e-4));
	REQUIRE_THAT(double(total.y.im), WithinAbs(0.0, 1e-4));
}

TEST_CASE("fresnelGoReflection combines the TE/TM bases and gammas exactly as complexSnellsLaw predicts, "
		  "for a lossy dielectric at oblique incidence",
		  "[raytracing][math][fresnel]")
{
	const double theta_i = 50.0 * prop::PI_V<double> / 180.0;
	const prop::Float3 norm{0.0f, 0.0f, 1.0f};
	const prop::Float3 u_inc{float(std::sin(theta_i)), 0.0f, float(-std::cos(theta_i))};
	const prop::Float3 u_refl = rt::go::reflect(u_inc, norm);
	const rt::Material mat{5.0f, 0.05f};
	constexpr float frequency = 3e9f;

	const auto basis = prop::incidencePlaneBasis(norm, u_inc);
	const auto e_tm_out = cross(basis.e_te, u_refl);
	const prop::CFloat3 e_inc = basis.e_te * prop::CFloat{0.2f, 0.4f} + basis.e_tm_in * prop::CFloat{-0.6f, 0.1f};

	const auto n2 = rt::go::indexOfRefraction(mat.relative_permittivity, mat.conductivity, frequency);
	prop::CFloat gamma_tm, gamma_te;
	rt::go::complexSnellsLaw(norm, u_inc, n2, gamma_tm, gamma_te);

	const prop::Complex elec_te = dot(e_inc, basis.e_te);
	const prop::Complex elec_tm = dot(e_inc, basis.e_tm_in);
	const prop::CFloat3 expected = basis.e_te * (gamma_te * elec_te) + e_tm_out * (gamma_tm * elec_tm);

	const auto got = rt::go::fresnelGoReflection(e_inc, norm, u_inc, u_refl, &mat, frequency);

	REQUIRE_THAT(double(got.x.re), WithinAbs(double(expected.x.re), 1e-5));
	REQUIRE_THAT(double(got.x.im), WithinAbs(double(expected.x.im), 1e-5));
	REQUIRE_THAT(double(got.y.re), WithinAbs(double(expected.y.re), 1e-5));
	REQUIRE_THAT(double(got.y.im), WithinAbs(double(expected.y.im), 1e-5));
	REQUIRE_THAT(double(got.z.re), WithinAbs(double(expected.z.re), 1e-5));
	REQUIRE_THAT(double(got.z.im), WithinAbs(double(expected.z.im), 1e-5));
}
