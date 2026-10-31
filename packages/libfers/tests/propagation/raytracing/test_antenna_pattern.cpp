#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <highfive/highfive.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "antenna/antenna_factory.h"
#include "math/geometry_ops.h"
#include "propagation/raytracing/antenna_model.h"
#include "propagation/raytracing/sbr_impl.h"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
namespace rt = propagation::raytracing;
namespace prop = propagation;

namespace
{
	std::filesystem::path tempFilePath(const std::string& prefix, const std::string& extension)
	{
		const auto name =
			prefix + "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + extension;
		return std::filesystem::temp_directory_path() / name;
	}

	// A local direction, expressed as a unit Cartesian vector, matching SVec3(1, azimuth,
	// elevation)'s convention exactly (boresight = local +X).
	prop::Float3 localDir(const RealType azimuth, const RealType elevation)
	{
		return prop::Float3{float(std::cos(azimuth) * std::cos(elevation)), float(std::sin(azimuth) * std::cos(elevation)),
						  float(std::sin(elevation))};
	}

	// Evaluates the CPU antenna's own getGain() in its local frame (zero-rotation boresight,
	// matching how buildAntennaModel/sampleAntennaModel are meant to agree with it).
	RealType cpuGain(const antenna::Antenna& antenna, const RealType azimuth, const RealType elevation,
					 const RealType wavelength)
	{
		return antenna.getGain(math::SVec3(1.0, azimuth, elevation), math::SVec3(1.0, 0.0, 0.0), wavelength);
	}

	// Evaluates the device-side model exactly as the ray tracer would: through SbrParams, which
	// holds a Grid2D model's baked gains.
	float gpuGain(const rt::AntennaModel& model, const std::vector<float>& gains_buffer, const RealType azimuth,
				 const RealType elevation, const RealType wavelength)
	{
		rt::SbrParams params{};
		params.antenna_gains = gains_buffer.data();
		return sampleAntennaModel(params, model, localDir(azimuth, elevation), float(azimuth), float(elevation),
								  float(wavelength));
	}

	// The (azimuth, elevation) of grid point (j, i) of an az_count x el_count grid, matching
	// buildAntennaModel's own baking convention exactly (see antenna_model.cpp's bakeGrid).
	std::pair<RealType, RealType> gridPoint(uint32_t j, uint32_t i, uint32_t az_count, uint32_t el_count)
	{
		const RealType azimuth =
			(RealType(j) / RealType(az_count - 1)) * 2.0 * prop::PI_V<RealType> - prop::PI_V<RealType>;
		const RealType elevation =
			(RealType(i) / RealType(el_count - 1)) * prop::PI_V<RealType> - prop::PI_V<RealType> / 2.0;
		return {azimuth, elevation};
	}
}

TEST_CASE("buildAntennaModel captures Isotropic as a trivial model", "[raytracing][antenna_pattern]")
{
	antenna::Isotropic antenna("iso", 1);
	antenna.setEfficiencyFactor(0.6);

	std::vector<float> gains_buffer;
	const auto model = rt::buildAntennaModel(antenna, gains_buffer);
	REQUIRE(model.kind == rt::AntennaKind::Isotropic);
	REQUIRE_THAT(model.efficiency, WithinAbs(0.6, 1e-6));
	REQUIRE_THAT(gpuGain(model, gains_buffer, 0.3, -0.2, 0.03), WithinRel(0.6, 1e-6));
}

TEST_CASE("buildAntennaModel's Sinc matches antenna::Sinc::getGain exactly", "[raytracing][antenna_pattern]")
{
	antenna::Sinc antenna("sinc", 2.0, 1.5, 2.0, 1);
	antenna.setEfficiencyFactor(0.7);

	std::vector<float> gains_buffer;
	const auto model = rt::buildAntennaModel(antenna, gains_buffer);
	REQUIRE(model.kind == rt::AntennaKind::Sinc);

	for (const auto& [az, el] : {std::pair{0.0, 0.0}, std::pair{0.3, -0.1}, std::pair{-0.5, 0.2}})
	{
		const auto expected = cpuGain(antenna, az, el, 0.03);
		const auto actual = gpuGain(model, gains_buffer, az, el, 0.03);
		REQUIRE_THAT(actual, WithinRel(expected, 1e-6));
	}
}

TEST_CASE("buildAntennaModel's Gaussian matches antenna::Gaussian::getGain exactly", "[raytracing][antenna_pattern]")
{
	antenna::Gaussian antenna("gauss", 0.8, 1.3, 1);
	antenna.setEfficiencyFactor(0.4);

	std::vector<float> gains_buffer;
	const auto model = rt::buildAntennaModel(antenna, gains_buffer);
	REQUIRE(model.kind == rt::AntennaKind::Gaussian);

	for (const auto& [az, el] : {std::pair{0.0, 0.0}, std::pair{0.25, -0.15}, std::pair{-0.3, 0.1}})
	{
		const auto expected = cpuGain(antenna, az, el, 0.03);
		const auto actual = gpuGain(model, gains_buffer, az, el, 0.03);
		REQUIRE_THAT(actual, WithinRel(expected, 1e-6));
	}
}

TEST_CASE("buildAntennaModel's SquareHorn matches antenna::SquareHorn::getGain exactly",
		  "[raytracing][antenna_pattern]")
{
	antenna::SquareHorn antenna("horn", 0.4, 1);
	antenna.setEfficiencyFactor(0.9);

	std::vector<float> gains_buffer;
	const auto model = rt::buildAntennaModel(antenna, gains_buffer);
	REQUIRE(model.kind == rt::AntennaKind::SquareHorn);

	constexpr RealType wavelength = 0.1;
	for (const auto& [az, el] : {std::pair{0.0, 0.0}, std::pair{0.1, 0.0}, std::pair{0.0, -0.05}})
	{
		const auto expected = cpuGain(antenna, az, el, wavelength);
		const auto actual = gpuGain(model, gains_buffer, az, el, wavelength);
		// Not an exact match: the GPU path's boresight angle passes through a float32 acosf() before
		// squareHornGain's otherwise-double sinc, so it picks up float rounding - loosened accordingly.
		REQUIRE_THAT(actual, WithinRel(expected, 1e-5));
	}
}

TEST_CASE("buildAntennaModel's Parabolic matches antenna::Parabolic::getGain to the approximation's accuracy",
		  "[raytracing][antenna_pattern]")
{
	antenna::Parabolic antenna("dish", 1.2, 1);
	antenna.setEfficiencyFactor(0.8);

	std::vector<float> gains_buffer;
	const auto model = rt::buildAntennaModel(antenna, gains_buffer);
	REQUIRE(model.kind == rt::AntennaKind::Parabolic);

	constexpr RealType wavelength = 0.3;
	for (const auto& [az, el] : {std::pair{0.0, 0.0}, std::pair{0.1, 0.0}, std::pair{0.0, 0.05}})
	{
		const auto expected = cpuGain(antenna, az, el, wavelength);
		const auto actual = gpuGain(model, gains_buffer, az, el, wavelength);
		// Not an exact match: the GPU path uses the portable besselJ1Approx, the CPU one uses exact
		// libm j1() (see antenna_gain.h) - loosened accordingly.
		REQUIRE_THAT(actual, WithinRel(expected, 1e-5));
	}
}

TEST_CASE("buildAntennaModel bakes an XmlAntenna into a Grid2D model matching its samples exactly",
		  "[raytracing][antenna_pattern]")
{
	const auto path = tempFilePath("xml_antenna_model", ".xml");
	{
		std::ofstream out(path, std::ios::binary);
		REQUIRE(out.is_open());
		out << R"(<?xml version="1.0" encoding="UTF-8"?><antenna>
			<azimuth>
				<gainsample><angle>0.0</angle><gain>8.0</gain></gainsample>
				<gainsample><angle>1.0</angle><gain>4.0</gain></gainsample>
				<gainsample><angle>2.0</angle><gain>2.0</gain></gainsample>
			</azimuth>
			<elevation>
				<gainsample><angle>0.0</angle><gain>8.0</gain></gainsample>
				<gainsample><angle>1.0</angle><gain>2.0</gain></gainsample>
				<gainsample><angle>2.0</angle><gain>1.0</gain></gainsample>
			</elevation>
		</antenna>)";
	}

	antenna::XmlAntenna antenna("xml", path.string(), 1);
	antenna.setEfficiencyFactor(0.5);

	constexpr uint32_t az_count = 9, el_count = 5;
	std::vector<float> gains_buffer;
	const auto model = rt::buildAntennaModel(antenna, gains_buffer, az_count, el_count);
	std::filesystem::remove(path);

	REQUIRE(model.kind == rt::AntennaKind::Grid2D);
	REQUIRE(model.params.grid_2d.grid.az_count == az_count);
	REQUIRE(model.params.grid_2d.grid.el_count == el_count);
	REQUIRE(gains_buffer.size() == static_cast<size_t>(az_count * el_count));

	// Sampling exactly at a grid vertex incurs no bilinear interpolation, so it should reproduce
	// the baked-in getGain() sample (mod float rounding) rather than merely approximating it.
	for (uint32_t i = 0; i < el_count; ++i)
	{
		for (uint32_t j = 0; j < az_count; ++j)
		{
			const auto [az, el] = gridPoint(j, i, az_count, el_count);
			const auto expected = cpuGain(antenna, az, el, 0.03);
			const auto actual = gpuGain(model, gains_buffer, az, el, 0.03);
			REQUIRE_THAT(actual, WithinRel(expected, 1e-5));
		}
	}
}

TEST_CASE("buildAntennaModel's Grid2D wraps an H5Antenna pattern", "[raytracing][antenna_pattern]")
{
	const auto path = tempFilePath("h5_antenna_model", ".h5");
	{
		const std::vector<std::vector<RealType>> pattern{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
		HighFive::File file(path.string(), HighFive::File::Overwrite);
		file.createDataSet<RealType>("antenna", HighFive::DataSpace::From(pattern)).write(pattern);
	}

	antenna::H5Antenna antenna("h5", path.string(), 1);
	antenna.setEfficiencyFactor(0.3);

	std::vector<float> gains_buffer;
	const auto model = rt::buildAntennaModel(antenna, gains_buffer, 10, 6);
	std::filesystem::remove(path);

	REQUIRE(model.kind == rt::AntennaKind::Grid2D);
	REQUIRE(model.params.grid_2d.grid.az_count == 10);
	REQUIRE(model.params.grid_2d.grid.el_count == 6);
	REQUIRE(gains_buffer.size() == 60);
}

TEST_CASE("buildAntennaModel rejects a degenerate grid resolution for H5Antenna", "[raytracing][antenna_pattern]")
{
	const auto path = tempFilePath("h5_antenna_model_bad", ".h5");
	{
		const std::vector<std::vector<RealType>> pattern{{1.0, 2.0}, {3.0, 4.0}};
		HighFive::File file(path.string(), HighFive::File::Overwrite);
		file.createDataSet<RealType>("antenna", HighFive::DataSpace::From(pattern)).write(pattern);
	}

	const antenna::H5Antenna antenna("h5", path.string(), 1);
	std::vector<float> gains_buffer;
	REQUIRE_THROWS_AS(rt::buildAntennaModel(antenna, gains_buffer, 1, 10), std::invalid_argument);
	REQUIRE_THROWS_AS(rt::buildAntennaModel(antenna, gains_buffer, 10, 1), std::invalid_argument);

	std::filesystem::remove(path);
}
