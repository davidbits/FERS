#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>

#include "core/sim_id.h"
#include "math/coord.h"
#include "radar/platform.h"
#include "radar/target.h"

using Catch::Matchers::WithinAbs;

namespace
{
	math::SVec3 unitDirection(const RealType azimuth, const RealType elevation) { return {1.0, azimuth, elevation}; }

	class FixedRcsModel final : public radar::RcsModel
	{
	public:
		explicit FixedRcsModel(RealType value) : _value(value) {}

		RealType sampleModel() override { return _value; }

	private:
		RealType _value;
	};

	std::string uniqueFileName(const std::string& prefix)
	{
		return prefix + "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".xml";
	}

	std::filesystem::path tempFilePath(const std::string& filename)
	{
		return std::filesystem::temp_directory_path() / filename;
	}

	void removeIfExists(const std::filesystem::path& path)
	{
		std::error_code ec;
		std::filesystem::remove(path, ec);
	}

	void writeXmlTargetFile(const std::filesystem::path& path, const std::string& body)
	{
		std::ofstream out(path, std::ios::binary);
		REQUIRE(out.is_open());
		out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << body;
	}

	std::string xmlTargetFixture()
	{
		return R"(<target>
		<azimuth>
			<rcssample><angle>0.0</angle><rcs>2.0</rcs></rcssample>
			<rcssample><angle>1.0</angle><rcs>6.0</rcs></rcssample>
			<rcssample><angle>2.0</angle><rcs>10.0</rcs></rcssample>
		</azimuth>
		<elevation>
			<rcssample><angle>0.0</angle><rcs>3.0</rcs></rcssample>
			<rcssample><angle>1.0</angle><rcs>5.0</rcs></rcssample>
			<rcssample><angle>2.0</angle><rcs>7.0</rcs></rcssample>
		</elevation>
	</target>)";
	}

	void setStaticRotation(radar::Platform& platform, const RealType azimuth = 0.0, const RealType elevation = 0.0)
	{
		platform.getRotationPath()->addCoord(math::RotationCoord(azimuth, elevation, 0.0));
		platform.getRotationPath()->finalize();
	}

	void setConstantRotation(radar::Platform& platform, const RealType start_azimuth, const RealType start_elevation,
							 const RealType rate_azimuth, const RealType rate_elevation)
	{
		platform.getRotationPath()->setConstantRate(math::RotationCoord(start_azimuth, start_elevation, 0.0),
													math::RotationCoord(rate_azimuth, rate_elevation, 0.0));
	}
}

TEST_CASE("RcsConst returns unity", "[radar][target]")
{
	radar::RcsConst model;
	REQUIRE_THAT(model.sampleModel(), WithinAbs(1.0, 1e-12));
}

TEST_CASE("RcsChiSquare reports configured degrees of freedom", "[radar][target]")
{
	std::mt19937 rng(12345);
	radar::RcsChiSquare model(rng, 2.0);
	REQUIRE_THAT(model.getK(), WithinAbs(2.0, 1e-12));

	const RealType sample = model.sampleModel();
	REQUIRE(sample > 0.0);
}

TEST_CASE("RcsChiSquare sample mean matches configured mean power", "[radar][target]")
{
	std::mt19937 rng(24680);
	radar::RcsChiSquare model(rng, 4.0);

	constexpr int sample_count = 50000;
	RealType sum = 0.0;
	RealType minimum = model.sampleModel();
	sum += minimum;

	for (int i = 1; i < sample_count; ++i)
	{
		const RealType sample = model.sampleModel();
		minimum = std::min(minimum, sample);
		sum += sample;
	}

	REQUIRE(minimum >= 0.0);
	REQUIRE_THAT(sum / static_cast<RealType>(sample_count), WithinAbs(4.0, 0.15));
}

TEST_CASE("Target with constant (isotropic) RCS returns it", "[radar][target]")
{
	radar::Platform platform("TargetPlatform");
	radar::Target target(&platform, "Iso", 9002);
	target.setRcs(std::make_unique<radar::IsoTargetRcs>(12.5, 42));

	math::SVec3 in_angle(1.0, 0.0, 0.0);
	math::SVec3 out_angle(1.0, 0.0, 0.0);

	const auto* iso = dynamic_cast<const radar::IsoTargetRcs*>(target.getRcsSpec());
	REQUIRE(iso != nullptr);
	REQUIRE_THAT(iso->getConstRcs(), WithinAbs(12.5, 1e-12));

	const auto rcs = target.getRcs(in_angle, out_angle, 0.0);
	REQUIRE(rcs.has_value());
	REQUIRE_THAT(rcs.value(), WithinAbs(12.5, 1e-12));
	REQUIRE(target.getId() == 9002);
}

TEST_CASE("createIsoTarget attaches a constant RCS", "[radar][target]")
{
	radar::Platform platform("TargetPlatform");
	const auto target = radar::createIsoTarget(&platform, "IsoFactory", 9.0, 77, 8080);

	const auto* iso_rcs = dynamic_cast<const radar::IsoTargetRcs*>(target->getRcsSpec());
	REQUIRE(iso_rcs != nullptr);
	REQUIRE_THAT(iso_rcs->getConstRcs(), WithinAbs(9.0, 1e-12));
	REQUIRE(target->getId() == 8080);
}

TEST_CASE("Constant RCS applies fluctuation model", "[radar][target]")
{
	radar::Platform platform("TargetPlatform");
	radar::Target target(&platform, "Iso");
	target.setRcs(std::make_unique<radar::IsoTargetRcs>(3.0, 7));

	auto model = std::make_unique<FixedRcsModel>(2.0);
	const auto* model_ptr = model.get();
	target.getRcsSpec()->setFluctuationModel(std::move(model));

	math::SVec3 in_angle(1.0, 0.0, 0.0);
	math::SVec3 out_angle(1.0, 0.0, 0.0);

	REQUIRE(target.getRcsSpec()->getFluctuationModel() == model_ptr);
	const auto rcs = target.getRcs(in_angle, out_angle, 0.0);
	REQUIRE(rcs.has_value());
	REQUIRE_THAT(rcs.value(), WithinAbs(6.0, 1e-12));
}

TEST_CASE("Target's RCS RNG is deterministic per seed", "[radar][target]")
{
	radar::Platform platform("TargetPlatform");
	radar::Target target_a(&platform, "IsoA");
	target_a.setRcs(std::make_unique<radar::IsoTargetRcs>(1.0, 1337));
	radar::Target target_b(&platform, "IsoB");
	target_b.setRcs(std::make_unique<radar::IsoTargetRcs>(1.0, 1337));

	const auto first_a = target_a.getRcsSpec()->getRngEngine()();
	const auto first_b = target_b.getRcsSpec()->getRngEngine()();

	REQUIRE(first_a == first_b);
}

TEST_CASE("Target with constant RCS defaults to no fluctuation model and target-typed id", "[radar][target]")
{
	radar::Platform platform("TargetPlatform");
	radar::Target target(&platform, "Iso");
	target.setRcs(std::make_unique<radar::IsoTargetRcs>(1.0, 99));

	REQUIRE(target.getRcsSpec()->getFluctuationModel() == nullptr);
	REQUIRE(SimIdGenerator::getType(target.getId()) == ObjectType::Target);
}

TEST_CASE("Target with no RCS has no RCS spec", "[radar][target]")
{
	radar::Platform platform("TargetPlatform");
	radar::Target const target(&platform, "Bare", 12021);

	REQUIRE(target.getRcsSpec() == nullptr);

	math::SVec3 in_angle(1.0, 0.0, 0.0);
	math::SVec3 out_angle(1.0, 0.0, 0.0);
	REQUIRE_FALSE(target.getRcs(in_angle, out_angle, 0.0).has_value());
}

TEST_CASE("FileTarget multiplies interpolated azimuth and elevation RCS in the target frame", "[radar][target]")
{
	const std::filesystem::path path = tempFilePath(uniqueFileName("target_rcs_interp"));
	removeIfExists(path);
	writeXmlTargetFile(path, xmlTargetFixture());

	radar::Platform platform("TargetPlatform");
	setStaticRotation(platform);

	constexpr SimId explicit_id = 4001;
	radar::Target target(&platform, "File", explicit_id);
	target.setRcs(std::make_unique<radar::FileTargetRcs>(path.string(), 12));

	math::SVec3 in_angle = unitDirection(0.6, 0.2);
	math::SVec3 out_angle = unitDirection(0.4, 0.6);

	const RealType expected_azimuth_rcs = 2.0 + 4.0 * 0.5;
	const RealType expected_elevation_rcs = 3.0 + 2.0 * 0.4;
	const RealType expected_rcs = expected_azimuth_rcs * expected_elevation_rcs;

	const auto* file_rcs = dynamic_cast<const radar::FileTargetRcs*>(target.getRcsSpec());
	REQUIRE(file_rcs != nullptr);
	REQUIRE(target.getId() == explicit_id);
	REQUIRE(file_rcs->getFilename() == path.string());
	const auto rcs = target.getRcs(in_angle, out_angle, 0.0);
	REQUIRE(rcs.has_value());
	REQUIRE_THAT(rcs.value(), WithinAbs(expected_rcs, 1e-12));

	removeIfExists(path);
}

TEST_CASE("FileTarget uses time-dependent platform rotation to look up body-frame RCS", "[radar][target]")
{
	const std::filesystem::path path = tempFilePath(uniqueFileName("target_rcs_rotation"));
	removeIfExists(path);
	writeXmlTargetFile(path, xmlTargetFixture());

	radar::Platform platform("TargetPlatform");
	setConstantRotation(platform, 0.1, 0.2, 0.2, 0.1);

	radar::Target target(&platform, "File", 5002);
	target.setRcs(std::make_unique<radar::FileTargetRcs>(path.string(), 34));

	math::SVec3 in_angle = unitDirection(0.4, 0.2);
	math::SVec3 out_angle = unitDirection(0.5, 0.4);

	constexpr RealType time = 2.0;
	const RealType expected_azimuth_rcs = 2.0 + 4.0 * 0.2;
	const RealType expected_elevation_rcs = 3.0 + 2.0 * 0.1;
	const RealType expected_rcs = expected_azimuth_rcs * expected_elevation_rcs;

	const auto rcs = target.getRcs(in_angle, out_angle, time);
	REQUIRE(rcs.has_value());
	REQUIRE_THAT(rcs.value(), WithinAbs(expected_rcs, 1e-12));

	removeIfExists(path);
}

TEST_CASE("FileTarget applies fluctuation model as a multiplicative RCS term", "[radar][target]")
{
	const std::filesystem::path path = tempFilePath(uniqueFileName("target_rcs_fluctuation"));
	removeIfExists(path);
	writeXmlTargetFile(path, xmlTargetFixture());

	radar::Platform platform("TargetPlatform");
	setStaticRotation(platform);

	radar::Target target(&platform, "File", 6003);
	target.setRcs(std::make_unique<radar::FileTargetRcs>(path.string(), 56));
	auto model = std::make_unique<FixedRcsModel>(1.5);
	const auto* model_ptr = model.get();
	target.getRcsSpec()->setFluctuationModel(std::move(model));

	math::SVec3 in_angle = unitDirection(1.0, 1.0);
	math::SVec3 out_angle = unitDirection(1.0, 1.0);

	REQUIRE(target.getRcsSpec()->getFluctuationModel() == model_ptr);
	const auto rcs = target.getRcs(in_angle, out_angle, 0.0);
	REQUIRE(rcs.has_value());
	REQUIRE_THAT(rcs.value(), WithinAbs(30.0 * 1.5, 1e-12));

	removeIfExists(path);
}

TEST_CASE("createFileTarget constructs a target with explicit id and filename", "[radar][target]")
{
	const std::filesystem::path path = tempFilePath(uniqueFileName("target_factory"));
	removeIfExists(path);
	writeXmlTargetFile(path, xmlTargetFixture());

	radar::Platform platform("TargetPlatform");
	setStaticRotation(platform);

	const auto target = radar::createFileTarget(&platform, "FileFactory", path.string(), 78, 7004);
	const auto* file_rcs = dynamic_cast<const radar::FileTargetRcs*>(target->getRcsSpec());

	REQUIRE(file_rcs != nullptr);
	REQUIRE(target->getId() == 7004);
	REQUIRE(file_rcs->getFilename() == path.string());

	removeIfExists(path);
}

TEST_CASE("FileTarget skips incomplete sample nodes and uses the valid samples that remain", "[radar][target]")
{
	const std::filesystem::path path = tempFilePath(uniqueFileName("target_partial_samples"));
	removeIfExists(path);
	writeXmlTargetFile(path,
					   R"(<target>
		<azimuth>
			<rcssample><angle>0.0</angle></rcssample>
			<rcssample><angle>1.0</angle><rcs>4.0</rcs></rcssample>
		</azimuth>
		<elevation>
			<rcssample><rcs>9.0</rcs></rcssample>
			<rcssample><angle>1.0</angle><rcs>3.0</rcs></rcssample>
		</elevation>
	</target>)");

	radar::Platform platform("TargetPlatform");
	setStaticRotation(platform);

	radar::Target target(&platform, "File", 8005);
	target.setRcs(std::make_unique<radar::FileTargetRcs>(path.string(), 90));
	math::SVec3 in_angle = unitDirection(1.0, 1.0);
	math::SVec3 out_angle = unitDirection(1.0, 1.0);

	const auto rcs = target.getRcs(in_angle, out_angle, 0.0);
	REQUIRE(rcs.has_value());
	REQUIRE_THAT(rcs.value(), WithinAbs(12.0, 1e-12));

	removeIfExists(path);
}

TEST_CASE("FileTargetRcs throws for missing target description file", "[radar][target]")
{
	const std::filesystem::path path = tempFilePath(uniqueFileName("target_missing"));
	removeIfExists(path);

	REQUIRE_THROWS_AS(radar::FileTargetRcs(path.string(), 111), std::runtime_error);
}

TEST_CASE("FileTargetRcs throws for malformed target description file", "[radar][target]")
{
	const std::filesystem::path path = tempFilePath(uniqueFileName("target_malformed"));
	removeIfExists(path);

	{
		std::ofstream out(path, std::ios::binary);
		REQUIRE(out.is_open());
		out << "<target><azimuth><rcssample></target>";
	}

	REQUIRE_THROWS_AS(radar::FileTargetRcs(path.string(), 222), std::runtime_error);

	removeIfExists(path);
}

TEST_CASE("FileTarget currently throws at lookup time when an RCS axis has no samples", "[radar][target]")
{
	const std::filesystem::path path = tempFilePath(uniqueFileName("target_missing_axis"));
	removeIfExists(path);
	writeXmlTargetFile(path,
					   R"(<target>
		<azimuth>
			<rcssample><angle>0.0</angle><rcs>2.0</rcs></rcssample>
		</azimuth>
		<elevation></elevation>
	</target>)");

	radar::Platform platform("TargetPlatform");
	setStaticRotation(platform);

	radar::Target target(&platform, "File", 11008);
	target.setRcs(std::make_unique<radar::FileTargetRcs>(path.string(), 333));
	math::SVec3 in_angle = unitDirection(0.0, 0.0);
	math::SVec3 out_angle = unitDirection(0.0, 0.0);

	REQUIRE_THROWS_AS(target.getRcs(in_angle, out_angle, 0.0), std::runtime_error);

	removeIfExists(path);
}

TEST_CASE("Target's RCS exposes initial seed", "[radar][target]")
{
	radar::Platform platform("TargetPlatform");
	radar::Target target(&platform, "Iso");
	target.setRcs(std::make_unique<radar::IsoTargetRcs>(1.0, 12345));
	REQUIRE(target.getRcsSpec()->getSeed() == 12345);
}

// TODO: FileTargetRcs accepts XML files with missing/empty azimuth or elevation axes
// during construction and only fails later in computeRcs(). Once the source validates
// required sample sets in the constructor, replace the lookup-time failure test
// with a constructor-level validation test.
