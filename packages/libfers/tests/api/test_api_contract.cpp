#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <libfers/api.h>
#include <string_view>

#include "api_test_helpers.h"

using Catch::Matchers::ContainsSubstring;

TEST_CASE("API destroying a null context is a safe no-op", "[api][core]")
{
	api_test::clearLastError();

	fers_context_destroy(nullptr);

	api_test::ApiString const error = api_test::lastError();
	REQUIRE(error.get() == nullptr);
}

TEST_CASE("API context creation returns a valid handle", "[api][core]")
{
	api_test::Context const context;
	REQUIRE(context.get() != nullptr);

	api_test::ApiString const error = api_test::lastError();
	REQUIRE(error.get() == nullptr);
}

TEST_CASE("API exposes the shared repo version", "[api][core]")
{
	const char* version = fers_get_version();

	REQUIRE(version != nullptr);
	REQUIRE(std::string_view(version) == FERS_TEST_EXPECTED_VERSION);
}

TEST_CASE("API last error is null when no failure has occurred", "[api][core]")
{
	api_test::clearLastError();

	api_test::ApiString const error = api_test::lastError();
	REQUIRE(error.get() == nullptr);
}

TEST_CASE("API free helpers accept null pointers", "[api][core]")
{
	api_test::clearLastError();

	fers_free_string(nullptr);
	fers_free_interpolated_motion_path(nullptr);
	fers_free_interpolated_rotation_path(nullptr);
	fers_free_antenna_pattern_data(nullptr);
	fers_free_preview_links(nullptr);

	api_test::ApiString const error = api_test::lastError();
	REQUIRE(error.get() == nullptr);
}

TEST_CASE("API set thread count clears stale error state on success", "[api][core]")
{
	api_test::ParamGuard const guard;
	api_test::clearLastError();

	REQUIRE(fers_set_thread_count(0) != 0);
	api_test::ApiString error = api_test::lastError();
	REQUIRE(error.get() != nullptr);

	REQUIRE(fers_set_thread_count(2) == 0);
	error = api_test::lastError();
	REQUIRE(error.get() == nullptr);
}

TEST_CASE("API set ray tracing engine updates params and clears stale error state", "[api][core]")
{
	api_test::ParamGuard const guard;
	api_test::clearLastError();

	REQUIRE(fers_set_ray_tracing_engine(FERS_RT_ENGINE_OPTIX) == 0);
	CHECK(params::rayTracingEnginePreference() == params::RayTracingEnginePreference::OptiX);
	const api_test::ApiString error = api_test::lastError();
	REQUIRE(error.get() == nullptr);

	REQUIRE(fers_set_ray_tracing_engine(FERS_RT_ENGINE_EMBREE) == 0);
	CHECK(params::rayTracingEnginePreference() == params::RayTracingEnginePreference::Embree);

	REQUIRE(fers_set_ray_tracing_engine(FERS_RT_ENGINE_AUTO) == 0);
	CHECK(params::rayTracingEnginePreference() == params::RayTracingEnginePreference::Auto);
}

namespace
{
	template <typename E>
	E make_invalid_enum(std::underlying_type_t<E> raw) noexcept
	{
		static_assert(std::is_enum_v<E>);
		E e;
		std::memcpy(&e, &raw, sizeof e);
		return e;
	};
}

TEST_CASE("API set ray tracing engine rejects an out-of-range value", "[api][core]")
{
	api_test::ParamGuard const guard;
	api_test::clearLastError();

	REQUIRE(fers_set_ray_tracing_engine(make_invalid_enum<fers_ray_tracing_engine_t>(99)) != 0);
	const api_test::ApiString error = api_test::lastError();
	REQUIRE(error.get() != nullptr);
}

TEST_CASE("API successful calls clear stale thread-local errors beyond thread-count changes", "[api][core]")
{
	api_test::Context const context;
	REQUIRE(context.get() != nullptr);

	REQUIRE(fers_get_scenario_as_json(nullptr) == nullptr);
	api_test::ApiString error = api_test::lastError();
	REQUIRE_THAT(error.str(), ContainsSubstring("Invalid context provided to fers_get_scenario_as_json"));

	const std::string xml = api_test::minimalScenarioXml();
	REQUIRE(fers_load_scenario_from_xml_string(context.get(), xml.c_str(), 0) == 0);

	error = api_test::lastError();
	REQUIRE(error.get() == nullptr);
}

TEST_CASE("API antenna pattern rejects sample counts smaller than two", "[api][preview]")
{
	api_test::clearLastError();

	REQUIRE(fers_get_antenna_pattern(nullptr, 1, 1, 2, 1.0) == nullptr);
	api_test::ApiString error = api_test::lastError();
	REQUIRE(error.get() != nullptr);
	REQUIRE_THAT(error.str(), ContainsSubstring("sample counts must be >= 2"));

	REQUIRE(fers_get_antenna_pattern(nullptr, 1, 2, 1, 1.0) == nullptr);
	error = api_test::lastError();
	REQUIRE(error.get() != nullptr);
	REQUIRE_THAT(error.str(), ContainsSubstring("sample counts must be >= 2"));
}
