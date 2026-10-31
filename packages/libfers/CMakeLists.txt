project(libfers)

function(fers_append_static_library out_var candidate)
	set(_result "${${out_var}}")

	if (EXISTS "${candidate}")
		get_filename_component(_candidate_ext "${candidate}" EXT)
		string(TOLOWER "${_candidate_ext}" _candidate_ext)

		if (_candidate_ext STREQUAL ".a" OR _candidate_ext STREQUAL ".lib")
			list(APPEND _result "${candidate}")
		endif ()
	endif ()

	set(${out_var} "${_result}" PARENT_SCOPE)
endfunction()

# vcpkg config files express some of their dependencies as generator expressions,
# which CMake only expands at generate time, after every CMakeLists.txt has been
# processed, so a configure-time traversal like the one below sees them as opaque
# text. Resolve the forms that can be decided at configure time so the GENEX_STRIP
# fallback in fers_collect_static_link_libraries() does not silently discard a real
# library.
# Assimp, for instance, declares its minizip dependency as
#   $<LINK_ONLY:$<IF:$<TARGET_EXISTS:MINIZIP::minizip>,MINIZIP::minizip,MINIZIP::minizipstatic>>
# and dropping it leaves Assimp's unz* calls (IFCLoader, ZipArchiveIOSystem)
# undefined when Cargo links the static library into the Tauri binary.
function(fers_resolve_link_genex out_var item)
	set(_item "${item}")
	set(_resolved ON)

	while (_resolved)
		set(_resolved OFF)

		if (_item MATCHES "^\\$<LINK_ONLY:(.+)>$")
			set(_item "${CMAKE_MATCH_1}")
			set(_resolved ON)
		elseif (_item MATCHES "^\\$<IF:\\$<TARGET_EXISTS:([^>,]+)>,([^>,]*),([^>,]*)>$")
			if (TARGET "${CMAKE_MATCH_1}")
				set(_item "${CMAKE_MATCH_2}")
			else ()
				set(_item "${CMAKE_MATCH_3}")
			endif ()
			set(_resolved ON)
		elseif (_item MATCHES "^\\$<TARGET_NAME_IF_EXISTS:([^>,]+)>$")
			if (TARGET "${CMAKE_MATCH_1}")
				set(_item "${CMAKE_MATCH_1}")
			else ()
				set(_item "")
			endif ()
			set(_resolved ON)
		elseif (_item MATCHES "^\\$<TARGET_NAME:(.+)>$")
			set(_item "${CMAKE_MATCH_1}")
			set(_resolved ON)
		elseif (_item MATCHES "^\\$<\\$<BOOL:(.*)>:(.*)>$")
			set(_condition "${CMAKE_MATCH_1}")
			set(_value "${CMAKE_MATCH_2}")
			if (_condition)
				set(_item "${_value}")
			else ()
				set(_item "")
			endif ()
			set(_resolved ON)
		endif ()
	endwhile ()

	# Anything still carrying a generator expression is about to be stripped to
	# nothing by the caller. Say so loudly: a missing archive only shows up much
	# later, as undefined symbols from an unrelated-looking Rust link step.
	if (_item MATCHES "\\$<")
		message(WARNING
				"Unresolved generator expression while collecting static link libraries: '${item}'. "
				"If it names a library, it will be missing from vcpkg_links.txt and linking "
				"the Rust binary may fail with undefined symbols.")
	endif ()

	set(${out_var} "${_item}" PARENT_SCOPE)
endfunction()

function(fers_collect_static_link_libraries out_var visited_var)
	set(_result "${${out_var}}")
	set(_visited "${${visited_var}}")
	set(_items ${ARGN})
	set(_mode "all")

	while (_items)
		list(POP_FRONT _items _item)

		if (_item STREQUAL "optimized")
			set(_mode "release")
			continue()
		elseif (_item STREQUAL "debug")
			set(_mode "debug")
			continue()
		elseif (_item STREQUAL "general")
			set(_mode "all")
			continue()
		endif ()

		set(_process_item ON)
		if (_mode STREQUAL "debug" AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
			set(_process_item OFF)
		elseif (_mode STREQUAL "release" AND CMAKE_BUILD_TYPE STREQUAL "Debug")
			set(_process_item OFF)
		endif ()
		set(_mode "all")

		if (NOT _process_item)
			continue()
		endif ()

		fers_resolve_link_genex(_item_for_resolution "${_item}")

		string(GENEX_STRIP "${_item_for_resolution}" _item_stripped)
		if (NOT _item_stripped)
			continue()
		endif ()

		if (TARGET "${_item_stripped}")
			list(FIND _visited "${_item_stripped}" _visited_index)
			if (NOT _visited_index EQUAL -1)
				continue()
			endif ()
			list(APPEND _visited "${_item_stripped}")

			get_target_property(_is_imported "${_item_stripped}" IMPORTED)
			if (_is_imported)
				set(_location)

				if (CMAKE_BUILD_TYPE)
					string(TOUPPER "${CMAKE_BUILD_TYPE}" _config_upper)
					get_target_property(_location "${_item_stripped}" "IMPORTED_LOCATION_${_config_upper}")
					if (NOT _location)
						get_target_property(_location "${_item_stripped}" "IMPORTED_IMPLIB_${_config_upper}")
					endif ()
				endif ()

				if (NOT _location)
					get_target_property(_location "${_item_stripped}" IMPORTED_LOCATION)
				endif ()
				if (NOT _location)
					get_target_property(_location "${_item_stripped}" IMPORTED_IMPLIB)
				endif ()

				if (_location AND NOT _location MATCHES "-NOTFOUND$")
					fers_append_static_library(_result "${_location}")
				endif ()
			endif ()

			foreach (_prop IN ITEMS
					 INTERFACE_LINK_LIBRARIES
					 LINK_LIBRARIES
					 IMPORTED_LINK_INTERFACE_LIBRARIES)
				get_target_property(_deps "${_item_stripped}" ${_prop})
				if (_deps AND NOT _deps MATCHES "-NOTFOUND$")
					fers_collect_static_link_libraries(_result _visited ${_deps})
				endif ()
			endforeach ()
		elseif (IS_ABSOLUTE "${_item_stripped}")
			fers_append_static_library(_result "${_item_stripped}")
		endif ()
	endwhile ()

	# REMOVE_DUPLICATES preserves the first occurrence of each entry. The recursion
	# appends a target's own IMPORTED_LOCATION before descending into its dependencies,
	# so dependents remain ordered before their dependencies, which static linkers require.
	list(REMOVE_DUPLICATES _result)
	set(${out_var} "${_result}" PARENT_SCOPE)
	set(${visited_var} "${_visited}" PARENT_SCOPE)
endfunction()

# Helper function to apply common settings to a library target
function(fers_configure_library_target target_name)
	# Apply project-wide compiler warnings and flags
	apply_fers_warnings(${target_name})

	# --- Source and Header Files ---
	target_sources(${target_name} PRIVATE
				   ${LIBFERS_SOURCES}
				   ${LIBFERS_PUBLIC_HEADERS}
				   ${LIBFERS_PRIVATE_HEADERS}
	)

	# --- Include Directories ---
	target_include_directories(${target_name} PUBLIC
							   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
							   $<INSTALL_INTERFACE:include>
	)
	target_include_directories(${target_name} PRIVATE
							   ${CMAKE_CURRENT_SOURCE_DIR}/src
	)

	# --- Schema Headers ---
	add_dependencies(${target_name} generate_schemas)
	target_include_directories(${target_name} PRIVATE
							   $<BUILD_INTERFACE:${GENERATED_HEADERS_DIR}>
	)

	# --- Optional GPU Ray Tracing Backend Build Steps ---
	if (FERS_ENABLE_OPTIX)
		add_dependencies(${target_name} ${FERS_OPTIX_GENERATE_TARGET})
		# Only define this when the option is actually on, so `#if defined(FERS_ENABLE_OPTIX)`
		# checks in code are correct (a macro defined as `0` via $<BOOL:...> would still be
		# `defined`, which is the mismatch that used to gate the OptiX branch off entirely).
		target_compile_definitions(${target_name} PRIVATE FERS_ENABLE_OPTIX)
	endif ()

	if (FERS_ENABLE_EMBREE)
		# Same reasoning as FERS_ENABLE_OPTIX above: a bare compile definition so
		# `#if defined(FERS_ENABLE_EMBREE)` / `#ifdef FERS_ENABLE_EMBREE` works.
		# No add_dependencies() here - unlike OptiX's PTX embedding, Embree has no
		# generated-header build step.
		target_compile_definitions(${target_name} PRIVATE FERS_ENABLE_EMBREE)
	endif ()

	# --- Third-Party Dependencies ---
	target_link_libraries(${target_name} PRIVATE
						  ${FERS_NATIVE_LINK_DEPENDENCIES}
	)
	target_compile_definitions(${target_name} PRIVATE
							   HAVE_LIBHDF5
	)

	set_target_properties(${target_name} PROPERTIES
						  SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
						  VERSION ${CMAKE_PROJECT_VERSION}
	)
endfunction()


# --- Schema Headers Generation ---
set(SCHEMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../schema)
set(GENERATED_HEADERS_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_headers)

file(MAKE_DIRECTORY ${GENERATED_HEADERS_DIR})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/fers_version.h.in
			   ${GENERATED_HEADERS_DIR}/fers_version.h
			   @ONLY)

add_custom_command(
		OUTPUT
		${GENERATED_HEADERS_DIR}/fers_xml_dtd.h
		${GENERATED_HEADERS_DIR}/fers_xml_xsd.h
		${GENERATED_HEADERS_DIR}/antenna_pattern_dtd.h
		${GENERATED_HEADERS_DIR}/antenna_pattern_xsd.h
		COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_HEADERS_DIR}
		COMMAND ${CMAKE_COMMAND}
				"-DINPUT=${SCHEMA_DIR}/fers-xml.dtd"
				"-DOUTPUT=${GENERATED_HEADERS_DIR}/fers_xml_dtd.h"
				-DVAR=fers_xml_dtd
				-P "${CMAKE_SOURCE_DIR}/cmake/FersEmbedFile.cmake"
		COMMAND ${CMAKE_COMMAND}
				"-DINPUT=${SCHEMA_DIR}/fers-xml.xsd"
				"-DOUTPUT=${GENERATED_HEADERS_DIR}/fers_xml_xsd.h"
				-DVAR=fers_xml_xsd
				-P "${CMAKE_SOURCE_DIR}/cmake/FersEmbedFile.cmake"
		COMMAND ${CMAKE_COMMAND}
				"-DINPUT=${SCHEMA_DIR}/antenna-pattern.dtd"
				"-DOUTPUT=${GENERATED_HEADERS_DIR}/antenna_pattern_dtd.h"
				-DVAR=antenna_pattern_dtd
				-P "${CMAKE_SOURCE_DIR}/cmake/FersEmbedFile.cmake"
		COMMAND ${CMAKE_COMMAND}
				"-DINPUT=${SCHEMA_DIR}/antenna-pattern.xsd"
				"-DOUTPUT=${GENERATED_HEADERS_DIR}/antenna_pattern_xsd.h"
				-DVAR=antenna_pattern_xsd
				-P "${CMAKE_SOURCE_DIR}/cmake/FersEmbedFile.cmake"
		DEPENDS
		${SCHEMA_DIR}/fers-xml.dtd
		${SCHEMA_DIR}/fers-xml.xsd
		${SCHEMA_DIR}/antenna-pattern.dtd
		${SCHEMA_DIR}/antenna-pattern.xsd
		COMMENT "Converting XML schema files to C headers"
)
add_custom_target(generate_schemas ALL DEPENDS
				  ${GENERATED_HEADERS_DIR}/fers_xml_dtd.h
				  ${GENERATED_HEADERS_DIR}/fers_xml_xsd.h
				  ${GENERATED_HEADERS_DIR}/antenna_pattern_dtd.h
				  ${GENERATED_HEADERS_DIR}/antenna_pattern_xsd.h
)

# --- Gather Source and Header Files ---
# (This must come after the Schema Headers Generation above and before the
# Third-Party Dependencies below: when FERS_ENABLE_OPTIX is set, this pulls in
# cmake/FersOptiX.cmake, which writes a generated header into
# GENERATED_HEADERS_DIR and needs OptiX::OptiX / CUDA::cuda_driver to exist
# before FERS_NATIVE_LINK_DEPENDENCIES is assembled below.)
include(include/libfers/CMakeLists.txt)
include(src/CMakeLists.txt)

# --- Third-Party Dependencies (vcpkg) ---
find_package(HighFive CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(GeographicLib CONFIG REQUIRED)
find_package(LibXml2 REQUIRED)
find_package(HDF5 COMPONENTS C HL REQUIRED)
find_package(assimp CONFIG REQUIRED)

set(FERS_NATIVE_LINK_DEPENDENCIES
	HighFive
	nlohmann_json::nlohmann_json
	GeographicLib::GeographicLib
	LibXml2::LibXml2
	assimp::assimp
	# Using this format instead of hdf5::hdf5/hdf5::hdf5_hl due to macOS linking issues
	${HDF5_C_LIBRARIES}
	${HDF5_HL_LIBRARIES}
)

if (FERS_ENABLE_OPTIX)
	list(APPEND FERS_NATIVE_LINK_DEPENDENCIES ${FERS_OPTIX_LINK_LIBRARIES})
endif ()

if (FERS_ENABLE_EMBREE)
	list(APPEND FERS_NATIVE_LINK_DEPENDENCIES ${FERS_EMBREE_LINK_LIBRARIES})
endif ()

if (WIN32)
	list(APPEND FERS_NATIVE_LINK_DEPENDENCIES ws2_32)
endif ()

# Roots for the Rust static-link manifest. This is deliberately distinct from
# FERS_NATIVE_LINK_DEPENDENCIES: the manifest must list libraries in the order a
# static linker expects (dependents before their dependencies), and it must
# recurse into transitive imported targets (szip, aec, zlib) which are only
# reachable through the hdf5::*-static target graph, not through the raw
# HDF5_C_LIBRARIES paths used above. HighFive and nlohmann_json are omitted
# because they are header-only.
set(FERS_NATIVE_LINK_MANIFEST_TARGETS
	GeographicLib::GeographicLib
	LibXml2::LibXml2
	assimp::assimp
	hdf5::hdf5_hl-static
	hdf5::hdf5-static
)

# --- Create Library Targets ---

# Compile sources exactly once into an OBJECT library.
# Both static and shared targets consume these objects.
add_library(fers_objects OBJECT)
set_target_properties(fers_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
fers_configure_library_target(fers_objects)
target_precompile_headers(fers_objects PRIVATE
	<nlohmann/json.hpp>
)

if (FERS_BUILD_STATIC_LIBS)
	add_library(fers_static STATIC $<TARGET_OBJECTS:fers_objects>)
	set_target_properties(fers_static PROPERTIES OUTPUT_NAME "fers")
	# Ensure the static library is built with Position-Independent Code,
	# which is required when linking it into a shared library (like the Rust cdylib).
	set_target_properties(fers_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

	# Include directories and link dependencies must be set on the
	# final library targets so that consumers inherit them correctly.
	target_include_directories(fers_static PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
	)
	target_link_libraries(fers_static PRIVATE ${FERS_NATIVE_LINK_DEPENDENCIES})
	target_compile_definitions(fers_static PRIVATE HAVE_LIBHDF5)

	set_target_properties(fers_static PROPERTIES
		SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
		VERSION ${CMAKE_PROJECT_VERSION}
	)
	add_library(fers::fers_static ALIAS fers_static)

	set(_fers_native_static_links)
	set(_fers_seen_link_targets)
	fers_collect_static_link_libraries(
		_fers_native_static_links
		_fers_seen_link_targets
		${FERS_NATIVE_LINK_MANIFEST_TARGETS}
	)

	string(REPLACE ";" "\n" _fers_native_static_links "${_fers_native_static_links}")
	if (_fers_native_static_links)
		string(APPEND _fers_native_static_links "\n")
	endif ()

	file(WRITE
		 ${CMAKE_CURRENT_BINARY_DIR}/vcpkg_links.txt
		 "${_fers_native_static_links}"
	)
endif ()

if (FERS_BUILD_SHARED_LIBS)
	add_library(fers_shared SHARED $<TARGET_OBJECTS:fers_objects>)
	set_target_properties(fers_shared PROPERTIES OUTPUT_NAME "fers")
	if (WIN32)
		# Avoid an MSVC Ninja conflict with the static library's fers.lib.
		set_target_properties(fers_shared PROPERTIES ARCHIVE_OUTPUT_NAME "fers_shared")
	endif ()

	target_include_directories(fers_shared PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
	)
	target_link_libraries(fers_shared PRIVATE ${FERS_NATIVE_LINK_DEPENDENCIES})
	target_compile_definitions(fers_shared PRIVATE HAVE_LIBHDF5)

	set_target_properties(fers_shared PROPERTIES
		SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
		VERSION ${CMAKE_PROJECT_VERSION}
	)
	add_library(fers::fers_shared ALIAS fers_shared)
endif ()

# Create a generic 'libfers' INTERFACE target that downstream internal targets
# (like fers-cli) can link to. It prefers the shared library if available.
add_library(libfers INTERFACE)
if (FERS_BUILD_SHARED_LIBS)
	target_link_libraries(libfers INTERFACE fers::fers_shared)
else ()
	target_link_libraries(libfers INTERFACE fers::fers_static)
endif ()
add_library(fers::fers ALIAS libfers)

# Group sources for IDEs
source_group("Source Files" FILES ${LIBFERS_SOURCES})
source_group("Header Files\\Public" FILES ${LIBFERS_PUBLIC_HEADERS})
source_group("Header Files\\Private" FILES ${LIBFERS_PRIVATE_HEADERS})

# --- Installation ---
if (FERS_BUILD_STATIC_LIBS)
	install(TARGETS fers_static
			EXPORT libfers-targets
			ARCHIVE DESTINATION lib
	)
endif ()

if (FERS_BUILD_SHARED_LIBS)
	install(TARGETS fers_shared
			EXPORT libfers-targets
			LIBRARY DESTINATION lib
			RUNTIME DESTINATION bin
			ARCHIVE DESTINATION lib
	)
endif ()

install(FILES ${LIBFERS_PUBLIC_HEADERS}
		DESTINATION include/libfers
)

install(EXPORT libfers-targets
		FILE libfers-config.cmake
		NAMESPACE fers::
		DESTINATION lib/cmake/libfers
)

# --- Testing ---
if (FERS_BUILD_TESTS)
	add_subdirectory(tests)
endif ()
