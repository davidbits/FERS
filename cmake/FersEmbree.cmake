# FERS Embree CPU ray-tracing backend.
#
# Included (once) from packages/libfers/src/CMakeLists.txt when
# FERS_ENABLE_EMBREE is ON. Embree is sourced via vcpkg's "embree" manifest
# feature (see vcpkg.json) and exports its own CMake config package, so no
# custom Find module is needed (unlike OptiX).
#
# Sets, in the caller's scope:
#   FERS_EMBREE_LINK_LIBRARIES - libraries to add to FERS_NATIVE_LINK_DEPENDENCIES

find_package(embree CONFIG REQUIRED)

set(_fers_embree_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/propagation/raytracing/embree)

set(FERS_EMBREE_PRIVATE_HEADERS
	${_fers_embree_dir}/kernel.h
  ${_fers_embree_dir}/embree.h
  ${_fers_embree_dir}/utils.h
)
set(FERS_EMBREE_SOURCES
	${_fers_embree_dir}/kernel.cpp
  ${_fers_embree_dir}/embree.cpp
)

set(FERS_EMBREE_LINK_LIBRARIES unofficial::embree::embree)
