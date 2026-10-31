# FERS OptiX GPU ray-tracing backend.
#
# Included (once) from packages/libfers/src/CMakeLists.txt when
# FERS_ENABLE_OPTIX is ON. Locates OptiX + CUDA, compiles kernel.cu to PTX with
# nvcc, and embeds the PTX into a generated header (via FersEmbedFile.cmake,
# the same tool used for the XML schema headers) so optix.cpp can hand the
# module bytes straight to optixModuleCreate() without FERS needing to locate
# a loose .ptx file next to the binary at runtime.
#
# Expects GENERATED_HEADERS_DIR to already be set by the including
# CMakeLists.txt. Sets, in the caller's scope:
#   FERS_OPTIX_SOURCES         - .cpp files to add to LIBFERS_SOURCES
#   FERS_OPTIX_PRIVATE_HEADERS - headers to add to LIBFERS_PRIVATE_HEADERS
#   FERS_OPTIX_LINK_LIBRARIES  - libraries to add to FERS_NATIVE_LINK_DEPENDENCIES
#   FERS_OPTIX_GENERATE_TARGET - target the library target must add_dependencies() on

# Find OptiX with cmake/FindOptiX.cmake
find_package(OptiX REQUIRED)
# Find the CUDA toolkit so we can compile device programs to PTX.
find_package(CUDAToolkit REQUIRED)

set(_fers_optix_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/propagation/raytracing/optix)

set(FERS_OPTIX_PRIVATE_HEADERS
	${_fers_optix_dir}/optix.h
)
set(FERS_OPTIX_SOURCES
	${_fers_optix_dir}/optix.cpp
)
set(FERS_OPTIX_LINK_LIBRARIES
	OptiX::OptiX
	CUDA::cuda_driver
	CUDA::cudart
)

# -------------------- PTX Generation -------------------- #

set(_fers_optix_kernel_cu ${_fers_optix_dir}/kernel.cu)
set(_fers_optix_kernel_ptx ${CMAKE_CURRENT_BINARY_DIR}/optix_ptx/kernel.ptx)
set(_fers_optix_kernel_ptx_header ${GENERATED_HEADERS_DIR}/kernel_ptx.h)

add_custom_command(
	OUTPUT ${_fers_optix_kernel_ptx}
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/optix_ptx
	COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
			--ptx
			--expt-relaxed-constexpr
			-lineinfo
			-std=c++${CMAKE_CXX_STANDARD}
			-Xcompiler -Wall,-Wextra,-Wshadow,-Wconversion,-Wsign-conversion,-Wold-style-cast
			--Wreorder --Wext-lambda-captures-this --Wdefault-stream-launch
			-I "${OptiX_INCLUDE_DIR}"
			-I "${_fers_optix_dir}"
			-I "${CMAKE_CURRENT_SOURCE_DIR}/src"
			-o "${_fers_optix_kernel_ptx}"
			"${_fers_optix_kernel_cu}"
	DEPENDS
		${_fers_optix_kernel_cu}
		${_fers_optix_dir}/kernel_defs.h
		${_fers_optix_dir}/utils.h
		${CMAKE_CURRENT_SOURCE_DIR}/src/propagation/raytracing/sbr_shared.h
		${CMAKE_CURRENT_SOURCE_DIR}/src/propagation/raytracing/sbr_impl.h
		${CMAKE_CURRENT_SOURCE_DIR}/src/propagation/math.h
	COMMENT "Compiling kernel.cu to PTX with nvcc"
	VERBATIM
)

add_custom_command(
	OUTPUT ${_fers_optix_kernel_ptx_header}
	COMMAND ${CMAKE_COMMAND}
			"-DINPUT=${_fers_optix_kernel_ptx}"
			"-DOUTPUT=${_fers_optix_kernel_ptx_header}"
			-DVAR=devicePrograms_ptx
			-P "${CMAKE_SOURCE_DIR}/cmake/FersEmbedFile.cmake"
	DEPENDS ${_fers_optix_kernel_ptx} "${CMAKE_SOURCE_DIR}/cmake/FersEmbedFile.cmake"
	COMMENT "Embedding kernel.ptx into kernel_ptx.h"
)

# ALL so that a plain `cmake --build` regenerates it even before anything
# depends on it yet (matches generate_schemas' behaviour below in the parent
# CMakeLists.txt) - useful for e.g. warming up an IDE's build directory.
add_custom_target(generate_optix_ptx ALL DEPENDS ${_fers_optix_kernel_ptx_header})
set(FERS_OPTIX_GENERATE_TARGET generate_optix_ptx)

unset(_fers_optix_dir)
unset(_fers_optix_kernel_cu)
unset(_fers_optix_kernel_ptx)
unset(_fers_optix_kernel_ptx_header)
