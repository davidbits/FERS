# Searches OPTIX_ROOT, environment variable, and default install locations
find_path(OptiX_INCLUDE_DIR
    NAMES optix.h
    PATHS
        ${OPTIX_ROOT}
        $ENV{OPTIX_ROOT}
        /opt/optix
        /opt/nvidia/optix
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0"
    PATH_SUFFIXES include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX
    REQUIRED_VARS OptiX_INCLUDE_DIR
    VERSION_VAR OptiX_VERSION
)

if(OptiX_FOUND AND NOT TARGET OptiX::OptiX)
    add_library(OptiX::OptiX INTERFACE IMPORTED)
    target_include_directories(OptiX::OptiX INTERFACE ${OptiX_INCLUDE_DIR})
endif()
