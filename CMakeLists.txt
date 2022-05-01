cmake_minimum_required(VERSION 3.16)
project(gappp LANGUAGES C CXX CUDA)
set(SOVERSION 0)
set(VERSION ${SOVERSION}.0.1)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -march=x86-64-v3 -msse4 -mavx -mavx2 -Wno-volatile")
set(CMAKE_SHARED_LINKER_FLAGS " -Wl,--as-needed ${CMAKE_SHARED_LINKER_FLAGS}")
set(INSTALL_BIN_DIR ${CMAKE_INSTALL_PREFIX}/bin CACHE PATH "Installation directory for executables")
set(INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib CACHE PATH "Installation directory for libraries")
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH ${INSTALL_LIB_DIR})

set_property(GLOBAL PROPERTY POSITION_INDEPENDENT_CODE ON)

# Load macro definitions
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda.cmake)

# CUDA
set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86 87)
find_package(CUDA)
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)

# DPDK
find_package(PkgConfig REQUIRED)
pkg_check_modules(DPDK REQUIRED libdpdk)
set(DPDK_LIBS "${DPDK_LDFLAGS}")
include_directories(${DPDK_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cuda-src)

# fmt
option(GAPPP_USE_SYSTEM_FMT "Use system fmtlib" ON)
if (GAPPP_USE_SYSTEM_FMT)
	find_package(fmt REQUIRED)
	set(fmt_LIBS fmt::fmt)
else()
	set(FMT_DEBUG_POSTFIX "")
	add_subdirectory(fmt)
	set(fmt_LIBS fmt::fmt-header-only)
endif()

# optimization options
option(GAPPP_LTO "Use link time optimization" ON)
if (GAPPP_LTO)
	include(CheckIPOSupported)
	check_ipo_supported()
	set_property(GLOBAL PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
	set_property(GLOBAL PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
endif ()

add_subdirectory(cuda-src)

add_subdirectory(src)
set(gappp::libgappp libgappp ALIAS)
target_link_libraries(libgappp PRIVATE gappp-cuda)