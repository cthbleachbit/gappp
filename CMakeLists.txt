cmake_minimum_required(VERSION 3.16)
project(gappp LANGUAGES C CXX CUDA)
set(SOVERSION 0)
set(VERSION ${SOVERSION}.0.1)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -march=x86-64-v3 -msse4 -mavx -mavx2 -Wno-volatile ")
set(CMAKE_SHARED_LINKER_FLAGS " -Wl,--as-needed ${CMAKE_SHARED_LINKER_FLAGS}")
set(INSTALL_BIN_DIR ${CMAKE_INSTALL_PREFIX}/bin CACHE PATH "Installation directory for executables")
set(INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib CACHE PATH "Installation directory for libraries")
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH ${INSTALL_LIB_DIR})

SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_definitions("_GNU_SOURCE")

# Load macro definitions
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda.cmake)

# CUDA
set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86 87)
find_package(CUDA)
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
set(CUDA_NVCC_EXECUTABLE ${CMAKE_CUDA_COMPILER})
set(CMAKE_CUDA_FLAGS "-ccbin ${CMAKE_CXX_COMPILER} ")

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
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ")
	set_property(GLOBAL PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
	set_property(GLOBAL PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
endif ()

# silence mode
option(GAPPP_SILENT "Do not print anything - for benchmark only?" OFF)
if (GAPPP_SILENT)
	add_compile_definitions(GAPPP_SILENT)
endif()

# GPU direct
option(GAPPP_GPU_DIRECT "Build with GPU direct support" ON)
if (GAPPP_GPU_DIRECT)
	if(NOT DEFINED pkgcfg_lib_DPDK_rte_gpudev)
		message(WARNING "DPDK gpudev support not found - GPU direct turned off!")
		set(GAPPP_GPU_DIRECT OFF)
	else()
		add_compile_definitions(GAPPP_GPU_DIRECT)
		add_compile_definitions(ALLOW_EXPERIMENTAL_API)
	endif()
endif()

add_subdirectory(cuda-src)

add_subdirectory(src)
target_link_libraries(libgappp PUBLIC gappp-cuda)
target_link_libraries(gappp-main PUBLIC gappp-cuda)

set(gappp::libgappp libgappp ALIAS)