cmake_minimum_required(VERSION 3.16)
project(GAPPP)
set(SOVERSION 0)
set(VERSION ${SOVERSION}.0.1)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_SHARED_LINKER_FLAGS " -Wl,--as-needed ${CMAKE_SHARED_LINKER_FLAGS}")
set(INSTALL_BIN_DIR ${CMAKE_INSTALL_PREFIX}/bin CACHE PATH "Installation directory for executables")
set(INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib CACHE PATH "Installation directory for libraries")
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH ${INSTALL_LIB_DIR})

# Load macro definitions
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda.cmake)

# CUDA
set(GAPPP_CUDA_BINARIES_ARCH sm_60 sm_61 sm_62 sm_70 sm_72 sm_75 sm_80 sm_86 sm_87 CACHE STRING "CUDA architectures to build binaries for")
find_package(CUDA)

# DPDK
find_package(PkgConfig REQUIRED)
pkg_check_modules(DPDK REQUIRED libdpdk)

# fmt
find_package(fmt REQUIRED)

add_subdirectory(src)