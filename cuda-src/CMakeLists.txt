set(GAPPP_CUDA_SOURCES l3fwd.cu l3fwd.h selftest.cu selftest.h dummy.cu dummy.h)

add_library(gappp-cuda SHARED ${GAPPP_CUDA_SOURCES})
target_link_libraries(gappp-cuda PRIVATE fmt::fmt)
# target_link_libraries(gappp-cuda PRIVATE cudadevrt)