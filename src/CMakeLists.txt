set(LIBGAPPP_SOURCES Router.cpp GPUHelm.cpp RouteTable.h RouteTable.cpp)

if (GAPPP_GPU_DIRECT)
	list(APPEND LIBGAPPP_SOURCES GPUDirectHelm.cpp)
endif()

add_library(libgappp SHARED ${LIBGAPPP_SOURCES})
set_target_properties(libgappp PROPERTIES OUTPUT_NAME gappp)
target_link_libraries(libgappp PUBLIC ${fmt_LIBS})
target_link_libraries(libgappp PUBLIC pthread)
target_link_libraries(libgappp PUBLIC ${DPDK_LIBS})
target_link_libraries(libgappp PUBLIC ${CUDA_CUDART_LIBRARY})

add_executable(gappp-main gappp-main.cpp)
set_target_properties(gappp-main PROPERTIES OUTPUT_NAME gappp)
target_link_libraries(gappp-main PUBLIC libgappp)