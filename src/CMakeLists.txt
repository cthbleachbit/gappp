set(LIBGAPPP_SOURCES Router.cpp GPUHelm.cpp RouteTable.h RouteTable.cpp)

add_library(libgappp SHARED ${LIBGAPPP_SOURCES})
set_target_properties(libgappp PROPERTIES OUTPUT_NAME gappp)
target_link_libraries(libgappp PUBLIC ${fmt_LIBS})
target_link_libraries(libgappp PUBLIC ${DPDK_LIBS})
target_link_libraries(libgappp PUBLIC pthread)

add_executable(gappp-main gappp-main.cpp)
set_target_properties(gappp-main PROPERTIES OUTPUT_NAME gappp)
target_link_libraries(gappp-main PRIVATE libgappp)
