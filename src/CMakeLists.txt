set(LIBGAPPP_SOURCES Router.cpp Router.h Logging.h)

add_library(libgappp SHARED ${LIBGAPPP_SOURCES})
set_target_properties(libgappp PROPERTIES OUTPUT_NAME gappp)
target_link_libraries(libgappp PUBLIC fmt::fmt)
target_link_libraries(libgappp PUBLIC ${DPDK_LIBS})

add_executable(gappp-main gappp-main.cpp)
set_target_properties(gappp-main PROPERTIES OUTPUT_NAME gappp)
target_link_libraries(gappp-main PRIVATE libgappp)
