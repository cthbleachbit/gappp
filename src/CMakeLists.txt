add_executable(gappp-main gappp-main.cpp)
set_target_properties(gappp-main PROPERTIES OUTPUT_NAME gappp)
target_link_libraries(gappp-main PRIVATE fmt::fmt)
target_link_libraries(gappp-main PRIVATE ${DPDK_LIBS})