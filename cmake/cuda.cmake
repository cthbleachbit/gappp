macro(GAPPP_CUDA_KERNEL_ADD arch prev_arch name flags sources)
	if(${arch} MATCHES "compute_.*")
		set(format "ptx")
	else()
		set(format "cubin")
	endif()
	set(cuda_file ${name}_${arch}.${format})
	set(kernel_sources ${sources})
	if(NOT ${prev_arch} STREQUAL "none")
		if(${prev_arch} MATCHES "compute_.*")
			set(kernel_sources ${kernel_sources} ${name}_${prev_arch}.ptx)
		else()
			set(kernel_sources ${kernel_sources} ${name}_${prev_arch}.cubin)
		endif()
	endif()
	set(cuda_kernel_src "${CMAKE_CURRENT_SOURCE_DIR}/${name}.cu")
	set(cuda_flags ${flags}
		-D NVCC
		-m ${CUDA_BITS}
		-I ${CMAKE_CURRENT_SOURCE_DIR}
		--use_fast_math
		-o ${CMAKE_CURRENT_BINARY_DIR}/${cuda_file})

	if(WITH_CYCLES_CUBIN_COMPILER)
		string(SUBSTRING ${arch} 3 -1 CUDA_ARCH)

		# Needed to find libnvrtc-builtins.so. Can't do it from inside
		# cycles_cubin_cc since the env variable is read before main()
		set(CUBIN_CC_ENV ${CMAKE_COMMAND}
			-E env LD_LIBRARY_PATH="${cuda_toolkit_root_dir}/lib")

		add_custom_command(
			OUTPUT ${cuda_file}
			COMMAND ${CUBIN_CC_ENV}
			"$<TARGET_FILE:gappp_cubin_cc>"
			-target ${CUDA_ARCH}
			-i ${CMAKE_CURRENT_SOURCE_DIR}${cuda_kernel_src}
			${cuda_flags}
			-v
			-cuda-toolkit-dir "${cuda_toolkit_root_dir}"
			DEPENDS ${kernel_sources} cycles_cubin_cc)
	else()
		add_custom_command(
			OUTPUT ${cuda_file}
			COMMAND ${cuda_nvcc_executable}
			-arch=${arch}
			${CUDA_NVCC_FLAGS}
			--${format}
			${CMAKE_CURRENT_SOURCE_DIR}${cuda_kernel_src}
			--ptxas-options="-v"
			${cuda_flags}
			DEPENDS ${kernel_sources})
	endif()
	install("${CMAKE_CURRENT_BINARY_DIR}" "${cuda_file}" ${INSTALL_LIB_DIR})
	list(APPEND cuda_cubins ${cuda_file})
endmacro()

# vim: set noexpandtab:
