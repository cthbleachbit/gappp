#ifndef __global__
#define __global__
#endif

#include "gappp_types.h"
#include "l3fwd.h"
#include "Logging.h"

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <iostream>

#define GAPPP_LOG_SELFTEST "GPU Self Test"

namespace GAPPP {
	namespace selftest {
		static int device_id = -1;

		__global__ void GenVec(double base, double*A) {
			int i = threadIdx.x;
			A[i] = pow(2, base + 2 * i);
		}

		template<typename T>
		struct aligned_vec128 {
			alignas(4096) T v[128];
		};

		// Mapping CPU memory to GPU:
		// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/simpleZeroCopy/simpleZeroCopy.cu
		int invoke(unsigned int nbr_tasks, struct rte_mbuf **packets) {
			int device_count = 0;
			cudaDeviceProp device_properties{};

			cudaError_t ret = cudaGetDeviceCount(&device_count);
			if (device_count < 0) {
				whine(Severity::CRIT, "No CUDA devices found", GAPPP_LOG_SELFTEST);
			} else {
				whine(Severity::INFO, fmt::format("{} CUDA devices found", device_count), GAPPP_LOG_SELFTEST);
			}
			device_id = 0;
			cudaSetDevice(device_id);
			ret = cudaGetDeviceProperties(&device_properties, device_id);
			if (ret < 0) {
				whine(Severity::CRIT, "Failed to obtain CUDA device properties", GAPPP_LOG_SELFTEST);
			}

			whine(Severity::INFO, fmt::format("Using CUDA device {}", device_properties.name), GAPPP_LOG_SELFTEST);

			ret = cudaSetDeviceFlags(cudaDeviceMapHost);
			if (ret < 0) {
				whine(Severity::CRIT, "Device does not support mapping from host memory", GAPPP_LOG_SELFTEST);
			}

			auto va = new aligned_vec128<double>(); // 4K aligned allocation
			double *da; // GPU memory pointer - not valid under CPU content

			// Maps va->v into GPU memory space
			ret = cudaHostRegister(va->v, 128 * sizeof(double), cudaHostRegisterMapped);
			if (ret < 0) {
				whine(Severity::CRIT, "Failed to map from host memory", GAPPP_LOG_SELFTEST);
			}

			// Obtain memory pointers in GPU memory space - this pointer will be passed to CUDA kernel routines
			ret = cudaHostGetDevicePointer((void **) &da, (void *)va->v, 0);
			if (ret < 0) {
				whine(Severity::CRIT, "Failed to obtain mapped GPU memory address", GAPPP_LOG_SELFTEST);
			}

			dim3 block(256);
			dim3 grid((unsigned int)ceil(128 / (float)block.x));
			GenVec<<<grid, block>>>(0, da);
			cudaDeviceSynchronize();
			ret = cudaGetLastError();
			if (ret < 0) {
				whine(Severity::CRIT, "Self test vector generation failed", GAPPP_LOG_SELFTEST);
			}

			// Compare results
			bool results = va->v[0] == 1.0f && va->v[1] == 4.0f;
			delete va;

			return results ? 0 : -1;
		}
	}
}