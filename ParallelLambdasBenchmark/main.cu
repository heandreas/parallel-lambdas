#include <iostream>
#include <math.h>
#include <functional>
#include <chrono>
#include <iostream>
#include <string>
#include <omp.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include "../include/parallel_utils_openmp.h"
#include "../include/parallel_utils_cuda.cu"

//! Inspired by Go's defer keyword (https://blog.golang.org/defer-panic-and-recover).
//! Executes the passed function when the object is deleted.
class Defer
{
	std::function<void()> m_DeferredFunc;

public:
	Defer(const std::function<void()>& func) : m_DeferredFunc(func) {}
	~Defer() { m_DeferredFunc(); }
};

class Timer
{
public:
	template<typename T>
	static double toMilliseconds(T d)
	{
		return std::chrono::duration_cast<std::chrono::duration<double, std::chrono::milliseconds::period>>(d).count();
	}

	using Clock = std::chrono::high_resolution_clock;
	using TimePoint = Clock::time_point;

	//! Returns the current time.
	static TimePoint now() { return Clock::now(); }

	// Returns the time in milliseconds that passed since the given reference point.
	static double getTimeSince(const TimePoint& referencePoint) { return toMilliseconds(now() - referencePoint); }

	static Defer logScopeTiming(const std::string& str) {
		auto ts = now(); return Defer([ts, str]() { std::cout << str << " " << getTimeSince(ts) << " ms" << std::endl; });
	}
};

__host__ __device__ int ceilDivision(int a, int b)
{
	return (a + b - 1) / b;
}

template<typename T, typename F>
__global__ void opCuda(int n, T *x, float *y, F op)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = op(x[i]);
}

template<typename T, typename F>
__global__ void opCudaNested(int n, T *x, float *y, F op)
{
	if (threadIdx.x == 0)
	{
		int elementsPerBlock = ceilDivision(n, gridDim.x);
		int offset = blockIdx.x * elementsPerBlock;
		int numElements = min(elementsPerBlock, n - offset);
		int blockSize = 256;
		int numBlocks = min(1 << 15, ceilDivision(numElements, blockSize));
		opCuda << <numBlocks, blockSize >> > (numElements, x + offset, y + offset, op);
	}
}

struct InputData
{
	float x;
	int numElements;
};

int main(void)
{
	int N = 1 << 25;
	std::cout << "Number of elements " << N << std::endl;

	std::vector<InputData> input(N);
	std::vector<float> y(N);
	std::vector<float> control(N);

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++)
	{
		input[i].numElements = 20 + i % 11;
		input[i].x = (float)i;
	}

	InputData* inputGPU;
	float* yGPU;
	cudaMalloc(&inputGPU, N * sizeof(InputData));
	cudaMalloc(&yGPU, N * sizeof(float));

	{
		auto timer = Timer::logScopeTiming("Transfer CPU to GPU");
		cudaMemcpy(inputGPU, input.data(), input.size() * sizeof(InputData), cudaMemcpyHostToDevice);
	}

	omp_set_num_threads(4);
	std::cout << "OMP num threads: " << omp_get_max_threads() << std::endl;

	auto gpuLambda = []__host__ __device__(const InputData& data)
	{
		float accu = 0.0f;
		for (int i = 0; i < data.numElements; i++)
			accu += std::sqrt(data.x * (float)i);
		return accu;
	};
	auto cpuLambda = [](const InputData& data)
	{
		float accu = 0.0f;
		for (int i = 0; i < data.numElements; i++)
			accu += std::sqrt(data.x * (float)i);
		return accu;
	};

	// for (int i = 0; i < 3; i++)
	{
		{
			auto timer = Timer::logScopeTiming("Single threaded CPU");
			for (int i = 0; i < N; i++)
				y[i] = cpuLambda(input[i]);
		}
		{
			auto timer = Timer::logScopeTiming("Multi threaded CPU");
			ParallelUtilsOpenMP::createTeam()->map(N, [&](size_t i) { y[i] = cpuLambda(input[i]); });
		}
		{
			auto timer = Timer::logScopeTiming("GPU");
			int blockSize = 256;
			int numBlocks = std::min(1 << 15, ceilDivision(N, blockSize));
			opCuda << <numBlocks, blockSize >> > (N, inputGPU, yGPU, gpuLambda);
			cudaDeviceSynchronize();
		}
		{
			auto timer = Timer::logScopeTiming("GPU 2");
			opCudaNested << <256, 1 >> > (N, inputGPU, yGPU, gpuLambda);
			cudaDeviceSynchronize();
		}
		{
			auto timer = Timer::logScopeTiming("GPU 3");
			ParallelUtilsCuda::createTeam()->map(N, [=] __device__ (int i) { yGPU[i] = gpuLambda(inputGPU[i]); });
		}
	}

	{
		auto timer = Timer::logScopeTiming("Transfer GPU to CPU");
		cudaMemcpy(control.data(), yGPU, control.size() * sizeof(float), cudaMemcpyDeviceToHost);
	}

	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
	{
		float error = std::abs(control[i] - y[i]);
		// if (error > 0.000001f)
		//	std::cout << i << " " << y[i] << " " << control[i] << std::endl;
		maxError = std::max(maxError, error);
	}
	std::cout << "Max error CPU vs GPU: " << maxError << std::endl;

	// Free memory
	cudaFree(inputGPU);
	cudaFree(yGPU);

	std::cin.get();

	return 0;
}