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
#include <thrust/system/omp/execution_policy.h>
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
	uint64_t offset;
	int numElements;
};

int main(void)
{
	int N = 1 << 22;
	std::cout << "Number of elements " << N << std::endl;

	std::vector<InputData> cpuInput(N);
	std::vector<int> cpuNeighborIndices;
	std::vector<float> cpuFloats(N);
	std::vector<float> cpuOutput(N);
	std::vector<float> control(N);

	InputData* gpuInput;
	int* gpuNeighborIndices;
	float* gpuFloats;
	float* gpuOutput;
	cudaMalloc(&gpuInput, N * sizeof(InputData));
	cudaMalloc(&gpuFloats, N * sizeof(float));
	cudaMalloc(&gpuOutput, N * sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++)
	{
		int numElements = 20 + i % 11;
		cpuInput[i].offset = cpuNeighborIndices.size();
		cpuFloats[i] = (i % 52) * 0.1f;
		int minI = std::max(0, i - numElements / 2);
		int maxI = std::min(N, i + numElements / 2);
		cpuInput[i].numElements = maxI - minI;
		for (int index = minI; index < maxI; index++)
		{
			cpuNeighborIndices.push_back(index);
		}
	}

	cudaMemcpy(gpuInput, cpuInput.data(), N * sizeof(InputData), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuFloats, cpuFloats.data(), N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&gpuNeighborIndices, cpuNeighborIndices.size() * sizeof(int));
	cudaMemcpy(gpuNeighborIndices, cpuNeighborIndices.data(), cpuNeighborIndices.size() * sizeof(int), cudaMemcpyHostToDevice);

	omp_set_num_threads(omp_get_num_procs());
	std::cout << "OMP num threads: " << omp_get_max_threads() << std::endl;

	auto gpuLambda = [=] __device__(float* particles, const InputData& neighbors)
	{
		float accu = 0.0f;
		int* neighborListEnd = gpuNeighborIndices + neighbors.offset + neighbors.numElements;
		for (int* it = gpuNeighborIndices + neighbors.offset; it != neighborListEnd; it++)
		{
			float data = particles[*it];
			if (data > 0.0f)
				accu += 1.0f / std::sqrt(data);
		}
		return accu;
	};
	auto cpuLambda = [&](float* particles, const InputData& neighbors)
	{
		float accu = 0.0f;
		int* neighborListEnd = (int*)cpuNeighborIndices.data() + neighbors.offset + neighbors.numElements;
		for (int* it = (int*)cpuNeighborIndices.data() + neighbors.offset; it != neighborListEnd; it++)
		{
			float data = particles[*it];
			if (data > 0.0f)
				accu += 1.0f / std::sqrt(data);
		}
		return accu;
	};

	auto cpuLambda2 = [&] (int i)
	{
		float accu = 0.0f;
		int numNeighbors = 20 + i % 11;
		int halfNeighbors = numNeighbors / 2;
		int minI = max(0, i - halfNeighbors);
		int maxI = min(N, i + halfNeighbors);
		for (int index = minI; index < maxI; index++)
		{
			float data = cpuFloats[index];
			if (data > 0.0f)
				accu += 1.0f / std::sqrt(data);
		}
		return accu;
	};

	auto gpuLambda2 = [=] __device__(int i)
	{
		float accu = 0.0f;
		int numNeighbors = 20 + i % 11;
		int halfNeighbors = numNeighbors / 2;
		int minI = max(0, i - halfNeighbors);
		int maxI = min(N, i + halfNeighbors);
		for (int index = minI; index < maxI; index++)
		{
			float data = gpuFloats[index];
			if (data > 0.0f)
				accu += 1.0f / std::sqrt(data);
		}
		return accu;
	};

	// for (int i = 0; i < 3; i++)
	{
		{
			auto timer = Timer::logScopeTiming("Single threaded CPU");
			for (int i = 0; i < N; i++)
				cpuOutput[i] = cpuLambda(cpuFloats.data(), cpuInput[i]);
		}
		{
			auto timer = Timer::logScopeTiming("Multi threaded CPU");
			ParallelUtilsOpenMP::createTeam()->map(N, [&](size_t i) { cpuOutput[i] = cpuLambda(cpuFloats.data(), cpuInput[i]); });
		}
		{
			auto timer = Timer::logScopeTiming("Multi threaded CPU 2");
			ParallelUtilsOpenMP::createTeam()->map(N, [&](size_t i) { cpuOutput[i] = cpuLambda2(i); });
		}
		{
			auto timer = Timer::logScopeTiming("GPU");
			ParallelUtilsCuda::createTeam()->map(N, [=] __device__(int i) { gpuOutput[i] = gpuLambda(gpuFloats, gpuInput[i]); });
		}
		{
			auto timer = Timer::logScopeTiming("GPU 2");
			ParallelUtilsCuda::createTeam()->map(N, [=] __device__(int i) { gpuOutput[i] = gpuLambda2(i); });
		}
	}

	{
		auto timer = Timer::logScopeTiming("Transfer GPU to CPU");
		cudaMemcpy(control.data(), gpuOutput, control.size() * sizeof(float), cudaMemcpyDeviceToHost);
	}

	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
	{
		float error = std::abs(control[i] - cpuOutput[i]);
		if (error > 0.000001f)
		{
			std::cout << i << " " << cpuOutput[i] << " " << control[i] << std::endl;
			int a;
			std::cin >> a;
		}
		maxError = std::max(maxError, error);
	}
	std::cout << "Max error CPU vs GPU: " << maxError << std::endl;

	cudaFree(gpuFloats);
	cudaFree(gpuOutput);
	cudaFree(gpuInput);
	cudaFree(gpuNeighborIndices);

	std::cin.get();

	return 0;
}