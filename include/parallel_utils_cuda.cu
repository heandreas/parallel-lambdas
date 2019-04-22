#pragma once

#include <vector>
#include <type_traits>
#include <algorithm>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

template<typename T, typename F>
__global__  void mapKernel(int n, T* pThreadData, F mapper)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	pThreadData += index;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		mapper(i, *pThreadData);
}

class ParallelUtilsCuda
{
public:
	template<typename Int>
	static Int ceilDivision(Int a, Int b)
	{
		return (a + b - 1) / b;
	}

	struct ScheduleHints
	{
		bool m_UseDynamicScheduling = false;
		size_t m_DynamicSchedulingChunkSize = 1;
		ScheduleHints(bool useDynamicScheduling = false, size_t dynamicSchedulingChunkSize = 1) : m_UseDynamicScheduling(useDynamicScheduling), m_DynamicSchedulingChunkSize(dynamicSchedulingChunkSize) {}
		static ScheduleHints dynamic(size_t dynamicSchedulingChunkSize = 1) { return ScheduleHints(true, dynamicSchedulingChunkSize); }
	};

	template<class CreateData>
	using ReturnOfCreateData = typename std::result_of<CreateData(int)>::type;

	template<class CreateThreadLocalDataFunc>
	class Team
	{
		using ThreadLocalData = ReturnOfCreateData<CreateThreadLocalDataFunc>;
		using SelfType = Team<CreateThreadLocalDataFunc>;
		CreateThreadLocalDataFunc m_CreateThreadLocalDataFunc;
		int m_NumThreads;
		thrust::device_vector<ThreadLocalData> m_PerThreadData;

		template<class Mapper>
		Team* _mapWithLocalData(size_t numElements, const Mapper& mapper, const ScheduleHints& scheduleHints = ScheduleHints())
		{
			int blockSize = 256;
			int numBlocks = std::min(m_NumThreads, (int)ceilDivision((int)numElements, blockSize));
			mapKernel << <numBlocks, blockSize >> > ((int)numElements, thrust::raw_pointer_cast(&m_PerThreadData[0]), mapper);
			cudaDeviceSynchronize();
			return this;
		}

	public:
		Team(const CreateThreadLocalDataFunc& func, int numThreads) : m_CreateThreadLocalDataFunc(func), m_NumThreads(numThreads) {}

		//! Performs a parallel map operation on the range [0, numElements). The mapper should take an index and the thread-local data as argument.
		template<class Mapper>
		Team* mapWithLocalData(size_t numElements, const Mapper& mapper, const ScheduleHints& scheduleHints = ScheduleHints())
		{
			if (m_PerThreadData.empty()) m_PerThreadData.resize(m_NumThreads);
			return _mapWithLocalData(numElements, mapper, scheduleHints);
		}
		//! Performs a parallel map operation on the range [0, numElements). The mapper should take an index as argument.
		template<class Mapper>
		Team* map(size_t numElements, const Mapper& mapper, const ScheduleHints& scheduleHints = ScheduleHints())
		{
			return _mapWithLocalData(numElements, [mapper] __device__ (int i, ThreadLocalData&) { mapper(i); }, scheduleHints);
		}
		template<class FoldFunc>
		Team* fold(const FoldFunc& foldFunc, ThreadLocalData& tOut)
		{
			tOut = thrust::reduce(thrust::device, m_PerThreadData.begin(), m_PerThreadData.end(), foldFunc);
			return this;
		}

		//! Adds a user-defined operation to the OpList. The CustomAdder class must implement a static function addOp(OpList*, params...).
		template<typename CustomAdder, typename... Params>
		Team* customOp(const Params& ... params)
		{
			CustomAdder::addOp(this, params...);
			return this;
		}
	};

	struct NoThreadLocalData { int operator()(int) const { return 0; } };

	//! @param createThreadLocalDataFunc is a function that initializes and returns the thread-local data passed to the mappers. The thread index is passed as an int parameter to the createData function.
	//! @param numThreads is the number of threads that should be used for the computation.
	template<class CreateThreadLocalDataFunc = NoThreadLocalData>
	static std::unique_ptr<Team<CreateThreadLocalDataFunc>> createTeam(const CreateThreadLocalDataFunc& createThreadLocalDataFunc = NoThreadLocalData(), int numThreads = 1 << 15)
	{
		return std::make_unique<Team<CreateThreadLocalDataFunc>>(createThreadLocalDataFunc, numThreads);
	}
};
