#pragma once

#include <omp.h>
#include <array>
#include <vector>
#include <type_traits>
#include <atomic>
#include <memory>

class ParallelUtilsOpenMP
{
public:
	template<typename Int>
	static Int ceilDivision(Int a, Int b)
	{
		return (a + b - 1) / b;
	}

	class StaticScheduler
	{
		size_t m_NumElements;
		int m_NumThreads;
		size_t m_NumElementsPerThread;

	public:
		StaticScheduler(size_t numElements, int numThreads = omp_get_max_threads()) : m_NumElements(numElements), m_NumThreads(numThreads)
		{
			m_NumElementsPerThread = ceilDivision(numElements, (size_t)numThreads);
		}

		//! Retrieves the range of indices for the given thread index.
		bool getThreadRange(int threadIndex, size_t* start, size_t* end, size_t offset = 0) const
		{
			*start = offset + std::min((size_t)threadIndex * m_NumElementsPerThread, m_NumElements);
			*end = offset + std::min((size_t)(threadIndex + 1) * m_NumElementsPerThread, m_NumElements);
			return (*end > *start);
		}

		template<class F>
		void mapThreadLocal(int threadIndex, const F& mapper)
		{
			size_t localStart, localEnd;
			if (!getThreadRange(threadIndex, &localStart, &localEnd)) return;
			for (size_t j = localStart; j < localEnd; j++)
				mapper(j);
		}

		int numThreads() const { return m_NumThreads; }
	};

	template<class PerThreadData>
	class PerThreadOperation
	{
	public:
		virtual ~PerThreadOperation() {}
		virtual int doWork(int numThreads, int ownThreadNum, PerThreadData& perThreadDat) = 0;
	};

	struct ScheduleHints
	{
		bool m_UseDynamicScheduling = false;
		size_t m_DynamicSchedulingChunkSize = 1;
		ScheduleHints(bool useDynamicScheduling = false, size_t dynamicSchedulingChunkSize = 1) : m_UseDynamicScheduling(useDynamicScheduling), m_DynamicSchedulingChunkSize(dynamicSchedulingChunkSize) {}
		static ScheduleHints dynamic(size_t dynamicSchedulingChunkSize = 1) { return ScheduleHints(true, dynamicSchedulingChunkSize); }
	};

	template<class PerThreadData, class Mapper>
	class MapWithLocalDataOp : public PerThreadOperation<PerThreadData>
	{
	protected:
		size_t m_NumElements;
		Mapper m_Mapper;
		std::vector<std::atomic<size_t>> m_JobCounters;
		ScheduleHints m_ScheduleHints;

	public:
		virtual ~MapWithLocalDataOp() {}
		MapWithLocalDataOp(size_t numElements, const Mapper& mapper, size_t numThreads, const ScheduleHints& scheduleHints = ScheduleHints())
			: m_NumElements(numElements), m_Mapper(mapper), m_JobCounters(scheduleHints.m_UseDynamicScheduling ? numThreads : 0), m_ScheduleHints(scheduleHints)
		{
			for (auto& counter : m_JobCounters)
				counter.store(0);
		}
		virtual int doWork(int numThreads, int ownThreadNum, PerThreadData& perThreadData) override
		{
			if (numThreads == 1)
			{
				for (size_t i = 0; i < m_NumElements; i++)
					m_Mapper(i, perThreadData);
				return 0;
			}
			if (!m_ScheduleHints.m_UseDynamicScheduling || m_NumElements <= (size_t)numThreads)   // Dynamic scheduling makes no sense if there is less data than threads.
			{
				StaticScheduler scheduleHelper(m_NumElements, numThreads);
				#pragma omp for
				for (int i = 0; i < scheduleHelper.numThreads(); i++)
				{
					scheduleHelper.mapThreadLocal(i, [&](size_t j) { m_Mapper(j, perThreadData); });
				}
			}
			else
			{
				size_t numChunks = ceilDivision(m_NumElements, m_ScheduleHints.m_DynamicSchedulingChunkSize);
				StaticScheduler scheduleHelper(numChunks, numThreads);
				size_t threadRangeStart, threadRangeEnd;
				scheduleHelper.getThreadRange(ownThreadNum, &threadRangeStart, &threadRangeEnd);
				int activeThreadNum = ownThreadNum;
				int threadOffset = 1;
				bool skipLeft = false;
				auto getNextNeighborThread = [&]()
				{
					bool leftAvailable = ownThreadNum - threadOffset >= 0;
					bool rightAvailable = ownThreadNum + threadOffset < numThreads;
					if (!leftAvailable && !rightAvailable) return false;
					if (!skipLeft && leftAvailable)
					{
						activeThreadNum = ownThreadNum - threadOffset;
						if (!rightAvailable)
							threadOffset++;
						else skipLeft = true;
						return true;
					}
					activeThreadNum = ownThreadNum + threadOffset++;
					skipLeft = false;
					return true;
				};
				while (true)
				{
					size_t jobId = threadRangeStart + m_JobCounters[activeThreadNum]++;
					if (jobId >= threadRangeEnd)
					{
						if (!getNextNeighborThread()) break;
						scheduleHelper.getThreadRange(activeThreadNum, &threadRangeStart, &threadRangeEnd);
						continue;
					}
					if (m_ScheduleHints.m_DynamicSchedulingChunkSize == 1)
						m_Mapper(jobId, perThreadData);
					else
					{
						size_t elementStart = jobId * m_ScheduleHints.m_DynamicSchedulingChunkSize;
						size_t elementEnd = std::min(m_NumElements, elementStart + m_ScheduleHints.m_DynamicSchedulingChunkSize);
						for (size_t index = elementStart; index < elementEnd; index++)
							m_Mapper(index, perThreadData);
					}
				}
				#pragma omp barrier
			}
			return 0;
		}
	};

	template<class PerThreadData, class FoldFunc>
	class FoldOp : public PerThreadOperation<PerThreadData>
	{
		FoldFunc m_FoldFunc;
		bool m_Ordered = false;
	public:
		virtual int doWork(int numThreads, int, PerThreadData& perThreadData) override
		{
			if (numThreads == 1)
			{
				m_FoldFunc(perThreadData);
				return 0;
			}
			else if (m_Ordered)
			{
				#pragma omp for ordered schedule(static, 1)
				for (int i = 0; i < numThreads; i++)
				{
					#pragma omp ordered
					m_FoldFunc(perThreadData);
				}
			}
			else
			{
				#pragma omp critical
				m_FoldFunc(perThreadData);
			}
			return 0;
		}

	public:
		FoldOp(const FoldFunc& func, bool ordered = false) : m_FoldFunc(func), m_Ordered(ordered) {}
	};

	template<class CreateData>
	using ReturnOfCreateData = typename std::result_of<CreateData(int)>::type;

	template<class CreateThreadLocalDataFunc>
	class Team
	{
		using ThreadLocalData = ReturnOfCreateData<CreateThreadLocalDataFunc>;
		CreateThreadLocalDataFunc m_CreateThreadLocalDataFunc;
		int m_NumThreads;
		std::vector<std::unique_ptr<PerThreadOperation<ThreadLocalData>>> m_List;
		bool m_Executed = false;

	public:
		Team(const CreateThreadLocalDataFunc& func, int numThreads) : m_CreateThreadLocalDataFunc(func), m_NumThreads(numThreads) {}
		~Team() { if (!m_Executed) execute(); }

		//! Performs a parallel map operation on the range [0, numElements). The mapper should take an index and the thread-local data as argument.
		template<class Mapper>
		Team* mapWithLocalData(size_t numElements, const Mapper& mapper, const ScheduleHints& scheduleHints = ScheduleHints())
		{
			m_List.emplace_back(new MapWithLocalDataOp<ThreadLocalData, Mapper>(numElements, mapper, m_NumThreads, scheduleHints));
			return this;
		}
		//! Performs a parallel map operation on the range [0, numElements). The mapper should take an index as argument.
		template<class Mapper>
		Team* map(size_t numElements, const Mapper& mapper, const ScheduleHints& scheduleHints = ScheduleHints())
		{
			return mapWithLocalData(numElements, [mapper](size_t i, ThreadLocalData&) { mapper(i); }, scheduleHints);
		}
		//! Calls the given lambda synchronized for each thread, passing the thread-local data. Intended for fold / reduce operations.
		template<class FoldFunc>
		Team* fold(const FoldFunc& foldFunc, bool ordered = false)
		{
			m_List.emplace_back(new FoldOp<ThreadLocalData, FoldFunc>(foldFunc, ordered));
			return this;
		}
		//! Adds a user-defined operation to the OpList. The CustomAdder class must implement a static function addOp(OpList*, params...).
		template<typename CustomAdder, typename... Params>
		Team* customOp(const Params& ... params)
		{
			CustomAdder::addOp(this, params...);
			return this;
		}

		void execute()
		{
			if (m_NumThreads == 1)
			{   // Single threaded version.
				auto localData = m_CreateThreadLocalDataFunc(0);
				for (size_t i = 0; i < m_List.size(); i++)
					i += m_List[i]->doWork(1, 0, localData);
			}
			else
			{   // Multithreaded version.
				#pragma omp parallel num_threads(m_NumThreads)
				{
					int ownThreadNum = omp_get_thread_num();
					auto localData = m_CreateThreadLocalDataFunc(ownThreadNum);
					for (size_t i = 0; i < m_List.size(); i++)
					{
						i += m_List[i]->doWork(m_NumThreads, ownThreadNum, localData);
					}
				}
			}
			m_Executed = true;
		}
	};

	struct NoThreadLocalData { int operator()(int) const { return 0; } };

	//! @param createThreadLocalDataFunc is a function that initializes and returns the thread-local data passed to the mappers. The thread index is passed as an int parameter to the createData function.
	//! @param numThreads is the number of threads that should be used for the computation.
	template<class CreateThreadLocalDataFunc = NoThreadLocalData>
	static std::unique_ptr<Team<CreateThreadLocalDataFunc>> createTeam(const CreateThreadLocalDataFunc& createThreadLocalDataFunc = NoThreadLocalData(), int numThreads = omp_get_max_threads())
	{
		return std::make_unique<Team<CreateThreadLocalDataFunc>>(createThreadLocalDataFunc, numThreads);
	}
};
