/**
 * @file parallel.h
 * @brief Portable parallel_for and parallel_reduce with a persistent thread pool.
 *
 * Threads are created once and reused across all calls.
 * No external dependencies — works on macOS, Linux, Windows.
 */
#pragma once

#include <thread>
#include <vector>
#include <functional>
#include <algorithm>
#include <mutex>
#include <condition_variable>

namespace sloshing {

/**
 * @brief Persistent thread pool with generation-based synchronization.
 *
 * Singleton pool that creates threads once at startup and reuses them
 * for all parallel operations. Uses condition variables (no busy-wait)
 * and a caller-participates pattern where the calling thread runs the
 * last chunk of work.
 */
class ThreadPool {
public:
    /// @brief Get the singleton thread pool instance.
    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    /// @brief Number of worker threads (including the caller thread).
    unsigned size() const { return n_workers_; }

    /**
     * @brief Distribute a range [begin, end) across worker threads.
     *
     * The callable func(tid, lo, hi) is invoked once per thread with
     * a disjoint sub-range. The calling thread participates as the
     * last worker. Blocks until all threads complete.
     *
     * @tparam Func Callable with signature void(unsigned tid, int lo, int hi).
     * @param begin Start of the iteration range.
     * @param end End of the iteration range (exclusive).
     * @param func Work function invoked per thread with its sub-range.
     */
    template <typename Func>
    void run(int begin, int end, Func&& func) {
        int count = end - begin;
        unsigned n = std::min(n_workers_, static_cast<unsigned>(count));

        // Wait for all workers to be idle from the previous run
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_idle_.wait(lock, [this] { return idle_count_ == n_workers_ - 1; });

            // Publish new task
            task_begin_ = begin;
            task_count_ = count;
            task_n_ = n;
            task_ = [&func](unsigned tid, int lo, int hi) { func(tid, lo, hi); };
            idle_count_ = 0;
            generation_++;
        }
        cv_start_.notify_all();

        // Caller runs the last chunk
        int caller_lo, caller_hi;
        computeRange(n - 1, begin, count, n, caller_lo, caller_hi);
        func(n - 1, caller_lo, caller_hi);

        // Wait for all workers to become idle (finished + back in wait state)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_idle_.wait(lock, [this] { return idle_count_ == n_workers_ - 1; });
        }

        task_ = nullptr;
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
            generation_++;
        }
        cv_start_.notify_all();
        for (auto& t : workers_) t.join();
    }

private:
    ThreadPool() {
        n_workers_ = std::max(1u, std::thread::hardware_concurrency());
        idle_count_ = 0; // workers will increment as they enter wait state
        for (unsigned i = 0; i < n_workers_ - 1; ++i) {
            workers_.emplace_back([this, i] { workerLoop(i); });
        }
    }

    static void computeRange(unsigned tid, int begin, int count, unsigned n,
                              int& lo, int& hi) {
        int chunk = count / n;
        int remainder = count % n;
        lo = begin;
        for (unsigned t = 0; t < tid; ++t)
            lo += chunk + (t < static_cast<unsigned>(remainder) ? 1 : 0);
        hi = lo + chunk + (tid < static_cast<unsigned>(remainder) ? 1 : 0);
    }

    void workerLoop(unsigned tid) {
        unsigned local_gen = 0;
        while (true) {
            // Signal idle and wait for new work
            {
                std::unique_lock<std::mutex> lock(mutex_);
                idle_count_++;
                cv_idle_.notify_one();

                cv_start_.wait(lock, [this, local_gen] {
                    return generation_ != local_gen || shutdown_;
                });
                local_gen = generation_;
                if (shutdown_) return;
            }

            // Do work if this thread has a chunk (caller handles chunk task_n_-1)
            if (tid + 1 < task_n_) {
                int lo, hi;
                computeRange(tid, task_begin_, task_count_, task_n_, lo, hi);
                task_(tid, lo, hi);
            }

            // Loop back to signal idle
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    unsigned n_workers_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_start_;
    std::condition_variable cv_idle_;
    std::function<void(unsigned, int, int)> task_;
    int task_begin_ = 0;
    int task_count_ = 0;
    unsigned task_n_ = 0;
    unsigned idle_count_ = 0;
    unsigned generation_ = 0;
    bool shutdown_ = false;
};

/// @brief Return the number of threads in the pool.
inline unsigned parallel_thread_count() {
    return ThreadPool::instance().size();
}

/**
 * @brief Parallel for-loop over [begin, end).
 *
 * Distributes iterations across the thread pool. Falls back to
 * sequential execution for single-element ranges or single-thread pools.
 *
 * @tparam Func Callable with signature void(int i).
 * @param begin Start index (inclusive).
 * @param end End index (exclusive).
 * @param func Body to execute for each index.
 */
template <typename Func>
void parallel_for(int begin, int end, Func&& func) {
    int count = end - begin;
    if (count <= 0) return;

    auto& pool = ThreadPool::instance();
    if (pool.size() <= 1 || count <= 1) {
        for (int i = begin; i < end; ++i) func(i);
        return;
    }

    pool.run(begin, end, [&func](unsigned /*tid*/, int lo, int hi) {
        for (int i = lo; i < hi; ++i) func(i);
    });
}

/**
 * @brief Parallel reduction over [begin, end).
 *
 * Each thread accumulates into its own partial result (no false sharing),
 * then partial results are combined with the reduction operator.
 *
 * @tparam T Result type.
 * @tparam Func Callable with signature void(int i, T& accumulator).
 * @tparam ReduceOp Callable with signature T(T, T).
 * @param begin Start index (inclusive).
 * @param end End index (exclusive).
 * @param identity Initial value for each thread's accumulator.
 * @param func Body to execute for each index, accumulating into a thread-local result.
 * @param op Binary operator to combine partial results.
 * @return Combined result across all threads.
 */
template <typename T, typename Func, typename ReduceOp>
T parallel_reduce(int begin, int end, T identity, Func&& func, ReduceOp&& op) {
    int count = end - begin;
    if (count <= 0) return identity;

    auto& pool = ThreadPool::instance();
    if (pool.size() <= 1 || count <= 1) {
        T result = identity;
        for (int i = begin; i < end; ++i) func(i, result);
        return result;
    }

    unsigned n = std::min(pool.size(), static_cast<unsigned>(count));
    std::vector<T> partial(n, identity);

    pool.run(begin, end, [&func, &partial](unsigned tid, int lo, int hi) {
        for (int i = lo; i < hi; ++i) func(i, partial[tid]);
    });

    T result = identity;
    for (unsigned t = 0; t < n; ++t)
        result = op(result, partial[t]);
    return result;
}

} // namespace sloshing
