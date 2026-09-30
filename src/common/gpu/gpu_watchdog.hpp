#pragma once
// GPU work watchdog.
//
// A hung GPU kernel blocks the worker thread in the driver. The thread cannot
// check a deadline. The thread cannot release the model lock. This watchdog
// runs on a separate thread. It watches a progress heartbeat. If the heartbeat
// stops, the watchdog stops the process and writes a message. A fast, visible
// exit is better than a server that accepts requests and never answers.
//
// A timeout of 0 turns the watchdog off. Each heartbeat records the current
// stage. In this way, the log names the operation that stopped.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

class GpuWatchdog {
public:
    GpuWatchdog(std::string name, double timeout_s)
        : name_(std::move(name)), timeout_s_(timeout_s) {
        if (!(timeout_s_ > 0.0)) return;
        last_ms_.store(now_ms(), std::memory_order_relaxed);
        stage_.store("start", std::memory_order_relaxed);
        stop_.store(false, std::memory_order_relaxed);
        worker_ = std::thread([this] { run(); });
    }

    ~GpuWatchdog() { stop(); }

    GpuWatchdog(const GpuWatchdog&) = delete;
    GpuWatchdog& operator=(const GpuWatchdog&) = delete;

    // Record progress. Call this function before and after a long GPU operation.
    void beat(const char* stage) {
        if (!(timeout_s_ > 0.0)) return;
        stage_.store(stage, std::memory_order_relaxed);
        last_ms_.store(now_ms(), std::memory_order_relaxed);
    }

private:
    static int64_t now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    void stop() {
        if (!worker_.joinable()) return;
        stop_.store(true, std::memory_order_relaxed);
        worker_.join();
    }

    void run() {
        const auto poll = std::chrono::milliseconds(200);
        while (!stop_.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(poll);
            const int64_t last = last_ms_.load(std::memory_order_relaxed);
            const double age = (now_ms() - last) / 1e3;
            if (age >= timeout_s_) {
                const char* stage = stage_.load(std::memory_order_relaxed);
                std::fprintf(stderr,
                             "[watchdog] '%s' has no GPU progress for %.1fs at stage "
                             "'%s'. The watchdog stops the process. The model lock is "
                             "released. The GPU can be hung. A host reboot can be "
                             "necessary.\n",
                             name_.c_str(), age, stage ? stage : "?");
                std::fflush(stderr);
                std::_Exit(70);
            }
        }
    }

    std::string name_;
    double timeout_s_ = 0.0;
    std::atomic<int64_t> last_ms_{0};
    std::atomic<const char*> stage_{nullptr};
    std::atomic<bool> stop_{true};
    std::thread worker_;
};
