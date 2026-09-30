// gpu_watchdog_check — checks the GPU work watchdog in a separate process.
// It shows that a disabled watchdog and a healthy heartbeat do not stop the
// process. It also shows that a stalled heartbeat stops the process with code
// 70.
//
// Build and run the test in the arcaine-dev-1 container:
//   docker exec arcaine-dev-1 sh -c 'cd /workspace && icpx -O2 -pthread \
//     src/bench/gpu_watchdog_check.cpp \
//     -o /tmp/gpu_watchdog_check && /tmp/gpu_watchdog_check'
#include "../common/gpu/gpu_watchdog.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {
int failures = 0;
void check(const char* name, bool ok) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++failures;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--timeout") {
        GpuWatchdog wd("watchdog-check", 1.0);   // never beat
        std::this_thread::sleep_for(std::chrono::seconds(5));
        std::printf("[FAIL] watchdog did not exit\n");
        return 0;
    }

    // Disabled: must not exit even with no beats.
    { GpuWatchdog wd("disabled", 0.0);
      std::this_thread::sleep_for(std::chrono::milliseconds(1500)); }
    check("disabled watchdog does not exit", true);

    // Healthy: beats every 200 ms for 2 s with a 1 s timeout.
    { GpuWatchdog wd("healthy", 1.0);
      for (int i = 0; i < 10; ++i) {
          wd.beat("loop");
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
      } }
    check("watchdog with regular beats does not exit", true);

    // Stalled: child must exit 70.
    pid_t pid = fork();
    if (pid == 0) {
        execl(argv[0], argv[0], "--timeout", (char*)nullptr);
        _exit(127);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    check("watchdog exits 70 on a stalled heartbeat",
          WIFEXITED(status) && WEXITSTATUS(status) == 70);

    std::printf(failures ? "\n%d FAILURES\n" : "\nall watchdog checks passed\n",
                failures);
    return failures ? 1 : 0;
}
