#!/usr/bin/env bash
#
# bootstrap.sh -- generates the oneDNN + SYCL (Intel GPU) C++ dev environment.
# Run from an empty project directory:  bash bootstrap.sh
#
set -euo pipefail

echo "Scaffolding oneDNN + SYCL dev environment into $(pwd) ..."

mkdir -p .devops src

# ---- .devops/Dockerfile ------------------------------------------
cat > .devops/Dockerfile << 'DOCKERFILE_EOF'
# syntax=docker/dockerfile:1
#
# Development image for a C++ project using oneDNN with SYCL on Intel GPUs.
#
# oneDNN is BUILT FROM SOURCE here (not the binary package). This is the
# oneDNN maintainers' recommended path as of the 2026.0 release: serious SYCL
# consumers (PyTorch, TensorFlow, OpenVINO) build from source to control the
# GPU and threading runtimes, and mixing the binary distribution with an
# open-source build causes SYMBOL CONFLICTS. Building from source also means
# the base image version no longer matters for oneDNN (2026.0 dropped the
# binary from Deep Learning Essentials, but we don't depend on it).
#
# The base still carries what we DO need: the DPC++/SYCL compiler (icx/icpx)
# and oneMKL.

ARG ONEAPI_VERSION=2025.3.3-0-devel-ubuntu24.04

# ==========================================================================
# Stage 1: build oneDNN from source with SYCL CPU + GPU runtimes
# ==========================================================================
FROM intel/deep-learning-essentials:${ONEAPI_VERSION} AS onednn-build

ARG DEBIAN_FRONTEND=noninteractive

# Pin the oneDNN release. NOTE: confirm the current tag at
#   https://github.com/uxlfoundation/oneDNN/releases
# (releases land roughly quarterly; v3.5 confirmed, 2026 releases scheduled
# Feb/May/Aug). Bump this arg to the version you want.
ARG ONEDNN_VERSION=v3.12

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    # The base image already exports the oneAPI compiler paths. Sourcing
    # setvars.sh under /bin/sh with `set -e` exits early on current images.
    git clone --depth 1 --branch "${ONEDNN_VERSION}" \
        https://github.com/uxlfoundation/oneDNN.git /tmp/oneDNN; \
    cmake -S /tmp/oneDNN -B /tmp/oneDNN/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=icx \
        -DCMAKE_CXX_COMPILER=icpx \
        -DDNNL_CPU_RUNTIME=SYCL \
        -DDNNL_GPU_RUNTIME=SYCL \
        -DDNNL_BUILD_TESTS=OFF \
        -DDNNL_BUILD_EXAMPLES=OFF \
        -DCMAKE_INSTALL_PREFIX=/opt/onednn; \
    cmake --build /tmp/oneDNN/build -j"$(nproc)"; \
    cmake --install /tmp/oneDNN/build; \
    rm -rf /tmp/oneDNN

# ==========================================================================
# Stage 2: the dev image you actually work in
# ==========================================================================
FROM intel/deep-learning-essentials:${ONEAPI_VERSION} AS dev

ARG DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------------
# 1. Base build tooling
# --------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git pkg-config gdb \
        ca-certificates gnupg wget curl \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# 2. Bring in the source-built oneDNN (headers + libs + DNNLConfig.cmake)
# --------------------------------------------------------------------------
COPY --from=onednn-build /opt/onednn /opt/onednn
ENV CMAKE_PREFIX_PATH=/opt/onednn:${CMAKE_PREFIX_PATH} \
    LD_LIBRARY_PATH=/opt/onednn/lib:${LD_LIBRARY_PATH}

# --------------------------------------------------------------------------
# 3. Intel GPU userspace runtime (so SYCL kernels can target the GPU)
#    The base image already includes the current Intel OpenCL and Level Zero GPU
#    runtime stack. Install only the small tools/headers we need here; pulling an
#    extra Intel GPU repo can mix incompatible libigc/libigdfcl package families.
# --------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        libze-dev \
        clinfo \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# 4. Rust toolchain (LOW priority -- the Rust layer will bind a C `extern "C"`
#    shim over your C++ later. Installed so the container is ready when you
#    get to it. Disable with --build-arg INSTALL_RUST=0.)
# --------------------------------------------------------------------------
ARG INSTALL_RUST=1
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN if [ "$INSTALL_RUST" = "1" ]; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
          | sh -s -- -y --no-modify-path --default-toolchain stable; \
        rustup component add rust-src; \
    fi

# --------------------------------------------------------------------------
# 5. Activate the oneAPI environment (compilers, oneMKL) for every shell.
#    oneDNN is already on CMAKE_PREFIX_PATH / LD_LIBRARY_PATH via the ENV above.
# --------------------------------------------------------------------------
COPY .devops/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
DOCKERFILE_EOF

# ---- .devops/entrypoint.sh ---------------------------------------
cat > .devops/entrypoint.sh << 'ENTRYPOINT_EOF'
#!/usr/bin/env bash
set -e

# Activate the Intel oneAPI environment so icx/icpx, oneDNN and oneMKL are on
# PATH / LD_LIBRARY_PATH / CMAKE_PREFIX_PATH for whatever command runs next.
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
fi

exec "$@"
ENTRYPOINT_EOF
chmod +x .devops/entrypoint.sh

# ---- docker-compose.yml ------------------------------------------
cat > docker-compose.yml << 'COMPOSE_EOF'
# docker-compose.yml -- interactive GPU development environment.
#
# Usage:
#   docker compose build
#   docker compose run --rm dev          # drops you into a shell in /workspace
# then inside the container, see README.md for the cmake build commands.

services:
  dev:
    build:
      context: .
      dockerfile: .devops/Dockerfile
      args:
        ONEAPI_VERSION: 2025.3.3-0-devel-ubuntu24.04
        ONEDNN_VERSION: v3.12
        INSTALL_RUST: "1"
    image: myproject-dev:latest

    # --- Intel GPU passthrough --------------------------------------------
    # Exposes the host's render/card device nodes to the container so SYCL can
    # target the GPU. Requires the host to have the Intel GPU kernel driver
    # (i915 or xe) loaded and /dev/dri present.
    devices:
      - /dev/dri:/dev/dri
    group_add:
      # Only needed when running as a NON-root user inside the container.
      # The 'render' group on the HOST owns /dev/dri/renderD*. Find its GID:
      #   getent group render | cut -d: -f3
      # then export it before `docker compose ...`, e.g. RENDER_GID=110.
      - "${RENDER_GID:-render}"

    # --- Source + build artifacts -----------------------------------------
    volumes:
      - .:/workspace:cached          # edit on host, build in container
    working_dir: /workspace

    # Keep an interactive shell available.
    stdin_open: true
    tty: true
    command: bash
COMPOSE_EOF

# ---- CMakeLists.txt ----------------------------------------------
cat > CMakeLists.txt << 'CMAKE_EOF'
cmake_minimum_required(VERSION 3.20)
project(myproject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # for clangd / IDE tooling

# Build with the Intel DPC++/SYCL compiler. Configure with:
#   cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx
#
# oneDNN is built from source and installed to /opt/onednn in the image,
# which is on CMAKE_PREFIX_PATH (set via ENV in the Dockerfile), so this
# find_package picks up its DNNLConfig.cmake automatically.
find_package(dnnl CONFIG REQUIRED)

add_executable(gpu_check src/gpu_check.cpp)

# -fsycl on both compile and link so the same compiler builds device code.
target_compile_options(gpu_check PRIVATE -fsycl)
target_link_options(gpu_check PRIVATE -fsycl)

target_link_libraries(gpu_check PRIVATE DNNL::dnnl)
CMAKE_EOF

# ---- src/gpu_check.cpp -------------------------------------------
cat > src/gpu_check.cpp << 'GPUCHECK_EOF'
// gpu_check.cpp
// Minimal sanity check: confirms oneDNN can see and create a GPU engine.
// Build via the project CMake, then run ./build/gpu_check inside the
// container (with /dev/dri passed through). Verifies the whole chain:
// SYCL runtime -> Intel GPU userspace -> oneDNN GPU engine.

#include <iostream>
#include "oneapi/dnnl/dnnl.hpp"

int main() {
    using namespace dnnl;
    try {
        const auto gpu_count = engine::get_count(engine::kind::gpu);
        std::cout << "oneDNN GPU engines available: " << gpu_count << "\n";

        if (gpu_count > 0) {
            engine eng(engine::kind::gpu, 0);
            std::cout << "Created a oneDNN GPU engine successfully.\n";
        } else {
            std::cout << "No GPU engine found; falling back to CPU.\n";
            engine eng(engine::kind::cpu, 0);
            std::cout << "Created a oneDNN CPU engine successfully.\n";
        }
    } catch (const dnnl::error &e) {
        std::cerr << "oneDNN error (status " << static_cast<int>(e.status)
                  << "): " << e.message << "\n";
        return 1;
    }
    return 0;
}
GPUCHECK_EOF

# ---- README.md ---------------------------------------------------
cat > README.md << 'README_EOF'
# oneDNN + SYCL (Intel GPU) C++ dev environment

A containerized development setup for a C++ project that uses **oneDNN** with
**SYCL on Intel GPUs**. A Rust layer is planned on top but is low priority for
now; the image ships a Rust toolchain so it's ready when you get there.

## Layout

```
.
├── .devops/
│   ├── Dockerfile        # dev image: SYCL toolchain + oneDNN + Intel GPU userspace + Rust
│   └── entrypoint.sh     # sources the oneAPI environment for every shell
├── docker-compose.yml    # GPU passthrough + source bind mount
├── CMakeLists.txt        # find_package(dnnl) + SYCL flags
└── src/
    └── gpu_check.cpp      # oneDNN GPU engine sanity check
```

## Host prerequisites (Linux only)

GPU passthrough needs a Linux host with an Intel GPU:

1. Intel GPU kernel driver loaded (`i915` or `xe`) and `/dev/dri` present.
2. Your host user in the `render` (and usually `video`) group.
3. Docker + the Compose plugin.

Confirm the host sees the GPU:

```bash
ls -l /dev/dri              # expect renderD128, card0, ...
getent group render         # note the GID for RENDER_GID below
```

## Build & run

```bash
# Optional: match the container's GPU group to the host's render GID
export RENDER_GID=$(getent group render | cut -d: -f3)

docker compose build
docker compose run --rm dev          # interactive shell in /workspace
```

Inside the container:

```bash
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx
cmake --build build -j"$(nproc)"
./build/gpu_check
```

Expected output when the GPU is visible:

```
oneDNN GPU engines available: 1
Created a oneDNN GPU engine successfully.
```

If it reports 0 GPU engines, the chain falls back to CPU — check that
`/dev/dri` is passed through and run `clinfo` / `sycl-ls` in the container to
see what devices the runtime detects.

## Notes

- **oneDNN is built from source** (SYCL CPU + GPU runtimes) in a builder stage
  and installed to `/opt/onednn`. This is the oneDNN maintainers' recommended
  path as of 2026.0: source builds let you control the GPU/threading runtimes,
  and mixing the *binary* oneDNN distribution with an open-source build causes
  **symbol conflicts**. It also decouples you from the base image version —
  2026.0 dropped binary oneDNN from Deep Learning Essentials, but we don't use
  it. Pin the oneDNN release via `ONEDNN_VERSION` and the base via
  `ONEAPI_VERSION` (you can move the base to a 2026.x tag freely now).
- **Build config**: `-DDNNL_CPU_RUNTIME=SYCL -DDNNL_GPU_RUNTIME=SYCL` with
  `icx`/`icpx`. If you later want a different mix (e.g. OpenMP/TBB CPU runtime,
  or OpenCL GPU runtime), change those flags in the builder stage.
- **Intel GPU userspace package names** in the graphics repo change over time.
  If that Dockerfile layer fails, consult https://dgpu-docs.intel.com for the
  current package set, or pin `.deb` URLs from the intel/compute-runtime
  releases (the approach llama.cpp's Dockerfile takes).
- **Rust**: when you add it, keep oneDNN on the C++ side. Expose a narrow
  `extern "C"` interface from your C++ and bind *that* from Rust (bindgen or
  cxx) — the Rust crate never links `dnnl` directly. Build wiring can be
  glued together with `corrosion` so one `cmake --build` drives both.
```
README_EOF

cat <<"DONE"

Done. Project scaffolded:
  .devops/Dockerfile     .devops/entrypoint.sh
  docker-compose.yml     CMakeLists.txt
  src/gpu_check.cpp      README.md

Next steps:
  1. Confirm the current oneDNN tag at
       https://github.com/uxlfoundation/oneDNN/releases
     and set ONEDNN_VERSION in docker-compose.yml if needed.
  2. export RENDER_GID=$(getent group render | cut -d: -f3)
  3. docker compose build        # first build compiles oneDNN; slow, then cached
  4. docker compose run --rm dev  # interactive shell in /workspace
     then: cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx \
           && cmake --build build -j"$(nproc)" && ./build/gpu_check
DONE
