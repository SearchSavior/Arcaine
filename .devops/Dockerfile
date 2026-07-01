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
ARG ONEDNN_VERSION=v3.14

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
        -DONEDNN_EXPERIMENTAL_GROUPED_MEMORY=ON \
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
