#!/usr/bin/env bash
set -e

# Activate the Intel oneAPI environment so icx/icpx, oneDNN and oneMKL are on
# PATH / LD_LIBRARY_PATH / CMAKE_PREFIX_PATH for whatever command runs next.
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
fi

exec "$@"
