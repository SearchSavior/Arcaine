---
name: docker-exec
description: >
  Use `docker exec` to run commands in the specified container.
---

Use this container for all development and code changes; DO NOT run commands on the host.

# Notes

- All project dependencies live in the container.
- base directory is /workspace/


## Example to run commands in the container

`docker exec arcaine-dev-run`

## Example to build Arcaine

```
docker exec -i arcaine-dev-1 sh -c 'cmake -B build -G Ninja \
  -DCMAKE_CXX_COMPILER=icpx \
  -DARCAINE_SYCL_TARGETS=intel_gpu_bmg_g31 && \
  cmake --build build -j"$(nproc)"'
```

## Restrictions

- DO NOT run commands on the host.