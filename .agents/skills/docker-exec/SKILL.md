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


## Restrictions

- DO NOT run commands on the host.
- ALWAYS preface a command with `docker exec`