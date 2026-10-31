#!/usr/bin/env bash
#
# docker.sh -- launch the interactive GPU dev container with the model store
# mounted into the workspace.
#
# Usage:
#   ./docker.sh
#
# Requires ARCAINE_MODELS to point at the host model store (see ~/.bashrc);
# it is mounted at /workspace/models inside the container. Override per-run:
#   ARCAINE_MODELS=/path/to/models ./docker.sh
#
# A second model store is mounted at /workspace/models1; it defaults to the
# AutoRound W4A16 Flash-Next checkpoint and can be overridden per-run:
#   ARCAINE_MODELS1=/path/to/other ./docker.sh
#
set -euo pipefail

: "${ARCAINE_MODELS:?ARCAINE_MODELS is not set; export it in ~/.bashrc}"
: "${ARCAINE_MODELS1:=/home/echo/qfn/Qwen3.8-Flash-Next-W4A16-AutoRound}"

# Host render-group GID so the container user can access /dev/dri. Kept local
# to this script rather than committed to ~/.bashrc.
export RENDER_GID="$(getent group render | cut -d: -f3)"

exec docker compose run --rm --service-ports \
  -v "${ARCAINE_MODELS}:/workspace/models" \
  -v "${ARCAINE_MODELS1}:/workspace/models1" \
  dev \
  bash
