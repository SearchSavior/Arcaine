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
set -euo pipefail

: "${ARCAINE_MODELS:?ARCAINE_MODELS is not set; export it in ~/.bashrc}"

# Host render-group GID so the container user can access /dev/dri. Kept local
# to this script rather than committed to ~/.bashrc.
export RENDER_GID="$(getent group render | cut -d: -f3)"

exec docker compose run --rm --service-ports \
  -v "${ARCAINE_MODELS}:/workspace/models" \
  dev \
  bash
