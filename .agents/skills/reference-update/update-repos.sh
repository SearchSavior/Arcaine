#!/usr/bin/env bash
#
# update-repos.sh -- fetch and fast-forward every git repository under
# reference/ to its upstream branch.
#
# Run from anywhere; the reference directory is resolved relative to this
# script so it works both on the host and inside the container (/workspace).
#
# Usage:
#   bash .agents/skills/reference-update/update-repos.sh
#
# Exit status is non-zero if any repository failed to update.
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REFERENCE_DIR="${REPO_ROOT}/reference"

if [ ! -d "${REFERENCE_DIR}" ]; then
  echo "error: reference directory not found at ${REFERENCE_DIR}" >&2
  exit 1
fi

updated=0
skipped=0
failed=0

for repo in "${REFERENCE_DIR}"/*/; do
  [ -d "${repo}/.git" ] || continue
  name="$(basename "${repo}")"

  branch="$(git -C "${repo}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  if [ -z "${branch}" ] || [ "${branch}" = "HEAD" ]; then
    echo "skip   ${name} (detached HEAD)"
    skipped=$((skipped + 1))
    continue
  fi

  upstream="$(git -C "${repo}" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || true)"
  if [ -z "${upstream}" ]; then
    echo "skip   ${name} (no upstream tracking branch)"
    skipped=$((skipped + 1))
    continue
  fi
  remote="${upstream%%/*}"
  remote_branch="${upstream#*/}"

  # Only the tracked branch (main), no tags, no other branches -- the
  # reference mirrors (e.g. intel-llvm) are huge and we only ever
  # fast-forward the current branch.
  if ! git -C "${repo}" fetch --no-tags "${remote}" "${remote_branch}"; then
    echo "fail   ${name} (fetch failed)"
    failed=$((failed + 1))
    continue
  fi

  if git -C "${repo}" merge --ff-only "${upstream}" >/dev/null 2>&1; then
    echo "ok     ${name} (${branch})"
    updated=$((updated + 1))
  else
    echo "fail   ${name} (cannot fast-forward ${branch}: divergence or local changes)"
    failed=$((failed + 1))
  fi
done

echo
echo "updated: ${updated}  skipped: ${skipped}  failed: ${failed}"
[ "${failed}" -eq 0 ]
