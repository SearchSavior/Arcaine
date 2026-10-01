---
name: reference-update
description: >
  Update every git repository under reference/ to its latest upstream commit.
---

## What it does

Fetch and fast-forward each git checkout under `reference/` to its upstream
branch, reporting per-repository status and a summary of updated/skipped/failed
repositories. Non-git entries and detached-HEAD checkouts are skipped.

## How to use

```
bash .agents/skills/reference-update/update-repos.sh
```

## Restrictions

- Run inside the container (`/workspace`), not on the host.
- Do not force-update or discard local changes; the script only fast-forwards.

## Always

- Stop and ask the user what to do if a repository cannot fast-forward
  (divergence or local changes) or if `reference/` is missing.
