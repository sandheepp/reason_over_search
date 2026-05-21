---
title: TODO index
tags: [todo]
source: internal
created: 2026-05-10
updated: 2026-05-10
---

# `docs/todo/` — TODO tracking

Files and jobs:

| File | Type | Purpose |
|---|---|---|
| [`TODO.md`](TODO.md) | evergreen | Quick-glance pending-work list. New TODOs land here; close by deleting the line. |
| [`TODO_2026-05-10.md`](TODO_2026-05-10.md) | dated catch-up | Current snapshot of project state + active path. Refresh by creating a new dated file when state shifts materially. |
| [`TODO_2026-05-04.md`](TODO_2026-05-04.md) | dated catch-up | Frozen historical handoff (Jose/Sanju onboarding). Kept because the recipe-ablation priority list it contains is still valid post-M5. |
| [`HANDOFF_M4.4_2026-05-10.md`](HANDOFF_M4.4_2026-05-10.md) | scoped handoff | Cross-machine handoff for the M4.4 prompt-search sub-phase. Read first when picking up M4.4 on a fresh box. |

When the user says "add to TODOs" / "track this for later" without specifying a file, write to [`TODO.md`](TODO.md). Date-stamped catch-up files (`TODO_YYYY-MM-DD.md`) are point-in-time snapshots; `HANDOFF_*.md` files are scoped, sub-phase-specific bring-up notes.
