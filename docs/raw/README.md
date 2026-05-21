---
title: raw/ source-of-truth dump
tags: [meta, schema]
source: internal
created: 2026-05-06
updated: 2026-05-06
---

# `docs/raw/` - immutable source documents

This folder is the wiki's source-of-truth dump. Anything in here is treated as an artifact, not a working document.

## What lives here

- PDFs of papers we cite.
- Screenshots (W&B plots, terminal output, errors worth keeping).
- Fetched articles (HTML or markdown), gist exports, blog snapshots.
- Supervisor / advisor emails or meeting recordings (text exports).
- Raw experiment exports (CSVs, JSON dumps) that don't already live next to their generating code.

## Rules

1. **Never edit a file in `raw/`.** If a source needs correction, add a sibling `.md` note explaining the issue; leave the original alone.
2. **Naming**: `YYYY-MM-DD_short-slug.ext`. The date is when the source was captured, not when it was written. Use lowercase, hyphens for spaces.
3. **Provenance**: for non-obvious sources (URLs, emails), drop a sibling `.md` with the same slug:

   ```markdown
   ---
   title: <human title>
   source: <URL or origin>
   captured: YYYY-MM-DD
   captured_by: <who>
   ---

   Why this was kept and what wiki page(s) reference it.
   ```

4. **Don't re-derive**: if a paper is already in `docs/research/` as a structured note, don't also dump the PDF here unless the PDF itself is the artifact (e.g., the version we cite in the thesis).

## How `raw/` interacts with the wiki

The ingest cycle (see [`../SCHEMA.md`](../SCHEMA.md)):

1. Human drops a source here.
2. LLM reads it, files distilled facts into one or more topical wiki pages (under `docs/research/`, `docs/training/`, etc.).
3. LLM appends a one-line entry to [`../log.md`](../log.md) with the path of the source and the pages it touched.

The `raw/` file is the citation target for any claim that came from it; wiki pages should link back via `[source](../raw/<file>)`.

## What does NOT belong here

- Working notes. Those go in topical subdirs.
- The `CONVERSATION_CONTEXT.md` snapshots. Those are wiki pages.
- Generated artifacts that have a canonical home in code (`evaluation_search_r1/results/`, W&B run dirs, etc.). Link to them from a wiki page; don't copy them here.
