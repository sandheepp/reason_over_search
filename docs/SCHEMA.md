---
title: Wiki schema
tags: [meta, schema]
source: internal
created: 2026-05-06
updated: 2026-05-06
---

# Wiki schema

This wiki sits at `docs/`. It follows Andrej Karpathy's [personal wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) workflow, adapted for a multi-strand thesis project. The schema below tells humans and LLMs how to read, write, and maintain the wiki.

## Directory roles

- [`raw/`](raw/README.md) - immutable source documents. Papers, screenshots, fetched articles, supervisor emails. Never edited; only added to. Citation target.
- [`log.md`](log.md) - append-only chronological record. One entry per ingest, decision, learning, or lint pass. Newest day on top.
- **Topical sections** (each is a wiki "section"; treat as siblings, not as an inheritance hierarchy):
  - [`report/`](report/) - thesis-facing record (supervisor meetings, progress reports, results bundles for v0/v1).
  - [`research/`](research/) - literature reviews and algorithm/systems engineering notes.
  - [`papers/`](papers/) - deep, single-paper ingest notes (one file per paper). Companion to `research/SURVEY.md`. Convention: filename is `<arxiv-id>_<slug>.md`; PDFs live in [`raw/papers/`](raw/papers/). See [`papers/README.md`](papers/README.md).
  - [`training/`](training/) - Phase-2 NeMo-RL training pipeline (data, configs, smoke results, knobs).
  - [`milestone_one/`](milestone_one/), [`milestone_two/`](milestone_two/) - phase-bound runbooks and comparison plans.
  - [`eval/`](eval/) - evaluation pipeline operations and audits.
  - [`retriever/`](retriever/) - FAISS retriever architecture and benchmarking.
  - [`edu/`](edu/) - deep-dives explaining the *why* behind hyperparameters (RNG, batch math, GRPO lifecycle).
  - [`setup/`](setup/) - bootstrap and hardware guides.
  - [`internship/`](internship/) - Alstom internship strand.
  - [`sankar/`](sankar/) - CV / job-search / personal context.
  - [`zeta_alpha/`](zeta_alpha/) - internship-related primer + indexed content.
  - [`archive/`](archive/) - discarded experiments and historical snapshots.
- **Index pages** within each section: `README.md` or `00_INDEX.md` is the section landing page; `CONVERSATION_CONTEXT.md` (where present) is the long-running state snapshot for that strand.

## Frontmatter contract

Every page outside `raw/` should have a YAML frontmatter block:

```yaml
---
title: <human-readable title>
tags: []
source: <relative path under raw/, full URL, or "internal">
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

- `title` - readable, not the slug.
- `tags` - populated organically as pages are touched. No mandatory taxonomy. Common tags so far: `meta`, `schema`, `results`, `recipe`, `eval`, `training`, `survey`, `decision`.
- `source` - `internal` for synthesis pages; relative path under `raw/` for ingest derivatives; full URL for direct external citations.
- `created` / `updated` - ISO date. `updated` should change when the page is meaningfully edited (not for typo fixes).

Pages are skipped by the lint if they already have frontmatter; the init script (`scripts/wiki_frontmatter_init.py`) seeds skeletons for new pages without touching existing ones.

## Linking conventions

- Use **relative markdown links**: e.g. from this file, `[label](research/SURVEY_FOCUSED.md)`. No Obsidian-style `[[wikilinks]]` (so the wiki renders correctly on GitHub, in plain editors, and in Obsidian alike).
- Anchor specific lines in code with `[file:line](../../training/src/foo.py#L42)` per the project's existing convention.
- Cite sources back to `raw/` with the full relative path. The lint catches broken links.

## Workflow cycles

Three cycles, executed by humans and LLMs together. The LLM does the bulk of the writing; the human curates direction and asks questions.

### Ingest cycle

Triggered when a new source is captured.

1. Human drops the source under `docs/raw/` with the documented naming convention.
2. LLM reads the source. For non-trivial sources, also writes a sibling `.md` in `raw/` capturing provenance (URL, fetch date, why captured).
3. LLM identifies the wiki pages this source affects (1 to N pages across topical sections). For each:
   - If a relevant page exists, append or revise a section, citing the raw file with a relative link.
   - If no page exists and the source is substantial, create a new page in the right topical section with frontmatter.
4. LLM appends an entry to `log.md`:
   ```
   - Ingested: ../raw/2026-05-06_paper-foo.pdf -> research/SURVEY.md, training/PAPER_VS_OURS_TRAINING.md
   ```

### Query cycle

Triggered when the human asks a question or wants synthesis across pages.

1. LLM reads the relevant section's `README.md` / `CONVERSATION_CONTEXT.md` first (these are the indexes).
2. LLM searches topical subdirs for pages that touch the question. Cross-section reasoning is allowed and expected.
3. LLM synthesises an answer with `file:line` citations.
4. If the synthesis is non-trivial and likely to be asked again, file it back as a new page or as an explicit section in an existing page. Add a `log.md` entry:
   ```
   - Learned: <one-liner>, filed in <page>
   ```

### Lint cycle

Triggered manually (`python scripts/wiki_lint.py docs/`) or before a significant docs PR.

Reports four categories:
- **Broken links**: hard error; fix or remove.
- **Orphans**: pages that nothing else links to. Triage: link them in, archive them, or accept as a deliberate root.
- **Stale claims**: pages with `updated` older than 90 days that mention a date / version / "as of" string from outside that window. Triage: update or annotate.
- **Frontmatter validity**: pages outside `raw/` that lack a parseable frontmatter block.

Lint never auto-fixes. It surfaces; the LLM (or human) resolves.

## When to create a new page vs append

- **Append** when the new content directly extends an existing page's narrative (e.g., results from a follow-up run on the same recipe).
- **New page** when the content is a distinct artifact (a new ablation block, a new paper review, a new audit). Cross-link both ways.
- Don't fragment for the sake of frontmatter. A 500-word note is fine inline if it doesn't deserve its own page.

## What the wiki is NOT

- It is not a replacement for code documentation. Code-internal docs live next to code; the wiki references them.
- It is not a personal knowledge base across all of life. The `sankar/` strand is the one exception; everything else is project-bound.
- It is not append-only outside `log.md`. Topical pages are revised in place; the git history is the audit trail.
- It is not a database. No schemas to migrate, no queries beyond grep + Obsidian Dataview.

## Pointers

- Project schema for code + wiki: [`../claude/CLAUDE.md`](../claude/CLAUDE.md).
- Daily catch-up entry point: [`TODO_2026-05-04.md`](todo/TODO_2026-05-04.md).
- Per-strand state snapshots: `report/CONVERSATION_CONTEXT.md`, `training/CONVERSATION_CONTEXT.md`, `research/CONVERSATION_CONTEXT.md`, `internship/CONVERSATION_CONTEXT.md`, `sankar/00_CONVERSATION_CONTEXT.md`.
