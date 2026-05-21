#!/usr/bin/env python3
"""Seed YAML frontmatter on wiki pages that lack it.

Walks docs/**/*.md; for every file that does not already start with a `---`
frontmatter block, prepends a skeleton block with `title`, empty `tags`,
`source: internal`, and `created`/`updated` derived from git log.

Idempotent: re-running is a no-op (files with frontmatter are skipped).

Usage:
    python scripts/wiki_frontmatter_init.py            # apply
    python scripts/wiki_frontmatter_init.py --dry-run  # preview
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"
SKIP_DIRS = {"raw"}  # raw/ files are sources, not wiki pages
TODAY = dt.date.today().isoformat()


def has_frontmatter(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("---\n") or stripped.startswith("---\r\n")


def slug_to_title(stem: str) -> str:
    cleaned = re.sub(r"[_\-]+", " ", stem).strip()
    if cleaned.isupper():
        return cleaned
    return cleaned[:1].upper() + cleaned[1:] if cleaned else stem


def git_date(path: Path, kind: str) -> str:
    flag = "--reverse" if kind == "created" else ""
    cmd = ["git", "log"] + ([flag] if flag else []) + [
        "--format=%ad",
        "--date=short",
        "--",
        str(path.relative_to(REPO_ROOT)),
    ]
    try:
        out = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()
    except Exception:
        return TODAY
    if not out:
        return TODAY
    if kind == "created":
        return out.splitlines()[0]
    return out.splitlines()[0]  # most-recent first by default


def build_frontmatter(path: Path) -> str:
    title = slug_to_title(path.stem)
    created = git_date(path, "created")
    updated = git_date(path, "updated")
    return (
        "---\n"
        f"title: {title}\n"
        "tags: []\n"
        "source: internal\n"
        f"created: {created}\n"
        f"updated: {updated}\n"
        "---\n\n"
    )


def iter_targets() -> list[Path]:
    files = []
    for p in DOCS_ROOT.rglob("*.md"):
        rel = p.relative_to(DOCS_ROOT)
        if rel.parts and rel.parts[0] in SKIP_DIRS:
            continue
        files.append(p)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = iter_targets()
    seeded, skipped = 0, 0

    for path in targets:
        text = path.read_text(encoding="utf-8")
        if has_frontmatter(text):
            skipped += 1
            continue
        if not text.strip():
            skipped += 1
            continue
        block = build_frontmatter(path)
        new_text = block + text
        rel = path.relative_to(REPO_ROOT)
        if args.dry_run:
            print(f"[would-seed] {rel}")
        else:
            path.write_text(new_text, encoding="utf-8")
            print(f"[seeded] {rel}")
        seeded += 1

    verb = "would seed" if args.dry_run else "seeded"
    print(f"\n{verb}: {seeded} file(s); skipped (already have frontmatter): {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
