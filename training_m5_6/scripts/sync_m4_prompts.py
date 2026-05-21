#!/usr/bin/env python3
"""Materialise an M4 prompt template into M5.1's prompt files.

The M4 eval pipeline iterates on prompt candidates centrally in
`evaluation_qwen35/flashrag/search_r1/templates.py:QWEN35_TEMPLATES`. To
keep training-time rollouts byte-identical to whichever candidate the M4
sweep locks (mandated by docs/milestone_5/MILESTONE_5_6.md §"Prompt + tag
scheme (carry from M4)"), this script copies the chosen template's text
into `src/prompts/m5_qwen35_{user,system}.txt`, where the training config
points.

Run once when M4 picks a winner; re-run if it changes.

Examples (run from the repo root):

    # Current M4.2 canonical (user-locus, no system, tools=auto):
    python training_m5_6/scripts/sync_m4_prompts.py --mode qwen35_minimal

    # An M4.4 system-locus candidate (system + user "Question: {q}", tools=auto):
    python training_m5_6/scripts/sync_m4_prompts.py --mode qwen35_recall_port

    # List available modes:
    python training_m5_6/scripts/sync_m4_prompts.py --list

Important caveats:

- The training processor calls `prompt.format(question)` (positional), so
  this script substitutes `{prompt}` -> `{}` on write. Don't add new format
  placeholders in M5.1 prompts without also updating the processor.
- Mode `qwen35_minimal_no_system` ALSO requires the training processor to
  stop passing `tools=[SEARCH_TOOL]` to `apply_chat_template` (the M4
  pipeline does that for this mode). That knob does NOT exist yet in the
  processor; if M4.4 picks this mode the processor needs a `pass_tools=False`
  branch added. The script warns when this mode is selected.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_QWEN35_ROOT = REPO_ROOT / "evaluation_qwen35"
if str(EVAL_QWEN35_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_QWEN35_ROOT))

# Imported after sys.path is patched.
from flashrag.search_r1.templates import QWEN35_TEMPLATES  # noqa: E402

OUT_DIR = REPO_ROOT / "training_m5_6" / "src" / "prompts"
USER_TXT = OUT_DIR / "m5_qwen35_user.txt"
SYSTEM_TXT = OUT_DIR / "m5_qwen35_system.txt"

# Modes that require the M4 pipeline to SKIP `tools=[QWEN35_SEARCH_TOOL]`
# (format spec inlined into the user message instead). The training processor
# does not yet honour this; flag and exit non-zero.
MODES_REQUIRING_NO_TOOLS = ("qwen35_minimal_no_system",)


def is_user_locus(template: str) -> bool:
    """User-locus templates contain `{prompt}`; system-locus templates do not."""
    return "{prompt}" in template


def translate_for_positional_format(template: str) -> str:
    """Processor calls `.format(question)` (positional); rewrite `{prompt}` -> `{}`."""
    return template.replace("{prompt}", "{}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--mode", default=None, help="prompt mode key in QWEN35_TEMPLATES")
    ap.add_argument("--list", action="store_true", help="list available modes and exit")
    ap.add_argument(
        "--no-pre-stage-only",
        action="store_true",
        help="(internal) skip writing files; used by tests",
    )
    args = ap.parse_args()

    if args.list:
        print("Available modes:")
        for k in sorted(QWEN35_TEMPLATES):
            t = QWEN35_TEMPLATES[k]
            locus = "user" if is_user_locus(t) else "system"
            extra = " [NEEDS pass_tools=False]" if k in MODES_REQUIRING_NO_TOOLS else ""
            print(f"  {k:<32s}  locus={locus}{extra}")
        return 0

    if not args.mode:
        print("error: --mode is required (use --list to see available modes)", file=sys.stderr)
        return 2

    if args.mode not in QWEN35_TEMPLATES:
        print(f"error: unknown mode {args.mode!r}; use --list to see options", file=sys.stderr)
        return 2

    template = QWEN35_TEMPLATES[args.mode]

    if args.mode in MODES_REQUIRING_NO_TOOLS:
        print(
            f"WARNING: mode {args.mode!r} requires the M4 pipeline to skip\n"
            "  `tools=[QWEN35_SEARCH_TOOL]` in apply_chat_template. The M5.1\n"
            "  training processor currently always passes tools=; using this\n"
            "  mode without a processor edit produces train/eval drift.",
            file=sys.stderr,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    user_locus = is_user_locus(template)
    if user_locus:
        user_text = translate_for_positional_format(template)
        system_text = None
        print(f"mode {args.mode!r} is user-locus")
    else:
        # System-locus: template is the system message; user message is the
        # canonical M4 `Question: {q}\n` (same shape M4's _is_qwen35 branch uses).
        user_text = "Question: {}\n"
        system_text = template
        print(f"mode {args.mode!r} is system-locus")

    USER_TXT.write_text(user_text, encoding="utf-8")
    print(f"  wrote {USER_TXT.relative_to(REPO_ROOT)} ({len(user_text)} chars)")

    if system_text is not None:
        SYSTEM_TXT.write_text(system_text, encoding="utf-8")
        print(f"  wrote {SYSTEM_TXT.relative_to(REPO_ROOT)} ({len(system_text)} chars)")
    elif SYSTEM_TXT.exists():
        # Clear stale system file from a previous system-locus mode.
        SYSTEM_TXT.unlink()
        print(f"  removed stale {SYSTEM_TXT.relative_to(REPO_ROOT)} (mode has no system msg)")

    print()
    print("Config knobs to set in your training YAML:")
    print(f"  data.train.prompt_file: training_m5_6/src/prompts/{USER_TXT.name}")
    if system_text is not None:
        print(f"  data.train.system_prompt_file: training_m5_6/src/prompts/{SYSTEM_TXT.name}")
    else:
        print("  data.train.system_prompt_file: null  # mode has no system message")
    return 0


if __name__ == "__main__":
    sys.exit(main())
