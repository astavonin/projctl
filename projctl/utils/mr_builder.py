"""Shared MR/PR command argument builders."""

from typing import Any, List


def append_common_mr_flags(cmd: List[str], args: Any) -> None:
    """Append shared assignee/reviewer/label/milestone flags to a command list.

    Used by both the GitLab (glab mr create) and GitHub (gh pr create) handlers
    to avoid duplicating flag-building logic.  Target-branch flag is intentionally
    excluded because the two CLIs use different flag names (--target-branch vs
    --base).

    Args:
        cmd: Mutable command list to extend in-place.
        args: Parsed CLI arguments; expected to expose ``assignee``, ``reviewer``,
              ``label``, and ``milestone`` attributes.
    """
    for assignee in args.assignee or []:
        cmd.extend(["--assignee", assignee])

    for reviewer in args.reviewer or []:
        cmd.extend(["--reviewer", reviewer])

    for label in args.label or []:
        cmd.extend(["--label", label])

    if args.milestone:
        cmd.extend(["--milestone", args.milestone])
