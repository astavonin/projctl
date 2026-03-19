"""Merge request creation handler."""

import logging
import subprocess
from typing import List

logger = logging.getLogger(__name__)


def _build_create_mr_cmd(args) -> List[str]:
    """Build the glab mr create command from parsed arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        List of command tokens ready for subprocess.
    """
    cmd = ["glab", "mr", "create"]

    if args.title:
        cmd.extend(["--title", args.title])

    if args.description:
        cmd.extend(["--description", args.description])

    for flag in ["draft", "fill", "web"]:
        if getattr(args, flag, False):
            cmd.append(f"--{flag}")

    for assignee in args.assignee or []:
        cmd.extend(["--assignee", assignee])

    for reviewer in args.reviewer or []:
        cmd.extend(["--reviewer", reviewer])

    for label in args.label or []:
        cmd.extend(["--label", label])

    if args.milestone:
        cmd.extend(["--milestone", args.milestone])

    if args.target_branch:
        cmd.extend(["--target-branch", args.target_branch])

    return cmd


def cmd_create_mr(args) -> int:
    """Handle the 'create-mr' subcommand - create merge request from current branch.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        cmd = _build_create_mr_cmd(args)

        if args.dry_run:
            print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            return 0

        logger.debug("Executing: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        logger.info("✓ Merge request created")
        print(result.stdout.strip())
        return 0

    except subprocess.CalledProcessError as err:
        logger.error("Command failed: %s", err.stderr)
        return 1
    except (ValueError, OSError) as err:
        logger.error("Error: %s", err)
        return 1
