"""GitHub pull request creation handler."""

import logging
from typing import List

from ..config import Config
from ..exceptions import PlatformError
from ..utils.gh_runner import run_gh_command
from ..utils.mr_builder import append_common_mr_flags, validate_mr_args

logger = logging.getLogger(__name__)


def _build_create_pr_cmd(args) -> List[str]:
    """Build the gh pr create command from parsed arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        List of command tokens ready for subprocess.
    """
    cmd = ["pr", "create"]

    if args.title:
        cmd.extend(["--title", args.title])

    if args.description:
        cmd.extend(["--body", args.description])

    if getattr(args, "draft", False):
        cmd.append("--draft")

    append_common_mr_flags(cmd, args)

    if args.target_branch:
        cmd.extend(["--base", args.target_branch])

    return cmd


def cmd_create_pr(args, config: Config) -> int:
    """Handle the 'create-mr' subcommand for GitHub — create a pull request.

    Args:
        args: Parsed command-line arguments.
        config: Project configuration used for template validation.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        validate_mr_args(args, config)
        cmd = _build_create_pr_cmd(args)

        if args.dry_run:
            print(f"[DRY RUN] Would execute: gh {' '.join(cmd)}")
            return 0

        output = run_gh_command(cmd)
        logger.info("[github] Pull request created")
        print(output)
        return 0

    except PlatformError as err:
        logger.error("[github] %s", err)
        return 1
    except (ValueError, OSError) as err:
        logger.error("[github] Error: %s", err)
        return 1
