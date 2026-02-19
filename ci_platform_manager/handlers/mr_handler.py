"""Merge request creation handler."""

import logging
import subprocess

logger = logging.getLogger(__name__)


def cmd_create_mr(args) -> int:
    """Handle the 'create-mr' subcommand - create merge request from current branch.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        # Build glab mr create command
        cmd = ['glab', 'mr', 'create']

        if args.title:
            cmd.extend(['--title', args.title])

        if args.description:
            cmd.extend(['--description', args.description])

        # Add boolean flags
        for flag in ['draft', 'fill', 'web']:
            if getattr(args, flag, False):
                cmd.append(f'--{flag}')

        if args.assignee:
            for assignee in args.assignee:
                cmd.extend(['--assignee', assignee])

        if args.reviewer:
            for reviewer in args.reviewer:
                cmd.extend(['--reviewer', reviewer])

        if args.label:
            for label in args.label:
                cmd.extend(['--label', label])

        if args.milestone:
            cmd.extend(['--milestone', args.milestone])

        if args.target_branch:
            cmd.extend(['--target-branch', args.target_branch])

        if args.dry_run:
            print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            return 0

        # Execute command
        logger.debug("Executing: %s", ' '.join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        logger.info("✓ Merge request created")
        print(result.stdout.strip())
        return 0

    except subprocess.CalledProcessError as err:
        logger.error("Command failed: %s", err.stderr)
        return 1
    except (ValueError, OSError) as err:
        logger.error("Error: %s", err)
        return 1
